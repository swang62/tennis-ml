-- gold.rolling_features: post-match player snapshots.
--
-- ONE ROW PER (player_id, match_id): the post-match snapshot for that player's
-- completed match. snapshot_date is the match date. Downstream match_features
-- pairs each match with the player's immediately preceding snapshot
-- (player_match_number - 1), so the newest snapshot is immediately usable for
-- future inference.
--
-- All rolling windows are POST-MATCH and INCLUSIVE of the snapshot match: the
-- Nth snapshot's win_rate_5 is the average of match_won over the last 5
-- completed matches INCLUDING this one. Windows never include future matches.
--
-- Cold start: rates are computed over however many completed matches fall in
-- the window (the first snapshot's win_rate_5 is that match's own match_won).
-- There is NO zero-filling anywhere — ratio rates (ace_rate_*,
-- first_serve_pct_*, break_points_saved_pct_*, first_serve_win_pct_*,
-- second_serve_win_pct_*, serve_win_pct_*, df_rate_*, aces_per_svc_game_*) are
-- NULL when the window's denominator sum is 0, and surface rates are NULL until the
-- player has played on that surface. Honest averages over partial windows are
-- not zero-fills, so the plan's "no silent zero filling in the training
-- source" holds.
--
-- Surface-specific rates: clay_win_rate_10 / grass_win_rate_10 /
-- hard_win_rate_10 are computed independently over the last 10 matches ON that
-- surface (per (player_id, surface) partition, inclusive window — a
-- player_surface_matches CTE), then carried forward onto every snapshot row:
-- LAST_VALUE(CASE WHEN surface = 'clay' THEN rate END) IGNORE NULLS over the
-- player's whole history picks out the most recent clay rate at-or-before this
-- snapshot date. The current match's surface rate is therefore updated by this
-- match; the other two surfaces' rates are unchanged by it (this match is not
-- on those surfaces). A surface rate is NULL until the player's first match on
-- that surface.
--
-- win_streak = consecutive wins ending at and including this snapshot's match:
-- player_match_number minus the player_match_number of the most recent loss
-- (0 if none), same definition shape as the old match_features.sql formula
-- (`rn - max(streak_start where lost)`) but with the window INCLUDING the
-- current match. A loss resets it to 0 (never negative); the first match is
-- 1 if won, 0 if lost.
--
-- loss_streak is the exact mirror: consecutive losses ending at and including
-- this snapshot's match (player_match_number minus the most recent win's
-- number); a win resets it to 0. The two streaks are mutually exclusive by
-- construction — never both positive on one snapshot.
--
-- last_match_date = snapshot_date: the most recent completed match date in the
-- rolling history IS this match (windows are inclusive), so the plan's "last
-- completed match date" is simply the snapshot date itself.
--
-- weighted_form_10 is the same last-10-match window with exponential decay:
-- match_won weighted by POW(0.9, rn) where rn counts backwards from the newest
-- match (newest = 0, weight 1; each older match decays by 0.9 per step). The
-- reverse row number is computed once per player in the snapshots CTE (a
-- window cannot reference another window's row number) and reweighted per
-- window. Cold-start partial windows use whatever weights fall inside the
-- window; the ratio's NULLIF guards the (impossible) empty window.
--
-- rank_trend_10/20 are NOT computed here. avg_player_rank_10/20 store the
-- player's rolling average ranking; match_features derives rank trend by
-- subtracting the next match's current ranking from these. Unranked matches
-- (ranking NULL after the 0 -> NULL mapping in player_matches) are skipped by
-- the AVG window — a 10-match window containing 3 unranked matches averages
-- the 7 known ranks, never a bogus 0.

WITH player_surface_matches AS (
    -- One row per (player_id, match_id); the inclusive 10-match win rate on
    -- that row's own surface.
    SELECT
        player_id,
        match_id,
        surface,
        AVG(match_won) OVER (
            PARTITION BY player_id, surface
            ORDER BY match_date, match_id
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        ) AS surface_win_rate_10
    FROM {{ ref('player_matches') }}
),
snapshots AS (
    SELECT
        pm.player_id,
        pm.match_id,
        pm.match_date AS snapshot_date,
        pm.player_match_number,
        -- Reverse match ordinal per player (newest = 0) for weighted_form_10's
        -- exponential decay. Computed here in the CTE, not inside the window:
        -- a window can reference CTE columns but never another window's row
        -- number.
        ROW_NUMBER() OVER (
            PARTITION BY pm.player_id
            ORDER BY pm.match_date DESC, pm.match_id DESC
        ) - 1 AS match_rn_rev,
        pm.surface,
        pm.player_ranking,
        pm.opponent_ranking,
        pm.player_rank_points,
        pm.player_age,
        pm.match_won,
        pm.aces,
        pm.double_faults,
        pm.first_serves_made,
        pm.total_serve_points,
        pm.first_serve_points_won,
        pm.second_serve_points_won,
        pm.service_games,
        pm.break_points_saved,
        pm.break_points_faced,
        ps.surface AS rate_surface,
        ps.surface_win_rate_10
    FROM {{ ref('player_matches') }} pm
    LEFT JOIN player_surface_matches ps
        ON ps.player_id = pm.player_id
       AND ps.match_id = pm.match_id
)
SELECT
    player_id,
    match_id,
    snapshot_date,
    player_match_number,
    surface,

    -- Latest observed ranking, opponent ranking, rank points, and age (this
    -- match's values)
    player_ranking AS latest_player_ranking,
    opponent_ranking AS latest_opponent_ranking,
    player_rank_points AS latest_player_rank_points,
    player_age AS latest_player_age,

    -- Rolling win rates over the last 5/10/20 matches, including this one
    AVG(match_won) OVER w5  AS win_rate_5,
    AVG(match_won) OVER w10 AS win_rate_10,
    AVG(match_won) OVER w20 AS win_rate_20,

    -- Exponentially-decayed win rate over the last 10 matches: match_won
    -- weighted by POW(0.9, rn) where rn is the per-player reverse match
    -- ordinal (newest = 0, weight 1; each older match 0.9x the previous).
    -- The rn comes from the snapshots CTE; NULLIF guards the (impossible)
    -- empty window.
    SUM(match_won * POW(0.9, match_rn_rev)) OVER w10
        / NULLIF(SUM(POW(0.9, match_rn_rev)) OVER w10, 0) AS weighted_form_10,

    -- Ace rate: aces / first serves made, last 5/10 including this one
    SUM(aces) OVER w5  / NULLIF(SUM(first_serves_made) OVER w5,  0) AS ace_rate_5,
    SUM(aces) OVER w10 / NULLIF(SUM(first_serves_made) OVER w10, 0) AS ace_rate_10,

    -- First-serve percentage: first serves made / total serve points, 5/10
    SUM(first_serves_made) OVER w5
        / NULLIF(SUM(total_serve_points) OVER w5,  0) AS first_serve_pct_5,
    SUM(first_serves_made) OVER w10
        / NULLIF(SUM(total_serve_points) OVER w10, 0) AS first_serve_pct_10,

    -- Break-point save rate: break points saved / faced, last 5/10 incl. this one
    SUM(break_points_saved) OVER w5
        / NULLIF(SUM(break_points_faced) OVER w5,  0) AS break_points_saved_pct_5,
    SUM(break_points_saved) OVER w10
        / NULLIF(SUM(break_points_faced) OVER w10, 0) AS break_points_saved_pct_10,

    -- First-serve win rate: first-serve points won / first serves made, 5/10
    SUM(first_serve_points_won) OVER w5
        / NULLIF(SUM(first_serves_made) OVER w5,  0) AS first_serve_win_pct_5,
    SUM(first_serve_points_won) OVER w10
        / NULLIF(SUM(first_serves_made) OVER w10, 0) AS first_serve_win_pct_10,

    -- Second-serve win rate: second-serve points won / second serves made
    -- (total serve points minus first serves made), 5/10
    SUM(second_serve_points_won) OVER w5
        / NULLIF(
            SUM(total_serve_points - first_serves_made) OVER w5,  0
        ) AS second_serve_win_pct_5,
    SUM(second_serve_points_won) OVER w10
        / NULLIF(
            SUM(total_serve_points - first_serves_made) OVER w10, 0
        ) AS second_serve_win_pct_10,

    -- Serve win rate: (first + second serve points won) / total serve points, 5/10
    SUM(first_serve_points_won + second_serve_points_won) OVER w5
        / NULLIF(SUM(total_serve_points) OVER w5,  0) AS serve_win_pct_5,
    SUM(first_serve_points_won + second_serve_points_won) OVER w10
        / NULLIF(SUM(total_serve_points) OVER w10, 0) AS serve_win_pct_10,

    -- Double-fault rate: double faults / total serve points, 5/10 (same
    -- NULLIF convention as the other serve rates — NULL when the window has
    -- no serve points)
    SUM(double_faults) OVER w5
        / NULLIF(SUM(total_serve_points) OVER w5,  0) AS df_rate_5,
    SUM(double_faults) OVER w10
        / NULLIF(SUM(total_serve_points) OVER w10, 0) AS df_rate_10,

    -- Aces per service game, 5/10
    SUM(aces) OVER w5
        / NULLIF(SUM(service_games) OVER w5,  0) AS aces_per_svc_game_5,
    SUM(aces) OVER w10
        / NULLIF(SUM(service_games) OVER w10, 0) AS aces_per_svc_game_10,

    -- Rolling average player rank over the last 10/20 (rank_trend derived
    -- downstream in match_features, not here)
    AVG(player_ranking) OVER w10 AS avg_player_rank_10,
    AVG(player_ranking) OVER w20 AS avg_player_rank_20,

    -- Rolling average opponent rank (strength of schedule) over the last 5/10;
    -- AVG skips NULL opponent rankings, so unranked opponents inside the
    -- window never pollute the average
    AVG(opponent_ranking) OVER w5  AS avg_rank_faced_5,
    AVG(opponent_ranking) OVER w10 AS avg_rank_faced_10,

    -- Current consecutive-wins run including this match; 0 on a loss
    player_match_number - (
        MAX(CASE WHEN match_won = 0 THEN player_match_number ELSE 0 END) OVER (
            PARTITION BY player_id ORDER BY snapshot_date, match_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )
    ) AS win_streak,

    -- Current consecutive-losses run including this match; 0 on a win. Exact
    -- mirror of win_streak (most recent win resets it): the two can never be
    -- positive on the same snapshot.
    player_match_number - (
        MAX(CASE WHEN match_won = 1 THEN player_match_number ELSE 0 END) OVER (
            PARTITION BY player_id ORDER BY snapshot_date, match_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )
    ) AS loss_streak,

    -- Last completed match date = this snapshot's date (windows are inclusive)
    snapshot_date AS last_match_date,

    -- Surface-specific win rates over the last 10 matches on that surface,
    -- carried forward so every snapshot has all three values (see header).
    LAST_VALUE(CASE WHEN rate_surface = 'clay'  THEN surface_win_rate_10 END IGNORE NULLS)
        OVER (
            PARTITION BY player_id ORDER BY snapshot_date, match_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS clay_win_rate_10,
    LAST_VALUE(CASE WHEN rate_surface = 'grass' THEN surface_win_rate_10 END IGNORE NULLS)
        OVER (
            PARTITION BY player_id ORDER BY snapshot_date, match_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS grass_win_rate_10,
    LAST_VALUE(CASE WHEN rate_surface = 'hard'  THEN surface_win_rate_10 END IGNORE NULLS)
        OVER (
            PARTITION BY player_id ORDER BY snapshot_date, match_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS hard_win_rate_10

FROM snapshots
WINDOW
    w5  AS (PARTITION BY player_id ORDER BY snapshot_date, match_id
            ROWS BETWEEN 4  PRECEDING AND CURRENT ROW),
    w10 AS (PARTITION BY player_id ORDER BY snapshot_date, match_id
            ROWS BETWEEN 9  PRECEDING AND CURRENT ROW),
    w20 AS (PARTITION BY player_id ORDER BY snapshot_date, match_id
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
ORDER BY snapshot_date, match_id, player_id
