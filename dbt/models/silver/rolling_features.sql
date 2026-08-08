-- silver.rolling_features: post-match player snapshots.
--
-- ONE ROW PER (player_id, match_id): the post-match snapshot for that player's
-- completed match. snapshot_date is the match date. Downstream match_features
-- pairs each match with the player's immediately preceding snapshot
-- (player_match_number - 1), so the newest snapshot is immediately usable for
-- future inference.
--
-- All rolling windows are POST-MATCH and INCLUSIVE of the snapshot match: the
-- Nth snapshot's win_rate_10 is the average of match_won over the last 10
-- completed matches INCLUDING this one. Windows never include future matches.
--
-- Cold start: rates are computed over however many completed matches fall in
-- the window (the first snapshot's win_rate_10 is that match's own match_won).
-- There is NO zero-filling anywhere — ratio rates (ace_rate_10,
-- first_serve_pct_10, break_points_saved_pct_10, first_serve_win_pct_10,
-- second_serve_win_pct_10, serve_win_pct_10, return_points_won_pct_10,
-- df_rate_10, aces_per_svc_game_10)
-- are NULL when the window's denominator sum is 0, and surface rates are NULL
-- until the player has played on that surface. Honest averages over partial
-- windows are not zero-fills, so the plan's "no silent zero filling in the
-- training source" holds.
--
-- Surface-specific rates: clay_win_rate_10 / grass_win_rate_10 /
-- hard_win_rate_10 are computed independently over the last 10 matches ON that
-- surface (per (player_id, surface) partition, inclusive window — a
-- player_surface_matches CTE), then carried forward onto every snapshot row.
-- PostgreSQL has no LAST_VALUE IGNORE NULLS, so the carry-forward is
-- explicit: for each snapshot, a windowed conditional MAX picks the most
-- recent player_match_number with that surface at-or-before the snapshot
-- (surface_carry), and the per-surface rate CTE is joined back on that key.
-- This preserves the exact as-of semantics without correlated row-by-row
-- scans: the current match's surface rate is updated by this match; the other
-- two surfaces' rates are unchanged by it (this match is not on those
-- surfaces). A surface rate is NULL until the player's first match on that
-- surface.
--
-- streak is a SINGLE SIGNED current run: positive = consecutive wins ending at
-- and including this snapshot's match, negative = consecutive losses (the
-- mirror). player_match_number minus the player_match_number of the most recent
-- loss (0 if none) gives the win run; subtracting the most recent win's number
-- gives the loss run. A win pushes the value to +1/+2/..., a loss to -1/-2/...;
-- the first match is +1 if won, -1 if lost. The old separate win_streak /
-- loss_streak columns are removed (Task 6).
--
-- rank_trend_10 is NOT computed here. avg_player_rank_10 stores the player's
-- rolling average ranking; match_features derives rank trend by subtracting the
-- next match's current ranking from it. Unranked matches (ranking NULL after
-- the 0 -> NULL mapping in player_matches) are skipped by the AVG window — a
-- 10-match window containing unranked matches averages the known ranks, never a
-- bogus 0.
--
-- PostgreSQL division notes: SUM over SMALLINT columns returns BIGINT, and
-- BIGINT / BIGINT is INTEGER division (it truncates), so every ratio rate
-- casts its numerator to DOUBLE PRECISION — PostgreSQL would otherwise
-- silently truncate the floating-point rates. AVG-based rates (win_rate_10,
-- avg_player_rank_10, avg_rank_faced_10, surface rates) return NUMERIC, which
-- carries the same values exactly. The surface carry is the conditional-MAX +
-- join-back design described above.
--
-- Task 6 reductions: every `_5`/`_20` output, the separate win/loss streaks,
-- last_match_date (== snapshot_date by construction), and intermediate/source
-- outputs not consumed by gold.match_features or inference are removed.
-- days_since_last_match and matches_30d are computed on demand (not stored
-- here): match_features derives days from snapshot_date (365 on cold start),
-- and matches_30d is the current silver row's pre-match count. Current-match
-- serve/break analysis rates are derived on demand from bronze raw counts
-- (silver retains the raw counts), not stored here.

WITH player_surface_matches AS (
    -- One row per (player_id, match_id); the inclusive 10-match win rate on
    -- that row's own surface. player_match_number lets the carry join back
    -- onto each snapshot's carried key without another hop through silver.
    SELECT
        player_id,
        match_id,
        player_match_number,
        surface,
        AVG(match_won) OVER (
            PARTITION BY player_id, surface
            ORDER BY match_date, match_id
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        ) AS surface_win_rate_10
    FROM {{ ref('player_matches') }}
),
surface_carry AS (
    -- Per snapshot, the player_match_number of the most recent match on each
    -- surface at-or-before this snapshot (0 = none yet). player_match_number
    -- increases with (match_date, match_id), so MAX over the same unbounded
    -- frame the old LAST_VALUE used picks the exact same "latest at-or-before"
    -- row; ordering matches the snapshot ordering.
    SELECT
        player_id,
        match_id,
        MAX(CASE WHEN surface = 'clay'  THEN player_match_number ELSE 0 END) OVER (
            PARTITION BY player_id
            ORDER BY match_date, match_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS clay_last_match_number,
        MAX(CASE WHEN surface = 'grass' THEN player_match_number ELSE 0 END) OVER (
            PARTITION BY player_id
            ORDER BY match_date, match_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS grass_last_match_number,
        MAX(CASE WHEN surface = 'hard'  THEN player_match_number ELSE 0 END) OVER (
            PARTITION BY player_id
            ORDER BY match_date, match_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS hard_last_match_number
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
        pm.return_points_won,
        pm.return_points_available,
        sc.clay_last_match_number,
        sc.grass_last_match_number,
        sc.hard_last_match_number
    FROM {{ ref('player_matches') }} pm
    LEFT JOIN surface_carry sc
        ON sc.player_id = pm.player_id
       AND sc.match_id = pm.match_id
)
SELECT
    s.player_id,
    s.match_id,
    s.snapshot_date,
    s.player_match_number,
    s.surface,

    -- Latest observed ranking, rank points, and age (this match's values);
    -- opponent ranking is not kept (not a retained rolling/source value).
    s.player_ranking AS latest_player_ranking,
    s.player_rank_points AS latest_player_rank_points,
    s.player_age AS latest_player_age,

    -- Exponentially-decayed win rate over the last 10 matches: match_won
    -- weighted by POW(0.9, rn) where rn is the per-player reverse match
    -- ordinal (newest = 0, weight 1; each older match 0.9x the previous).
    -- The rn comes from the snapshots CTE; NULLIF guards the (impossible)
    -- empty window. POW is floating point, so no cast is needed here.
    SUM(s.match_won * POW(0.9, s.match_rn_rev)) OVER w10
        / NULLIF(SUM(POW(0.9, s.match_rn_rev)) OVER w10, 0) AS weighted_form_10,

    -- Rolling win rate over the last 10 matches, including this one
    AVG(s.match_won) OVER w10 AS win_rate_10,

    -- Ace rate: aces / first serves made, last 10 including this one. The
    -- numerator cast defeats PostgreSQL integer division (BIGINT/BIGINT
    -- truncates); the NULLIF zero-denominator convention is unchanged.
    CAST(SUM(s.aces) OVER w10 AS DOUBLE PRECISION)
        / NULLIF(SUM(s.first_serves_made) OVER w10, 0) AS ace_rate_10,

    -- First-serve percentage: first serves made / total serve points, 10
    CAST(SUM(s.first_serves_made) OVER w10 AS DOUBLE PRECISION)
        / NULLIF(SUM(s.total_serve_points) OVER w10, 0) AS first_serve_pct_10,

    -- Break-point save rate: break points saved / faced, last 10 incl. this
    CAST(SUM(s.break_points_saved) OVER w10 AS DOUBLE PRECISION)
        / NULLIF(SUM(s.break_points_faced) OVER w10, 0) AS break_points_saved_pct_10,

    -- First-serve win rate: first-serve points won / first serves made, 10
    CAST(SUM(s.first_serve_points_won) OVER w10 AS DOUBLE PRECISION)
        / NULLIF(SUM(s.first_serves_made) OVER w10, 0) AS first_serve_win_pct_10,

    -- Second-serve win rate: second-serve points won / second serves made
    -- (total serve points minus first serves made), 10
    CAST(SUM(s.second_serve_points_won) OVER w10 AS DOUBLE PRECISION)
        / NULLIF(SUM(s.total_serve_points - s.first_serves_made) OVER w10, 0)
        AS second_serve_win_pct_10,

    -- Serve win rate: (first + second serve points won) / total serve points, 10
    CAST(SUM(s.first_serve_points_won + s.second_serve_points_won) OVER w10 AS DOUBLE PRECISION)
        / NULLIF(SUM(s.total_serve_points) OVER w10, 0) AS serve_win_pct_10,

    -- Return-points-won rate: opponent serve points NOT won / opponent serve
    -- points, last 10 incl. this one. The opponent's serve points not won are
    -- exactly this player's return points won (return_points_won in
    -- player_matches); the denominator is the opponent's total serve points
    -- (return_points_available). Same NULLIF zero-denominator convention as
    -- the other rates.
    CAST(SUM(s.return_points_won) OVER w10 AS DOUBLE PRECISION)
        / NULLIF(SUM(s.return_points_available) OVER w10, 0)
        AS return_points_won_pct_10,

    -- Double-fault rate: double faults / total serve points, 10 (same
    -- NULLIF convention as the other serve rates — NULL when the window has
    -- no serve points)
    CAST(SUM(s.double_faults) OVER w10 AS DOUBLE PRECISION)
        / NULLIF(SUM(s.total_serve_points) OVER w10, 0) AS df_rate_10,

    -- Aces per service game, 10
    CAST(SUM(s.aces) OVER w10 AS DOUBLE PRECISION)
        / NULLIF(SUM(s.service_games) OVER w10, 0) AS aces_per_svc_game_10,

    -- Rolling average player rank over the last 10 (rank_trend derived
    -- downstream in match_features, not here)
    AVG(s.player_ranking) OVER w10 AS avg_player_rank_10,

    -- Rolling average opponent rank (strength of schedule) over the last 10;
    -- AVG skips NULL opponent rankings, so unranked opponents inside the
    -- window never pollute the average
    AVG(s.opponent_ranking) OVER w10 AS avg_rank_faced_10,

    -- Single signed current run including this match: positive on a win run,
    -- negative on a loss run. win run = player_match_number - most recent loss
    -- number (0 if none); loss run = player_match_number - most recent win
    -- number (0 if none). The sign selects which.
    CASE WHEN s.match_won = 1 THEN
        s.player_match_number - (
            MAX(CASE WHEN s.match_won = 0 THEN s.player_match_number ELSE 0 END) OVER (
                PARTITION BY s.player_id ORDER BY s.snapshot_date, s.match_id
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            )
        )
    ELSE
        -1 * (
            s.player_match_number - (
                MAX(CASE WHEN s.match_won = 1 THEN s.player_match_number ELSE 0 END) OVER (
                    PARTITION BY s.player_id ORDER BY s.snapshot_date, s.match_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                )
            )
        )
    END AS streak,

    -- Surface-specific win rates over the last 10 matches on that surface,
    -- carried forward so every snapshot has all three values (see header):
    -- the carried player_match_number (surface_carry) is joined back to the
    -- per-surface rate CTE. 0 carried keys join nothing -> NULL (cold start
    -- on that surface). The current snapshot's own match contributes to its
    -- own surface's rate; the other surfaces keep their prior value.
    psm_clay.surface_win_rate_10  AS clay_win_rate_10,
    psm_grass.surface_win_rate_10 AS grass_win_rate_10,
    psm_hard.surface_win_rate_10  AS hard_win_rate_10

FROM snapshots s
LEFT JOIN player_surface_matches psm_clay
    ON psm_clay.player_id = s.player_id
   AND psm_clay.surface = 'clay'
   AND psm_clay.player_match_number = s.clay_last_match_number
LEFT JOIN player_surface_matches psm_grass
    ON psm_grass.player_id = s.player_id
   AND psm_grass.surface = 'grass'
   AND psm_grass.player_match_number = s.grass_last_match_number
LEFT JOIN player_surface_matches psm_hard
    ON psm_hard.player_id = s.player_id
   AND psm_hard.surface = 'hard'
   AND psm_hard.player_match_number = s.hard_last_match_number
WINDOW
    w10 AS (PARTITION BY s.player_id ORDER BY s.snapshot_date, s.match_id
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW)
ORDER BY s.snapshot_date, s.match_id, s.player_id
