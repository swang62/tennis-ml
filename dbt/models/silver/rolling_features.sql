-- silver.rolling_features: post-match player snapshots.
--
-- One post-match snapshot per player and match; downstream reads the prior one.
--
-- Windows are post-match, inclusive, and never include future matches.
--
-- Partial windows use available matches. Every [0,1] probability rate is
-- smoothed with a fixed Beta(1,1) prior, (successes + 1) / (opportunities + 2),
-- so a first-match window yields the neutral 0.5 and a zero-opportunity window
-- is never NULL; `matches_10` exposes how many matches actually back the
-- 10-match rates (1 for the first match, up to 10). Surface rates are smoothed
-- the same way; they remain NULL for unseen surfaces (no surface match yet).
-- dominance_10 is the ratio of two smoothed rates,
-- return_points_won_pct_10 / (1 - serve_win_pct_10), expressing return
-- strength per unit of serve weakness; both inputs are strictly below 1 (the
-- Beta(1,1) smoothing caps the smoothed serve/return rates at < 1), so the
-- denominator is never zero and dominance_10 is never NULL. It is a ratio, not
-- a probability, so it is unbounded above.
-- aces_per_svc_game_10, weighted_form_10, streak, game_margin_10, and the
-- rank averages are NOT probabilities and are deliberately left unsmoothed.
-- game_margin_10 is the rolling average per-match game margin (games won minus
-- games lost) parsed from the winner-first bronze score: each completed-set
-- token "a-b" contributes a - b, so walkover/retirement tokens and missing
-- scores contribute nothing (a retired match counts only its completed sets).
-- The training source never silently zero-fills.
--
-- Per-surface windows are carried forward with conditional MAX because
-- PostgreSQL lacks LAST_VALUE IGNORE NULLS.
--
-- Query shape: the rolling 10-match rates and the signed streak share one
-- player-ordered window pass; the per-surface carries ride the snapshot
-- builder's own player-ordered pass; surface rates use their own (player,
-- surface) window; and the weighted-form decay derives its reversed exponent
-- from the player's max match ordinal, so no descending sort is needed.
--
-- Streak is the signed current win/loss run, including this match.
--
-- match_features derives rank trend; AVG skips unranked NULL values.
--
-- The smoothed rates use `+ 1.0`/`+ 2.0` numeric literals, so no integer
-- division applies; only the unsmoothed aces_per_svc_game_10 keeps its
-- explicit CAST + NULLIF guard.
--
-- Only gold/inference inputs remain; activity and current-match rates are derived on demand.
--
-- Incremental boundary: affected-player rebuilds, not append-only. A snapshot
-- depends on matches up to its own (match_date, match_id), so a historical
-- bronze insert that lands before existing matches changes the rolling values
-- of every later snapshot for that player. A run therefore recomputes the FULL
-- history and returns every row of each player whose (player_id, match_id) set
-- differs from the target; the delete+insert on the composite unique key then
-- replaces all stale snapshots for those players in place.
--
-- The window CTEs are always evaluated over the FULL player_matches history, so
-- each returned snapshot carries exactly the values a full rebuild gives.
-- player_matches is scanned once by the snapshot builder and once (indexed) by
-- the surface-rate windows; the carries ride the snapshot window pass.

{{ config(
    materialized="incremental",
    incremental_strategy="delete+insert",
    unique_key=["player_id", "match_id"],
) }}

WITH
{% if is_incremental() %}
-- Affected players have a missing snapshot or a changed ordinal. Checking the
-- ordinal makes the model recover when player_matches committed successfully
-- but a later model/test failed before rolling_features could be rebuilt.
changed_players AS (
    SELECT DISTINCT pm.player_id
    FROM {{ ref('player_matches') }} pm
    LEFT JOIN {{ this }} t
        ON t.player_id = pm.player_id
       AND t.match_id = pm.match_id
    WHERE t.match_id IS NULL
       OR t.player_match_number <> pm.player_match_number
),
{% endif %}
player_surface_matches AS (
    -- Inclusive 10-match rate on each match's own surface, Beta(1,1)-smoothed:
    -- (SUM(match_won) + 1) / (COUNT(*) + 2); a first surface match yields 0.5.
    SELECT
        player_id,
        match_id,
        player_match_number,
        surface,
        (SUM(match_won) OVER (
            PARTITION BY player_id, surface
            ORDER BY match_date, match_id
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        ) + 1.0) / (COUNT(*) OVER (
            PARTITION BY player_id, surface
            ORDER BY match_date, match_id
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        ) + 2.0) AS surface_win_rate_10
    FROM {{ ref('player_matches') }}
),
-- Winner-perspective per-match game margin parsed from the bronze score.
-- Tiebreak digits were stripped at ingest; every completed-set token "6-4"
-- contributes a - b, non-set tokens (W/O, RET) and missing scores are skipped.
-- The winner-first orientation is the ingest contract, so the sign follows
-- the row's match_won in snapshots below.
match_game_margins AS MATERIALIZED (
    SELECT
        sets.match_id,
        SUM(sets.winner_games - sets.loser_games) AS winner_game_margin
    FROM (
        SELECT
            m.match_id,
            split_part(t.token, '-', 1)::INT AS winner_games,
            split_part(t.token, '-', 2)::INT AS loser_games
        FROM {{ source('bronze', 'match_events') }} m
        CROSS JOIN LATERAL regexp_split_to_table(m.score, '\s+') AS t(token)
        WHERE m.score IS NOT NULL
          AND m.score <> ''
          AND t.token ~ '^[0-9]+-[0-9]+$'
    ) sets
    GROUP BY sets.match_id
),
-- Every snapshot is computed here over the FULL player_matches history:
-- window values and surface carries for a row depend on all of a player's
-- matches up to that row, so filtering earlier would silently corrupt the
-- new rows' values. The incremental filter is applied only in the outermost
-- SELECT, after every window has been evaluated.
snapshots AS (
    SELECT
        pm.player_id,
        pm.match_id,
        pm.match_date AS snapshot_date,
        pm.player_match_number,
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
        -- Game margin in this match's player perspective: winner-first score
        -- signed by the perspective's result (winner +, loser -).
        CASE WHEN pm.match_won = 1 THEN mgm.winner_game_margin
             ELSE -mgm.winner_game_margin END AS game_margin,
        -- Latest match number per surface at each snapshot (0 if unseen),
        -- carried forward with conditional MAX because PostgreSQL lacks
        -- LAST_VALUE IGNORE NULLS. Computed here, not in a separate scan of
        -- player_matches, so the surface-rate join keys exist as plain columns.
        MAX(CASE WHEN pm.surface = 'clay'  THEN pm.player_match_number ELSE 0 END) OVER (
            PARTITION BY pm.player_id
            ORDER BY pm.match_date, pm.match_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS clay_last_match_number,
        MAX(CASE WHEN pm.surface = 'grass' THEN pm.player_match_number ELSE 0 END) OVER (
            PARTITION BY pm.player_id
            ORDER BY pm.match_date, pm.match_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS grass_last_match_number,
        MAX(CASE WHEN pm.surface = 'hard'  THEN pm.player_match_number ELSE 0 END) OVER (
            PARTITION BY pm.player_id
            ORDER BY pm.match_date, pm.match_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS hard_last_match_number,
        -- Last loss/win match numbers feed the signed streak below; they share
        -- the same full-history window pass as the surface carries.
        MAX(CASE WHEN pm.match_won = 0 THEN pm.player_match_number ELSE 0 END) OVER (
            PARTITION BY pm.player_id
            ORDER BY pm.match_date, pm.match_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS last_loss_match_number,
        MAX(CASE WHEN pm.match_won = 1 THEN pm.player_match_number ELSE 0 END) OVER (
            PARTITION BY pm.player_id
            ORDER BY pm.match_date, pm.match_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS last_win_match_number
    FROM {{ ref('player_matches') }} pm
    LEFT JOIN match_game_margins mgm
        ON mgm.match_id = pm.match_id
),
computed AS (
SELECT
    s.player_id,
    s.match_id,
    s.snapshot_date,
    s.player_match_number,
    s.surface,

    -- Current ranking, rank points, and age.
    s.player_ranking AS latest_player_ranking,
    s.player_rank_points AS latest_player_rank_points,
    TRUNC(s.player_age::NUMERIC, 5)::DOUBLE PRECISION AS latest_player_age,

    -- Exponentially-decayed 10-match win rate; newest weight is 1. The decay
    -- weight per row is precomputed (match_decay) with an exponent equal to
    -- the reverse row number, so the frame stays ascending and needs no
    -- reversed sort pass; the values are identical to the original
    -- POW(0.9, reverse_row_number) formulation.
    TRUNC((SUM(s.match_won * s.match_decay) OVER w10
        / NULLIF(SUM(s.match_decay) OVER w10, 0))::NUMERIC, 5)::DOUBLE PRECISION
        AS weighted_form_10,

    -- Rolling win rate over the last 10 matches, including this one, smoothed
    -- with the fixed Beta(1,1) prior: (wins + 1) / (matches + 2). The first
    -- match yields the neutral 0.5, never 0 or 1.
    TRUNC(((SUM(s.match_won) OVER w10 + 1.0) / (COUNT(*) OVER w10 + 2.0))::NUMERIC, 5)
        AS win_rate_10,

    -- Ace rate: (aces + 1) / (first serves made + 2), last 10 incl. this one.
    -- Smoothed so a zero-opportunity window is never NULL (>= 2 denominator).
    TRUNC(((SUM(s.aces) OVER w10 + 1.0) / (SUM(s.first_serves_made) OVER w10 + 2.0))::NUMERIC, 5)
        AS ace_rate_10,

    -- First-serve percentage: (first serves made + 1) / (total serve points + 2)
    TRUNC(((SUM(s.first_serves_made) OVER w10 + 1.0)
        / (SUM(s.total_serve_points) OVER w10 + 2.0))::NUMERIC, 5)
        AS first_serve_pct_10,

    -- Break-point save rate: (saved + 1) / (faced + 2), last 10 incl. this one
    TRUNC(((SUM(s.break_points_saved) OVER w10 + 1.0)
        / (SUM(s.break_points_faced) OVER w10 + 2.0))::NUMERIC, 5)
        AS break_points_saved_pct_10,

    -- First-serve win rate: (first-serve points won + 1) / (first serves made + 2)
    TRUNC(((SUM(s.first_serve_points_won) OVER w10 + 1.0)
        / (SUM(s.first_serves_made) OVER w10 + 2.0))::NUMERIC, 5)
        AS first_serve_win_pct_10,

    -- Second-serve win rate: (second-serve points won + 1) / (second serves
    -- made + 2), where second serves made = total serve points - first serves
    TRUNC(((SUM(s.second_serve_points_won) OVER w10 + 1.0)
        / (SUM(s.total_serve_points - s.first_serves_made) OVER w10 + 2.0))::NUMERIC, 5)
        AS second_serve_win_pct_10,

    -- Serve win rate: ((first + second serve points won) + 1) / (total + 2)
    TRUNC(((SUM(s.first_serve_points_won + s.second_serve_points_won) OVER w10 + 1.0)
        / (SUM(s.total_serve_points) OVER w10 + 2.0))::NUMERIC, 5)
        AS serve_win_pct_10,

    -- Return points won: (return_points_won + 1) / (return_points_available + 2)
    TRUNC(((SUM(s.return_points_won) OVER w10 + 1.0)
        / (SUM(s.return_points_available) OVER w10 + 2.0))::NUMERIC, 5)
        AS return_points_won_pct_10,

    -- Double-fault rate: (double faults + 1) / (total serve points + 2)
    TRUNC(((SUM(s.double_faults) OVER w10 + 1.0)
        / (SUM(s.total_serve_points) OVER w10 + 2.0))::NUMERIC, 5)
        AS df_rate_10,

    -- Number of matches in the last-10 window backing the smoothed rates;
    -- 1 for the first match, up to 10. Carried to gold as matches_10 exposure.
    COUNT(*) OVER w10 AS matches_10,

    -- Aces per service game, 10
    TRUNC((CAST(SUM(s.aces) OVER w10 AS DOUBLE PRECISION)
        / NULLIF(SUM(s.service_games) OVER w10, 0))::NUMERIC, 5)::DOUBLE PRECISION
        AS aces_per_svc_game_10,

    -- Rolling average per-match game margin over the last 10 incl. this one.
    -- AVG skips matches without a parseable score (NULL game_margin); NULL
    -- until the player's first match with a completed-set score.
    TRUNC((AVG(s.game_margin) OVER w10)::NUMERIC, 5) AS game_margin_10,

    -- Rolling average player rank over the last 10 (rank_trend derived
    -- downstream in match_features, not here)
    TRUNC((AVG(s.player_ranking) OVER w10)::NUMERIC, 5) AS avg_player_rank_10,

    -- Rolling average opponent rank (strength of schedule) over the last 10;
    -- AVG skips NULL opponent rankings, so unranked opponents inside the
    -- window never pollute the average
    TRUNC((AVG(s.opponent_ranking) OVER w10)::NUMERIC, 5) AS avg_rank_faced_10,

    -- Signed current win/loss run, computed in the FROM subquery from the
    -- snapshot pass's last-loss/last-win ordinals.
    s.streak,

    -- Per-surface rates carried forward from the latest surface match.
    TRUNC(psm_clay.surface_win_rate_10::NUMERIC, 5)  AS clay_win_rate_10,
    TRUNC(psm_grass.surface_win_rate_10::NUMERIC, 5) AS grass_win_rate_10,
    TRUNC(psm_hard.surface_win_rate_10::NUMERIC, 5)  AS hard_win_rate_10

FROM (
    -- Player max match ordinal (the constant needed for the weighted-form
    -- decay; a partition-only window, so it needs no sort) plus the signed
    -- streak built from the full-history pass's last-loss/last-win ordinals.
    -- The join keys for the surface-rate joins below must live here: window
    -- results are not visible to the FROM/JOIN clauses that use them.
    SELECT s.*,
        MAX(s.player_match_number) OVER (PARTITION BY s.player_id)
            AS player_max_match_number,
        -- Precomputed decay weight so the window pass only sums, not pow()s
        -- over and over for each frame row.
        POW(0.9, MAX(s.player_match_number) OVER (PARTITION BY s.player_id)
            - s.player_match_number) AS match_decay,
        CASE WHEN s.match_won = 1
            THEN s.player_match_number - s.last_loss_match_number
            ELSE -1 * (s.player_match_number - s.last_win_match_number)
        END AS streak
    FROM snapshots s
) s
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
),
-- dominance_10: return strength per unit of serve weakness. Derived from the
-- two smoothed rates above (never the raw window sums) so the ratio matches
-- what gold/inference recompute from the stored rates cell for cell.
dominance AS (
    SELECT
        *,
        TRUNC((return_points_won_pct_10 / (1.0 - serve_win_pct_10))::NUMERIC, 5)
            AS dominance_10
    FROM computed
)
-- Trim to the affected players' full histories; every window above was
-- evaluated over the full history, so these rows carry exactly the values a
-- full rebuild gives and the temp relation holds unique (player_id, match_id)s.
SELECT * FROM dominance
{% if is_incremental() %}
WHERE player_id IN (SELECT player_id FROM changed_players)
{% endif %}
ORDER BY snapshot_date, match_id, player_id
