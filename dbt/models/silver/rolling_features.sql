-- silver.rolling_features: post-match player snapshots.
--
-- One post-match snapshot per player and match; downstream reads the prior one.
--
-- Windows are post-match, inclusive, and never include future matches.
--
-- Partial windows use available matches. Rates remain NULL for zero denominators
-- or unseen surfaces; the training source never silently zero-fills.
--
-- Per-surface windows are carried forward with conditional MAX because
-- PostgreSQL lacks LAST_VALUE IGNORE NULLS.
--
-- Streak is the signed current win/loss run, including this match.
--
-- match_features derives rank trend; AVG skips unranked NULL values.
--
-- Cast SUM numerators to double precision to avoid PostgreSQL integer division.
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
    -- Inclusive 10-match rate on each match's own surface.
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
    -- Latest match number per surface at each snapshot (0 if unseen).
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
        -- Reverse ordinal for weighted-form decay; window results cannot nest.
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
    s.player_age AS latest_player_age,

    -- Exponentially-decayed 10-match win rate; newest weight is 1.
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

    -- Return points won over opponent serve points, with the standard NULLIF guard.
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

    -- Signed current win/loss run.
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

    -- Per-surface rates carried forward from the latest surface match.
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
)
-- Trim to the affected players' full histories; every window above was
-- evaluated over the full history, so these rows carry exactly the values a
-- full rebuild gives and the temp relation holds unique (player_id, match_id)s.
SELECT * FROM computed
{% if is_incremental() %}
WHERE player_id IN (SELECT player_id FROM changed_players)
{% endif %}
ORDER BY snapshot_date, match_id, player_id
