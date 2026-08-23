-- silver.rolling_features: post-match player snapshots; downstream reads the
-- prior one. Windows are post-match and never include future matches.
-- Incremental: affected-player rebuild (delete+insert replaces stale snapshots).
-- [0,1] rates are Beta(1,1)-smoothed so a first-match window = 0.5 and a
-- zero-opportunity window is never NULL; surface rates stay NULL for unseen surfaces.
-- dominance is a lifetime Dominance Ratio, not a 10-match window.
-- aces_per_svc_game_10, weighted_form_10, streak, and rank averages are unsmoothed.
-- Per-surface windows carry forward via conditional MAX; streak is the signed run.

{{ config(
    materialized="incremental",
    incremental_strategy="delete+insert",
    unique_key=["player_id", "match_id"],
) }}

WITH
{% if is_incremental() %}
-- Affected players have a missing or changed-ordinal snapshot; checking the ordinal
-- recovers when player_matches committed but rolling_features didn't.
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
    -- Inclusive 10-match surface win rate, Beta(1,1)-smoothed.
    SELECT
        player_id,
        match_id,
        player_match_number,
        surface,
        (SUM(match_won) OVER (
            PARTITION BY player_id, surface
            ORDER BY match_date, match_num
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        ) + 1.0) / (COUNT(*) OVER (
            PARTITION BY player_id, surface
            ORDER BY match_date, match_num
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        ) + 2.0) AS surface_win_rate_10
    FROM {{ ref('player_matches') }}
),
snapshots AS (
    SELECT
        pm.player_id,
        pm.match_id,
        pm.match_num,
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
        -- Latest ordinal in full history; the gap to this row feeds the decay.
        MAX(pm.player_match_number) OVER (PARTITION BY pm.player_id) AS player_max_match_number,
        -- Latest ordinal per surface; carried forward via conditional MAX.
        MAX(CASE WHEN pm.surface = 'clay'  THEN pm.player_match_number ELSE 0 END) OVER (
            PARTITION BY pm.player_id
            ORDER BY pm.match_date, pm.match_num
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS clay_last_match_number,
        MAX(CASE WHEN pm.surface = 'grass' THEN pm.player_match_number ELSE 0 END) OVER (
            PARTITION BY pm.player_id
            ORDER BY pm.match_date, pm.match_num
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS grass_last_match_number,
        MAX(CASE WHEN pm.surface = 'hard'  THEN pm.player_match_number ELSE 0 END) OVER (
            PARTITION BY pm.player_id
            ORDER BY pm.match_date, pm.match_num
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS hard_last_match_number,
        -- Last loss/win ordinals feed the signed streak.
        MAX(CASE WHEN pm.match_won = 0 THEN pm.player_match_number ELSE 0 END) OVER (
            PARTITION BY pm.player_id
            ORDER BY pm.match_date, pm.match_num
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS last_loss_match_number,
        MAX(CASE WHEN pm.match_won = 1 THEN pm.player_match_number ELSE 0 END) OVER (
            PARTITION BY pm.player_id
            ORDER BY pm.match_date, pm.match_num
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS last_win_match_number
    FROM {{ ref('player_matches') }} pm
),
computed AS (
SELECT
    s.player_id,
    s.match_id,
    s.match_num,
    s.snapshot_date,
    s.player_match_number,
    s.surface,

    s.player_ranking AS latest_player_ranking,
    s.player_rank_points AS latest_player_rank_points,
    s.player_age AS latest_player_age,

    -- Decay exponent is the ordinal gap to the player's max (newest = 0).
    SUM(s.match_won * POW(0.9, s.player_max_match_number - s.player_match_number)) OVER w10
        / NULLIF(SUM(POW(0.9, s.player_max_match_number - s.player_match_number)) OVER w10, 0)
        AS weighted_form_10,

    -- Beta(1,1)-smoothed 10-match rates: (x + 1) / (denom + 2).
    (SUM(s.match_won) OVER w10 + 1.0) / (COUNT(*) OVER w10 + 2.0)
        AS win_rate_10,
    (SUM(s.aces) OVER w10 + 1.0) / (SUM(s.first_serves_made) OVER w10 + 2.0)
        AS ace_rate_10,
    (SUM(s.first_serves_made) OVER w10 + 1.0)
        / (SUM(s.total_serve_points) OVER w10 + 2.0)
        AS first_serve_pct_10,
    (SUM(s.break_points_saved) OVER w10 + 1.0)
        / (SUM(s.break_points_faced) OVER w10 + 2.0)
        AS break_points_saved_pct_10,
    (SUM(s.first_serve_points_won) OVER w10 + 1.0)
        / (SUM(s.first_serves_made) OVER w10 + 2.0)
        AS first_serve_win_pct_10,
    (SUM(s.second_serve_points_won) OVER w10 + 1.0)
        / (SUM(s.total_serve_points - s.first_serves_made) OVER w10 + 2.0)
        AS second_serve_win_pct_10,
    (SUM(s.first_serve_points_won + s.second_serve_points_won) OVER w10 + 1.0)
        / (SUM(s.total_serve_points) OVER w10 + 2.0)
        AS serve_win_pct_10,
    (SUM(s.return_points_won) OVER w10 + 1.0)
        / (SUM(s.return_points_available) OVER w10 + 2.0)
        AS return_points_won_pct_10,
    -- Lifetime (unbounded) Dominance Ratio, Beta(1,1)-smoothed over full history.
    -- Both inputs are smoothed rates below 1, so the denominator is never zero.
    (
        (SUM(s.return_points_won) OVER w_life + 1.0)
            / (SUM(s.return_points_available) OVER w_life + 2.0)
        / (1.0 - (SUM(s.first_serve_points_won + s.second_serve_points_won) OVER w_life + 1.0)
                    / (SUM(s.total_serve_points) OVER w_life + 2.0))
    )::NUMERIC AS dominance,
    (SUM(s.double_faults) OVER w10 + 1.0)
        / (SUM(s.total_serve_points) OVER w10 + 2.0)
        AS df_rate_10,

    -- Matches backing the smoothed rates (1 for the first, up to 10).
    COUNT(*) OVER w10 AS matches_10,

    -- Unsmoothed rates: aces per service game and rolling average ranks.
    CAST(SUM(s.aces) OVER w10 AS DOUBLE PRECISION)
        / NULLIF(SUM(s.service_games) OVER w10, 0)
        AS aces_per_svc_game_10,
    AVG(s.player_ranking) OVER w10 AS avg_player_rank_10,
    AVG(s.opponent_ranking) OVER w10 AS avg_rank_faced_10,

    s.streak,

    psm_clay.surface_win_rate_10  AS clay_win_rate_10,
    psm_grass.surface_win_rate_10 AS grass_win_rate_10,
    psm_hard.surface_win_rate_10  AS hard_win_rate_10

FROM (
    -- Streak derives here because window results aren't visible to the surface joins.
    SELECT s.*,
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
    w10 AS (PARTITION BY s.player_id ORDER BY s.snapshot_date, s.match_num
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW),
    w_life AS (PARTITION BY s.player_id ORDER BY s.snapshot_date, s.match_num
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
)
-- Trim to affected players' full histories; every window above was evaluated over
-- full history, so returned rows carry full-rebuild values.
SELECT * FROM computed
{% if is_incremental() %}
WHERE player_id IN (SELECT player_id FROM changed_players)
{% endif %}
ORDER BY snapshot_date, player_match_number, player_id
