-- gold.tour_averages: single-row (singleton_id = 1) imputation defaults + tour
-- benchmarks over all silver snapshots/matches. Plain table rebuilt in full:
-- a global aggregate, every new match shifts the pool.
--
-- Fallback semantics (matching the former feature_defaults): rank/streak-like
-- medians, continuous rates means; an empty pool falls back to explicit
-- constants (ranking 100, rank points 500, age 26, streak 0, rates/forms 0,
-- avg ranks 100, days since 365, matches 365d 0, rate 0.5, left-handed 0,
-- years-pro 8) so every column stays non-null and finite.
--
-- Intended, reported leakage: old cold-start/currently-missing cells use the
-- same full-pool singleton; verification reports the affected row count.
--
-- pool_as_of_date is one day after the latest snapshot, or CURRENT_DATE only
-- as metadata when the pool is empty. Weighted tour comparisons are SUM()/SUM()
-- from silver.player_matches (never per-player AVG); NULL only when the
-- denominator is zero, defaults never NULL.
--
-- Query shape: pool_stats mixes pool metadata and snapshot aggregates in one
-- scan of rolling_features; activity reduces to one latest snapshot per player
-- before the 30-day window join, so it runs per player not per snapshot.

WITH
-- Pool metadata + full-pool snapshot fallback aggregates (one scan).
pool_stats AS (
    SELECT
        COALESCE(MAX(snapshot_date), CURRENT_DATE) + 1 AS pool_as_of_date,
        COUNT(*) AS snapshot_pool_rows,
        COUNT(DISTINCT player_id) AS snapshot_pool_players,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latest_player_ranking))
            AS latest_player_ranking,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latest_player_rank_points))
            AS latest_player_rank_points,
        AVG(latest_player_age) AS latest_player_age,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY streak)) AS streak,
        AVG(weighted_form_10) AS weighted_form_10,
        AVG(win_rate_10) AS win_rate_10,
        AVG(ace_rate_10) AS ace_rate_10,
        AVG(first_serve_pct_10) AS first_serve_pct_10,
        AVG(break_points_saved_pct_10) AS break_points_saved_pct_10,
        AVG(first_serve_win_pct_10) AS first_serve_win_pct_10,
        AVG(second_serve_win_pct_10) AS second_serve_win_pct_10,
        AVG(serve_win_pct_10) AS serve_win_pct_10,
        AVG(return_points_won_pct_10) AS return_points_won_pct_10,
        AVG(dominance) AS dominance,
        AVG(df_rate_10) AS df_rate_10,
        AVG(aces_per_svc_game_10) AS aces_per_svc_game_10,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY avg_player_rank_10))
            AS avg_player_rank_10,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY avg_rank_faced_10))
            AS avg_rank_faced_10,
        AVG(clay_win_rate_10) AS clay_win_rate_10,
        AVG(grass_win_rate_10) AS grass_win_rate_10,
        AVG(hard_win_rate_10) AS hard_win_rate_10
    FROM {{ ref('rolling_features') }}
),
-- One latest snapshot per player: drives the 30-day window join per player.
latest_snapshots AS (
    SELECT
        player_id,
        MAX(snapshot_date) AS latest_snapshot_date
    FROM {{ ref('rolling_features') }}
    GROUP BY player_id
),
-- Cold-start activity defaults: rounded medians of per-player recency and
-- 30-day match count.
activity AS (
    SELECT
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY days_since))
            AS median_days_since,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY matches_30d))
            AS median_matches_30d
    FROM (
        SELECT
            ls.player_id,
            p.pool_as_of_date - ls.latest_snapshot_date AS days_since,
            COALESCE(w.matches_30d, 0) AS matches_30d
        FROM latest_snapshots ls
        CROSS JOIN pool_stats p
        LEFT JOIN LATERAL (
            SELECT COUNT(*) AS matches_30d
            FROM {{ ref('player_matches') }} pm
            WHERE pm.player_id = ls.player_id
              AND pm.match_date >= p.pool_as_of_date - INTERVAL '30 days'
              AND pm.match_date < p.pool_as_of_date
        ) w ON true
    ) per_player
),
-- Static profile pool: left-handed rate over known L/R, time-aware years-pro.
profile_aggregates AS (
    SELECT
        AVG(CASE WHEN prof.handedness = 'L' THEN 1
                 WHEN prof.handedness = 'R' THEN 0 END) AS left_handed_rate,
        AVG(EXTRACT(YEAR FROM p.pool_as_of_date) - prof.turned_pro) AS avg_years_pro,
        COUNT(*) AS profile_rows
    FROM {{ source('bronze', 'player_profiles') }} prof
    CROSS JOIN pool_stats p
),
-- Weighted tour benchmarks: SUM()/SUM() over all silver.player_matches rows.
tour_rates AS (
    SELECT
        COUNT(*) AS player_match_rows,
        CAST(SUM(aces) AS DOUBLE PRECISION)
            / NULLIF(SUM(first_serves_made), 0) AS tour_ace_rate,
        CAST(SUM(first_serves_made) AS DOUBLE PRECISION)
            / NULLIF(SUM(total_serve_points), 0) AS tour_first_serve_pct,
        CAST(SUM(break_points_saved) AS DOUBLE PRECISION)
            / NULLIF(SUM(break_points_faced), 0) AS tour_break_points_saved_pct,
        CAST(SUM(first_serve_points_won) AS DOUBLE PRECISION)
            / NULLIF(SUM(first_serves_made), 0) AS tour_first_serve_win_pct,
        CAST(SUM(second_serve_points_won) AS DOUBLE PRECISION)
            / NULLIF(SUM(total_serve_points - first_serves_made), 0)
            AS tour_second_serve_win_pct,
        CAST(SUM(first_serve_points_won + second_serve_points_won) AS DOUBLE PRECISION)
            / NULLIF(SUM(total_serve_points), 0) AS tour_serve_win_pct,
        CAST(SUM(return_points_won) AS DOUBLE PRECISION)
            / NULLIF(SUM(return_points_available), 0) AS tour_return_points_won_pct,
        CAST(SUM(double_faults) AS DOUBLE PRECISION)
            / NULLIF(SUM(total_serve_points), 0) AS tour_df_rate,
        CAST(SUM(aces) AS DOUBLE PRECISION)
            / NULLIF(SUM(service_games), 0) AS tour_aces_per_svc_game,
        CAST(SUM(break_points_faced) AS DOUBLE PRECISION)
            / NULLIF(SUM(service_games), 0)
            AS tour_break_point_opportunities_per_return_game,
        -- Return games won tour-wide: a converted break point ends the game.
        CAST(SUM(break_points_faced - break_points_saved) AS DOUBLE PRECISION)
            / NULLIF(SUM(service_games), 0)
            AS tour_return_games_won_pct
    FROM {{ ref('player_matches') }}
)
SELECT
    1 AS singleton_id,
    p.pool_as_of_date,
    p.snapshot_pool_rows,
    p.snapshot_pool_players,
    pa.profile_rows,
    tr.player_match_rows,

    -- Rank/streak-like medians; empty pool falls back to constants.
    ROUND(COALESCE(p.latest_player_ranking, 100)::NUMERIC, 5)::DOUBLE PRECISION
        AS latest_player_ranking,
    ROUND(COALESCE(p.latest_player_rank_points, 500)::NUMERIC, 5)::DOUBLE PRECISION
        AS latest_player_rank_points,
    ROUND(COALESCE(p.streak, 0)::NUMERIC, 5)::DOUBLE PRECISION AS streak,
    ROUND(COALESCE(p.avg_player_rank_10, 100)::NUMERIC, 5)::DOUBLE PRECISION
        AS avg_player_rank_10,
    ROUND(COALESCE(p.avg_rank_faced_10, 100)::NUMERIC, 5)::DOUBLE PRECISION
        AS avg_rank_faced_10,

    -- Continuous means over the full pool.
    ROUND(COALESCE(p.latest_player_age, 26.0)::NUMERIC, 5)::DOUBLE PRECISION
        AS latest_player_age,
    ROUND(COALESCE(p.weighted_form_10, 0.0)::NUMERIC, 5)::DOUBLE PRECISION
        AS weighted_form_10,
    ROUND(COALESCE(p.win_rate_10, 0.0)::NUMERIC, 5)::DOUBLE PRECISION
        AS win_rate_10,
    ROUND(COALESCE(p.ace_rate_10, 0.0)::NUMERIC, 5)::DOUBLE PRECISION
        AS ace_rate_10,
    ROUND(COALESCE(p.first_serve_pct_10, 0.0)::NUMERIC, 5)::DOUBLE PRECISION
        AS first_serve_pct_10,
    ROUND(COALESCE(p.break_points_saved_pct_10, 0.0)::NUMERIC, 5)::DOUBLE PRECISION
        AS break_points_saved_pct_10,
    ROUND(COALESCE(p.first_serve_win_pct_10, 0.0)::NUMERIC, 5)::DOUBLE PRECISION
        AS first_serve_win_pct_10,
    ROUND(COALESCE(p.second_serve_win_pct_10, 0.0)::NUMERIC, 5)::DOUBLE PRECISION
        AS second_serve_win_pct_10,
    ROUND(COALESCE(p.serve_win_pct_10, 0.0)::NUMERIC, 5)::DOUBLE PRECISION
        AS serve_win_pct_10,
    ROUND(COALESCE(p.return_points_won_pct_10, 0.0)::NUMERIC, 5)::DOUBLE PRECISION
        AS return_points_won_pct_10,
    -- Tour-average fallback dominance: mean lifetime Dominance Ratio over the
    -- full snapshot pool; 0.0 when the pool is empty.
    ROUND(COALESCE(p.dominance, 0.0)::NUMERIC, 5)::DOUBLE PRECISION
        AS dominance,
    ROUND(COALESCE(p.df_rate_10, 0.0)::NUMERIC, 5)::DOUBLE PRECISION
        AS df_rate_10,
    ROUND(COALESCE(p.aces_per_svc_game_10, 0.0)::NUMERIC, 5)::DOUBLE PRECISION
        AS aces_per_svc_game_10,
    ROUND(COALESCE(p.clay_win_rate_10, 0.0)::NUMERIC, 5)::DOUBLE PRECISION
        AS clay_win_rate_10,
    ROUND(COALESCE(p.grass_win_rate_10, 0.0)::NUMERIC, 5)::DOUBLE PRECISION
        AS grass_win_rate_10,
    ROUND(COALESCE(p.hard_win_rate_10, 0.0)::NUMERIC, 5)::DOUBLE PRECISION
        AS hard_win_rate_10,

    -- Cold-start activity defaults.
    ROUND(COALESCE(a.median_days_since, 365)::NUMERIC, 5)::DOUBLE PRECISION
        AS days_since_default,
    ROUND(COALESCE(a.median_matches_30d, 0)::NUMERIC, 5)::DOUBLE PRECISION
        AS matches_30d_default,

    -- Fixed constants and static profile-pool means.
    0.5 AS rate_default,
    ROUND(COALESCE(pa.left_handed_rate, 0.0)::NUMERIC, 5)::DOUBLE PRECISION
        AS left_handed_rate,
    ROUND(COALESCE(pa.avg_years_pro, 8.0)::NUMERIC, 5)::DOUBLE PRECISION
        AS avg_years_pro,

    -- Weighted tour benchmarks (may be NULL only when denominator is zero).
    ROUND(tr.tour_ace_rate::NUMERIC, 5)::DOUBLE PRECISION AS tour_ace_rate,
    ROUND(tr.tour_first_serve_pct::NUMERIC, 5)::DOUBLE PRECISION
        AS tour_first_serve_pct,
    ROUND(tr.tour_break_points_saved_pct::NUMERIC, 5)::DOUBLE PRECISION
        AS tour_break_points_saved_pct,
    ROUND(tr.tour_first_serve_win_pct::NUMERIC, 5)::DOUBLE PRECISION
        AS tour_first_serve_win_pct,
    ROUND(tr.tour_second_serve_win_pct::NUMERIC, 5)::DOUBLE PRECISION
        AS tour_second_serve_win_pct,
    ROUND(tr.tour_serve_win_pct::NUMERIC, 5)::DOUBLE PRECISION
        AS tour_serve_win_pct,
    ROUND(tr.tour_return_points_won_pct::NUMERIC, 5)::DOUBLE PRECISION
        AS tour_return_points_won_pct,
    ROUND(tr.tour_df_rate::NUMERIC, 5)::DOUBLE PRECISION AS tour_df_rate,
    ROUND(tr.tour_aces_per_svc_game::NUMERIC, 5)::DOUBLE PRECISION
        AS tour_aces_per_svc_game,
    ROUND(tr.tour_break_point_opportunities_per_return_game::NUMERIC, 5)
        ::DOUBLE PRECISION AS tour_break_point_opportunities_per_return_game,
    ROUND(tr.tour_return_games_won_pct::NUMERIC, 5)::DOUBLE PRECISION
        AS tour_return_games_won_pct
FROM pool_stats p
CROSS JOIN activity a
CROSS JOIN profile_aggregates pa
CROSS JOIN tour_rates tr