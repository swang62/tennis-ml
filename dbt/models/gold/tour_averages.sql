-- gold.tour_averages: single-row (singleton) imputation defaults + tour benchmarks.
--
-- This materializes EXACTLY ONE row (singleton_id = 1) and is the only shared
-- source for:
--   1. difficult model-feature fallbacks used by gold.match_features and
--      inference; and
--   2. weighted tour-wide rates used for player-profile comparisons.
--
-- It replaces the date-keyed gold.feature_defaults table. There are no
-- as_of_date rows and no date-expansion joins. The singleton is a full-pool
-- aggregate over ALL silver.rolling_features snapshots and ALL
-- silver.player_matches rows.
--
-- Materialization: plain table, rebuilt in full on every ETL run. This is a
-- global aggregate — every new match shifts the pool, so it is recomputed
-- globally (never incremental) by design.
--
-- Fallback semantics (identical destinations to the former feature_defaults):
--   - Rank / rank-points / streak-like values use the (rounded) median.
--   - Continuous rates, age, years-pro, and handedness rate use the mean.
--   - An empty snapshot pool falls back to explicit deterministic constants so
--     every fallback column is finite and non-null (ranking 100, rank points
--     500, age 26, streak 0, rates/forms 0, average ranks 100, days since 365,
--     matches in 30d 0, rate 0.5, left-handed 0, years-pro 8).
--
-- Limited historical leakage is intentional and documented: old cold-start or
-- otherwise missing cells use the same full-pool singleton as current rows.
-- Verification reports the affected row/cell count so the accepted bias is
-- visible without introducing another defaults table.
--
-- pool_as_of_date is one day after the latest snapshot for non-empty data;
-- CURRENT_DATE is used only as metadata when the snapshot pool is empty. All
-- fallback values still come from the explicit constants in that case.
--
-- Weighted tour comparison columns are SUM(numerator) / SUM(denominator)
-- ratios from silver.player_matches (NOT unweighted per-player AVG). They may
-- be NULL only when their source denominator is zero; defaults must never be
-- NULL.

WITH
-- Singleton metadata: pool_as_of_date and snapshot-pool observability counts.
pool_meta AS (
    SELECT
        COALESCE(MAX(snapshot_date), CURRENT_DATE) + 1 AS pool_as_of_date,
        COUNT(*) AS snapshot_pool_rows,
        COUNT(DISTINCT player_id) AS snapshot_pool_players
    FROM {{ ref('rolling_features') }}
),
-- Full-pool snapshot fallback aggregates over ALL rolling_features rows.
snapshot_aggregates AS (
    SELECT
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
-- Cold-start activity defaults: rounded medians of per-player latest-snapshot
-- recency and 30-day pre-pool match count.
activity AS (
    SELECT
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY days_since))
            AS median_days_since,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY matches_30d))
            AS median_matches_30d
    FROM (
        SELECT
            r.player_id,
            p.pool_as_of_date - MAX(r.snapshot_date) AS days_since,
            COUNT(pm.match_id) AS matches_30d
        FROM {{ ref('rolling_features') }} r
        CROSS JOIN pool_meta p
        LEFT JOIN {{ ref('player_matches') }} pm
            ON pm.player_id = r.player_id
           AND pm.match_date >= p.pool_as_of_date - INTERVAL '30 days'
           AND pm.match_date < p.pool_as_of_date
        GROUP BY r.player_id, p.pool_as_of_date
    ) per_player
),
-- Static profile pool: left-handed rate over known L/R only, time-aware
-- years-pro, and the profile row count. Metadata comes from bronze —
-- gold.player_profiles carries aggregates only.
profile_aggregates AS (
    SELECT
        AVG(CASE WHEN prof.handedness = 'L' THEN 1
                 WHEN prof.handedness = 'R' THEN 0 END) AS left_handed_rate,
        AVG(EXTRACT(YEAR FROM p.pool_as_of_date) - prof.turned_pro) AS avg_years_pro,
        COUNT(*) AS profile_rows
    FROM {{ source('bronze', 'player_profiles') }} prof
    CROSS JOIN pool_meta p
),
-- Weighted tour benchmarks: SUM(numerator) / SUM(denominator) over ALL
-- silver.player_matches rows. NULL when the denominator is zero.
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
        -- return games won tour-wide: every converted break point ends that
        -- service game, so (BP faced - BP saved) / service games
        CAST(SUM(break_points_faced - break_points_saved) AS DOUBLE PRECISION)
            / NULLIF(SUM(service_games), 0)
            AS tour_return_games_won_pct
    FROM {{ ref('player_matches') }}
)
SELECT
    -- Singleton identity + observability.
    1 AS singleton_id,
    p.pool_as_of_date,
    p.snapshot_pool_rows,
    p.snapshot_pool_players,
    pa.profile_rows,
    tr.player_match_rows,

    -- Rank/streak-like medians, rounded; empty pool falls back to constants.
    COALESCE(s.latest_player_ranking, 100)::DOUBLE PRECISION AS latest_player_ranking,
    COALESCE(s.latest_player_rank_points, 500)::DOUBLE PRECISION AS latest_player_rank_points,
    COALESCE(s.streak, 0)::DOUBLE PRECISION AS streak,
    COALESCE(s.avg_player_rank_10, 100)::DOUBLE PRECISION AS avg_player_rank_10,
    COALESCE(s.avg_rank_faced_10, 100)::DOUBLE PRECISION AS avg_rank_faced_10,

    -- Continuous means over the full pool.
    COALESCE(s.latest_player_age, 26.0) AS latest_player_age,
    COALESCE(s.weighted_form_10, 0.0) AS weighted_form_10,
    COALESCE(s.win_rate_10, 0.0) AS win_rate_10,
    COALESCE(s.ace_rate_10, 0.0) AS ace_rate_10,
    COALESCE(s.first_serve_pct_10, 0.0) AS first_serve_pct_10,
    COALESCE(s.break_points_saved_pct_10, 0.0) AS break_points_saved_pct_10,
    COALESCE(s.first_serve_win_pct_10, 0.0) AS first_serve_win_pct_10,
    COALESCE(s.second_serve_win_pct_10, 0.0) AS second_serve_win_pct_10,
    COALESCE(s.serve_win_pct_10, 0.0) AS serve_win_pct_10,
    COALESCE(s.return_points_won_pct_10, 0.0) AS return_points_won_pct_10,
    COALESCE(s.df_rate_10, 0.0) AS df_rate_10,
    COALESCE(s.aces_per_svc_game_10, 0.0) AS aces_per_svc_game_10,
    COALESCE(s.clay_win_rate_10, 0.0) AS clay_win_rate_10,
    COALESCE(s.grass_win_rate_10, 0.0) AS grass_win_rate_10,
    COALESCE(s.hard_win_rate_10, 0.0) AS hard_win_rate_10,

    -- Cold-start activity defaults; pre-rounded whole values.
    COALESCE(a.median_days_since, 365)::DOUBLE PRECISION AS days_since_default,
    COALESCE(a.median_matches_30d, 0)::DOUBLE PRECISION AS matches_30d_default,

    -- Explicit fixed constants and static profile-pool means.
    0.5 AS rate_default,
    COALESCE(pa.left_handed_rate, 0.0) AS left_handed_rate,
    COALESCE(pa.avg_years_pro, 8.0) AS avg_years_pro,

    -- Weighted tour benchmarks (may be NULL only when denominator is zero).
    tr.tour_ace_rate,
    tr.tour_first_serve_pct,
    tr.tour_break_points_saved_pct,
    tr.tour_first_serve_win_pct,
    tr.tour_second_serve_win_pct,
    tr.tour_serve_win_pct,
    tr.tour_return_points_won_pct,
    tr.tour_df_rate,
    tr.tour_aces_per_svc_game,
    tr.tour_break_point_opportunities_per_return_game,
    tr.tour_return_games_won_pct
FROM pool_meta p
CROSS JOIN snapshot_aggregates s
CROSS JOIN activity a
CROSS JOIN profile_aggregates pa
CROSS JOIN tour_rates tr
