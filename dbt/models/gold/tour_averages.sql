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
--
-- Query shape: pool_stats folds the pool metadata and the full-pool snapshot
-- aggregates into a single scan of silver.rolling_features, and activity
-- reduces the pool to one latest snapshot per player BEFORE joining the
-- 30-day match window — so the window join runs per player, not per snapshot
-- row (previously ~11.5M intermediate rows from 399,802 snapshots × their
-- in-window matches). The per-player 30-day count is now the true match count
-- for that player instead of the snapshot-multiplied join count; the rounded
-- median is unchanged on current data (verified 13347 / 0).
--
-- Numeric precision contract: every floating/statistical output below is
-- rounded to 3 decimal places at the final SELECT boundary via
-- ROUND(x::NUMERIC, 3)::DOUBLE PRECISION, so the pool/aggregate CTEs retain
-- full precision internally. singleton_id, counts, and pool_as_of_date are
-- unchanged.

WITH
-- Singleton metadata + full-pool snapshot fallback aggregates, one scan of
-- rolling_features.
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
-- One latest snapshot per player: the recency anchor for the cold-start
-- activity defaults. Reduces the 30-day window join from one row per
-- snapshot to one row per player.
latest_snapshots AS (
    SELECT
        player_id,
        MAX(snapshot_date) AS latest_snapshot_date
    FROM {{ ref('rolling_features') }}
    GROUP BY player_id
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
    CROSS JOIN pool_stats p
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

    -- Rank/streak-like medians (whole values), rounded to 3 dp per the
    -- singleton precision contract; empty pool falls back to constants.
    ROUND(COALESCE(p.latest_player_ranking, 100)::NUMERIC, 3)::DOUBLE PRECISION
        AS latest_player_ranking,
    ROUND(COALESCE(p.latest_player_rank_points, 500)::NUMERIC, 3)::DOUBLE PRECISION
        AS latest_player_rank_points,
    ROUND(COALESCE(p.streak, 0)::NUMERIC, 3)::DOUBLE PRECISION AS streak,
    ROUND(COALESCE(p.avg_player_rank_10, 100)::NUMERIC, 3)::DOUBLE PRECISION
        AS avg_player_rank_10,
    ROUND(COALESCE(p.avg_rank_faced_10, 100)::NUMERIC, 3)::DOUBLE PRECISION
        AS avg_rank_faced_10,

    -- Continuous means over the full pool, rounded to 3 dp.
    ROUND(COALESCE(p.latest_player_age, 26.0)::NUMERIC, 3)::DOUBLE PRECISION
        AS latest_player_age,
    ROUND(COALESCE(p.weighted_form_10, 0.0)::NUMERIC, 3)::DOUBLE PRECISION
        AS weighted_form_10,
    ROUND(COALESCE(p.win_rate_10, 0.0)::NUMERIC, 3)::DOUBLE PRECISION
        AS win_rate_10,
    ROUND(COALESCE(p.ace_rate_10, 0.0)::NUMERIC, 3)::DOUBLE PRECISION
        AS ace_rate_10,
    ROUND(COALESCE(p.first_serve_pct_10, 0.0)::NUMERIC, 3)::DOUBLE PRECISION
        AS first_serve_pct_10,
    ROUND(COALESCE(p.break_points_saved_pct_10, 0.0)::NUMERIC, 3)::DOUBLE PRECISION
        AS break_points_saved_pct_10,
    ROUND(COALESCE(p.first_serve_win_pct_10, 0.0)::NUMERIC, 3)::DOUBLE PRECISION
        AS first_serve_win_pct_10,
    ROUND(COALESCE(p.second_serve_win_pct_10, 0.0)::NUMERIC, 3)::DOUBLE PRECISION
        AS second_serve_win_pct_10,
    ROUND(COALESCE(p.serve_win_pct_10, 0.0)::NUMERIC, 3)::DOUBLE PRECISION
        AS serve_win_pct_10,
    ROUND(COALESCE(p.return_points_won_pct_10, 0.0)::NUMERIC, 3)::DOUBLE PRECISION
        AS return_points_won_pct_10,
    ROUND(COALESCE(p.df_rate_10, 0.0)::NUMERIC, 3)::DOUBLE PRECISION
        AS df_rate_10,
    ROUND(COALESCE(p.aces_per_svc_game_10, 0.0)::NUMERIC, 3)::DOUBLE PRECISION
        AS aces_per_svc_game_10,
    ROUND(COALESCE(p.clay_win_rate_10, 0.0)::NUMERIC, 3)::DOUBLE PRECISION
        AS clay_win_rate_10,
    ROUND(COALESCE(p.grass_win_rate_10, 0.0)::NUMERIC, 3)::DOUBLE PRECISION
        AS grass_win_rate_10,
    ROUND(COALESCE(p.hard_win_rate_10, 0.0)::NUMERIC, 3)::DOUBLE PRECISION
        AS hard_win_rate_10,

    -- Cold-start activity defaults (already whole medians), rounded to 3 dp.
    ROUND(COALESCE(a.median_days_since, 365)::NUMERIC, 3)::DOUBLE PRECISION
        AS days_since_default,
    ROUND(COALESCE(a.median_matches_30d, 0)::NUMERIC, 3)::DOUBLE PRECISION
        AS matches_30d_default,

    -- Explicit fixed constant (already exactly 3 dp) and profile-pool means.
    0.5 AS rate_default,
    ROUND(COALESCE(pa.left_handed_rate, 0.0)::NUMERIC, 3)::DOUBLE PRECISION
        AS left_handed_rate,
    ROUND(COALESCE(pa.avg_years_pro, 8.0)::NUMERIC, 3)::DOUBLE PRECISION
        AS avg_years_pro,

    -- Weighted tour benchmarks (may be NULL only when denominator is zero),
    -- rounded to 3 dp; NULLs preserved.
    ROUND(tr.tour_ace_rate::NUMERIC, 3)::DOUBLE PRECISION AS tour_ace_rate,
    ROUND(tr.tour_first_serve_pct::NUMERIC, 3)::DOUBLE PRECISION
        AS tour_first_serve_pct,
    ROUND(tr.tour_break_points_saved_pct::NUMERIC, 3)::DOUBLE PRECISION
        AS tour_break_points_saved_pct,
    ROUND(tr.tour_first_serve_win_pct::NUMERIC, 3)::DOUBLE PRECISION
        AS tour_first_serve_win_pct,
    ROUND(tr.tour_second_serve_win_pct::NUMERIC, 3)::DOUBLE PRECISION
        AS tour_second_serve_win_pct,
    ROUND(tr.tour_serve_win_pct::NUMERIC, 3)::DOUBLE PRECISION
        AS tour_serve_win_pct,
    ROUND(tr.tour_return_points_won_pct::NUMERIC, 3)::DOUBLE PRECISION
        AS tour_return_points_won_pct,
    ROUND(tr.tour_df_rate::NUMERIC, 3)::DOUBLE PRECISION AS tour_df_rate,
    ROUND(tr.tour_aces_per_svc_game::NUMERIC, 3)::DOUBLE PRECISION
        AS tour_aces_per_svc_game,
    ROUND(tr.tour_break_point_opportunities_per_return_game::NUMERIC, 3)
        ::DOUBLE PRECISION AS tour_break_point_opportunities_per_return_game,
    ROUND(tr.tour_return_games_won_pct::NUMERIC, 3)::DOUBLE PRECISION
        AS tour_return_games_won_pct
FROM pool_stats p
CROSS JOIN activity a
CROSS JOIN profile_aggregates pa
CROSS JOIN tour_rates tr