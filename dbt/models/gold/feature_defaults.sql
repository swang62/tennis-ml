-- gold.feature_defaults: date-keyed imputation defaults for model features.
--
-- One row per as-of date (every historical match date plus the dbt run date).
-- Every pool value is computed ONLY from silver.rolling_features snapshots
-- strictly before that date (`snapshot_date < as_of_date`), so no current-match
-- or future state can leak into the defaults. Missing pool values (e.g. an
-- empty prior pool on the first match date) fall back to explicit constants.
--
-- Median is used for rank / rank-points / streak-like values (rounded to a
-- whole value); mean for rates, age, years-pro, and handedness rate.
-- `rate_default` is the fixed constant for the unknown/0 surface win rate and
-- empty-pool rates. `snapshot_pool_rows` / `snapshot_pool_players` /
-- `profile_rows` are meta-only observability counts.
--
-- Both gold.match_features and scalar/bulk inference read the NEWEST row with
-- as_of_date <= the match/as-of date (the oldest row covers pre-history dates,
-- the dbt run-date row covers future dates). This replaces on-demand
-- AVG/PERCENTILE imputation queries with a materialized lookup.

WITH as_of_dates AS (
    SELECT DISTINCT match_date AS as_of_date
    FROM {{ source('bronze', 'match_events') }}
    WHERE match_date IS NOT NULL
    UNION
    SELECT CURRENT_DATE AS as_of_date
),
-- Every snapshot strictly before each as-of date: the shared prior pool.
eligible_pool AS (
    SELECT
        d.as_of_date,
        r.player_id,
        r.snapshot_date,
        r.latest_player_ranking,
        r.latest_player_rank_points,
        r.latest_player_age,
        r.streak,
        r.weighted_form_10,
        r.win_rate_10,
        r.ace_rate_10,
        r.first_serve_pct_10,
        r.break_points_saved_pct_10,
        r.first_serve_win_pct_10,
        r.second_serve_win_pct_10,
        r.serve_win_pct_10,
        r.return_points_won_pct_10,
        r.df_rate_10,
        r.aces_per_svc_game_10,
        r.avg_player_rank_10,
        r.avg_rank_faced_10,
        r.clay_win_rate_10,
        r.grass_win_rate_10,
        r.hard_win_rate_10
    FROM as_of_dates d
    JOIN {{ ref('rolling_features') }} r
      ON r.snapshot_date < d.as_of_date
),
snapshot_aggregates AS (
    SELECT
        as_of_date,
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
        AVG(hard_win_rate_10) AS hard_win_rate_10,
        COUNT(*) AS snapshot_pool_rows,
        COUNT(DISTINCT player_id) AS snapshot_pool_players
    FROM eligible_pool
    GROUP BY as_of_date
),
-- Median days since each player's latest eligible snapshot.
median_days AS (
    SELECT
        as_of_date,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY days_since))
            AS median_days_since
    FROM (
        SELECT
            as_of_date,
            player_id,
            as_of_date - MAX(snapshot_date) AS days_since
        FROM eligible_pool
        GROUP BY as_of_date, player_id
    ) per_player
    GROUP BY as_of_date
),
-- Median 30-day pre-as-of match count across players with eligible snapshots.
median_matches AS (
    SELECT
        as_of_date,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY matches_30d))
            AS median_matches_30d
    FROM (
        SELECT
            e.as_of_date,
            e.player_id,
            COUNT(pm.match_id) AS matches_30d
        FROM (
            SELECT DISTINCT as_of_date, player_id
            FROM eligible_pool
        ) e
        LEFT JOIN {{ ref('player_matches') }} pm
            ON pm.player_id = e.player_id
           AND pm.match_date >= e.as_of_date - INTERVAL '30 days'
           AND pm.match_date < e.as_of_date
        GROUP BY e.as_of_date, e.player_id
    ) per_player
    GROUP BY as_of_date
),
-- Static profile pool, averaged at each as-of year (years-pro is time-aware).
profile_aggregates AS (
    SELECT
        d.as_of_date,
        AVG(CASE WHEN prof.handedness = 'L' THEN 1 ELSE 0 END) AS left_handed_rate,
        AVG(EXTRACT(YEAR FROM d.as_of_date) - prof.turned_pro) AS avg_years_pro,
        COUNT(*) AS profile_rows
    FROM as_of_dates d
    CROSS JOIN gold.player_profiles prof
    GROUP BY d.as_of_date
)
SELECT
    d.as_of_date,
    -- Rank/streak-like medians, rounded; empty pool falls back to constants.
    COALESCE(s.latest_player_ranking, 100)::DOUBLE PRECISION AS latest_player_ranking,
    COALESCE(s.latest_player_rank_points, 500)::DOUBLE PRECISION AS latest_player_rank_points,
    COALESCE(s.streak, 0)::DOUBLE PRECISION AS streak,
    COALESCE(s.avg_player_rank_10, 100)::DOUBLE PRECISION AS avg_player_rank_10,
    COALESCE(s.avg_rank_faced_10, 100)::DOUBLE PRECISION AS avg_rank_faced_10,
    -- Continuous means.
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
    -- Date-keyed defaults for cold-start activity; pre-rounded whole values.
    COALESCE(md.median_days_since, 365)::DOUBLE PRECISION AS days_since_default,
    COALESCE(mm.median_matches_30d, 0)::DOUBLE PRECISION AS matches_30d_default,
    -- Explicit fixed constants.
    0.0 AS rate_default,
    COALESCE(pa.left_handed_rate, 0.0) AS left_handed_rate,
    COALESCE(pa.avg_years_pro, 8.0) AS avg_years_pro,
    -- Meta-only observability counts.
    COALESCE(s.snapshot_pool_rows, 0) AS snapshot_pool_rows,
    COALESCE(s.snapshot_pool_players, 0) AS snapshot_pool_players,
    COALESCE(pa.profile_rows, 0) AS profile_rows
FROM as_of_dates d
LEFT JOIN snapshot_aggregates s
    ON s.as_of_date = d.as_of_date
LEFT JOIN median_days md
    ON md.as_of_date = d.as_of_date
LEFT JOIN median_matches mm
    ON mm.as_of_date = d.as_of_date
LEFT JOIN profile_aggregates pa
    ON pa.as_of_date = d.as_of_date
ORDER BY d.as_of_date
