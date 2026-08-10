-- Assert the tour_averages singleton contract: exactly one row, identity,
-- finite fallback defaults, non-negative counts, and a non-null pool anchor.
-- Any returned row is a violation. Only the first check column is aliased
-- `violation` (PostgreSQL rejects duplicate column names); the others are
-- unnamed and only the first failing check per row is reported.

WITH checks AS (
    SELECT
        CASE WHEN singleton_id != 1 THEN 'singleton_id must be 1' END AS violation,
        CASE WHEN latest_player_ranking IS NULL
                  OR latest_player_ranking = 'NaN'::DOUBLE PRECISION
                  OR latest_player_ranking = 'Infinity'::DOUBLE PRECISION
                  OR latest_player_ranking = '-Infinity'::DOUBLE PRECISION
             THEN 'latest_player_ranking must be non-null and finite' END,
        CASE WHEN latest_player_rank_points IS NULL
                  OR latest_player_rank_points = 'NaN'::DOUBLE PRECISION
                  OR latest_player_rank_points = 'Infinity'::DOUBLE PRECISION
                  OR latest_player_rank_points = '-Infinity'::DOUBLE PRECISION
             THEN 'latest_player_rank_points must be non-null and finite' END,
        CASE WHEN latest_player_age IS NULL
                  OR latest_player_age = 'NaN'::DOUBLE PRECISION
                  OR latest_player_age = 'Infinity'::DOUBLE PRECISION
                  OR latest_player_age = '-Infinity'::DOUBLE PRECISION
             THEN 'latest_player_age must be non-null and finite' END,
        CASE WHEN streak IS NULL
                  OR streak = 'NaN'::DOUBLE PRECISION
                  OR streak = 'Infinity'::DOUBLE PRECISION
                  OR streak = '-Infinity'::DOUBLE PRECISION
             THEN 'streak must be non-null and finite' END,
        CASE WHEN weighted_form_10 IS NULL
                  OR weighted_form_10 = 'NaN'::DOUBLE PRECISION
                  OR weighted_form_10 = 'Infinity'::DOUBLE PRECISION
                  OR weighted_form_10 = '-Infinity'::DOUBLE PRECISION
             THEN 'weighted_form_10 must be non-null and finite' END,
        CASE WHEN win_rate_10 IS NULL
                  OR win_rate_10 = 'NaN'::DOUBLE PRECISION
                  OR win_rate_10 = 'Infinity'::DOUBLE PRECISION
                  OR win_rate_10 = '-Infinity'::DOUBLE PRECISION
             THEN 'win_rate_10 must be non-null and finite' END,
        CASE WHEN ace_rate_10 IS NULL
                  OR ace_rate_10 = 'NaN'::DOUBLE PRECISION
                  OR ace_rate_10 = 'Infinity'::DOUBLE PRECISION
                  OR ace_rate_10 = '-Infinity'::DOUBLE PRECISION
             THEN 'ace_rate_10 must be non-null and finite' END,
        CASE WHEN first_serve_pct_10 IS NULL
                  OR first_serve_pct_10 = 'NaN'::DOUBLE PRECISION
                  OR first_serve_pct_10 = 'Infinity'::DOUBLE PRECISION
                  OR first_serve_pct_10 = '-Infinity'::DOUBLE PRECISION
             THEN 'first_serve_pct_10 must be non-null and finite' END,
        CASE WHEN break_points_saved_pct_10 IS NULL
                  OR break_points_saved_pct_10 = 'NaN'::DOUBLE PRECISION
                  OR break_points_saved_pct_10 = 'Infinity'::DOUBLE PRECISION
                  OR break_points_saved_pct_10 = '-Infinity'::DOUBLE PRECISION
             THEN 'break_points_saved_pct_10 must be non-null and finite' END,
        CASE WHEN first_serve_win_pct_10 IS NULL
                  OR first_serve_win_pct_10 = 'NaN'::DOUBLE PRECISION
                  OR first_serve_win_pct_10 = 'Infinity'::DOUBLE PRECISION
                  OR first_serve_win_pct_10 = '-Infinity'::DOUBLE PRECISION
             THEN 'first_serve_win_pct_10 must be non-null and finite' END,
        CASE WHEN second_serve_win_pct_10 IS NULL
                  OR second_serve_win_pct_10 = 'NaN'::DOUBLE PRECISION
                  OR second_serve_win_pct_10 = 'Infinity'::DOUBLE PRECISION
                  OR second_serve_win_pct_10 = '-Infinity'::DOUBLE PRECISION
             THEN 'second_serve_win_pct_10 must be non-null and finite' END,
        CASE WHEN serve_win_pct_10 IS NULL
                  OR serve_win_pct_10 = 'NaN'::DOUBLE PRECISION
                  OR serve_win_pct_10 = 'Infinity'::DOUBLE PRECISION
                  OR serve_win_pct_10 = '-Infinity'::DOUBLE PRECISION
             THEN 'serve_win_pct_10 must be non-null and finite' END,
        CASE WHEN return_points_won_pct_10 IS NULL
                  OR return_points_won_pct_10 = 'NaN'::DOUBLE PRECISION
                  OR return_points_won_pct_10 = 'Infinity'::DOUBLE PRECISION
                  OR return_points_won_pct_10 = '-Infinity'::DOUBLE PRECISION
             THEN 'return_points_won_pct_10 must be non-null and finite' END,
        CASE WHEN df_rate_10 IS NULL
                  OR df_rate_10 = 'NaN'::DOUBLE PRECISION
                  OR df_rate_10 = 'Infinity'::DOUBLE PRECISION
                  OR df_rate_10 = '-Infinity'::DOUBLE PRECISION
             THEN 'df_rate_10 must be non-null and finite' END,
        CASE WHEN aces_per_svc_game_10 IS NULL
                  OR aces_per_svc_game_10 = 'NaN'::DOUBLE PRECISION
                  OR aces_per_svc_game_10 = 'Infinity'::DOUBLE PRECISION
                  OR aces_per_svc_game_10 = '-Infinity'::DOUBLE PRECISION
             THEN 'aces_per_svc_game_10 must be non-null and finite' END,
        CASE WHEN avg_player_rank_10 IS NULL
                  OR avg_player_rank_10 = 'NaN'::DOUBLE PRECISION
                  OR avg_player_rank_10 = 'Infinity'::DOUBLE PRECISION
                  OR avg_player_rank_10 = '-Infinity'::DOUBLE PRECISION
             THEN 'avg_player_rank_10 must be non-null and finite' END,
        CASE WHEN avg_rank_faced_10 IS NULL
                  OR avg_rank_faced_10 = 'NaN'::DOUBLE PRECISION
                  OR avg_rank_faced_10 = 'Infinity'::DOUBLE PRECISION
                  OR avg_rank_faced_10 = '-Infinity'::DOUBLE PRECISION
             THEN 'avg_rank_faced_10 must be non-null and finite' END,
        CASE WHEN clay_win_rate_10 IS NULL
                  OR clay_win_rate_10 = 'NaN'::DOUBLE PRECISION
                  OR clay_win_rate_10 = 'Infinity'::DOUBLE PRECISION
                  OR clay_win_rate_10 = '-Infinity'::DOUBLE PRECISION
             THEN 'clay_win_rate_10 must be non-null and finite' END,
        CASE WHEN grass_win_rate_10 IS NULL
                  OR grass_win_rate_10 = 'NaN'::DOUBLE PRECISION
                  OR grass_win_rate_10 = 'Infinity'::DOUBLE PRECISION
                  OR grass_win_rate_10 = '-Infinity'::DOUBLE PRECISION
             THEN 'grass_win_rate_10 must be non-null and finite' END,
        CASE WHEN hard_win_rate_10 IS NULL
                  OR hard_win_rate_10 = 'NaN'::DOUBLE PRECISION
                  OR hard_win_rate_10 = 'Infinity'::DOUBLE PRECISION
                  OR hard_win_rate_10 = '-Infinity'::DOUBLE PRECISION
             THEN 'hard_win_rate_10 must be non-null and finite' END,
        CASE WHEN days_since_default IS NULL
                  OR days_since_default = 'NaN'::DOUBLE PRECISION
                  OR days_since_default = 'Infinity'::DOUBLE PRECISION
                  OR days_since_default = '-Infinity'::DOUBLE PRECISION
             THEN 'days_since_default must be non-null and finite' END,
        CASE WHEN matches_30d_default IS NULL
                  OR matches_30d_default = 'NaN'::DOUBLE PRECISION
                  OR matches_30d_default = 'Infinity'::DOUBLE PRECISION
                  OR matches_30d_default = '-Infinity'::DOUBLE PRECISION
             THEN 'matches_30d_default must be non-null and finite' END,
        CASE WHEN rate_default IS NULL
                  OR rate_default = 'NaN'::DOUBLE PRECISION
                  OR rate_default = 'Infinity'::DOUBLE PRECISION
                  OR rate_default = '-Infinity'::DOUBLE PRECISION
             THEN 'rate_default must be non-null and finite' END,
        CASE WHEN left_handed_rate IS NULL
                  OR left_handed_rate = 'NaN'::DOUBLE PRECISION
                  OR left_handed_rate = 'Infinity'::DOUBLE PRECISION
                  OR left_handed_rate = '-Infinity'::DOUBLE PRECISION
             THEN 'left_handed_rate must be non-null and finite' END,
        CASE WHEN avg_years_pro IS NULL
                  OR avg_years_pro = 'NaN'::DOUBLE PRECISION
                  OR avg_years_pro = 'Infinity'::DOUBLE PRECISION
                  OR avg_years_pro = '-Infinity'::DOUBLE PRECISION
             THEN 'avg_years_pro must be non-null and finite' END,
        -- Weighted tour benchmarks may be NULL (zero denominator) but must be
        -- finite when present.
        CASE WHEN tour_break_point_opportunities_per_return_game IS NOT NULL
                  AND (tour_break_point_opportunities_per_return_game = 'NaN'::DOUBLE PRECISION
                       OR tour_break_point_opportunities_per_return_game = 'Infinity'::DOUBLE PRECISION
                       OR tour_break_point_opportunities_per_return_game = '-Infinity'::DOUBLE PRECISION)
             THEN 'tour_break_point_opportunities_per_return_game must be finite when present' END,
        -- Observability counts are never negative.
        CASE WHEN snapshot_pool_rows < 0 THEN 'snapshot_pool_rows must be non-negative' END,
        CASE WHEN snapshot_pool_players < 0 THEN 'snapshot_pool_players must be non-negative' END,
        CASE WHEN profile_rows < 0 THEN 'profile_rows must be non-negative' END,
        CASE WHEN player_match_rows < 0 THEN 'player_match_rows must be non-negative' END,
        CASE WHEN pool_as_of_date IS NULL THEN 'pool_as_of_date must be non-null' END
    FROM {{ ref('tour_averages') }}
)
SELECT violation FROM checks WHERE violation IS NOT NULL
UNION ALL
SELECT 'too many rows' FROM {{ ref('tour_averages') }} HAVING COUNT(*) != 1
