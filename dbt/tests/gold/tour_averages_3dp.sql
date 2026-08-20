-- Assert the tour_averages precision contract: every emitted
-- floating/statistical output is rounded to 3 decimal places at the model's
-- output boundary (ROUND(x::NUMERIC, 3)::DOUBLE PRECISION in the final
-- SELECT). Any returned row violates the contract.
--
-- The comparison happens in the double domain on purpose: a double produced
-- by ROUND(x::NUMERIC, 3)::DOUBLE PRECISION re-rounds to itself exactly,
-- while any value carrying more than 3 decimals differs. Weighted tour
-- benchmark columns may be NULL (zero denominator) and are skipped by
-- IS DISTINCT FROM; fallback columns are non-null (tour_averages_contract).

{% set float_cols = [
    "latest_player_ranking",
    "latest_player_rank_points",
    "latest_player_age",
    "streak",
    "weighted_form_10",
    "win_rate_10",
    "ace_rate_10",
    "first_serve_pct_10",
    "break_points_saved_pct_10",
    "first_serve_win_pct_10",
    "second_serve_win_pct_10",
    "serve_win_pct_10",
    "return_points_won_pct_10",
    "df_rate_10",
    "aces_per_svc_game_10",
    "avg_player_rank_10",
    "avg_rank_faced_10",
    "clay_win_rate_10",
    "grass_win_rate_10",
    "hard_win_rate_10",
    "days_since_default",
    "matches_30d_default",
    "rate_default",
    "left_handed_rate",
    "avg_years_pro",
    "tour_ace_rate",
    "tour_first_serve_pct",
    "tour_break_points_saved_pct",
    "tour_first_serve_win_pct",
    "tour_second_serve_win_pct",
    "tour_serve_win_pct",
    "tour_return_points_won_pct",
    "tour_df_rate",
    "tour_aces_per_svc_game",
    "tour_break_point_opportunities_per_return_game",
    "tour_return_games_won_pct",
] %}

WITH violations AS (
    {% for col in float_cols %}
    SELECT '{{ col }}' AS column_name,
           {{ col }}::DOUBLE PRECISION AS value
    FROM {{ ref('tour_averages') }}
    WHERE {{ col }}::DOUBLE PRECISION
        IS DISTINCT FROM ROUND({{ col }}::NUMERIC, 3)::DOUBLE PRECISION
    {% if not loop.last %}UNION ALL{% endif %}
    {% endfor %}
)
SELECT column_name, value FROM violations