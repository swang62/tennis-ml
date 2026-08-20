-- Assert the gold.match_features precision contract: every emitted
-- floating/statistical column — matchup differences, absolute weighted-form
-- state values, the h2h advantage, the years-pro state, and the
-- similarity-only serve/return columns — is truncated to 5 decimal places at
-- the final SELECT boundary (TRUNC(x::NUMERIC, 5)). Any returned row
-- violates the contract.
--
-- The comparison happens in the double domain on purpose: a value produced
-- by TRUNC(x::NUMERIC, 5)::DOUBLE PRECISION re-truncates to itself exactly,
-- while one carrying more than 5 decimals differs. Every FEATURE_COLS cell is
-- non-null (match_features_no_null_model_features), so NULLs are not
-- expected; IS DISTINCT FROM keeps the test NULL-safe regardless.
-- Integer identifiers, match_won (the label), counts, the 0/1 flags
-- (player/opponent_is_left_handed, is_clay/is_grass/is_hard, is_indoor),
-- encoded categoricals, and the integer-valued rank/rank-points/streak
-- differences are exempt.

{% set float_cols = [
    "age_diff",
    "win_rate_diff",
    "ace_rate_diff",
    "first_serve_pct_diff",
    "break_points_saved_pct_diff",
    "first_serve_win_pct_diff",
    "second_serve_win_pct_diff",
    "serve_win_pct_diff",
    "return_points_won_pct_diff",
    "df_rate_diff",
    "aces_per_svc_game_diff",
    "rank_trend_diff",
    "avg_rank_faced_diff",
    "surface_form_diff",
    "days_since_last_match_diff",
    "recent_game_margin_diff",
    "player_weighted_form_10",
    "opponent_weighted_form_10",
    "player_years_pro",
    "opponent_years_pro",
    "h2h_advantage",
    "player_first_serve_pct_10",
    "opponent_first_serve_pct_10",
    "player_first_serve_win_pct_10",
    "opponent_first_serve_win_pct_10",
    "player_second_serve_win_pct_10",
    "opponent_second_serve_win_pct_10",
    "player_serve_win_pct_10",
    "opponent_serve_win_pct_10",
    "player_return_points_won_pct_10",
    "opponent_return_points_won_pct_10",
] %}

WITH violations AS (
    {% for col in float_cols %}
    SELECT '{{ col }}' AS column_name,
           {{ col }}::DOUBLE PRECISION AS value
    FROM {{ ref('match_features') }}
    WHERE {{ col }}::DOUBLE PRECISION
        IS DISTINCT FROM TRUNC({{ col }}::NUMERIC, 5)::DOUBLE PRECISION
    {% if not loop.last %}UNION ALL{% endif %}
    {% endfor %}
)
SELECT column_name, value FROM violations