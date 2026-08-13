-- Contract: every FEATURE_COLS cell of gold.match_features is non-null and
-- finite. Any returned row is a violation. This enforces the finalized
-- model-ready contract: dbt build/test fails when the imputation contract is
-- broken (NULL, NaN, or infinite value in any model feature column).
--
-- The similarity-only appended columns (player/opponent_first_serve_pct_10,
-- ..., player/opponent_return_points_won_pct_10) are intentionally NOT
-- checked: they are not model features.
{% set feature_cols = [
    "rank_diff",
    "rank_points_diff",
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
    "streak_diff",
    "player_weighted_form_10",
    "opponent_weighted_form_10",
    "player_days_since_last_match",
    "opponent_days_since_last_match",
    "player_matches_30d",
    "opponent_matches_30d",
    "player_surface_win_rate_10",
    "opponent_surface_win_rate_10",
    "player_is_left_handed",
    "opponent_is_left_handed",
    "player_years_pro",
    "opponent_years_pro",
    "h2h_exposure",
    "h2h_advantage",
    "is_clay",
    "is_grass",
    "is_hard",
    "is_carpet",
    "is_indoor",
    "tournament_level",
    "round_encoded",
] %}
WITH violations AS (
    {% for col in feature_cols %}
    SELECT match_id, '{{ col }}' AS feature, {{ col }} AS value
    FROM {{ ref('match_features') }}
    WHERE {{ col }} IS NULL
       OR {{ col }} = 'NaN'::DOUBLE PRECISION
       OR {{ col }} = 'Infinity'::DOUBLE PRECISION
       OR {{ col }} = '-Infinity'::DOUBLE PRECISION
    {{ 'UNION ALL' if not loop.last }}
    {% endfor %}
)
SELECT * FROM violations
ORDER BY match_id, feature
