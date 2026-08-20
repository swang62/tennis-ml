-- Assert the silver.rolling_features precision contract: every emitted
-- floating/statistical column is truncated to 5 decimal places at the
-- computed boundary (TRUNC(x::NUMERIC, 5) in the computed CTE, which the
-- outermost SELECT passes through unchanged). Any returned row violates the
-- contract.
--
-- The comparison happens in the double domain on purpose: a double produced
-- by TRUNC(x::NUMERIC, 5)::DOUBLE PRECISION re-truncates to itself exactly,
-- while a value carrying more than 5 decimals differs. NULLs (unseen
-- surfaces, unscored matches, unranked-opponent windows) are skipped by
-- IS DISTINCT FROM. Integer identifiers, ordinals, counts, dates, ranks,
-- rank points, and the signed streak are not float columns and are exempt.

{% set float_cols = [
    "latest_player_age",
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
    "game_margin_10",
    "avg_player_rank_10",
    "avg_rank_faced_10",
    "clay_win_rate_10",
    "grass_win_rate_10",
    "hard_win_rate_10",
] %}

WITH violations AS (
    {% for col in float_cols %}
    SELECT '{{ col }}' AS column_name,
           {{ col }}::DOUBLE PRECISION AS value
    FROM {{ ref('rolling_features') }}
    WHERE {{ col }}::DOUBLE PRECISION
        IS DISTINCT FROM TRUNC({{ col }}::NUMERIC, 5)::DOUBLE PRECISION
    {% if not loop.last %}UNION ALL{% endif %}
    {% endfor %}
)
SELECT column_name, value FROM violations