-- Assert the tour_averages singleton contract: every fallback default that
-- feeds gold.match_features imputation and live inference is non-null and
-- finite (NULL alone does not catch NaN/Infinity), and the relation holds
-- exactly one row (the yml unique/not_null/accepted_values on singleton_id
-- covers duplicates and identity but passes vacuously on an empty table).
--
-- Unlike the old single-row-of-CASE version (which reported only the first
-- failing column per row), this uses a UNION ALL of one SELECT per fallback
-- column, each uniformly aliasing `column_name`/`value` and filtering its own
-- NULL/NaN/inf condition — so a violation in ANY fallback column returns a
-- row, not just the first checked one. Any returned row is a violation.

{% set fallback_cols = [
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
] %}

WITH violations AS (
    {% for col in fallback_cols %}
    {{ non_null_finite(col) }}
    {% if not loop.last %}UNION ALL{% endif %}
    {% endfor %}
    UNION ALL
    -- Weighted tour benchmarks may be NULL (zero denominator) but must be
    -- finite when present.
    SELECT 'tour_break_point_opportunities_per_return_game' AS column_name,
           tour_break_point_opportunities_per_return_game AS value
    FROM {{ ref('tour_averages') }}
    WHERE tour_break_point_opportunities_per_return_game IS NOT NULL
      AND (tour_break_point_opportunities_per_return_game = 'NaN'::DOUBLE PRECISION
           OR tour_break_point_opportunities_per_return_game = 'Infinity'::DOUBLE PRECISION
           OR tour_break_point_opportunities_per_return_game = '-Infinity'::DOUBLE PRECISION)
    UNION ALL
    SELECT 'tour_return_games_won_pct' AS column_name,
           tour_return_games_won_pct AS value
    FROM {{ ref('tour_averages') }}
    WHERE tour_return_games_won_pct IS NOT NULL
      AND (tour_return_games_won_pct = 'NaN'::DOUBLE PRECISION
           OR tour_return_games_won_pct = 'Infinity'::DOUBLE PRECISION
           OR tour_return_games_won_pct = '-Infinity'::DOUBLE PRECISION)
)
SELECT column_name, value FROM violations
UNION ALL
SELECT 'too many rows', NULL::DOUBLE PRECISION
FROM {{ ref('tour_averages') }} HAVING COUNT(*) != 1
