-- Assert the gold.player_profiles precision contract: every emitted
-- floating/statistical aggregate (weighted service/return/surface/career
-- rates and the latest rolling win rate) is truncated to 5 decimal places at
-- the final SELECT boundary (TRUNC(x::NUMERIC, 5)). Any returned row
-- violates the contract.
--
-- The comparison happens in the double domain on purpose: a value produced
-- by TRUNC(x::NUMERIC, 5)::DOUBLE PRECISION re-truncates to itself exactly,
-- while one carrying more than 5 decimals differs. Aggregates may be NULL for
-- zero-match players or zero denominators; NULLs are skipped by IS DISTINCT
-- FROM. Integer identifiers, counts, rank points, dates, and current_rank
-- are exempt.

{% set float_cols = [
    "first_serve_in_pct",
    "aces_per_first_serve",
    "first_serve_points_won_pct",
    "second_serve_points_won_pct",
    "overall_serve_points_won_pct",
    "double_faults_per_serve_point",
    "aces_per_service_game",
    "break_points_saved_pct",
    "return_points_won_pct",
    "first_serve_return_points_won_pct",
    "second_serve_return_points_won_pct",
    "return_games_won_pct",
    "break_point_conversion_pct",
    "break_point_opportunities_per_return_game",
    "hard_win_rate",
    "clay_win_rate",
    "grass_win_rate",
    "career_win_rate",
    "win_rate_10",
] %}

WITH violations AS (
    {% for col in float_cols %}
    SELECT '{{ col }}' AS column_name,
           {{ col }}::DOUBLE PRECISION AS value
    FROM {{ ref('player_profiles') }}
    WHERE {{ col }}::DOUBLE PRECISION
        IS DISTINCT FROM TRUNC({{ col }}::NUMERIC, 5)::DOUBLE PRECISION
    {% if not loop.last %}UNION ALL{% endif %}
    {% endfor %}
)
SELECT column_name, value FROM violations