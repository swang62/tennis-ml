-- Regression: avg_rank_faced_5/10 skip NULL opponent rankings — an unranked
-- opponent inside the window must never pollute the strength-of-schedule
-- average (a COALESCE(rank, 0) or a non-skipping COUNT formulation would
-- diverge). Recompute the rolling average over the same window with AVG
-- (which ignores NULLs) and compare against the stored snapshot value.
WITH windowed AS (
    SELECT
        pm.player_id,
        pm.match_id,
        AVG(pm.opponent_ranking) OVER (
            PARTITION BY pm.player_id ORDER BY pm.match_date, pm.match_id
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ) AS expected_5,
        AVG(pm.opponent_ranking) OVER (
            PARTITION BY pm.player_id ORDER BY pm.match_date, pm.match_id
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        ) AS expected_10
    FROM {{ ref('player_matches') }} pm
)
SELECT
    r.player_id,
    r.match_id,
    r.player_match_number,
    r.avg_rank_faced_5 AS actual_5,
    w.expected_5,
    r.avg_rank_faced_10 AS actual_10,
    w.expected_10
FROM {{ ref('rolling_features') }} r
JOIN windowed w
    ON w.player_id = r.player_id AND w.match_id = r.match_id
WHERE r.avg_rank_faced_5 IS DISTINCT FROM w.expected_5
   OR r.avg_rank_faced_10 IS DISTINCT FROM w.expected_10
