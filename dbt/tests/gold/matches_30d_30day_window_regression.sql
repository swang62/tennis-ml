-- Regression: matches_30d_before counts only prior matches STRICTLY within
-- the 30-day window [match_date - 30 days, match_date). Recompute the count
-- per player_matches row and compare; the old buggy ROWS-frame formulation
-- counted every preceding row instead.
WITH expected AS (
    SELECT
        pm.player_id,
        pm.match_id,
        COUNT(prior.match_id) AS expected_count
    FROM {{ ref('player_matches') }} pm
    LEFT JOIN {{ ref('player_matches') }} prior
      ON prior.player_id = pm.player_id
     AND prior.match_date >= pm.match_date - INTERVAL '30 days'
     AND prior.match_date <  pm.match_date
    GROUP BY pm.player_id, pm.match_id
)
SELECT
    pm.player_id,
    pm.match_id,
    pm.matches_30d_before AS actual,
    expected.expected_count
FROM {{ ref('player_matches') }} pm
JOIN expected
  ON expected.player_id = pm.player_id AND expected.match_id = pm.match_id
WHERE pm.matches_30d_before IS DISTINCT FROM expected.expected_count
