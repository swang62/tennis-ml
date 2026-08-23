SELECT me.match_id
FROM {{ source('bronze', 'match_events') }} me
LEFT JOIN (
    SELECT match_id, COUNT(*) AS n
    FROM {{ ref('player_matches') }}
    GROUP BY match_id
) pm
    ON pm.match_id = me.match_id
WHERE me.match_id IS NOT NULL
  AND me.match_date IS NOT NULL
  AND (pm.n IS NULL OR pm.n <> 2)
