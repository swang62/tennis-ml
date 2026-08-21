-- Assert every dated bronze match yields exactly two directional rows. Guards
-- against unranked players (rank NULL) dropping a match from training, and
-- against an incremental run materializing only one perspective.
SELECT me.match_id
FROM {{ source('bronze', 'match_events') }} me
LEFT JOIN {{ ref('match_features') }} mf
    ON mf.match_id = me.match_id
WHERE me.match_id IS NOT NULL
  AND me.match_date IS NOT NULL
GROUP BY me.match_id
HAVING COUNT(mf.match_id) <> 2
