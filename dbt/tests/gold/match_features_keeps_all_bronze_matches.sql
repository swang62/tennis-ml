-- Assert every bronze match with a usable date produces exactly one
-- match_features row. Guards against silent row drops — e.g. unranked players
-- (rank NULL after the 0 -> NULL mapping) must never remove a match from
-- training, which is what the old `ranking > 0` filter did.
SELECT me.match_id
FROM {{ source('bronze', 'match_events') }} me
LEFT JOIN {{ ref('match_features') }} mf
    ON mf.match_id = me.match_id
WHERE me.match_id IS NOT NULL
  AND me.match_date IS NOT NULL
  AND mf.match_id IS NULL
