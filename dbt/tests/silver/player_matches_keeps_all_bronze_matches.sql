-- Assert every bronze match with a usable date produces exactly two
-- player_matches rows (one per player). Guards the incremental append
-- boundary: if the match_id filter silently dropped or duplicated a match,
-- this diverges from bronze and fails.
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
