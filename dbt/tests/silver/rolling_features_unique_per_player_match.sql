-- Assert exactly one rolling snapshot per (player_id, match_id). Duplicate
-- source rows (the surface-match join fanning out over stale ordinals) break
-- the composite unique key and the delete+insert; this guards the mirror
-- direction of rolling_features_one_per_player_match (orphans).
SELECT player_id, match_id
FROM {{ ref('rolling_features') }}
GROUP BY player_id, match_id
HAVING COUNT(*) > 1
