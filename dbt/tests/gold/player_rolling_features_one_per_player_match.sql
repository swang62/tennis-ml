-- Assert every player_matches row has exactly one snapshot: rows in
-- player_matches without a matching player_rolling_features row are orphans.
SELECT pm.player_id, pm.match_id
FROM {{ ref('player_matches') }} pm
LEFT JOIN {{ ref('player_rolling_features') }} pr
  ON pr.player_id = pm.player_id AND pr.match_id = pm.match_id
WHERE pr.player_id IS NULL
