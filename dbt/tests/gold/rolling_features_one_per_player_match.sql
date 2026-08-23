SELECT pm.player_id, pm.match_id
FROM {{ ref('player_matches') }} pm
LEFT JOIN {{ ref('rolling_features') }} pr
  ON pr.player_id = pm.player_id AND pr.match_id = pm.match_id
WHERE pr.player_id IS NULL
