-- Assert one snapshot per (player_id, match_id) in player_rolling_features.
SELECT player_id, match_id FROM {{ ref('player_rolling_features') }} GROUP BY 1, 2 HAVING COUNT(*) > 1
