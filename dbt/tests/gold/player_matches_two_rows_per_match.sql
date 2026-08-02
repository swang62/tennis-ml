-- Assert every match in player_matches has exactly two rows (one per player).
SELECT match_id FROM {{ ref('player_matches') }} GROUP BY match_id HAVING COUNT(*) <> 2
