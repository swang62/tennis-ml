-- Assert (match_id, player_id) is unique in player_matches.
SELECT match_id, player_id FROM {{ ref('player_matches') }} GROUP BY 1, 2 HAVING COUNT(*) > 1
