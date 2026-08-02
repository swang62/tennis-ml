-- Assert (player_id, player_match_number) is unique in player_matches.
SELECT player_id, player_match_number FROM {{ ref('player_matches') }} GROUP BY 1, 2 HAVING COUNT(*) > 1
