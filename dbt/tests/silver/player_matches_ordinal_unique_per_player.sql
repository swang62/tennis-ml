SELECT player_id, player_match_number
FROM {{ ref('player_matches') }}
GROUP BY player_id, player_match_number
HAVING COUNT(*) > 1
