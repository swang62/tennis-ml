-- Assert each player's player_match_number is unique; a stale ordinal from an
-- append-only incremental run after a historical bronze insert would collide
-- with the newly numbered row.
SELECT player_id, player_match_number
FROM {{ ref('player_matches') }}
GROUP BY player_id, player_match_number
HAVING COUNT(*) > 1
