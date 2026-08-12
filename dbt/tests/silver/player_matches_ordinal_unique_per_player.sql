-- Assert each player's player_match_number is unique. A stale ordinal left by
-- an append-only incremental run after a historical bronze insert collides
-- with the newly numbered row; the affected-player rebuild exists to prevent
-- exactly that.
SELECT player_id, player_match_number
FROM {{ ref('player_matches') }}
GROUP BY player_id, player_match_number
HAVING COUNT(*) > 1
