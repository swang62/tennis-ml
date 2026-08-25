SELECT
    a.match_id,
    a.player_id,
    a.opponent_id,
    a.match_won
FROM {{ ref('match_features') }} a
LEFT JOIN {{ ref('match_features') }} b
    ON b.match_id = a.match_id
   AND b.player_id = a.opponent_id
   AND b.opponent_id = a.player_id
   AND b.match_won = 1 - a.match_won
WHERE b.match_id IS NULL
