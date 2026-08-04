-- Assert win_streak and loss_streak are never both positive on one snapshot.
-- They are mutually exclusive by construction (a loss resets win_streak to 0
-- and extends loss_streak, and vice versa); any returned row is a violation.
SELECT
    player_id,
    match_id,
    win_streak,
    loss_streak
FROM {{ ref('rolling_features') }}
WHERE win_streak > 0 AND loss_streak > 0
