-- Assert the signed `streak` always encodes a consistent run: a streak of k>0
-- consecutive wins or k<0 consecutive losses, and never 0 mid-history. Any
-- returned row is a violation.
--   * The sign of streak matches the snapshot's outcome (win -> >0, loss -> <0).
--   * The run is contiguous: within a player's history, consecutive same-result
--     snapshots move streak toward 0 by 1 (|streak_n - streak_{n-1}| = 1 when
--     the result matches the current streak's direction, and the run resets
--     when the result flips).
-- The strongest exact check is: streak == (player_match_number - LAG_position)
-- where LAG_position is the most recent snapshot (inclusive) of the run. We
-- recompute it from the raw match_won sequence.
WITH runs AS (
    SELECT
        r.player_id,
        r.match_id,
        r.player_match_number,
        pm.match_won,
        r.streak,
        -- Recompute the signed run from the raw wins directly.
        r.player_match_number
            - COALESCE(
                MAX(CASE WHEN pm.match_won = 0 THEN r.player_match_number END)
                    OVER (PARTITION BY r.player_id ORDER BY r.player_match_number
                          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
                0
              ) AS win_run,
        r.player_match_number
            - COALESCE(
                MAX(CASE WHEN pm.match_won = 1 THEN r.player_match_number END)
                    OVER (PARTITION BY r.player_id ORDER BY r.player_match_number
                          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
                0
              ) AS loss_run
    FROM {{ ref('rolling_features') }} r
    JOIN {{ ref('player_matches') }} pm
        ON pm.player_id = r.player_id AND pm.match_id = r.match_id
)
SELECT player_id, match_id, player_match_number, streak
FROM runs
WHERE streak IS DISTINCT FROM (CASE WHEN match_won = 1 THEN win_run ELSE -loss_run END)