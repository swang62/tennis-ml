-- Assert every match_features H2H value is exactly re-derivable from prior
-- meetings, so no row's H2H counts include the current match itself:
--   * matches strictly before the row's match_date (same-date excluded),
--   * deduped to distinct match_ids (silver has 2 rows per match),
--   * restricted to the most recent 5 (match_date DESC, match_id DESC).
-- Also asserts the cross-side invariants: player_h2h_matches ==
-- opponent_h2h_matches, wins sum to matches, and win rates sum to 1
-- (or are both 0.5 when there are zero prior meetings). Any returned row is
-- a violation.
WITH pair_meetings AS (
    -- One row per distinct match between a canonical pair; a_won is the
    -- canonical a-side's outcome (both perspective rows agree, so MAX is safe).
    SELECT
        CASE WHEN player_id < opponent_id THEN player_id ELSE opponent_id END AS a,
        CASE WHEN player_id < opponent_id THEN opponent_id ELSE player_id END AS b,
        match_id,
        match_date,
        MAX(CASE WHEN player_id < opponent_id THEN match_won
                 ELSE 1 - match_won END) AS a_won
    FROM {{ ref('player_matches') }}
    GROUP BY 1, 2, 3, 4
),
prior_meeting_rows AS (
    SELECT
        mf.match_id,
        meeting.a_won,
        ROW_NUMBER() OVER (
            PARTITION BY mf.match_id
            ORDER BY meeting.match_date DESC, meeting.match_id DESC
        ) AS rn
    FROM {{ ref('match_features') }} mf
    JOIN pair_meetings meeting
        ON meeting.a = mf.player_id
       AND meeting.b = mf.opponent_id
       AND meeting.match_date < mf.match_date
),
derived AS (
    SELECT
        match_id,
        COUNT(*) AS exp_matches,
        COALESCE(SUM(a_won), 0) AS exp_player_wins
    FROM prior_meeting_rows
    WHERE rn <= 5
    GROUP BY match_id
)
SELECT
    mf.match_id,
    mf.player_id,
    mf.opponent_id,
    mf.player_h2h_matches,
    mf.player_h2h_wins,
    mf.opponent_h2h_wins,
    COALESCE(d.exp_matches, 0) AS exp_matches,
    COALESCE(d.exp_player_wins, 0) AS exp_player_wins
FROM {{ ref('match_features') }} mf
LEFT JOIN derived d USING (match_id)
WHERE mf.player_h2h_matches IS DISTINCT FROM COALESCE(d.exp_matches, 0)
   OR mf.opponent_h2h_matches IS DISTINCT FROM mf.player_h2h_matches
   OR mf.player_h2h_wins + mf.opponent_h2h_wins <> mf.player_h2h_matches
   OR mf.player_h2h_wins IS DISTINCT FROM COALESCE(d.exp_player_wins, 0)
   OR (mf.player_h2h_matches > 0
       AND ABS(mf.player_h2h_win_rate + mf.opponent_h2h_win_rate - 1) > 1e-9)
   OR (mf.player_h2h_matches = 0
       AND (mf.player_h2h_win_rate <> 0.5 OR mf.opponent_h2h_win_rate <> 0.5))
