-- Assert every match_features H2H value is exactly re-derivable from strictly
-- prior meetings (same-date excluded), so H2H never includes the current match.
-- h2h_exposure = count of the five most recent deduped meetings; h2h_advantage =
-- (prior wins + 1)/(prior meetings + 2) - 0.5 from the same window. Compared at
-- full precision (1e-6 tolerance). Any returned row is a violation.
WITH pair_meetings AS (
    -- One row per distinct unordered pair match; a_won from the lower-id side.
    SELECT
        LEAST(player_id, opponent_id) AS a,
        GREATEST(player_id, opponent_id) AS b,
        match_id,
        match_date,
        MAX(match_num) AS match_num,
        MAX(CASE WHEN player_id < opponent_id THEN match_won
                 ELSE 1 - match_won END) AS a_won
    FROM {{ ref('player_matches') }}
    GROUP BY 1, 2, 3, 4
),
prior_meeting_rows AS (
    SELECT
        mf.match_id,
        mf.player_id,
        mf.opponent_id,
        meeting.a_won,
        ROW_NUMBER() OVER (
            PARTITION BY mf.match_id, mf.player_id
            ORDER BY meeting.match_date DESC, meeting.match_num DESC, meeting.match_id DESC
        ) AS rn
    FROM {{ ref('match_features') }} mf
    JOIN {{ ref('player_matches') }} pm
      ON pm.match_id = mf.match_id AND pm.player_id = mf.player_id
    JOIN pair_meetings meeting
        ON meeting.a = LEAST(mf.player_id, mf.opponent_id)
       AND meeting.b = GREATEST(mf.player_id, mf.opponent_id)
       AND (meeting.match_date, meeting.match_num, meeting.match_id)
           < (pm.match_date, pm.match_num, pm.match_id)
),
derived AS (
    SELECT
        match_id,
        player_id,
        COUNT(*) FILTER (WHERE rn <= 5) AS exp_recent_meetings,
        SUM(CASE WHEN rn <= 5 AND player_id < opponent_id THEN a_won
                 WHEN rn <= 5 THEN 1 - a_won END) AS exp_player_wins
    FROM prior_meeting_rows
    GROUP BY match_id, player_id
)
SELECT
    mf.match_id,
    mf.player_id,
    mf.opponent_id,
    mf.h2h_exposure,
    mf.h2h_advantage,
    COALESCE(d.exp_recent_meetings, 0) AS exp_recent_meetings,
    COALESCE(d.exp_player_wins, 0) AS exp_player_wins,
    (COALESCE(d.exp_player_wins, 0) + 1.0)
        / (COALESCE(d.exp_recent_meetings, 0) + 2.0) - 0.5 AS exp_advantage
FROM {{ ref('match_features') }} mf
LEFT JOIN derived d
    ON d.match_id = mf.match_id
   AND d.player_id = mf.player_id
WHERE mf.h2h_exposure IS DISTINCT FROM COALESCE(d.exp_recent_meetings, 0)
   OR ABS(mf.h2h_advantage - ((
       (COALESCE(d.exp_player_wins, 0) + 1.0)
       / (COALESCE(d.exp_recent_meetings, 0) + 2.0) - 0.5
   )::DOUBLE PRECISION)) > 1e-6