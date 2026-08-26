WITH prior_meeting_rows AS (
    SELECT
        mf.match_id,
        mf.player_id,
        mf.opponent_id,
        mf.surface AS current_surface,
        (meeting.winner_id = mf.player_id)::INTEGER AS player_won,
        meeting.surface AS meeting_surface,
        meeting.player1_id,
        meeting.player2_id,
        ROW_NUMBER() OVER (
            PARTITION BY mf.match_id, mf.player_id,
                meeting.player1_id, meeting.player2_id
            ORDER BY meeting.match_date DESC, meeting.match_num DESC, meeting.match_id DESC
        ) AS recency
    FROM {{ ref('match_features') }} mf
    JOIN {{ ref('player_matches') }} pm
      ON pm.match_id = mf.match_id AND pm.player_id = mf.player_id
    JOIN {{ source('bronze', 'match_events') }} meeting
      ON ((meeting.player1_id = mf.player_id AND meeting.player2_id = mf.opponent_id)
       OR (meeting.player1_id = mf.opponent_id AND meeting.player2_id = mf.player_id))
     AND (meeting.match_date, meeting.match_num, meeting.match_id)
           < (pm.match_date, pm.match_num, pm.match_id)
),
derived AS (
    SELECT
        match_id,
        player_id,
        COUNT(*) AS exp_recent_meetings,
        SUM(player_won) AS exp_player_wins,
        COUNT(*) FILTER (WHERE meeting_surface = current_surface) AS exp_surface_meetings,
        SUM(CASE WHEN meeting_surface = current_surface AND player_won = 1 THEN 1 ELSE 0 END)
            AS exp_surface_wins
    FROM prior_meeting_rows
    WHERE recency <= 10
    GROUP BY match_id, player_id
)
SELECT
    mf.match_id,
    mf.player_id,
    mf.opponent_id,
    mf.h2h_exposure,
    mf.h2h_advantage,
    mf.h2h_surface_advantage,
    COALESCE(d.exp_recent_meetings, 0) AS exp_recent_meetings,
    COALESCE(d.exp_player_wins, 0) AS exp_player_wins,
    (COALESCE(d.exp_player_wins, 0) + 1.0)
        / (COALESCE(d.exp_recent_meetings, 0) + 2.0) - 0.5 AS exp_advantage,
    (COALESCE(d.exp_surface_wins, 0) + 1.0)
        / (COALESCE(d.exp_surface_meetings, 0) + 2.0) - 0.5 AS exp_surface_advantage
FROM {{ ref('match_features') }} mf
LEFT JOIN derived d
    ON d.match_id = mf.match_id
   AND d.player_id = mf.player_id
WHERE mf.h2h_exposure IS DISTINCT FROM COALESCE(d.exp_recent_meetings, 0)
    OR ABS(mf.h2h_advantage - ((
        (COALESCE(d.exp_player_wins, 0) + 1.0)
        / (COALESCE(d.exp_recent_meetings, 0) + 2.0) - 0.5
    )::DOUBLE PRECISION)) > 1e-6
    OR ABS(mf.h2h_surface_advantage - ((
        (COALESCE(d.exp_surface_wins, 0) + 1.0)
        / (COALESCE(d.exp_surface_meetings, 0) + 2.0) - 0.5
    )::DOUBLE PRECISION)) > 1e-6
