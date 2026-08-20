-- Assert every match_features H2H value is exactly re-derivable from prior
-- meetings, so no row's H2H counts include the current match itself:
--   * meetings strictly before the row's match_date (same-date excluded),
--   * deduped to distinct match_ids (silver has 2 rows per match),
--   * oriented per directional row, keyed on (match_id, player_id).
-- h2h_exposure is the count of the FIVE most recent such meetings (identical
-- for both mirrors); h2h_advantage is the Beta(1,1)-smoothed directional value
-- built from the same bounded recent-five window (match_date DESC, match_id
-- DESC): (row player's prior wins + 1) / (prior meetings + 2) - 0.5 and
-- negates across mirrors. Neither value is truncated by the model, so the
-- derived values are compared at full precision (1e-6 tolerance). Any returned
-- row is a violation.
WITH pair_meetings AS (
    -- One row per distinct match between an unordered pair; a_won is the
    -- lower-id (LEAST/GREATEST canonicalization for lookup only) side's
    -- outcome (both perspective rows agree, so MAX is safe).
    SELECT
        LEAST(player_id, opponent_id) AS a,
        GREATEST(player_id, opponent_id) AS b,
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
        mf.player_id,
        mf.opponent_id,
        meeting.a_won,
        ROW_NUMBER() OVER (
            PARTITION BY mf.match_id, mf.player_id
            ORDER BY meeting.match_date DESC, meeting.match_id DESC
        ) AS rn
    FROM {{ ref('match_features') }} mf
    JOIN pair_meetings meeting
        ON meeting.a = LEAST(mf.player_id, mf.opponent_id)
       AND meeting.b = GREATEST(mf.player_id, mf.opponent_id)
       AND meeting.match_date < mf.match_date
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