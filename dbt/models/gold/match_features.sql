-- gold.match_features: canonical match-level training table.
--
-- One row per match. Rolling values use the prior snapshot; current rankings,
-- rank points, age, and 30-day activity come from player_matches. The lower
-- ATP id is canonical `player_*`, making raw player order irrelevant.
--
-- match_won = 1 iff the canonical player_* side won, else 0. It is the LABEL,
-- not a feature.
--
-- H2H uses the five most recent distinct, strictly-prior canonical meetings.
--
-- Cold starts retain NULL rolling values; days_since_last_match defaults to 365.
--
-- Unranked players are retained; training imputes NULL ranking values.
--
-- Model columns are metadata plus FEATURE_COLS; appended serve/return columns
-- are for similarity only. Height and current-match rates are not model features.

WITH player_match_enriched AS (
    SELECT
        pm.match_id,
        pm.match_date,
        bron.match_date       AS bronze_date,
        bron.tournament,
        bron.round,
        pm.surface,
        bron.is_indoor,
        pm.player_id,
        pm.opponent_id,
        pm.match_won,
        -- CURRENT event rankings/rank points/age, not the prior snapshot's
        pm.player_ranking,
        pm.opponent_ranking,
        pm.player_rank_points,
        pm.player_age,
        pm.player_match_number,

        -- Correct pre-match activity count (0 on first match; it is a COUNT)
        CAST(pm.matches_30d_before AS INTEGER) AS matches_30d,

        -- 365 on a first match.
        CAST(COALESCE(pm.match_date - pr.snapshot_date, 365)
            AS INTEGER) AS days_since_last_match,

        -- Prior rolling snapshot; NULL on cold start.
        pr.weighted_form_10,
        pr.win_rate_10,
        pr.ace_rate_10,
        pr.first_serve_pct_10,
        pr.break_points_saved_pct_10,
        pr.first_serve_win_pct_10,
        pr.second_serve_win_pct_10,
        pr.serve_win_pct_10,
        pr.return_points_won_pct_10,
        pr.df_rate_10,
        pr.aces_per_svc_game_10,

        -- Rank trend: prior rolling avg ranking minus CURRENT event ranking
        pr.avg_player_rank_10 - pm.player_ranking AS rank_trend_10,

        -- Strength of schedule: prior snapshot's rolling average opponent rank
        pr.avg_rank_faced_10,

        CAST(pr.streak AS INTEGER) AS streak,

        -- Surface-specific form: prior snapshot's rate on the current surface
        CASE pm.surface
            WHEN 'clay'  THEN pr.clay_win_rate_10
            WHEN 'grass' THEN pr.grass_win_rate_10
            WHEN 'hard'  THEN pr.hard_win_rate_10
        END AS surface_win_rate_10,

        -- Missing or non-L/R handedness stays NULL for train-time imputation.
        CAST(CASE WHEN prof.handedness = 'L' THEN 1
                  WHEN prof.handedness = 'R' THEN 0 END AS SMALLINT)
            AS is_left_handed,
        -- Time-aware years pro.
        CAST(EXTRACT(YEAR FROM pm.match_date) - prof.turned_pro AS INTEGER) AS years_pro

    FROM {{ ref('player_matches') }} pm
    LEFT JOIN {{ ref('rolling_features') }} pr
        ON pr.player_id = pm.player_id
       AND pr.player_match_number = pm.player_match_number - 1
    LEFT JOIN {{ source('bronze', 'match_events') }} bron
        ON bron.match_id = pm.match_id
    LEFT JOIN gold.player_profiles prof
        ON prof.player_id = pm.player_id
),
-- Dedupe player perspectives to canonical meetings for H2H aggregation.
pair_meetings AS (
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
-- Strictly-prior H2H rows, limited to the five most recent per match.
prior_meeting_rows AS (
    SELECT
        current_match.match_id,
        meeting.a_won,
        ROW_NUMBER() OVER (
            PARTITION BY current_match.match_id
            ORDER BY meeting.match_date DESC, meeting.match_id DESC
        ) AS rn
    FROM player_match_enriched current_match
    JOIN pair_meetings meeting
        ON meeting.a = current_match.player_id
       AND meeting.b = current_match.opponent_id
       AND meeting.match_date < current_match.match_date
),
prior_h2h AS (
    SELECT
        match_id,
        COUNT(*) AS player_h2h_matches,
        COALESCE(SUM(a_won), 0) AS player_h2h_wins
    FROM prior_meeting_rows
    WHERE rn <= 5
    GROUP BY match_id
)
-- Collapse perspectives into one lower-id canonical row per match.
SELECT
    p.match_id,
    p.match_date,
    p.player_id,
    p.opponent_id,
    p.tournament,
    p.round,
    p.surface,
    p.match_won,

    -- ── Matchup differences (canonical side minus opponent) ──
    p.player_ranking - o.player_ranking AS rank_diff,
    p.player_rank_points - o.player_rank_points AS rank_points_diff,
    p.player_age - o.player_age AS age_diff,
    p.win_rate_10 - o.win_rate_10 AS win_rate_diff,
    p.ace_rate_10 - o.ace_rate_10 AS ace_rate_diff,
    p.first_serve_pct_10 - o.first_serve_pct_10 AS first_serve_pct_diff,
    p.break_points_saved_pct_10 - o.break_points_saved_pct_10
        AS break_points_saved_pct_diff,
    p.first_serve_win_pct_10 - o.first_serve_win_pct_10
        AS first_serve_win_pct_diff,
    p.second_serve_win_pct_10 - o.second_serve_win_pct_10
        AS second_serve_win_pct_diff,
    p.serve_win_pct_10 - o.serve_win_pct_10 AS serve_win_pct_diff,
    p.df_rate_10 - o.df_rate_10 AS df_rate_diff,
    p.aces_per_svc_game_10 - o.aces_per_svc_game_10 AS aces_per_svc_game_diff,
    p.rank_trend_10 - o.rank_trend_10 AS rank_trend_diff,
    p.avg_rank_faced_10 - o.avg_rank_faced_10 AS avg_rank_faced_diff,
    p.streak - o.streak AS streak_diff,

    -- ── Absolute state values where both sides matter ──
    p.weighted_form_10      AS player_weighted_form_10,
    o.weighted_form_10      AS opponent_weighted_form_10,
    p.days_since_last_match AS player_days_since_last_match,
    o.days_since_last_match AS opponent_days_since_last_match,
    p.matches_30d           AS player_matches_30d,
    o.matches_30d           AS opponent_matches_30d,
    p.surface_win_rate_10   AS player_surface_win_rate_10,
    o.surface_win_rate_10   AS opponent_surface_win_rate_10,
    p.is_left_handed        AS player_is_left_handed,
    o.is_left_handed        AS opponent_is_left_handed,
    p.years_pro             AS player_years_pro,
    o.years_pro             AS opponent_years_pro,

    -- ── Canonical-player head-to-head history (0/0 when never met) ──
    COALESCE(h.player_h2h_matches, 0) AS player_h2h_matches,
    COALESCE(h.player_h2h_wins, 0)    AS player_h2h_wins,

    -- ── Numeric match context (one-hot surface for linear and neural models) ──
    CAST(CASE WHEN p.surface = 'clay'  THEN 1 ELSE 0 END AS SMALLINT) AS is_clay,
    CAST(CASE WHEN p.surface = 'grass' THEN 1 ELSE 0 END AS SMALLINT) AS is_grass,
    CAST(CASE WHEN p.surface = 'hard'  THEN 1 ELSE 0 END AS SMALLINT) AS is_hard,
    CAST(CASE WHEN p.surface = 'carpet' THEN 1 ELSE 0 END AS SMALLINT) AS is_carpet,
    b.is_indoor,
    CAST(CASE p.tournament
        WHEN 'grand_slam' THEN 4 WHEN 'masters' THEN 3
        WHEN 'atp_500' THEN 2 WHEN 'atp_250' THEN 1 ELSE 0
    END AS SMALLINT) AS tournament_level,
    CAST(CASE p.round
        WHEN 'r128' THEN 1 WHEN 'r64' THEN 2 WHEN 'r32' THEN 3 WHEN 'r16' THEN 4
        WHEN 'qf' THEN 5 WHEN 'sf' THEN 6 WHEN 'f' THEN 7 ELSE 0
    END AS SMALLINT) AS round_encoded,

    -- ── Similarity-analysis serve/return percentages (NOT model features) ──
    -- Appended style signals for PlayerSimilarity, never FEATURE_COLS.
    p.first_serve_pct_10        AS player_first_serve_pct_10,
    o.first_serve_pct_10        AS opponent_first_serve_pct_10,
    p.first_serve_win_pct_10    AS player_first_serve_win_pct_10,
    o.first_serve_win_pct_10    AS opponent_first_serve_win_pct_10,
    p.second_serve_win_pct_10   AS player_second_serve_win_pct_10,
    o.second_serve_win_pct_10   AS opponent_second_serve_win_pct_10,
    p.serve_win_pct_10          AS player_serve_win_pct_10,
    o.serve_win_pct_10          AS opponent_serve_win_pct_10,
    p.return_points_won_pct_10  AS player_return_points_won_pct_10,
    o.return_points_won_pct_10  AS opponent_return_points_won_pct_10
FROM player_match_enriched p
JOIN player_match_enriched o
    ON o.match_id = p.match_id
   AND o.player_id = p.opponent_id
LEFT JOIN bronze.match_events b
    ON b.match_id = p.match_id
LEFT JOIN prior_h2h h
    ON h.match_id = p.match_id
WHERE p.player_id < o.player_id
ORDER BY p.match_date, p.match_id
