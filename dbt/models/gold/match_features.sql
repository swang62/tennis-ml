-- gold.match_features: canonical match-level training table.
--
-- ONE ROW PER MATCH (not per player). Each player-match row is paired with
-- that player's IMMEDIATELY PRECEDING post-match snapshot
-- (player_match_number - 1) from gold.player_rolling_features, so all rolling
-- history values come strictly from completed matches BEFORE the current
-- event. The current event (gold.player_matches) supplies match context,
-- CURRENT rankings, and the correct pre-match 30-day activity count.
-- The two perspectives then collapse into one canonical row: the lower/stable
-- ATP id is the `player_*` side, the other is `opponent_*`, so the table is
-- order-invariant (swapping player1/player2 in bronze changes nothing).
--
-- match_won = 1 iff the canonical player_* side won, else 0. It is the LABEL,
-- not a feature: no current-match serve/break/outcome values are exposed.
--
-- Cold start: a player's first match has no prior snapshot, so all rolling
-- features are NULL (no zero filling). The only documented fallback is
-- days_since_last_match = 365; matches_30d is naturally 0 (it is a COUNT).

WITH player_match_enriched AS (
    SELECT
        pm.match_id,
        pm.match_date,
        pm.tournament,
        pm.round,
        pm.surface,
        pm.player_id,
        pm.opponent_id,
        pm.match_won,
        -- CURRENT event rankings, not the prior snapshot's
        pm.player_ranking,
        pm.opponent_ranking,
        pm.player_match_number,

        -- Correct pre-match activity count (0 on first match; it is a COUNT)
        CAST(pm.matches_30d_before AS INTEGER) AS matches_30d,

        -- Days since the player's previous completed match; 365 on first match
        CAST(COALESCE(DATEDIFF('day', pr.snapshot_date, pm.match_date), 365)
            AS INTEGER) AS days_since_last_match,

        -- Rolling form from the immediately preceding snapshot (NULL on cold
        -- start — no zero filling)
        pr.win_rate_5,
        pr.win_rate_10,
        pr.win_rate_20,
        pr.ace_rate_5,
        pr.ace_rate_10,
        pr.first_serve_pct_5,
        pr.first_serve_pct_10,
        pr.break_pct_5,
        pr.break_pct_10,

        -- Rank trend: prior rolling avg ranking minus CURRENT event ranking
        pr.avg_player_rank_10 - pm.player_ranking AS rank_trend_10,
        pr.avg_player_rank_20 - pm.player_ranking AS rank_trend_20,

        CAST(pr.win_streak AS UTINYINT) AS win_streak,

        -- Surface-specific form: prior snapshot's rate on the current surface
        CASE pm.surface
            WHEN 'clay'  THEN pr.clay_win_rate_10
            WHEN 'grass' THEN pr.grass_win_rate_10
            WHEN 'hard'  THEN pr.hard_win_rate_10
        END AS surface_win_rate_10,

        -- Profile-derived identity (static; NULL when the player has no
        -- profile or a cell is missing, same no-zero-filling policy as the
        -- rolling features — train-time imputation handles the NULLs)
        prof.height,
        -- Non-L/R handedness (including NULL) stays NULL so train-time
        -- imputation treats missing handedness like any other missing cell
        CAST(CASE WHEN prof.handedness = 'L' THEN 1
                  WHEN prof.handedness = 'R' THEN 0 END AS UTINYINT)
            AS is_left_handed,
        -- Years pro AT THIS MATCH (time-aware), not the raw turned_pro year
        CAST(YEAR(pm.match_date) - prof.turned_pro AS INTEGER) AS years_pro

    FROM {{ ref('player_matches') }} pm
    LEFT JOIN {{ ref('player_rolling_features') }} pr
        ON pr.player_id = pm.player_id
       AND pr.player_match_number = pm.player_match_number - 1
    LEFT JOIN gold.player_profiles prof
        ON prof.player_id = pm.player_id
    WHERE pm.player_ranking > 0
      AND pm.opponent_ranking > 0
)
-- Collapse the two player-perspective rows of each match into one canonical
-- row: the lower ATP id is the player_* side (p), the higher is the
-- opponent_* side (o). Exactly one row per match, regardless of the raw
-- player1/player2 assignment.
SELECT
    p.match_id,
    p.match_date,
    p.player_id,
    p.opponent_id,
    p.tournament,
    p.round,
    p.surface,
    p.match_won,

    -- Canonical player side (lower ATP id)
    p.player_ranking,
    p.win_rate_5            AS player_win_rate_5,
    p.win_rate_10           AS player_win_rate_10,
    p.win_rate_20           AS player_win_rate_20,
    p.ace_rate_5            AS player_ace_rate_5,
    p.ace_rate_10           AS player_ace_rate_10,
    p.first_serve_pct_5     AS player_first_serve_pct_5,
    p.first_serve_pct_10    AS player_first_serve_pct_10,
    p.break_pct_5           AS player_break_pct_5,
    p.break_pct_10          AS player_break_pct_10,
    p.rank_trend_10         AS player_rank_trend_10,
    p.rank_trend_20         AS player_rank_trend_20,
    p.win_streak            AS player_win_streak,
    p.days_since_last_match AS player_days_since_last_match,
    p.matches_30d           AS player_matches_30d,
    p.surface_win_rate_10   AS player_surface_win_rate_10,

    -- Profile-derived identity (canonical player side)
    p.height            AS player_height,
    p.is_left_handed    AS player_is_left_handed,
    p.years_pro         AS player_years_pro,

    -- Opponent side (higher ATP id)
    o.player_ranking AS opponent_ranking,
    o.win_rate_5            AS opponent_win_rate_5,
    o.win_rate_10           AS opponent_win_rate_10,
    o.win_rate_20           AS opponent_win_rate_20,
    o.ace_rate_5            AS opponent_ace_rate_5,
    o.ace_rate_10           AS opponent_ace_rate_10,
    o.first_serve_pct_5     AS opponent_first_serve_pct_5,
    o.first_serve_pct_10    AS opponent_first_serve_pct_10,
    o.break_pct_5           AS opponent_break_pct_5,
    o.break_pct_10          AS opponent_break_pct_10,
    o.rank_trend_10         AS opponent_rank_trend_10,
    o.rank_trend_20         AS opponent_rank_trend_20,
    o.win_streak            AS opponent_win_streak,
    o.days_since_last_match AS opponent_days_since_last_match,
    o.matches_30d           AS opponent_matches_30d,
    o.surface_win_rate_10   AS opponent_surface_win_rate_10,

    -- Profile-derived identity (opponent side)
    o.height            AS opponent_height,
    o.is_left_handed    AS opponent_is_left_handed,
    o.years_pro         AS opponent_years_pro,

    -- Comparison (differential) features: canonical side minus opponent side
    p.player_ranking - o.player_ranking AS rank_diff,
    p.win_rate_10 - o.win_rate_10 AS win_rate_diff,
    p.ace_rate_10 - o.ace_rate_10 AS ace_rate_diff,
    p.break_pct_10 - o.break_pct_10 AS break_pct_diff,
    CAST(p.win_streak AS INTEGER) - CAST(o.win_streak AS INTEGER) AS win_streak_diff,
    p.matches_30d - o.matches_30d AS matches_30d_diff,
    p.surface_win_rate_10 - o.surface_win_rate_10 AS surface_win_rate_diff,
    p.rank_trend_10 - o.rank_trend_10 AS rank_trend_diff,
    p.height - o.height AS height_diff,
    CAST(p.is_left_handed AS INTEGER) - CAST(o.is_left_handed AS INTEGER)
        AS handedness_diff,
    p.years_pro - o.years_pro AS years_pro_diff,

    -- Context
    CAST(CASE WHEN p.surface = 'clay'  THEN 1 ELSE 0 END AS UTINYINT) AS is_clay,
    CAST(CASE WHEN p.surface = 'grass' THEN 1 ELSE 0 END AS UTINYINT) AS is_grass,
    CAST(CASE WHEN p.surface = 'hard'  THEN 1 ELSE 0 END AS UTINYINT) AS is_hard,
    CAST(CASE p.tournament
        WHEN 'grand_slam' THEN 4 WHEN 'masters' THEN 3
        WHEN 'atp_500' THEN 2 WHEN 'atp_250' THEN 1 ELSE 0
    END AS UTINYINT) AS tournament_level,
    CAST(CASE p.round
        WHEN 'r128' THEN 1 WHEN 'r64' THEN 2 WHEN 'r32' THEN 3 WHEN 'r16' THEN 4
        WHEN 'qf' THEN 5 WHEN 'sf' THEN 6 WHEN 'f' THEN 7 ELSE 0
    END AS UTINYINT) AS round_encoded
FROM player_match_enriched p
JOIN player_match_enriched o
    ON o.match_id = p.match_id
   AND o.player_id = p.opponent_id
WHERE p.player_id < o.player_id
ORDER BY p.match_date, p.match_id
