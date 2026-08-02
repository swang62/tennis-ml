-- gold.match_features: canonical match-level training table.
--
-- ONE ROW PER MATCH (not per player). The two sides are canonicalized by ATP
-- id ordering: the lower/stable ATP id becomes the canonical `player_*` side,
-- the other becomes `opponent_*`. The full balanced feature set is included:
-- canonical player-side rolling stats, opponent-side rolling stats, plus
-- differential (comparison) and context features.
--
-- match_won = 1 iff the canonical player_* side won, else 0.
--
-- Per-player rolling features are computed in an internal expansion (two
-- player-perspective rows per match) and then collapsed back to one canonical
-- row per match, so the table is order-invariant: swapping player1/player2 in
-- bronze.match_events changes nothing.

-- 1) Expand each raw match into two player-perspective rows (internal only).
WITH expanded AS (
    SELECT
        match_id, match_date, tournament, round, surface,
        player1_id AS player_id,
        player2_id AS opponent_id,
        player1_ranking AS player_ranking,
        player2_ranking AS opponent_ranking,
        player1_wins_last_10 AS wins_last_10,
        player1_matches_last_10 AS matches_last_10,
        player1_aces AS aces,
        player1_double_faults AS double_faults,
        player1_first_serves_made AS first_serves_made,
        player1_total_serve_points AS total_serve_points,
        player1_break_points_won AS break_points_won,
        player1_break_points_total AS break_points_total,
        CASE WHEN winner_id = player1_id THEN 1 ELSE 0 END AS match_won
    FROM {{ source('bronze', 'match_events') }}

    UNION ALL

    SELECT
        match_id, match_date, tournament, round, surface,
        player2_id AS player_id,
        player1_id AS opponent_id,
        player2_ranking AS player_ranking,
        player1_ranking AS opponent_ranking,
        player2_wins_last_10 AS wins_last_10,
        player2_matches_last_10 AS matches_last_10,
        player2_aces AS aces,
        player2_double_faults AS double_faults,
        player2_first_serves_made AS first_serves_made,
        player2_total_serve_points AS total_serve_points,
        player2_break_points_won AS break_points_won,
        player2_break_points_total AS break_points_total,
        CASE WHEN winner_id = player2_id THEN 1 ELSE 0 END AS match_won
    FROM {{ source('bronze', 'match_events') }}
),
base AS (
    SELECT
        *,
        wins_last_10 / NULLIF(matches_last_10, 0) AS win_rate_last_10,
        aces / NULLIF(first_serves_made, 0) AS ace_rate,
        double_faults / NULLIF(total_serve_points, 0) AS double_fault_rate,
        first_serves_made / NULLIF(total_serve_points, 0) AS first_serve_pct,
        break_points_won / NULLIF(break_points_total, 0) AS break_points_converted_pct
    FROM expanded
    WHERE match_id IS NOT NULL
      AND match_date IS NOT NULL
      AND player_ranking > 0
      AND opponent_ranking > 0
),
with_windows AS (
    SELECT
        *,

        -- Rolling aggregates (exclude current match)
        avg(match_won)  OVER w5  AS win_rate_5,
        avg(match_won)  OVER w10 AS win_rate_10,
        avg(match_won)  OVER w20 AS win_rate_20,

        sum(aces) OVER w5  / NULLIF(sum(first_serves_made) OVER w5,  0) AS ace_rate_5,
        sum(aces) OVER w10 / NULLIF(sum(first_serves_made) OVER w10, 0) AS ace_rate_10,

        sum(first_serves_made) OVER w5
            / NULLIF(sum(total_serve_points) OVER w5,  0) AS first_serve_pct_5,
        sum(first_serves_made) OVER w10
            / NULLIF(sum(total_serve_points) OVER w10, 0) AS first_serve_pct_10,

        sum(break_points_won) OVER w5
            / NULLIF(sum(break_points_total) OVER w5,  0) AS break_pct_5,
        sum(break_points_won) OVER w10
            / NULLIF(sum(break_points_total) OVER w10, 0) AS break_pct_10,

        avg(player_ranking) OVER w10 - player_ranking AS rank_trend_10,
        avg(player_ranking) OVER w20 - player_ranking AS rank_trend_20,

        -- Days since player's last match
        DATEDIFF('day', LAG(match_date) OVER w_all, match_date) AS days_since_last_match,

        -- Matches in last 30 days before this match
        SUM(CASE WHEN match_date >= match_date - INTERVAL '30 days' THEN 1 ELSE 0 END)
            OVER (PARTITION BY player_id ORDER BY match_date
                  ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS matches_30d,

        -- Surface-specific win rate (rolling 10 on same surface)
        avg(match_won) OVER w_surface10 AS surface_win_rate_10

    FROM base

    WINDOW
        w_all  AS (PARTITION BY player_id ORDER BY match_date
                   ROWS BETWEEN UNBOUNDED PRECEDING AND 0 PRECEDING),
        w5     AS (PARTITION BY player_id ORDER BY match_date
                   ROWS BETWEEN 5  PRECEDING AND 1 PRECEDING),
        w10    AS (PARTITION BY player_id ORDER BY match_date
                   ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING),
        w20    AS (PARTITION BY player_id ORDER BY match_date
                   ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING),
        w_surface10 AS (PARTITION BY player_id, surface ORDER BY match_date
                        ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING)
),
with_win_streak AS (
    SELECT
        *,
        row_number() OVER (PARTITION BY player_id ORDER BY match_date) AS rn
    FROM with_windows
),
with_streak AS (
    SELECT
        *,
        rn - (
            MAX(CASE WHEN match_won = 0 THEN rn ELSE 0 END) OVER (
                PARTITION BY player_id ORDER BY match_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            )
        ) - 1 AS win_streak
    FROM with_win_streak
),
player_rows AS (
    SELECT
        match_id, match_date, tournament, round, surface,
        player_id, opponent_id, player_ranking, opponent_ranking,
        wins_last_10, matches_last_10,
        aces, double_faults, first_serves_made, total_serve_points,
        break_points_won, break_points_total, match_won,

        win_rate_last_10, ace_rate, double_fault_rate,
        first_serve_pct, break_points_converted_pct,

        win_rate_5, win_rate_10, win_rate_20,
        ace_rate_5, ace_rate_10,
        first_serve_pct_5, first_serve_pct_10,
        break_pct_5, break_pct_10,
        rank_trend_10, rank_trend_20,
        CAST(win_streak AS UTINYINT) AS win_streak,
        CAST(COALESCE(days_since_last_match, 365) AS INTEGER) AS days_since_last_match,
        CAST(COALESCE(matches_30d, 0) AS UTINYINT) AS matches_30d,
        COALESCE(surface_win_rate_10, 0) AS surface_win_rate_10
    FROM with_streak
)
-- 2) Collapse the two player-perspective rows of each match into one
--    canonical row: the lower ATP id is the player_* side (p), the higher
--    is the opponent_* side (o). Exactly one row per match, regardless of
--    the raw player1/player2 assignment.
SELECT
    p.match_id,
    p.match_date,
    p.player_id,
    p.opponent_id,
    p.tournament,
    p.round,
    p.surface,
    p.match_won,

    -- Canonical player side
    p.player_ranking,
    p.wins_last_10, p.matches_last_10,
    p.aces, p.double_faults, p.first_serves_made, p.total_serve_points,
    p.break_points_won, p.break_points_total,
    p.win_rate_last_10, p.ace_rate, p.double_fault_rate,
    p.first_serve_pct, p.break_points_converted_pct,
    p.win_rate_5 AS player_win_rate_5, p.win_rate_10 AS player_win_rate_10,
    p.win_rate_20 AS player_win_rate_20,
    p.ace_rate_5 AS player_ace_rate_5, p.ace_rate_10 AS player_ace_rate_10,
    p.first_serve_pct_5 AS player_first_serve_pct_5, p.first_serve_pct_10 AS player_first_serve_pct_10,
    p.break_pct_5 AS player_break_pct_5, p.break_pct_10 AS player_break_pct_10,
    p.rank_trend_10 AS player_rank_trend_10, p.rank_trend_20 AS player_rank_trend_20,
    p.win_streak AS player_win_streak,
    p.days_since_last_match AS player_days_since_last_match,
    p.matches_30d AS player_matches_30d,
    p.surface_win_rate_10 AS player_surface_win_rate_10,

    -- Opponent side
    o.player_ranking AS opponent_ranking,
    o.wins_last_10 AS opponent_wins_last_10, o.matches_last_10 AS opponent_matches_last_10,
    o.aces AS opponent_aces, o.double_faults AS opponent_double_faults,
    o.first_serves_made AS opponent_first_serves_made, o.total_serve_points AS opponent_total_serve_points,
    o.break_points_won AS opponent_break_points_won, o.break_points_total AS opponent_break_points_total,
    o.win_rate_last_10 AS opponent_win_rate_last_10, o.ace_rate AS opponent_ace_rate,
    o.double_fault_rate AS opponent_double_fault_rate,
    o.first_serve_pct AS opponent_first_serve_pct,
    o.break_points_converted_pct AS opponent_break_points_converted_pct,
    o.win_rate_5 AS opponent_win_rate_5, o.win_rate_10 AS opponent_win_rate_10,
    o.win_rate_20 AS opponent_win_rate_20,
    o.ace_rate_5 AS opponent_ace_rate_5, o.ace_rate_10 AS opponent_ace_rate_10,
    o.first_serve_pct_5 AS opponent_first_serve_pct_5, o.first_serve_pct_10 AS opponent_first_serve_pct_10,
    o.break_pct_5 AS opponent_break_pct_5, o.break_pct_10 AS opponent_break_pct_10,
    o.rank_trend_10 AS opponent_rank_trend_10, o.rank_trend_20 AS opponent_rank_trend_20,
    o.win_streak AS opponent_win_streak,
    o.days_since_last_match AS opponent_days_since_last_match,
    o.matches_30d AS opponent_matches_30d,
    o.surface_win_rate_10 AS opponent_surface_win_rate_10,

    -- Comparison (differential) features: canonical side minus opponent side
    p.player_ranking - o.player_ranking AS rank_diff,
    p.win_rate_10 - o.win_rate_10 AS win_rate_diff,
    p.ace_rate_10 - o.ace_rate_10 AS ace_rate_diff,
    p.break_pct_10 - o.break_pct_10 AS break_pct_diff,
    CAST(p.win_streak AS INTEGER) - CAST(o.win_streak AS INTEGER) AS win_streak_diff,
    CAST(p.matches_30d AS INTEGER) - CAST(o.matches_30d AS INTEGER) AS matches_30d_diff,
    p.surface_win_rate_10 - o.surface_win_rate_10 AS surface_win_rate_diff,
    p.rank_trend_10 - o.rank_trend_10 AS rank_trend_diff,

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
FROM player_rows p
JOIN player_rows o
    ON o.match_id = p.match_id
   AND o.player_id = p.opponent_id
WHERE p.player_id < o.player_id
ORDER BY p.match_date, p.match_id
