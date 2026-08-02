-- gold.player_matches: normalized player-perspective match rows.
--
-- Each bronze.match_events row (one row per match holding both players' stats)
-- expands into exactly two player-perspective rows: one with player1_id as the
-- player, one with player2_id as the player. Rows keep the player's current
-- match context, ranking, raw serve/break stats, and outcome only — no rolling
-- features, no pairwise differentials, no encoded context columns. This is the
-- expansion step that match_features.sql previously inlined; downstream models
-- build rolling features from here.
--
-- Event-relative activity fields (per player, ordered by match_date, match_id):
--   * player_match_number: 1-based ordinal of this player's matches.
--   * previous_match_date: date of this player's immediately preceding match;
--     NULL for the player's first match.
--   * matches_30d_before: count of this player's PRIOR matches with match_date
--     in [match_date - 30 days, match_date). Excludes the current match and any
--     match on the same date (only strictly earlier dates qualify); 0 for the
--     player's first match. Computed with a RANGE frame so the 30-day cutoff is
--     relative to the current row's date — the ROWS-based formulation in the
--     old match_features.sql compared each frame row against itself, counting
--     every preceding row instead.
--
-- No canonicalization: player/opponent orientation is the raw player1/player2
-- assignment from bronze.

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
        winner_id,
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
        winner_id,
        CASE WHEN winner_id = player2_id THEN 1 ELSE 0 END AS match_won
    FROM {{ source('bronze', 'match_events') }}
)
SELECT
    match_id,
    match_date,
    tournament,
    round,
    surface,
    player_id,
    opponent_id,
    player_ranking,
    opponent_ranking,
    winner_id,
    match_won,
    aces,
    double_faults,
    first_serves_made,
    total_serve_points,
    break_points_won,
    break_points_total,
    wins_last_10,
    matches_last_10,
    ROW_NUMBER() OVER (
        PARTITION BY player_id ORDER BY match_date, match_id
    ) AS player_match_number,
    LAG(match_date) OVER (
        PARTITION BY player_id ORDER BY match_date, match_id
    ) AS previous_match_date,
    COUNT(*) OVER (
        PARTITION BY player_id ORDER BY match_date
        RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING
    ) AS matches_30d_before
FROM expanded
WHERE match_id IS NOT NULL
  AND match_date IS NOT NULL
ORDER BY match_date, match_id, player_id
