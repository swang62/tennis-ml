-- silver.player_matches: normalized player-perspective match rows.
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
--   * matches_30d_before: count of this player's PRIOR matches with match_date
--     in [match_date - 30 days, match_date). Excludes the current match and any
--     match on the same date (only strictly earlier dates qualify); 0 for the
--     player's first match. Computed with a RANGE frame so the 30-day cutoff is
--     relative to the current row's date — the ROWS-based formulation in the
--     old match_features.sql compared each frame row against itself, counting
--     every preceding row instead.
--
-- Ranking semantics: ATP rank 0 is the CSV missing marker for unranked
-- players, so player_ranking/opponent_ranking are NULLIF'd to NULL. Rank 0
-- would otherwise read as "better than rank 1" and corrupt rank_diff,
-- rank_trend, and the rolling average-rank windows downstream. NULLs are
-- imputed at train time (median) and by the inference pool (median).
-- The same 0 -> NULL mapping applies to rank_points (0 rank points is the CSV
-- missing marker) and age (0 is the missing marker): a 0 would otherwise
-- corrupt rank_points_diff downstream.
--
-- No canonicalization: player/opponent orientation is the raw player1/player2
-- assignment from bronze.
--
-- Task 6 reductions: tournament/round/is_indoor are JOINED from bronze by
-- match_id in the consumers that need them (kept out of this table; only
-- surface stays because rolling surface form needs it). winner_id,
-- opponent_rank_points, opponent_age, the ATP-provided wins_last_10 /
-- matches_last_10, and previous_match_date are removed. Each player row keeps
-- its own player_rank_points / player_age, so the collapse in
-- gold.match_features still has both sides' rank points and age via the two
-- perspective rows.

WITH expanded AS (
    SELECT
        match_id, match_date, surface,
        player1_id AS player_id,
        player2_id AS opponent_id,
        NULLIF(player1_ranking, 0) AS player_ranking,
        NULLIF(player2_ranking, 0) AS opponent_ranking,
        NULLIF(player1_rank_points, 0) AS player_rank_points,
        NULLIF(player1_age, 0) AS player_age,
        player1_aces AS aces,
        player1_double_faults AS double_faults,
        player1_first_serves_made AS first_serves_made,
        player1_total_serve_points AS total_serve_points,
        player1_first_serve_points_won AS first_serve_points_won,
        player1_second_serve_points_won AS second_serve_points_won,
        player1_service_games AS service_games,
        player1_break_points_saved AS break_points_saved,
        player1_break_points_faced AS break_points_faced,
        CASE WHEN winner_id = player1_id THEN 1 ELSE 0 END AS match_won
    FROM {{ source('bronze', 'match_events') }}

    UNION ALL

    SELECT
        match_id, match_date, surface,
        player2_id AS player_id,
        player1_id AS opponent_id,
        NULLIF(player2_ranking, 0) AS player_ranking,
        NULLIF(player1_ranking, 0) AS opponent_ranking,
        NULLIF(player2_rank_points, 0) AS player_rank_points,
        NULLIF(player2_age, 0) AS player_age,
        player2_aces AS aces,
        player2_double_faults AS double_faults,
        player2_first_serves_made AS first_serves_made,
        player2_total_serve_points AS total_serve_points,
        player2_first_serve_points_won AS first_serve_points_won,
        player2_second_serve_points_won AS second_serve_points_won,
        player2_service_games AS service_games,
        player2_break_points_saved AS break_points_saved,
        player2_break_points_faced AS break_points_faced,
        CASE WHEN winner_id = player2_id THEN 1 ELSE 0 END AS match_won
    FROM {{ source('bronze', 'match_events') }}
)
SELECT
    match_id,
    match_date,
    surface,
    player_id,
    opponent_id,
    player_ranking,
    opponent_ranking,
    player_rank_points,
    player_age,
    match_won,
    aces,
    double_faults,
    first_serves_made,
    total_serve_points,
    first_serve_points_won,
    second_serve_points_won,
    service_games,
    break_points_saved,
    break_points_faced,
    ROW_NUMBER() OVER (
        PARTITION BY player_id ORDER BY match_date, match_id
    ) AS player_match_number,
    COUNT(*) OVER (
        PARTITION BY player_id ORDER BY match_date
        RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING
    ) AS matches_30d_before
FROM expanded
WHERE match_id IS NOT NULL
  AND match_date IS NOT NULL
ORDER BY match_date, match_id, player_id