-- silver.player_matches: normalized player-perspective match rows.
--
-- Expand each bronze match into two raw player perspectives for downstream rolling features.
--
-- Activity fields are match ordinal and strictly-prior 30-day count. RANGE keeps
-- the cutoff relative to each match date.
--
-- ATP zero rank, points, and age mean missing, not a real value; map to NULL.
--
-- No canonicalization: player/opponent orientation is the raw player1/player2
-- assignment from bronze.
--
-- Context stays in bronze except surface; consumers rebuild both sides from perspectives.
--
-- Return points are derived from the opponent's raw serve totals for similarity rates.

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
        player2_total_serve_points
            - (player2_first_serve_points_won + player2_second_serve_points_won)
            AS return_points_won,
        player2_total_serve_points AS return_points_available,
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
        player1_total_serve_points
            - (player1_first_serve_points_won + player1_second_serve_points_won)
            AS return_points_won,
        player1_total_serve_points AS return_points_available,
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
    return_points_won,
    return_points_available,
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
