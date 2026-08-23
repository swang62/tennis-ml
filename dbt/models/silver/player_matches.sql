
{{ config(
    materialized="incremental",
    incremental_strategy="delete+insert",
    unique_key=["player_id", "match_id"],
) }}

WITH expanded AS (
    SELECT
        match_id, match_date, surface, match_num,
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
        match_id, match_date, surface, match_num,
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
),
numbered AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY player_id ORDER BY match_date, match_num, match_id
        ) AS player_match_number,
        COUNT(*) OVER (
            PARTITION BY player_id ORDER BY match_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING
        ) AS matches_30d_before
    FROM expanded
)
{% if is_incremental() %}
, changed_players AS (
    SELECT DISTINCT numbered.player_id
    FROM numbered
    LEFT JOIN {{ this }} t
        ON t.player_id = numbered.player_id
       AND t.match_id = numbered.match_id
    WHERE numbered.match_id IS NOT NULL
      AND numbered.match_date IS NOT NULL
      AND (
          t.match_id IS NULL
          OR t.player_match_number <> numbered.player_match_number
          OR t.matches_30d_before <> numbered.matches_30d_before
      )
)
{% endif %}
SELECT
    match_id,
    match_date,
    surface,
    match_num,
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
    player_match_number,
    matches_30d_before
FROM numbered
WHERE match_id IS NOT NULL
  AND match_date IS NOT NULL
{% if is_incremental() %}
  AND player_id IN (SELECT player_id FROM changed_players)
{% endif %}
ORDER BY match_date, match_num, match_id, player_id
