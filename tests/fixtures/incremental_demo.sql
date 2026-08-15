-- Incremental ETL demo fixture: ONE new bronze match.
--
-- Apply this INSERT (e.g. `psql "$DATABASE_URL" -f tests/fixtures/incremental_demo.sql`)
-- and re-run `just db-etl`. Expected deltas on the next dbt build:
--
--   bronze.match_events      +1  (this row)
--   silver.player_matches    +2  (one perspective per player)
--   silver.rolling_features  +2  (one post-match snapshot per perspective)
--   gold.match_features      +2  (two directional rows per match)
--   gold.tour_averages       refreshed (still exactly 1 singleton row)
--   gold.player_profiles     refreshed (recomputed globally)
--
-- Re-running dbt build with no further bronze changes is a no-op (idempotent).
--
-- Player1 is the winner by bronze convention (winner_id = player1_id, the
-- CHECK enforced in infra/postgres/schema.sql). Values use existing pool players
-- (S0AG, A0E2) and satisfy every bronze CHECK constraint.

INSERT INTO bronze.match_events (
    match_id, match_date, player1_id, player2_id, tournament, tournament_name,
    round, surface, is_indoor, player1_ranking, player2_ranking,
    player1_wins_last_10, player1_matches_last_10,
    player1_aces, player1_double_faults, player1_first_serves_made,
    player1_total_serve_points, player1_first_serve_points_won,
    player1_second_serve_points_won, player1_service_games,
    player1_break_points_saved, player1_break_points_faced,
    player2_wins_last_10, player2_matches_last_10,
    player2_aces, player2_double_faults, player2_first_serves_made,
    player2_total_serve_points, player2_first_serve_points_won,
    player2_second_serve_points_won, player2_service_games,
    player2_break_points_saved, player2_break_points_faced,
    player1_rank_points, player2_rank_points, player1_age, player2_age, winner_id
) VALUES (
    '20260714-2026-316-011', DATE '2026-07-14', 'S0AG', 'A0E2', 'atp_250', 'Bastad',
    'r32', 'clay', 0, 1, 12,
    NULL, NULL,
    8, 2, 55,
    78, 40,
    12, 12,
    5, 8,
    NULL, NULL,
    4, 3, 48,
    70, 31,
    10, 11,
    4, 9,
    13450, 2450, 24.903, 22.4, 'S0AG'
);
