-- One row per match: both players' stats in the row, winner_id names the victor.
-- Column order follows bronze.match_events in init.sql:
-- match_id, match_date, player1_id, player2_id, tournament, round, surface,
-- player1_ranking, player2_ranking,
-- player1: wins_last_10, matches_last_10, aces, double_faults,
--          first_serves_made, total_serve_points, break_points_won, break_points_total,
-- player2: wins_last_10, matches_last_10, aces, double_faults,
--          first_serves_made, total_serve_points, break_points_won, break_points_total,
-- winner_id
INSERT INTO bronze.match_events VALUES
('2026-ao-r1-001', '2026-01-13', 'A0E2', 'Z355', 'australian_open', 'r16', 'hard', 3, 7, 8, 10, 6, 2, 38, 65, 4, 8, 7, 10, 9, 2, 40, 66, 3, 7, 'A0E2'),
('2026-ao-r1-002', '2026-01-13', 'S0AG', 'R0DG', 'australian_open', 'r16', 'hard', 1, 5, 9, 10, 8, 1, 44, 72, 5, 9, 6, 10, 5, 3, 36, 55, 3, 8, 'S0AG'),
('2026-ao-r1-003', '2026-01-14', 'MM58', 'A0E2', 'australian_open', 'qf', 'hard', 8, 3, 6, 10, 2, 4, 38, 58, 3, 9, 8, 10, 8, 2, 41, 68, 4, 8, 'A0E2'),
('2026-ao-r1-004', '2026-01-14', 'Z355', 'R0DG', 'australian_open', 'qf', 'hard', 7, 5, 7, 10, 10, 3, 40, 68, 4, 7, 6, 10, 5, 3, 36, 55, 3, 8, 'Z355'),
('2026-ao-r1-005', '2026-01-15', 'S0AG', 'Z355', 'australian_open', 'sf', 'hard', 1, 7, 9, 10, 5, 1, 38, 64, 3, 6, 7, 10, 9, 2, 40, 66, 3, 7, 'S0AG'),
('2026-ao-r1-006', '2026-01-15', 'A0E2', 'R0DG', 'australian_open', 'sf', 'hard', 3, 5, 8, 10, 9, 0, 42, 70, 5, 8, 6, 10, 5, 3, 36, 55, 3, 8, 'A0E2'),
('2026-iw-r1-001', '2026-03-10', 'A0E2', 'S0AG', 'indian_wells', 'f', 'hard', 2, 1, 7, 10, 5, 3, 40, 66, 4, 8, 9, 10, 7, 1, 42, 70, 4, 8, 'A0E2'),
('2026-iw-r1-002', '2026-03-10', 'R0DG', 'MM58', 'indian_wells', 'r16', 'hard', 4, 7, 8, 10, 6, 1, 36, 58, 2, 6, 6, 10, 4, 3, 36, 56, 3, 8, 'R0DG'),
('2026-rg-r1-001', '2026-05-25', 'A0E2', 'S0AG', 'roland_garros', 'f', 'clay', 2, 1, 7, 10, 5, 3, 40, 62, 4, 9, 9, 10, 7, 1, 42, 70, 4, 8, 'A0E2'),
('2026-rg-r1-002', '2026-05-25', 'Z355', 'MM58', 'roland_garros', 'r16', 'clay', 6, 9, 6, 10, 7, 2, 34, 50, 2, 7, 6, 10, 4, 3, 36, 56, 3, 8, 'Z355'),
('2026-rg-r1-003', '2026-05-26', 'R0DG', 'S0AG', 'roland_garros', 'qf', 'clay', 4, 1, 5, 10, 4, 5, 36, 55, 3, 8, 9, 10, 7, 1, 42, 70, 4, 8, 'S0AG'),
('2026-rg-r1-004', '2026-05-26', 'A0E2', 'MM58', 'roland_garros', 'qf', 'clay', 2, 9, 8, 10, 8, 0, 42, 68, 5, 7, 6, 10, 4, 3, 36, 56, 3, 8, 'A0E2'),
('2026-wb-r1-001', '2026-07-01', 'S0AG', 'A0E2', 'wimbledon', 'f', 'grass', 1, 2, 9, 10, 12, 1, 46, 76, 4, 7, 8, 10, 8, 2, 41, 68, 4, 8, 'S0AG'),
('2026-wb-r1-002', '2026-07-01', 'Z355', 'R0DG', 'wimbledon', 'r16', 'grass', 5, 4, 7, 10, 14, 2, 38, 60, 2, 6, 6, 10, 5, 3, 36, 55, 3, 8, 'R0DG'),
('2026-wb-r1-003', '2026-07-02', 'MM58', 'S0AG', 'wimbledon', 'qf', 'grass', 8, 1, 4, 10, 6, 4, 32, 48, 1, 5, 9, 10, 7, 1, 42, 70, 4, 8, 'S0AG'),
('2026-wb-r1-004', '2026-07-02', 'A0E2', 'R0DG', 'wimbledon', 'qf', 'grass', 2, 4, 7, 10, 10, 2, 40, 66, 3, 6, 6, 10, 5, 3, 36, 55, 3, 8, 'A0E2'),
('2026-us-r1-001', '2026-08-26', 'S0AG', 'MM58', 'us_open', 'f', 'hard', 1, 6, 8, 10, 7, 1, 42, 70, 5, 8, 6, 10, 4, 3, 36, 56, 3, 8, 'S0AG'),
('2026-us-r1-002', '2026-08-26', 'A0E2', 'Z355', 'us_open', 'sf', 'hard', 2, 5, 8, 10, 11, 1, 44, 74, 6, 9, 7, 10, 9, 2, 40, 66, 3, 7, 'A0E2'),
('2026-us-r1-003', '2026-08-27', 'R0DG', 'MM58', 'us_open', 'r16', 'hard', 4, 6, 6, 10, 3, 3, 36, 52, 3, 8, 6, 10, 4, 3, 36, 56, 3, 8, 'MM58'),
('2026-us-r1-004', '2026-08-27', 'Z355', 'S0AG', 'us_open', 'qf', 'hard', 5, 1, 6, 10, 8, 2, 40, 62, 4, 9, 9, 10, 7, 1, 42, 70, 4, 8, 'S0AG');

-- Player identity backbone: canonical ATP ids/names and base metadata from
-- ATP_Database.csv (see data/raw/ATP_Database.csv). Enrichment columns
-- (summary, play_style, wiki_title, enriched_at) stay empty here and are
-- filled by the Wikipedia fallback in src/flows/ingest.py when needed.
INSERT INTO gold.player_profiles VALUES
('A0E2', 'Carlos Alcaraz', 'Carlos Alcaraz', DATE '2003-05-05', '74', '183', 2018, 'El Palmar, Spain', 'Juan Carlos Ferrero, Samuel Lopez', 'R', '2H', 'ESP', NULL, NULL, NULL, NULL),
('S0AG', 'Jannik Sinner', 'Jannik Sinner', DATE '2001-08-16', '77', '191', 2018, 'San Candido, Italy', 'Simone Vagnozzi, Darren Cahill', 'R', '2H', 'ITA', NULL, NULL, NULL, NULL),
('MM58', 'Daniil Medvedev', 'Daniil Medvedev', DATE '1996-02-11', '83', '198', 2014, 'Moscow, Russia', 'Gilles Cervara', 'R', '2H', 'RUS', NULL, NULL, NULL, NULL),
('R0DG', 'Holger Rune', 'Holger Rune', DATE '2003-04-29', '77', '188', 2020, 'Gentofte, Denmark', 'Lars Christensen, Kenneth Carlsen', 'R', '2H', 'DEN', NULL, NULL, NULL, NULL),
('Z355', 'Alexander Zverev', 'Alexander Zverev', DATE '1997-04-20', '90', '198', 2013, 'Hamburg, Germany', 'Alexander Zverev Sr.', 'R', '2H', 'GER', NULL, NULL, NULL, NULL);
