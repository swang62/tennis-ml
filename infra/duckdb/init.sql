CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;

-- Raw match data: one row per match with both players' stats in the row.
-- The gold layer expands each row into two player-perspective rows.
CREATE TABLE IF NOT EXISTS bronze.match_events (
    match_id                   VARCHAR,
    match_date                 DATE,
    player1_id                 VARCHAR,
    player2_id                 VARCHAR,
    tournament                 VARCHAR,
    round                      VARCHAR,
    surface                    VARCHAR,
    player1_ranking            INTEGER,
    player2_ranking            INTEGER,
    player1_wins_last_10       UTINYINT,
    player1_matches_last_10    UTINYINT,
    player1_aces               UTINYINT,
    player1_double_faults      UTINYINT,
    player1_first_serves_made  UTINYINT,
    player1_total_serve_points UTINYINT,
    player1_first_serve_points_won   UTINYINT,
    player1_second_serve_points_won  UTINYINT,
    player1_service_games      UTINYINT,
    player1_break_points_saved UTINYINT,
    player1_break_points_faced UTINYINT,
    player2_wins_last_10       UTINYINT,
    player2_matches_last_10    UTINYINT,
    player2_aces               UTINYINT,
    player2_double_faults      UTINYINT,
    player2_first_serves_made  UTINYINT,
    player2_total_serve_points UTINYINT,
    player2_first_serve_points_won   UTINYINT,
    player2_second_serve_points_won  UTINYINT,
    player2_service_games      UTINYINT,
    player2_break_points_saved UTINYINT,
    player2_break_points_faced UTINYINT,
    player1_rank_points        INTEGER,
    player2_rank_points        INTEGER,
    player1_age                DOUBLE,
    player2_age                DOUBLE,
    winner_id                  VARCHAR,
    PRIMARY KEY (match_id)
);

-- Secondary indexes for the common gold-layer expansion/rolling query
-- patterns (player1/player2 sides of bronze are scanned per player + date).
CREATE INDEX IF NOT EXISTS idx_match_events_p1_date
    ON bronze.match_events (player1_id, match_date);
CREATE INDEX IF NOT EXISTS idx_match_events_p2_date
    ON bronze.match_events (player2_id, match_date);

-- Identity backbone for players, sourced from the ATP player database
-- (data/ATP_player_database.csv, canonical ATP id/name + base metadata).
-- Enrichment columns at the bottom are left empty by the ATP load and filled
-- by the Wikipedia fallback in src/flows/ingest.py for players it covers.
CREATE TABLE IF NOT EXISTS gold.player_profiles (
    player_id    VARCHAR PRIMARY KEY,  -- canonical ATP id (ATP_Database.id)
    display_name VARCHAR,              -- canonical ATP name (ATP_Database.player)
    atp_name     VARCHAR,              -- alternate name variant (ATP_Database.atpname)
    birthdate    DATE,                 -- ATP_Database.birthdate (YYYYMMDD)
    weight       SMALLINT,             -- kg (ATP_Database.weight) NULL when unknown
    height       SMALLINT,             -- cm (ATP_Database.height) NULL when unknown
    turned_pro   INTEGER,              -- year turned pro (ATP_Database.turnedpro)
    birthplace   VARCHAR,              -- ATP_Database.birthplace
    coaches      VARCHAR,              -- ATP_Database.coaches
    handedness   VARCHAR,              -- ATP_Database.hand (R/L)
    backhand     VARCHAR,              -- ATP_Database.backhand (1H/2H)
    ioc          VARCHAR,              -- ATP_Database.ioc country code
    -- Wikipedia enrichment (fallback for incomplete ATP rows)
    summary      VARCHAR,
    enriched_at  TIMESTAMP
);
