CREATE SCHEMA IF NOT EXISTS bronze;
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
    player1_break_points_won   UTINYINT,
    player1_break_points_total UTINYINT,
    player2_wins_last_10       UTINYINT,
    player2_matches_last_10    UTINYINT,
    player2_aces               UTINYINT,
    player2_double_faults      UTINYINT,
    player2_first_serves_made  UTINYINT,
    player2_total_serve_points UTINYINT,
    player2_break_points_won   UTINYINT,
    player2_break_points_total UTINYINT,
    winner_id                  VARCHAR
);

-- Identity backbone for players, sourced from ATP_Database.csv
-- (canonical ATP id/name + base metadata). Enrichment columns at the bottom
-- are left empty by the ATP load and filled by the Wikipedia fallback in
-- src/flows/ingest.py for players missing from the ATP database.
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
    play_style   VARCHAR,
    wiki_title   VARCHAR,
    enriched_at  TIMESTAMP
);
