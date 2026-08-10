-- PostgreSQL bootstrap: structure only.
--
-- Runs on any standard PostgreSQL 18 instance: the Compose `postgres:18.4`
-- service executes it from /docker-entrypoint-initdb.d/init.sql on a fresh
-- data volume (never on restart), and host operators may apply it manually to
-- the configured local database, e.g.:
--
--   psql -U <user> -d <db> -f infra/postgres/init.sql
--
-- It is idempotent (CREATE ... IF NOT EXISTS), so re-running it is safe.

-- Creates structure only: the three schemas and the non-dbt-owned base tables
-- (bronze.match_events and bronze.player_profiles). No data is loaded here —
-- dbt owns silver.player_matches / silver.rolling_features /
-- gold.match_features / gold.tour_averages / gold.player_profiles, and data
-- is written later by just db-seed / db-etl. Nothing is baked into an image;
-- the Compose named volume persists everything.

CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;

-- Raw match data: one row per match with both players' stats in the row.
-- The gold layer expands each row into two player-perspective rows.
--
-- The small-integer count columns hold counts in the 0..255 range the row
-- validator (src.features.validate) enforces; that range is re-asserted here
-- as a CHECK. Rank 0 is this project's missing/unranked marker, so
-- ranking/rank_points allow 0.
CREATE TABLE IF NOT EXISTS bronze.match_events (
    match_id                   VARCHAR NOT NULL,
    match_date                 DATE    NOT NULL,
    player1_id                 VARCHAR NOT NULL,
    player2_id                 VARCHAR NOT NULL,
    tournament                 VARCHAR NOT NULL,
    tournament_name            VARCHAR,
    round                      VARCHAR,
    surface                    VARCHAR NOT NULL,
    is_indoor                  SMALLINT,
    player1_ranking            INTEGER NOT NULL,
    player2_ranking            INTEGER NOT NULL,
    player1_wins_last_10       SMALLINT,
    player1_matches_last_10    SMALLINT,
    player1_aces               SMALLINT NOT NULL,
    player1_double_faults      SMALLINT NOT NULL,
    player1_first_serves_made  SMALLINT NOT NULL,
    player1_total_serve_points SMALLINT NOT NULL,
    player1_first_serve_points_won   SMALLINT NOT NULL,
    player1_second_serve_points_won  SMALLINT NOT NULL,
    player1_service_games      SMALLINT NOT NULL,
    player1_break_points_saved SMALLINT NOT NULL,
    player1_break_points_faced SMALLINT NOT NULL,
    player2_wins_last_10       SMALLINT,
    player2_matches_last_10    SMALLINT,
    player2_aces               SMALLINT NOT NULL,
    player2_double_faults      SMALLINT NOT NULL,
    player2_first_serves_made  SMALLINT NOT NULL,
    player2_total_serve_points SMALLINT NOT NULL,
    player2_first_serve_points_won   SMALLINT NOT NULL,
    player2_second_serve_points_won  SMALLINT NOT NULL,
    player2_service_games      SMALLINT NOT NULL,
    player2_break_points_saved SMALLINT NOT NULL,
    player2_break_points_faced SMALLINT NOT NULL,
    player1_rank_points        INTEGER NOT NULL,
    player2_rank_points        INTEGER NOT NULL,
    player1_age                DOUBLE PRECISION NOT NULL,
    player2_age                DOUBLE PRECISION NOT NULL,
    winner_id                  VARCHAR NOT NULL,
    PRIMARY KEY (match_id),
    CONSTRAINT match_events_check_players_distinct CHECK (player1_id <> player2_id),
    CONSTRAINT match_events_check_winner         CHECK (winner_id = player1_id),
    CONSTRAINT match_events_check_ranking        CHECK (
        player1_ranking >= 0 AND player2_ranking >= 0
        AND player1_rank_points BETWEEN 0 AND 20000 AND player2_rank_points BETWEEN 0 AND 20000
    ),
    CONSTRAINT match_events_check_age             CHECK (
        player1_age BETWEEN 0 AND 100 AND player2_age BETWEEN 0 AND 100
    ),
    CONSTRAINT match_events_check_integer_counts  CHECK (
        player1_wins_last_10 BETWEEN 0 AND 255 AND player1_matches_last_10 BETWEEN 0 AND 255
        AND player1_aces BETWEEN 0 AND 255 AND player1_double_faults BETWEEN 0 AND 255
        AND player1_first_serves_made BETWEEN 0 AND 255 AND player1_total_serve_points BETWEEN 0 AND 255
        AND player1_first_serve_points_won BETWEEN 0 AND 255
        AND player1_second_serve_points_won BETWEEN 0 AND 255
        AND player1_service_games BETWEEN 0 AND 255
        AND player1_break_points_saved BETWEEN 0 AND 255 AND player1_break_points_faced BETWEEN 0 AND 255
        AND player2_wins_last_10 BETWEEN 0 AND 255 AND player2_matches_last_10 BETWEEN 0 AND 255
        AND player2_aces BETWEEN 0 AND 255 AND player2_double_faults BETWEEN 0 AND 255
        AND player2_first_serves_made BETWEEN 0 AND 255 AND player2_total_serve_points BETWEEN 0 AND 255
        AND player2_first_serve_points_won BETWEEN 0 AND 255
        AND player2_second_serve_points_won BETWEEN 0 AND 255
        AND player2_service_games BETWEEN 0 AND 255
        AND player2_break_points_saved BETWEEN 0 AND 255 AND player2_break_points_faced BETWEEN 0 AND 255
    ),
    CONSTRAINT match_events_check_indoor CHECK (is_indoor IS NULL OR is_indoor IN (0, 1))
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
--
-- bronze.player_profiles is the ingest-owned write target: ATP identity
-- loading and Wikipedia enrichment both UPSERT here (on player_id). dbt
-- materializes the enriched, aggregate gold.player_profiles from this source.
CREATE TABLE IF NOT EXISTS bronze.player_profiles (
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
    ioc          VARCHAR NOT NULL DEFAULT 'UNK',  -- ATP_Database.ioc; UNK when unknown
    -- Wikipedia enrichment (fallback for incomplete ATP rows)
    summary      VARCHAR,
    enriched_at  TIMESTAMP
);

-- IOC invariant: every bronze.player_profiles.ioc row is non-null. New rows
-- default to the UNK sentinel, and this idempotent upgrade backfills any
-- existing NULL/empty values (from databases created before the invariant)
-- to UNK, then enforces NOT NULL. Re-running is safe: the UPDATE is a no-op
-- once no NULL/empty rows remain, and SET DEFAULT/SET NOT NULL are no-ops
-- when already applied. Verified IOC values are never overwritten here.
UPDATE bronze.player_profiles SET ioc = 'UNK'
    WHERE ioc IS NULL OR ioc = '';
ALTER TABLE bronze.player_profiles ALTER COLUMN ioc SET DEFAULT 'UNK';
ALTER TABLE bronze.player_profiles ALTER COLUMN ioc SET NOT NULL;

-- Official weekly ATP rankings (rank 1-200 only), sourced from
-- data/raw/rankings/atp_rankings_*.csv. player_id is the canonical id resolved
-- through the approved ranking identity map (data/ranking_player_map.csv) — the
-- raw ranking source id is never stored. rank is the official value exactly as
-- read from the source; it is never estimated or interpolated. points is empty
-- (NULL) in early eras that predate published points.
CREATE TABLE IF NOT EXISTS bronze.rankings (
    ranking_date DATE     NOT NULL,   -- weekly ranking Monday
    player_id    VARCHAR  NOT NULL,   -- canonical player id (ranked identity)
    rank         SMALLINT NOT NULL CHECK (rank BETWEEN 1 AND 200),
    points       INTEGER,             -- NULL when the source era has no points
    PRIMARY KEY (ranking_date, player_id)
);

-- API lookup: a player's rank history ordered by date.
CREATE INDEX IF NOT EXISTS idx_rankings_player_date
    ON bronze.rankings (player_id, ranking_date);

-- gold.player_profiles is now dbt-owned (dbt/models/gold/player_profiles.sql
-- materializes it as a table from bronze.player_profiles + silver aggregates).
-- This definition is retained ONLY as a fresh-install bootstrap convenience;
-- dbt overwrites the relation (and its schema) on the first build.
--
-- ONE-TIME MIGRATION for existing installs: before the first `dbt build` with
-- this change, the operator must manually drop the legacy ingest-owned table
-- so dbt can recreate it with the materialized aggregate schema:
--
--   psql -U <user> -d <db> -c "DROP TABLE IF EXISTS gold.player_profiles;"
--
-- This is an explicit operator action; this file never drops data silently.
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