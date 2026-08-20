-- PostgreSQL bootstrap: structure only.
--
-- Runs on any standard PostgreSQL 18 instance: the Compose `postgres:18.4`
-- service executes it from /docker-entrypoint-initdb.d/schema.sql on a fresh
-- data volume (never on restart), and host operators may apply it manually to
-- the configured local database, e.g.:
--
--   psql -U <user> -d <db> -f infra/postgres/schema.sql
--
-- It is idempotent (CREATE ... IF NOT EXISTS), so re-running it is safe.

-- Creates structure only: the three schemas and the non-dbt-owned base tables
-- (bronze.match_events and bronze.player_profiles). No data is loaded here —
-- dbt owns silver.player_matches / silver.rolling_features /
-- gold.match_features / gold.tour_averages / gold.player_profiles, and data
-- is written later by just seed / etl. Nothing is baked into an image;
-- the Compose named volume persists everything.

-- Bento workers can start concurrently; serialize this idempotent migration.
SELECT pg_advisory_xact_lock(7910881);

CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;

-- Raw match data: one row per match with both players' stats in the row.
-- The gold layer expands each row into two player-perspective rows.
--
-- Match count columns use INTEGER: exceptional long matches exceed SMALLINT's
-- former 0..255 validation ceiling. Unknown match-time ranks are NULL; rank
-- points may be 0.
CREATE TABLE IF NOT EXISTS bronze.match_events (
    match_id                   VARCHAR NOT NULL,
    match_date                 DATE    NOT NULL,
    player1_id                 VARCHAR NOT NULL,
    player2_id                 VARCHAR NOT NULL,
    tournament                 VARCHAR NOT NULL,
    tournament_name            VARCHAR,
    round                      VARCHAR,
    surface                    VARCHAR NOT NULL,
    score                      VARCHAR,
    is_indoor                  SMALLINT,
    player1_ranking            INTEGER,
    player2_ranking            INTEGER,
    player1_wins_last_10       INTEGER,
    player1_matches_last_10    INTEGER,
    player1_aces               INTEGER NOT NULL,
    player1_double_faults      INTEGER NOT NULL,
    player1_first_serves_made  INTEGER NOT NULL,
    player1_total_serve_points INTEGER NOT NULL,
    player1_first_serve_points_won   INTEGER NOT NULL,
    player1_second_serve_points_won  INTEGER NOT NULL,
    player1_service_games      INTEGER NOT NULL,
    player1_break_points_saved INTEGER NOT NULL,
    player1_break_points_faced INTEGER NOT NULL,
    player2_wins_last_10       INTEGER,
    player2_matches_last_10    INTEGER,
    player2_aces               INTEGER NOT NULL,
    player2_double_faults      INTEGER NOT NULL,
    player2_first_serves_made  INTEGER NOT NULL,
    player2_total_serve_points INTEGER NOT NULL,
    player2_first_serve_points_won   INTEGER NOT NULL,
    player2_second_serve_points_won  INTEGER NOT NULL,
    player2_service_games      INTEGER NOT NULL,
    player2_break_points_saved INTEGER NOT NULL,
    player2_break_points_faced INTEGER NOT NULL,
    player1_rank_points        INTEGER NOT NULL,
    player2_rank_points        INTEGER NOT NULL,
    player1_age                DOUBLE PRECISION NOT NULL,
    player2_age                DOUBLE PRECISION NOT NULL,
    winner_id                  VARCHAR NOT NULL,
    PRIMARY KEY (match_id),
    CONSTRAINT match_events_check_players_distinct CHECK (player1_id <> player2_id),
    CONSTRAINT match_events_check_winner         CHECK (winner_id = player1_id),
    CONSTRAINT match_events_check_ranking        CHECK (
        (player1_ranking IS NULL OR player1_ranking >= 1)
        AND (player2_ranking IS NULL OR player2_ranking >= 1)
        AND player1_rank_points BETWEEN 0 AND 20000 AND player2_rank_points BETWEEN 0 AND 20000
    ),
    CONSTRAINT match_events_check_age             CHECK (
        player1_age BETWEEN 0 AND 100 AND player2_age BETWEEN 0 AND 100
    ),
    CONSTRAINT match_events_check_integer_counts  CHECK (
        player1_wins_last_10 BETWEEN 0 AND 20000 AND player1_matches_last_10 BETWEEN 0 AND 20000
        AND player1_aces BETWEEN 0 AND 20000 AND player1_double_faults BETWEEN 0 AND 20000
        AND player1_first_serves_made BETWEEN 0 AND 20000 AND player1_total_serve_points >= 0
        AND player1_first_serve_points_won BETWEEN 0 AND 20000
        AND player1_second_serve_points_won BETWEEN 0 AND 20000
        AND player1_service_games BETWEEN 0 AND 20000
        AND player1_break_points_saved BETWEEN 0 AND 20000 AND player1_break_points_faced BETWEEN 0 AND 20000
        AND player2_wins_last_10 BETWEEN 0 AND 20000 AND player2_matches_last_10 BETWEEN 0 AND 20000
        AND player2_aces BETWEEN 0 AND 20000 AND player2_double_faults BETWEEN 0 AND 20000
        AND player2_first_serves_made BETWEEN 0 AND 20000 AND player2_total_serve_points >= 0
        AND player2_first_serve_points_won BETWEEN 0 AND 20000
        AND player2_second_serve_points_won BETWEEN 0 AND 20000
        AND player2_service_games BETWEEN 0 AND 20000
        AND player2_break_points_saved BETWEEN 0 AND 20000 AND player2_break_points_faced BETWEEN 0 AND 20000
    ),
    CONSTRAINT match_events_check_indoor CHECK (is_indoor IS NULL OR is_indoor IN (0, 1)),
    CONSTRAINT match_events_check_surface CHECK (surface IN ('clay', 'grass', 'hard', 'carpet'))
);

-- Upgrade existing local databases created before match score was ingested.
ALTER TABLE bronze.match_events ADD COLUMN IF NOT EXISTS score VARCHAR;

-- Non-destructive upgrade for databases created with SMALLINT match counts.
-- ALTER TYPE widens the existing values in place; no rows or schemas are dropped.
ALTER TABLE bronze.match_events
    ALTER COLUMN player1_wins_last_10 TYPE INTEGER,
    ALTER COLUMN player1_matches_last_10 TYPE INTEGER,
    ALTER COLUMN player1_aces TYPE INTEGER,
    ALTER COLUMN player1_double_faults TYPE INTEGER,
    ALTER COLUMN player1_first_serves_made TYPE INTEGER,
    ALTER COLUMN player1_total_serve_points TYPE INTEGER,
    ALTER COLUMN player1_first_serve_points_won TYPE INTEGER,
    ALTER COLUMN player1_second_serve_points_won TYPE INTEGER,
    ALTER COLUMN player1_service_games TYPE INTEGER,
    ALTER COLUMN player1_break_points_saved TYPE INTEGER,
    ALTER COLUMN player1_break_points_faced TYPE INTEGER,
    ALTER COLUMN player2_wins_last_10 TYPE INTEGER,
    ALTER COLUMN player2_matches_last_10 TYPE INTEGER,
    ALTER COLUMN player2_aces TYPE INTEGER,
    ALTER COLUMN player2_double_faults TYPE INTEGER,
    ALTER COLUMN player2_first_serves_made TYPE INTEGER,
    ALTER COLUMN player2_total_serve_points TYPE INTEGER,
    ALTER COLUMN player2_first_serve_points_won TYPE INTEGER,
    ALTER COLUMN player2_second_serve_points_won TYPE INTEGER,
    ALTER COLUMN player2_service_games TYPE INTEGER,
    ALTER COLUMN player2_break_points_saved TYPE INTEGER,
    ALTER COLUMN player2_break_points_faced TYPE INTEGER;
ALTER TABLE bronze.match_events DROP CONSTRAINT IF EXISTS match_events_check_integer_counts;
ALTER TABLE bronze.match_events ADD CONSTRAINT match_events_check_integer_counts CHECK (
    player1_wins_last_10 BETWEEN 0 AND 20000 AND player1_matches_last_10 BETWEEN 0 AND 20000
    AND player1_aces BETWEEN 0 AND 20000 AND player1_double_faults BETWEEN 0 AND 20000
    AND player1_first_serves_made BETWEEN 0 AND 20000 AND player1_total_serve_points >= 0
    AND player1_first_serve_points_won BETWEEN 0 AND 20000
    AND player1_second_serve_points_won BETWEEN 0 AND 20000
    AND player1_service_games BETWEEN 0 AND 20000
    AND player1_break_points_saved BETWEEN 0 AND 20000 AND player1_break_points_faced BETWEEN 0 AND 20000
    AND player2_wins_last_10 BETWEEN 0 AND 20000 AND player2_matches_last_10 BETWEEN 0 AND 20000
    AND player2_aces BETWEEN 0 AND 20000 AND player2_double_faults BETWEEN 0 AND 20000
    AND player2_first_serves_made BETWEEN 0 AND 20000 AND player2_total_serve_points >= 0
    AND player2_first_serve_points_won BETWEEN 0 AND 20000
    AND player2_second_serve_points_won BETWEEN 0 AND 20000
    AND player2_service_games BETWEEN 0 AND 20000
    AND player2_break_points_saved BETWEEN 0 AND 20000 AND player2_break_points_faced BETWEEN 0 AND 20000
);

-- Upgrade existing local databases created before unknown ranks became NULL.
ALTER TABLE bronze.match_events ALTER COLUMN player1_ranking DROP NOT NULL;
ALTER TABLE bronze.match_events ALTER COLUMN player2_ranking DROP NOT NULL;
UPDATE bronze.match_events SET player1_ranking = NULL WHERE player1_ranking = 0;
UPDATE bronze.match_events SET player2_ranking = NULL WHERE player2_ranking = 0;
ALTER TABLE bronze.match_events DROP CONSTRAINT IF EXISTS match_events_check_ranking;
ALTER TABLE bronze.match_events ADD CONSTRAINT match_events_check_ranking CHECK (
    (player1_ranking IS NULL OR player1_ranking >= 1)
    AND (player2_ranking IS NULL OR player2_ranking >= 1)
    AND player1_rank_points BETWEEN 0 AND 20000 AND player2_rank_points BETWEEN 0 AND 20000
);

-- Surface invariant: exactly the four canonicals. Absent/unknown source values
-- normalize to hard at ingest and this idempotent backfill (e.g. the "0" / "nan"
-- / blank markers legacy ingestion wrote for blank Davis Cup cells), and the
-- CHECK rejects anything else at the DB boundary.
UPDATE bronze.match_events SET surface = 'hard'
    WHERE surface NOT IN ('clay', 'grass', 'hard', 'carpet');
ALTER TABLE bronze.match_events DROP CONSTRAINT IF EXISTS match_events_check_surface;
ALTER TABLE bronze.match_events ADD CONSTRAINT match_events_check_surface CHECK (
    surface IN ('clay', 'grass', 'hard', 'carpet')
);

-- Secondary indexes for the common gold-layer expansion/rolling query
-- patterns (player1/player2 sides of bronze are scanned per player + date).
CREATE INDEX IF NOT EXISTS idx_match_events_p1_date
    ON bronze.match_events (player1_id, match_date);
CREATE INDEX IF NOT EXISTS idx_match_events_p2_date
    ON bronze.match_events (player2_id, match_date);

-- Ordered direct-meeting reads used by /head_to_head. `match_id` completes
-- the endpoint's deterministic ORDER BY after same-day matches.
CREATE INDEX IF NOT EXISTS idx_match_events_p1_date_match
    ON bronze.match_events (player1_id, match_date DESC, match_id DESC);
CREATE INDEX IF NOT EXISTS idx_match_events_p2_date_match
    ON bronze.match_events (player2_id, match_date DESC, match_id DESC);

-- Exact unordered-pair seek indexes for the gold H2H five-meeting lookup:
-- (p1_id, p2_id) equality in (match_date DESC, match_id DESC) order lets each
-- directional row's bounded lateral scan stop after its five newest
-- strictly-prior meetings instead of OR-scanning either side's full history.
-- Also serve direct pair queries from /head_to_head-style endpoints.
CREATE INDEX IF NOT EXISTS idx_match_events_p1_p2_date_match
    ON bronze.match_events (player1_id, player2_id, match_date DESC, match_id DESC);
CREATE INDEX IF NOT EXISTS idx_match_events_p2_p1_date_match
    ON bronze.match_events (player2_id, player1_id, match_date DESC, match_id DESC);

-- Global latest-match lookup for the dynamic directory footer.
CREATE INDEX IF NOT EXISTS idx_match_events_date
    ON bronze.match_events (match_date);

-- Identity backbone for players, sourced from the ATP player database
-- (data/ATP_player_database.csv, canonical ATP id/name + base metadata).
-- Enrichment columns at the bottom are left empty by the ATP load and filled
-- by the Wikipedia fallback in src/db/ingest.py for players it covers.
--
-- bronze.player_profiles is the ingest-owned write target: ATP identity
-- loading and Wikipedia enrichment both UPSERT here (on player_id). dbt
-- derives the aggregate gold.player_profiles (player_id + derived aggregates
-- only) from this source; metadata is never duplicated in gold.
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
--
-- The composite PK (ranking_date, player_id) is the identity both ingestion
-- paths upsert on: src/db/ingest.py ingest_rankings (seed) and the weekly
-- rankings catch-up (src/flows/rankings.py), each ON CONFLICT (ranking_date,
-- player_id) — so re-running either is idempotent.
CREATE TABLE IF NOT EXISTS bronze.rankings (
    ranking_date DATE     NOT NULL,   -- weekly ranking Monday
    player_id    VARCHAR  NOT NULL,   -- canonical player id (ranked identity)
    rank         SMALLINT NOT NULL CHECK (rank BETWEEN 1 AND 200),
    points       INTEGER,             -- NULL when the source era has no points
    PRIMARY KEY (ranking_date, player_id)
);

-- Per-player rank access paths, both served by this one index:
--  * chronological rank history: WHERE player_id = ? ORDER BY ranking_date
--    (the /rank_history endpoint)
--  * latest rank per player: gold.player_profiles DISTINCT ON (player_id)
--    ORDER BY player_id, ranking_date DESC (backward scan of the same index)
CREATE INDEX IF NOT EXISTS idx_rankings_player_date
    ON bronze.rankings (player_id, ranking_date);
