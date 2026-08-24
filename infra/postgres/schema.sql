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
--
-- silver.elo_snapshots is Python-owned (not dbt): the per-player Elo
-- materialization. The Elo materializer writes it; reset clears it. The shared
-- bronze.etl_state timestamp watermark is the sole pipeline progress marker and
-- advances only after base dbt, Elo, and gold.match_features all succeed.

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
    best_of                    SMALLINT,
    surface                    VARCHAR NOT NULL,
    score                      VARCHAR,
    match_num                  INTEGER NOT NULL,  -- per-tournament sequence from raw CSV (required)
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
    ingested_at                TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
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
    CONSTRAINT match_events_check_surface CHECK (surface IN ('clay', 'grass', 'hard', 'carpet')),
    CONSTRAINT match_events_check_best_of CHECK (best_of IS NULL OR best_of IN (1, 3, 5))
);

-- Existing databases need the append watermark too. New and existing rows are
-- stamped once here; later inserts/overwrites advance it at ingestion time.
ALTER TABLE bronze.match_events
    ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP;

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

-- Persist best_of (the best-of-N format: 1, 3, or 5 sets). New inserts carry
-- it from the source payload; legacy/seed rows may be NULL until re-ingested,
-- so the column is nullable and the CHECK allows NULL.
ALTER TABLE bronze.match_events ADD COLUMN IF NOT EXISTS best_of SMALLINT;
ALTER TABLE bronze.match_events DROP CONSTRAINT IF EXISTS match_events_check_best_of;
ALTER TABLE bronze.match_events ADD CONSTRAINT match_events_check_best_of CHECK (
    best_of IS NULL OR best_of IN (1, 3, 5)
);

-- Persist match_num (per-tournament sequence from the raw CSV); required, so NOT NULL.
ALTER TABLE bronze.match_events ADD COLUMN IF NOT EXISTS match_num INTEGER;
-- Legacy rows predating match_num cannot be chronologically ordered and are
-- dropped here; `seed --reset` recreates them with match_num populated. No-op on
-- a fresh database. Keeps the migration idempotent on a database upgraded in place.
DELETE FROM bronze.match_events WHERE match_num IS NULL;
ALTER TABLE bronze.match_events ALTER COLUMN match_num SET NOT NULL;

-- match_date is the mandatory bronze key; enforce NOT NULL (idempotent no-op
-- on databases already created with the constraint).
ALTER TABLE bronze.match_events ALTER COLUMN match_date SET NOT NULL;

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

-- Fast incremental ETL boundary: only source changes since this watermark need
-- the expensive windowed dbt models. The watermark advances after a successful
-- full incremental build, never before it.
CREATE INDEX IF NOT EXISTS idx_match_events_ingested_at_match_id
    ON bronze.match_events (ingested_at, match_id);

CREATE TABLE IF NOT EXISTS bronze.etl_state (
    pipeline             VARCHAR PRIMARY KEY,
    source_watermark     TIMESTAMPTZ
);

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

-- Python-owned per-player Elo snapshots: two rows per physical match (one per
-- participant). Each row carries the causal match key, the player's requested
-- match surface, the pre/post global and current-surface Elo, the prior
-- overall/surface match counts, and the overall/surface K values applied. The
-- post-ratings are persisted so gold joins an exact causal lookup and so state
-- reconstruction / as-of inference read the latest completed snapshot without an
-- in-memory league.
--
-- The PK (player_id, match_id) is the exact join key for gold.match_features:
-- no row ever reads its own or a later snapshot. Exactly one snapshot exists
-- per (player_id, match_id).
CREATE TABLE IF NOT EXISTS silver.elo_snapshots (
    player_id              VARCHAR          NOT NULL,
    match_id               VARCHAR          NOT NULL,
    match_date             DATE             NOT NULL,  -- causal order key: date, then match_num
    match_num              INTEGER          NOT NULL,  -- causal order key: per-tournament sequence
    -- Causal order is (match_date, match_num, match_id). match_num is per-tournament,
    -- so (match_date, match_num) is NOT globally unique across tournaments; match_id
    -- (bronze.match_events PK) is the deterministic tie-breaker that makes the order
    -- unique. No unique constraint is placed on (match_date, match_num) by design.
    surface                VARCHAR          NOT NULL,  -- requested match surface
    pre_elo                DOUBLE PRECISION NOT NULL,  -- pre-match global Elo
    post_elo               DOUBLE PRECISION NOT NULL,  -- post-match global Elo
    pre_elo_surface        DOUBLE PRECISION NOT NULL,  -- pre-match surface Elo
    post_elo_surface       DOUBLE PRECISION NOT NULL,  -- post-match surface Elo
    prior_overall_matches  INTEGER          NOT NULL,  -- prior overall match count
    prior_surface_matches  INTEGER          NOT NULL,  -- prior surface match count
    k_overall             DOUBLE PRECISION NOT NULL,  -- overall K applied
    k_surface            DOUBLE PRECISION NOT NULL,  -- surface K applied
    source_hash          VARCHAR,                    -- sha256 of the source match's
                                                     -- Elo-relevant content; NULL
                                                     -- (legacy) fails validation closed
    PRIMARY KEY (player_id, match_id),
    CONSTRAINT elo_snapshots_check_surface CHECK (surface IN ('clay', 'grass', 'hard', 'carpet'))
);

-- Latest overall Elo as-of query: a player's newest completed post-rating
-- strictly before a date. The covering index returns post_elo plus the causal
-- state columns (prior overall count + overall K) without a table lookup, and
-- its ORDER BY matches the query so no sort is required.
CREATE INDEX IF NOT EXISTS idx_elo_snapshots_player_overall
    ON silver.elo_snapshots (player_id, match_date DESC, match_num DESC, match_id DESC)
    INCLUDE (post_elo, prior_overall_matches, k_overall);

-- Latest surface Elo as-of query: a player's newest completed post-rating on a
-- surface strictly before a date.
CREATE INDEX IF NOT EXISTS idx_elo_snapshots_player_surface
    ON silver.elo_snapshots (player_id, surface, match_date DESC, match_num DESC, match_id DESC)
    INCLUDE (post_elo_surface);

-- Upgrade databases whose dbt-built gold.match_features predates the elo_diff /
-- elo_surface_diff columns (the leakage test references them). dbt owns this
-- table and creates it with these columns on a fresh build, so only add them
-- when the table already exists (e.g. an upgraded local database). The DO block
-- skips fresh databases where dbt has not yet materialized the table, so the
-- migration never errors on a relation that does not exist. ADD COLUMN IF NOT
-- EXISTS keeps reruns safe and non-destructive.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'gold' AND table_name = 'match_features'
    ) THEN
        ALTER TABLE gold.match_features
            ADD COLUMN IF NOT EXISTS elo_diff DOUBLE PRECISION;
        ALTER TABLE gold.match_features
            ADD COLUMN IF NOT EXISTS elo_surface_diff DOUBLE PRECISION;
    END IF;
END $$;
