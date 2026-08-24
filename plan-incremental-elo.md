# Plan: Incremental Timestamped Elo Features

## Goal

Add append-only, per-player Elo snapshots after base dbt models run. Use the
pre-match global and surface Elo differences (`elo_diff`, `elo_surface_diff`)
in `gold.match_features` and final `FEATURE_COLS`, including as-of-date
inference rows.

## Scope

- Maintain global and surface-specific Elo ratings.
- Use 1500 for players with no prior Elo history.
- Use adaptive K: `min(62, 43 + 800 / (prior_matches + 1))`; surface K uses
  prior matches on that surface.
- Before a match, after 90 inactive days, regress the rating 1% of the
  remaining distance to 1500 per further 7 days, capped at 50% regression.
- Process only matches strictly after the persisted causal watermark
  `(match_date, match_num, match_id)`.
- Fail closed if a source match is introduced or changed at or before the Elo
  watermark. A historical correction requires an explicit Elo rebuild.
- Make `seed --reset` clear the dbt and Elo progress state, as well as the
  Elo snapshots, so the subsequent ETL rebuilds all derived state from the
  newly seeded bronze history.

## Non-goals

- Do not import or reuse `../greencode` code. Its useful invariant is
  pre-match read followed by post-match update; its notebook-only state and
  ordering are not safe for this project.
- Do not add a new dependency, service, or public inference parameter.
- Do not auto-rebuild historical Elo when append-only validation fails.

## Data model

`silver.elo_snapshots` is Python-owned and has two rows for every physical
match. Each row contains the causal match key, player id, match surface,
pre/post global Elo, pre/post current-surface Elo, prior overall/surface match
counts, and the overall/surface K values applied. Persisting pre-ratings makes
the training join an exact causal lookup; persisting post-ratings makes state
reconstruction and as-of inference efficient and auditable.

There is exactly one progress state: `bronze.etl_state` (its `source_watermark`
TIMESTAMPTZ). It is the sole pipeline watermark; ETL advances it only after base
dbt, Elo, and gold.match_features all succeed. The Elo materializer reads this
timestamp watermark to select new matches and to fail closed on historical
corrections; it never advances progress itself, so a rerun after a later-phase
failure rates no match twice (the `silver.elo_snapshots` PK guards re-rating).

The materializer reconstructs only the two participants' state from their
latest persisted rows: latest overall post-rating and latest post-rating on
the current surface. This carries the state of all players through the table
without loading a permanent in-memory league state.

## Tasks

### [x] Task 1: Add Elo tables, indexes, constants, and reset support

- **Description**:
  - In `infra/postgres/schema.sql`, add the Python-owned
    `silver.elo_snapshots` table using idempotent DDL. Do NOT add a separate
    Elo state table; `bronze.etl_state` is the sole progress watermark.
  - Enforce one snapshot per `(player_id, match_id)`.
  - Add btree indexes tailored to the two latest-as-of queries:
    - `(player_id, match_date DESC, match_num DESC, match_id DESC)` with the
      post-overall rating and causal state columns included.
    - `(player_id, surface, match_date DESC, match_num DESC, match_id DESC)`
      with the post-surface rating included.
    - Retain the primary key `(player_id, match_id)` for the exact gold join.
  - Update the schema ownership comments so these non-dbt tables are clear.
  - In `src/constants.py`, add table-name and Elo parameter constants.
  - In `src/db/seed.py`, make `--reset` clear `silver.elo_snapshots`,
    `bronze.etl_state` as part of the existing clean bronze reset. Do not alter
    these tables for a normal append seed.
- **Files**:
  - `infra/postgres/schema.sql`
  - `src/constants.py`
  - `src/db/seed.py`
  - Relevant hermetic seed/reset test file, after locating the existing seed
    test coverage.
- **Acceptance Criteria**:
  - Schema re-application is safe.
  - `seed --reset` guarantees no stale dbt or Elo watermark/snapshot survives.
  - Normal seeding preserves incremental state.
  - Latest-as-of overall and surface queries use matching ordered indexes with
    `ORDER BY match_date DESC, match_num DESC, match_id DESC LIMIT 1`, without
    a sort; the exact gold join uses the primary key.
- **Guardrails**:
  - Do not delete tables or user data outside the explicit `--reset` path.
  - Do not create a separate Elo watermark; the shared `bronze.etl_state`
    timestamp is authoritative for incremental processing.

### [x] Task 2: Implement atomic, append-only Elo materialization

- **Description**:
  - Add `src/features/elo.py` with a deterministic materializer that reads
    `bronze.match_events` in `(match_date, match_num, match_id)` order.
   - Before writes, validate every source match at/before the shared
     `bronze.etl_state` timestamp watermark is already snapshotted with matching
     content; reject historical or out-of-order input (fail closed).
  - For each match, obtain each participant's previous overall and requested
    surface state from persisted post-match snapshots, defaulting to 1500 and
    zero prior matches.
  - Apply the selected layoff regression before expected-score calculation:
    no regression through 90 days; then 1% of remaining distance to 1500 per
    7 days; cap at 50%.
   - Calculate overall and surface Elo separately, using their applicable
     prior-match counts and K values; write both players' pre/post rows in one
     transaction. Progress is advanced only by ETL after every phase succeeds.
  - Return/log processed-match and snapshot counts; a no-new-match call is a
    no-op.
- **Files**:
  - New: `src/features/elo.py`
  - New: `tests/test_elo.py`
- **Acceptance Criteria**:
  - First-match pre-ratings are 1500.
  - Winners rise and losers fall using their own adaptive K.
  - A hard-court result changes global and hard Elo only, not clay/grass/carpet
    Elo state.
  - A later same-day match reads the prior match's post-rating.
  - Re-running with no new match writes nothing.
  - A failure leaves snapshots and the shared `bronze.etl_state` watermark
    unchanged.
- **Guardrails**:
  - Never rate a physical match twice.
  - Do not silently repair historical data; fail before any mutation.

### [x] Task 3: Split ETL around the Elo phase

- **Description**:
  - Update `src/flows/etl.py` to run base dbt models first:
    `silver.player_matches`, `silver.rolling_features`,
    `gold.tour_averages`, and `gold.player_profiles`.
  - Run the Elo materializer next.
  - Run `gold.match_features` and its dbt tests last, after Elo data exists.
  - Preserve the profile-only shortcut when no bronze match has changed; it
    must skip both Elo and match-feature rebuilding.
  - Preserve phase-specific dbt diagnostics and advance the existing dbt
    watermark only after every phase completes successfully.
- **Files**:
  - `src/flows/etl.py`
  - Relevant ETL flow tests, if present.
- **Acceptance Criteria**:
  - One ETL run makes newly calculated Elo available to gold in the same run.
  - Failure in Elo or final gold build does not advance `bronze.etl_state`.
  - Profile-only refreshes do not touch Elo snapshots.
- **Guardrails**:
  - Keep Prefect orchestration intact.
  - Do not run `gold.match_features` before Elo materialization.

### [x] Task 4: Add causal Elo features to gold

- **Description**:
  - Declare `silver.elo_snapshots` as a dbt source in `dbt/models/sources.yml`.
  - In `dbt/models/gold/match_features.sql`, join each directional row to its
    own Elo snapshot and select `elo_diff` and `elo_surface_diff` from the
    stored pre-match ratings.
  - Add `elo_diff` and `elo_surface_diff` to `DIFF_COLS` in
    `src/features/columns.py`, making them final `FEATURE_COLS` entries and
    preserving the shared training/inference contract.
  - Correct `changed_match_ids` to compare the full causal tuple rather than
    only `pm.match_date > nm.match_date`, so same-day future matches rebuild.
  - Extend model documentation and causal-leakage validation.
- **Files**:
  - `dbt/models/sources.yml`
  - `dbt/models/gold/match_features.sql`
  - `dbt/models/gold/match_features.yml`
  - `dbt/tests/gold/match_features_no_current_match_leakage.sql`
  - `src/features/columns.py`
- **Acceptance Criteria**:
  - Gold retains exactly two reciprocal rows per match.
  - Reversing player perspective negates both Elo differences.
  - A gold row never reads its own or a later snapshot.
  - Same-day causal order is respected during incremental rebuilds.
- **Guardrails**:
  - Elo must be a training feature, not a similarity-only column.
  - Do not replace the project's existing rolling-feature or cold-start logic.

### [x] Task 5: Add matching as-of Elo inference

- **Description**:
  - Extend scalar and bulk paths in `src/features/inference.py` to fetch each
    player's latest completed overall snapshot through `as_of_date` and their
    latest snapshot on the requested surface.
  - Apply the same inactivity regression from the latest overall completed
    match through `as_of_date`; default missing overall or surface state to
    1500.
  - Add the two Elo differences to assembled scalar and bulk rows, retaining
    current endpoint semantics: the latest completed same-day match is
    included because the public API accepts a date, not a match sequence.
  - Expand training/inference parity fixtures and the DuckDB snapshot contract
    expectations for the new ordered feature columns.
- **Files**:
  - `src/features/inference.py`
  - `tests/test_inference_features.py`
  - `tests/test_snapshot.py`
- **Acceptance Criteria**:
  - Scalar and bulk inference output identical Elo values.
  - Inference and gold agree for an equivalent causal fixture.
  - A cold-start pair yields zero Elo differences and no null model feature.
  - The DuckDB training snapshot validates the new feature order.
- **Guardrails**:
  - Preserve player order; never canonicalize ids.
  - Do not change the public API contract or same-day date semantics.

## Dependencies

1. Task 1 before Tasks 2-5.
2. Task 2 before Task 3's final gold phase and Task 4.
3. Task 4 before Task 5 because `FEATURE_COLS` changes the model contract.
4. Retrain and promote models only after the rebuilt gold snapshot validates;
   existing champion artifacts remain tied to the old feature list until then.

## QA Scenarios

1. Two debutants play: both pre-ratings are 1500; the winner rises and loser
   falls.
2. A player returns after 160 days: their pre-rating is partially regressed
   before expected-score and K calculations.
3. A hard-court match does not mutate the player's clay rating state.
4. Two matches on the same date process in `match_num` then `match_id` order;
   the latter observes the former's Elo update.
5. A second materialization with no new rows produces no writes.
6. An event at or before the shared ETL timestamp watermark aborts before writes.
7. `seed --reset` clears old Elo rows and the shared dbt state row; the next
   full ETL rebuilds them from the reseeded bronze data.
8. End-to-end ETL produces finite gold Elo features and inference produces the
   same ordered feature contract.
