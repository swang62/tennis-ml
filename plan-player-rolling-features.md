# Plan: Player-Centric Rolling Features and Simple Inference

## Goal

Make dbt the single source of truth for player form features. Store reusable player-centric rolling history, derive the canonical match training matrix from that history, and provide an upstream inference helper that accepts player IDs plus minimal match context and returns the finalized Bento payload.

## High-Level Architecture

```text
CSV/API data
    ↓ separate ingest command
bronze.match_events
    ↓ separate Bronze → Gold ETL (`dbt build`)
gold.player_matches                 view, two player rows per match
    ↓
gold.player_rolling_features        table, post-match player snapshots
    ├─ gold.match_features          table, prior-snapshot matchup rows for training
    └─ build_inference_features()   on-demand latest/as-of lookup for two players
```

No `player_latest_features` or defaults table is added. Latest/as-of snapshots and fallback aggregates are queried on demand because inference only needs two players at a time.

## Scope

### In scope

- Normalize Bronze matches into player-perspective rows.
- Compute rolling features once in a player-centric dbt model.
- Rebuild `gold.match_features` from prior player snapshots without leakage.
- Correct the existing `matches_30d` behavior.
- Replace the unused dict-based inference helper with an ID-based DuckDB feature builder.
- Preserve the finalized 49-column Bento feature contract and canonical orientation.
- Retrain all models before deploying the new feature implementation.

### Out of scope

- Combining ingestion and ETL; they remain separate for debugging.
- Moving feature lookup into BentoML; Bento remains model-only.
- Streaming/file watchers.
- Incremental dbt materializations until full rebuild performance is measured.
- A materialized latest-player or defaults table.

## Tasks

### [x] Task 1: Add the normalized player-match model

- **Description**: Add a dbt model that expands every Bronze match into two player-perspective rows. Give each player a deterministic `player_match_number` ordered by `match_date, match_id`; retain current-match ranking, opponent ranking, surface/context, raw serve/break statistics, and outcome. Calculate event-relative activity fields such as previous-match date and the count of prior matches in the requested 30-day range using a correct date-range window.
- **Files**:
  - Add `dbt/models/gold/player_matches.sql`
  - Add `dbt/models/gold/player_matches.yml`
  - Update `dbt/models/sources.yml` descriptions only where the new dependency path makes existing wording stale
- **Acceptance Criteria**:
  - Exactly two rows exist for each eligible Bronze match.
  - `(match_id, player_id)` and `(player_id, player_match_number)` are unique.
  - Current outcome/stat columns belong to the player represented by each row.
  - `matches_30d_before` counts only matches in `[match_date - 30 days, match_date)`.
  - First-match activity fields have documented null/default semantics.
- **Guardrails**:
  - Do not canonicalize player/opponent orientation here; this model is player-perspective.
  - Do not calculate pairwise differential or encoded context columns here.
  - Do not change ingestion behavior or Bronze schema.

### [x] Task 2: Build post-match player rolling snapshots

- **Description**: Add one post-match snapshot per player event. Compute count-based rolling state inclusive of that completed match so the newest row is immediately usable for future inference. Store the intermediate state needed to reconstruct current match features, including rolling avg ranking values rather than prematurely binding rank trend to a future ranking.
- **Files**:
  - Add `dbt/models/gold/player_rolling_features.sql`
  - Add `dbt/models/gold/player_rolling_features.yml`
- **Snapshot grain**:
  - One row per `(player_id, player_match_number)` / `(player_id, match_id)`.
  - `snapshot_date` is the completed match date.
- **Feature state**:
  - Latest observed ranking and opponent ranking.
  - Win rates over 5/10/20 completed matches.
  - Ace, first-serve, and break rates over 5/10 completed matches.
  - Average opponent rank over 10/20 completed matches.
  - Average player rank over 10/20 completed matches, allowing `match_features` to derive rank trend against the next match's current ranking.
  - Current win streak.
  - Separate clay, grass, and hard win-rate-over-10 columns, carried forward on every snapshot.
  - Last completed match date.
- **Acceptance Criteria**:
  - Rolling windows include the snapshot match and never include future matches.
  - First rows produce nulls where history is insufficient; no silent zero filling in the training source.
  - Every rate is null or in `[0, 1]`.
  - Surface rates use the last 10 matches on that surface, not the last 10 overall matches filtered by surface.
  - One snapshot exists for every row in `gold.player_matches`.
- **Guardrails**:
  - Materialize as a table because it is expensive and reused by training and inference.
  - Do not add pre-match and post-match copies of the same rolling columns.
  - Do not add a latest snapshot table/view.

### [x] Task 3: Rebuild the canonical match training table from snapshots

- **Description**: Rewrite `gold.match_features` to pair each player-match row with that player's immediately preceding post-match snapshot (`player_match_number - 1`), then collapse the two perspectives into the existing canonical lower-ID orientation. Use the current event for current ranking/context and the prior snapshot for historical form.
- **Files**:
  - Rewrite `dbt/models/gold/match_features.sql`
  - Update `dbt/models/gold/match_features.yml`
- **Derived fields**:
  - `days_since_last_match`: current match date minus prior snapshot date, with the existing documented cold-start fallback.
  - `matches_30d`: from the normalized player's correct pre-match activity count.
  - `rank_trend_10/20`: prior rolling avg ranking minus current event ranking.
  - `surface_win_rate_10`: select clay/grass/hard rate from the prior snapshot according to current match surface.
  - Opponent-prefixed fields and all differential columns after canonical pairing.
  - Existing surface, tournament-level, and round encoding.
- **Acceptance Criteria**:
  - Exactly one row per eligible match.
  - The existing `FEATURE_COLS` names and order remain available without changing the 49-column Bento model input contract.
  - Player and opponent rolling values come strictly from completed matches before the target match.
  - Canonical orientation remains stable under raw player1/player2 reversal.
  - The corrected `matches_30d` values pass bounded known-data checks.
- **Guardrails**:
  - Remove the duplicated rolling-window implementation from `match_features.sql`; rolling formulas belong in `player_rolling_features.sql`.
  - Do not use current-match serve/break/outcome values as model features.
  - Do not preserve the known incorrect all-history `matches_30d` behavior.

### [x] Task 4: Add dbt integrity and leakage tests

- **Description**: Add schema and singular tests that prove model grain, rolling chronology, current-match exclusion, surface behavior, and final schema integrity.
- **Files**:
  - Update the three model YAML files above
  - Add focused SQL tests under `dbt/tests/`
- **Acceptance Criteria**:
  - Tests cover two player rows per match, unique snapshot grain, and one canonical match row.
  - A leakage test proves each match row uses snapshot sequence `current sequence - 1`.
  - A 30-day regression test proves old matches outside the interval are excluded.
  - A surface test proves each surface rate changes only after a match on that surface.
  - `dbt build --project-dir dbt --profiles-dir dbt` succeeds on test/seed data.
- **Guardrails**:
  - Prefer data invariants over tests coupled to a single full-dataset row count.

### [x] Task 5: Replace the stale inference helper with an ID-based builder

- **Description**: Keep feature column definitions in `src/features/rolling.py`, remove its unused dict-merging `build_inference_features`, and add a dedicated upstream builder that queries DuckDB by ID and as-of date.
- **Files**:
  - Add `src/features/inference.py`
  - Update `src/features/rolling.py`
  - Update `src/db/client.py` only if needed to expose parameterized DataFrame queries
  - Update relevant package/docs imports
- **Public contract**:
  - Required: `player_id`, `opponent_id`, `surface`.
  - Optional: `as_of_date` (defaults to today), `tournament_level=0`, `round_encoded=0`.
  - Return: one finalized DataFrame row containing `FEATURE_COLS + ["player_id", "opponent_id"]` in serving order.
- **Lookup behavior**:
  - Parameterized query selects each player's newest snapshot strictly before the as-of date.
  - No dedicated latest table/view.
  - Query `gold.player_matches` for date-dependent `days_since_last_match` and `matches_30d` values at the requested as-of date.
  - Canonicalize by stable player ID before assigning player/opponent sides.
  - Select the requested surface's rolling-rate column.
  - Compute differentials after imputation and canonicalization.
- **Missing-player behavior**:
  - Do not add a defaults table.
  - Compute fallback statistics on demand from snapshots eligible before the as-of date.
  - Follow the existing training rule: median for ranking/streak-related values and mean for other numerical values.
  - If both players are missing, both receive the same defaults so pairwise diffs are neutral.
  - Preserve the requested IDs so Bento's unknown bio lookup continues to produce zero bio vectors.
- **Acceptance Criteria**:
  - Swapping the two input IDs produces the same canonical feature values and orientation.
  - Surface one-hot columns are valid and exactly one is active.
  - Unsupported surfaces and invalid encoded context fail validation at the boundary.
  - No SQL string interpolation is used for player IDs or dates.
  - Output contains no missing feature columns or NaNs.
- **Guardrails**:
  - Do not query DuckDB from `src/serving/service.py`.
  - Do not duplicate rolling-window transformations in Python.
  - Keep Bento model execution unchanged.

### [x] Task 6: Add focused inference tests

- **Description**: Test the feature builder against a small deterministic DuckDB fixture or seed dataset.
- **Files**:
  - Add `tests/test_inference_features.py`
  - Update `pyproject.toml` only if a missing test dependency is required
- **Scenarios**:
  - Two known players on each supported surface.
  - Reversed player IDs produce the same canonical row.
  - Historical `as_of_date` excludes later snapshots.
  - Future/default-today date computes activity fields correctly.
  - One missing player receives aggregate defaults.
  - Both missing players produce neutral differentials.
  - Invalid surface/context is rejected.
- **Acceptance Criteria**:
  - Tests assert exact schema/order, not only selected values.
  - At least one regression case catches the old one-match-stale lookup.
  - At least one regression case catches the old `matches_30d` bug.

### [x] Task 7: Wire and document the separate ETL/debug workflow

- **Description**: Keep ingestion and ETL separate. Ensure the Bronze-to-Gold Prefect flow explicitly loads the selected environment and continues to run one `dbt build`, letting dbt dependencies refresh all Gold models.
- **Files**:
  - Update `src/flows/etl.py`
  - Update `README.md`
  - Update `AGENTS.md`
- **Acceptance Criteria**:
  - Ingest command only writes Bronze.
  - `just db-etl` builds `player_matches`, `player_rolling_features`, and `match_features` in dependency order.
  - Documentation shows where to inspect each stage during debugging.
  - `PREFECT_API_URL` and other selected environment values are loaded explicitly via `src.utils.load_env()`.
- **Guardrails**:
  - Do not trigger ETL automatically from ingestion.
  - Do not add a watcher, scheduler, or new service.

### [ ] Task 8: Rebuild data, retrain, and deploy a compatible model

- **Description**: The corrected activity feature changes model inputs semantically, even though the column contract remains stable. Rebuild Gold, run the complete 00–05 pipeline, evaluate against production, and only then package/deploy the promoted model.
- **Files/Artifacts**:
  - Regenerated DuckDB Gold tables
  - Regenerated `data/processed/*` training artifacts
  - New MLflow model versions
  - Bento produced through existing `src/flows/deploy.py` / `just deploy-bento`
- **Acceptance Criteria**:
  - `just db-etl` passes all dbt tests.
  - `just pipeline` completes notebooks 00–05.
  - 05 evaluates and records the promotion decision using the rebuilt features.
  - `just deploy-bento` packages only the promoted compatible model.
  - A smoke prediction built from IDs + surface passes Bento validation and returns probabilities.
- **Guardrails**:
  - Do not deploy an old model against newly corrected feature semantics.
  - Do not delete previous MLflow versions or historical artifacts.

## Dependencies

1. Task 1 precedes Tasks 2–4.
2. Task 2 precedes the `match_features` rewrite and inference builder.
3. Tasks 1–4 must pass before model retraining.
4. Task 5 can begin once the snapshot schema is fixed, but its integration tests depend on Tasks 1–4.
5. Deployment in Task 8 requires all data, dbt, and inference checks to pass.

## QA / Testing Scenarios

- **Leakage**: A player's outcome and serve stats in match N do not affect match N's feature row, but do affect match N+1.
- **Latest lookup**: Inference after match N includes match N in rolling form.
- **As-of lookup**: Inference dated before match N excludes match N and all later matches.
- **Surface isolation**: A clay match changes clay form without changing grass/hard form.
- **Activity window**: A match 31 days earlier is excluded from `matches_30d`; one 30 days minus one day earlier is included.
- **Canonicalization**: `(A, B)` and `(B, A)` generate one identical canonical payload.
- **Cold start**: Unknown player features use as-of mean/median aggregates without NaNs; two unknown players yield zero pairwise diffs.
- **Contract parity**: Finalized payload columns exactly equal `FEATURE_COLS + ids`; Bento still performs model execution only.
- **Operational separation**: Bronze can be inspected after ingest and before ETL; failed ETL can be rerun without re-ingesting.

## Migration Notes

- This is a semantic feature migration, not merely a refactor. Correcting `matches_30d` and changing snapshot construction requires full retraining.
- The 49-column serving schema should remain stable, minimizing Bento code changes.
- Start with full dbt table rebuilds. At the current ~80k-match scale, incremental complexity is not justified until measured.
