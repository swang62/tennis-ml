# Plan: Phase 2 local CSV-to-DuckDB pipeline

## Goal

Keep the platform local and simple. The authoritative match sources are the raw ATP CSV files:

```text
data/raw/*.csv                   ┐
                                ├─ validate raw ATP rows -> DuckDB bronze
                                                                -> dbt silver/gold
                                                                -> training -> MLflow
                                                                -> Docker Hub image -> Compose Bento
```

The eventual ingestion scope is all regular ATP CSVs in `data/raw`. Today the deterministic seed subset is `data/raw/2026.csv`, which `infra/duckdb/seed.py` loads first. There is no ongoing export/merge/append process; new CSVs are added as raw files and picked up by the seed run.

Phase 1 dbt migration is complete. Phase 2 keeps the existing single-node k3d services needed for Prefect and MLflow, while raw CSV ingestion writes directly to the local DuckDB bronze tables after validation. Production BentoML serving runs outside Kubernetes through Docker Compose and pulls the `latest` image from Docker Hub.

## Deployment boundaries

- **Raw data**: all regular ATP CSVs under `data/raw` are the source of truth. They are discovered in sorted filename order and loaded directly; non-regular competition files are excluded. `data/raw/2026.csv` is the current deterministic seed subset; the scope is all regular ATP CSVs, not only 2026 files.
- **DuckDB**: validated CSV rows load directly into `bronze.match_events`; dbt owns silver/gold transformations. The direct CSV path remains the production path.
- **k3d**: only the existing local Prefect/MLflow platform and supporting stateful services. Ingestion remains a host/local-file operation.
- **BentoML**: `src/flows/deploy.py` builds the promoted Bento, securely logs in to Docker Hub, pushes the `${DOCKER_IMAGE}:latest` tag, and runs `docker compose -f compose.production.yaml up -d --pull always`.
- **Bento access**: host-local `http://127.0.0.1:3000`; no Bento Kubernetes Deployment and no Bento Traefik route.
- **Credentials**: Settings live in the existing untracked `.env` file. It sets `DOCKER_REPO=swang62` and `IMAGE_NAME=tennis-ml`; `DOCKER_TOKEN` is read securely via stdin by `deploy.py` and is never printed or committed. Secrets must never be written to Git. `.env` is untracked and there is no `.env.example` requirement.

## Current baseline

- `src/flows/ingest.py` already maps ATP-format rows to the bronze contract and calls `run_ingestion_checks` before insertion.
- `insert_bronze_rows` uses `match_id` as the DuckDB primary key, so database re-ingestion of any authoritative CSV is idempotent.
- `infra/duckdb/seed.py` supports `--all`: it discovers CSVs sorted under `data/raw`, loads all rows chronologically, and is idempotent via `match_id`. Seeding never performs Wikipedia enrichment; full-corpus loading is not executed by this plan.
- All regular ATP CSVs under `data/raw` are the authoritative ingestion scope and are ingested directly. `data/raw/2026.csv` is the current deterministic seed subset. There are no ongoing CSV exports, no merge command, and no rewriting/append/dedupe maintenance phase.
- `src/flows/deploy.py` reads `DOCKER_TOKEN` securely via stdin to `docker login`, derives the username from `DOCKER_USERNAME` or the `DOCKER_REPO` owner, then pushes the `latest` tag.
- Training is standalone in `src/flows/pipeline.py`. MLflow aliases remain `@best` and `@champion`.
- `bentofile.yaml` remains the Bento template; deploy generates a pinned Bento file.

## Tasks

### [x] Task 1: Inventory the local platform and simplify environment commands

- **Description**: Remove obsolete distributed-ingestion assumptions from the Phase 2 architecture and document the compact local path: CSV -> validation -> DuckDB -> dbt -> training/MLflow. Keep only the k3d workloads that are still required for local orchestration and model tracking. Use the existing untracked `.env` for DuckDB, Prefect, MLflow, Docker Hub image name/tag, and Compose settings.
- **Files**:
  - `infra/k3d/config.yaml`
  - `infra/k3d/start.sh`
  - `infra/manifests/default/config-map.yaml`
  - `justfile`
  - `README.md`
- **Acceptance Criteria**:
  - No external ingestion dependency is required to run ETL or training.
  - `just setup`, `just db-init`, `just db-etl`, `just train`, and `just deploy-bento` have clear, minimal responsibilities.
  - Local DuckDB and pipeline artifacts persist across k3d restarts.
  - No k3d image registry or registry hostname remains in the deployment path.

### [x] Task 2: Load validated ATP CSVs directly into DuckDB without online enrichment

- **Description**: Seed the deterministic minimal dataset into DuckDB for development and validation. Preserve `seed.py --all` only as an explicit production data-load capability; no task, test, QA scenario, or documentation proof run invokes it. `seed.py` must never look up or enrich Wikipedia bios and must not expose an enrichment option. Bio enrichment belongs only to ETL as an explicit `--enrich` operation; default ETL execution remains offline. For each player, use the first paragraph of the Wikipedia `Playing style` section; fall back to the article lead paragraph only when that section/paragraph is unavailable. Validate raw rows before insertion, preserve the existing bronze contract, and keep database inserts idempotent by `match_id`. Do not introduce a message broker, object store, connector, or intermediate Parquet source.
- **Files**:
  - `src/flows/ingest.py`
  - `src/features/validate.py`
  - `src/features/columns.py`
  - `infra/duckdb/seed.py`
  - `src/flows/etl.py`
  - `tests/`
- **Acceptance Criteria**:
  - All regular ATP CSVs under `data/raw` can be ingested directly after raw validation; `data/raw/2026.csv` is the current deterministic seed subset.
  - The default seed loads only the deterministic minimal dataset without network enrichment; focused tests validate `--all` discovery/ordering logic without loading the full corpus and confirm non-regular competition files are excluded.
  - Seed, default ETL, tests, and validation perform no Wikipedia/network enrichment. Only an explicit ETL opt-in may enrich bios, and it is outside this plan's test scope.
  - `just db-etl --enrich` enriches player bios from the first `Playing style` paragraph, falling back to the article lead paragraph only when needed.
  - After all other plan tasks complete, final QA runs explicit ETL enrichment once against the seeded deterministic miniset; this is the only live-network enrichment check in the plan.
  - Columns and headers are schema-validated before ingestion; mismatched schemas fail before any bronze rows are written.
  - Invalid rows are dropped/reported before DuckDB insertion; valid rows remain available for dbt.
  - Re-ingesting any source does not create duplicate bronze rows.
  - `just db-etl` builds silver/gold from DuckDB bronze with no external ingestion service.

### [x] Task 3: Verify dbt, training, and MLflow against the CSV source of truth

- **Description**: Remove any remaining source assumptions that point ETL away from local DuckDB bronze. Verify silver/gold behavior against the deterministic minimal seed only, then keep training standalone and promotion alias-based.
- **Files**:
  - `dbt/models/sources.yml`
  - `dbt/models/silver/`
  - `dbt/models/gold/`
  - `dbt/profiles.yml`
  - `src/flows/etl.py`
  - `src/flows/pipeline.py`
  - `README.md`
- **Acceptance Criteria**:
  - dbt reads the local DuckDB bronze tables populated from validated ATP CSVs.
  - dbt tests cover required columns, allowed values, and match identity uniqueness.
  - A second dbt run with no new CSV matches produces identical gold output.
  - Training runs only after successful dbt validation and remains outside Prefect.
  - MLflow registration and `@champion` promotion continue to work.

### [x] Task 4: Build and serve Bento through Docker Hub and Compose

- **Description**: Replace the old Kubernetes/k3d Bento rollout with a host-executed deployment. Build the promoted Bento, securely log in to Docker Hub, push the `latest` tag, then pull and boot one Compose service. Compose must not build locally and must not require k3d. Image name/tag come from the untracked `.env` (`DOCKER_REPO=swang62`, `IMAGE_NAME=tennis-ml`); `deploy.py` reads `DOCKER_TOKEN` securely via stdin to `docker login`, deriving the username from `DOCKER_USERNAME` or the `DOCKER_REPO` owner, then pushes the `latest` tag. The token is never printed or committed.
- **Files**:
  - `src/flows/deploy.py`
  - `compose.production.yaml` (exists)
  - `justfile`
  - `README.md`
- **Acceptance Criteria**:
  - `just deploy-bento` performs Bento build -> secure Docker Hub login -> push `latest` -> Compose pull/up.
  - `deploy.py` reads `DOCKER_TOKEN` via stdin (never argv/env/log), logs in with a username derived from `DOCKER_USERNAME` or the `DOCKER_REPO` owner, and pushes only the `latest` tag.
  - `just deploy-bento --force` runs `uv run python src/flows/deploy.py --force` and bypasses build/image caches.
  - Compose defines the Bento service with `image: ${DOCKER_IMAGE}:latest`, `pull_policy: always`, `3000:3000`, `/healthz` healthcheck, and `restart: unless-stopped`; Task 5 adds its DuckDB companion service.
  - A clean Docker host can pull the `latest` image and boot Bento without k3d or Kubernetes.
  - `/healthz` and `/predict` pass after boot and after a container restart.
  - No Docker Hub token is written to Git, image labels, Compose YAML, or logs.

### [x] Task 5: Serve the production DuckDB through a Quack companion container

- **Description**: After the Bento Compose service is corrected, add a DuckDB companion service that owns and mounts the complete production `tennis.duckdb` file and serves it over DuckDB's official HTTP-based Quack protocol. Use one runtime switch, `ENVIRONMENT=dev|production`: dev (and the default local `bentoml serve`) opens the repository-local DuckDB file directly; production Compose connects Bento to the Quack companion. `TENNIS_DB_PATH`, `QUACK_URI`, and `QUACK_TOKEN` configure their respective backends but do not select the mode. Generic PostgreSQL JDBC/PGWire is explicitly out of scope because official Quack preserves DuckDB SQL semantics but is not PostgreSQL wire compatible.
- **Files**:
  - `compose.production.yaml`
  - `infra/duckdb/Dockerfile` (new)
  - `infra/duckdb/server.py` (new; owns the connection, starts Quack, handles shutdown/checkpoint)
  - `src/db/client.py`
  - `src/constants.py`
  - `src/flows/deploy.py`
  - `bentofile.yaml`
  - `dbt/profiles.yml`
  - `justfile`
  - `tests/test_db_client.py`
  - `README.md`
- **Acceptance Criteria**:
  - The DuckDB image pins a DuckDB release that ships Quack from the official `core` extension repository, opens `/data/tennis.duckdb`, loads Quack, and serves `quack:0.0.0.0:9494` with an explicit token.
  - Compose mounts the complete production database at `/data/tennis.duckdb`, persists it independently of the Bento image/container, exposes Quack only to the Compose network and optionally `127.0.0.1:9494` for host administration, and starts Bento only after a real database query healthcheck passes.
  - `ENVIRONMENT` defaults to `dev`; local Python, notebooks, dbt, tests, and `bentoml serve` therefore use `TENNIS_DB_PATH` or the existing `data/tennis.duckdb` default without extra configuration.
  - Production Compose explicitly sets `ENVIRONMENT=production` on Bento. Production mode creates a local DuckDB client session, loads Quack, attaches `QUACK_URI` (default `quack:duckdb:9494`) with `QUACK_TOKEN`, and makes it the default catalog so existing schema-qualified SQL such as `gold.match_features` executes unchanged.
  - `src/flows/deploy.py` builds without opening or packaging the database, then starts Compose with production runtime settings. A direct local `bentoml serve` remains in development mode and automatically uses the embedded repository database.
  - Unknown `ENVIRONMENT` values fail fast. Production mode fails readiness when `QUACK_URI`/`QUACK_TOKEN` are missing or the companion cannot answer a real query; it never silently falls back to an embedded database.
  - `src/db/client.py` preserves parameterized execution and the existing DataFrame-returning helpers in both modes; dbt uses the same target selection rather than introducing a second SQL dialect.
  - The production database is no longer copied into or fingerprinted as part of the Bento image. Deploying a new model does not duplicate the DuckDB file, and replacing/checkpointing the mounted database does not require rebuilding Bento.
  - Quack credentials remain in the untracked runtime environment, never Compose YAML, image layers, logs, or Git. The endpoint is not exposed publicly without a TLS-terminating reverse proxy.
  - The same representative bronze/silver/gold SELECTs, parameterized inference queries, and one read/write smoke transaction return equivalent results with `ENVIRONMENT=dev` and `ENVIRONMENT=production`.
- **Guardrails**:
  - Do not use the experimental PostgreSQL-wire extension or claim generic JDBC compatibility.
  - Do not run two processes that directly open the production DuckDB file; the Quack container is its sole owner and all production clients connect remotely.
  - Do not infer production mode from hostname, container detection, or Bento internals; `ENVIRONMENT` is the sole mode switch.
  - Do not bake production data or authentication tokens into either container image.

### [x] Task 6: Finalize and apply the schema/feature reduction

- **Description**: Apply the finalized storage and 36-column model contract below after Task 5 establishes the production database boundary. Remove `silver.player_rankings`; reduce `silver.player_matches`, `gold.rolling_features`, and `gold.match_features`; and refactor every consumer before deleting its source table or column. No feature-selection performance comparison is required.
- **Files**:
  - `dbt/models/silver/`
  - `dbt/models/gold/`
  - `dbt/models/silver/player_rankings.sql` (remove)
  - `src/constants.py`
  - `src/features/columns.py`
  - `src/features/inference.py`
  - `src/serving/service.py`
  - `notebooks/parameters/`
  - `data/processed/feature_cols.json`
  - `web/`
  - `tests/`
- **Acceptance Criteria**:
  - `silver.player_rankings` is removed; rank-history and rank-point consumers query equivalent player-oriented rows from `bronze.match_events` or the retained player-history source, including both raw player sides.
  - `silver.player_matches`, `gold.rolling_features`, `gold.match_features`, and `FEATURE_COLS` match the finalized retained-column contracts below.
  - Every removed relation/column has its endpoint, dashboard, inference, notebook, feature-manifest, serving, and test consumers refactored first. No reference to a deleted table or column remains.
  - Current-match analysis rates remain derivable on demand from bronze with the existing `NULLIF` zero-denominator behavior.
  - ID-based inference derives all differences from the latest eligible snapshots and emits exactly the final 36 columns in order, without current-match data leakage.
  - Surface remains numeric one-hot encoding for the shared linear/GBDT/neural contract.
  - Existing registered models with the old feature shape are rejected/not served; retraining occurs only when the user separately resumes training.

#### Required consumer-refactor rules

- Removing a relation or column always includes all downstream callers; deletion alone is incomplete.
- Rank history moves off `silver.player_rankings` to bronze/player history.
- Dashboard or analysis code needing removed current-match rates derives them from bronze raw counts.
- Match-model profile removals do not remove separate dashboard profile data; height remains in `gold.player_profiles` and the profile API.
- Diff-only model fields remove duplicated side values only from `gold.match_features`/`FEATURE_COLS`; their source values stay in `gold.rolling_features` so training and inference can derive the diff.

#### Final silver and rolling contracts

- **Keep in `silver.player_matches`:** match identity/date, player/opponent ids, `match_won`, surface, player ranking/rank points/age, opponent ranking, raw serve/break counts, `player_match_number`, and the activity state needed for final `matches_30d` behavior.
- **Remove from `silver.player_matches`:** `winner_id`, `opponent_rank_points`, `opponent_age`, ATP-provided `wins_last_10`/`matches_last_10`, and `previous_match_date`. Join `tournament`, `round`, and `is_indoor` from bronze by `match_id`; keep surface in silver because rolling surface form needs it.
- **Keep in `gold.rolling_features`:** identity/snapshot keys, latest player ranking/rank points/age, selected `_10` source values, `weighted_form_10`, `avg_player_rank_10`, `avg_rank_faced_10`, signed `streak`, and carried-forward per-surface `_10` rates.
- **Remove from `gold.rolling_features`:** every `_5`/`_20` output, separate win/loss streaks, and intermediate/source outputs not needed by the final match contract or inference. Raw serve/break counts remain in silver because retained rolling rates require them.

#### Final model feature contract

`FEATURE_COLS` has 36 numeric columns. The player/opponent rolling snapshots remain the source of the diff fields; these are model inputs only, not a request to delete snapshot data.

```python
FEATURE_COLS = [
    # Matchup differences: rolling values are all 10-match values.
    "rank_diff",
    "rank_points_diff",
    "age_diff",
    "win_rate_diff",
    "ace_rate_diff",
    "first_serve_pct_diff",
    "break_points_saved_pct_diff",
    "first_serve_win_pct_diff",
    "second_serve_win_pct_diff",
    "serve_win_pct_diff",
    "df_rate_diff",
    "aces_per_svc_game_diff",
    "rank_trend_diff",
    "avg_rank_faced_diff",
    "streak_diff",

    # Values where the absolute state of both players matters.
    "player_weighted_form_10",
    "opponent_weighted_form_10",
    "player_days_since_last_match",
    "opponent_days_since_last_match",
    "player_matches_30d",
    "opponent_matches_30d",
    "player_surface_win_rate_10",
    "opponent_surface_win_rate_10",
    "player_is_left_handed",
    "opponent_is_left_handed",
    "player_years_pro",
    "opponent_years_pro",

    # Canonical-player head-to-head history.
    "player_h2h_matches",
    "player_h2h_wins",

    # Numeric match context; keep one-hot surface for linear and neural models.
    "is_clay",
    "is_grass",
    "is_hard",
    "is_carpet",
    "is_indoor",
    "tournament_level",
    "round_encoded",
]
```

`match_id`, `match_date`, `player_id`, `opponent_id`, `tournament`, `round`, and `surface` remain metadata. `match_won` remains the label. Current-match serve/break analysis rates are omitted from gold but derive on demand from bronze raw counts.

**Execution gate:** apply this finalized contract after Task 5's embedded/Quack parity checks pass. No feature-selection training comparison is planned.

### [x] Task 7: Build and deploy the webapp in the same Compose stack

- **Description**: Add a production webapp container built from the `web/` application and deploy the webapp, Bento, and Quack DuckDB companion together through the existing `just deploy-bento` path and one tracked `compose.production.yaml`. The webapp must use the Compose Bento hostname, not `127.0.0.1`, for browser-to-API requests; host access remains through the single Compose-published web port and Bento remains an internal dependency unless the existing host-local API port is needed for administration. Compose must build the webapp image locally as part of deployment, while Bento continues pulling the Docker Hub `latest` image.
- **Files**:
  - `web/Dockerfile` (new or corrected)
  - `web/nginx.conf` or equivalent production server config (new or corrected)
  - `web/` frontend API/runtime configuration
  - `compose.production.yaml`
  - `justfile`
  - `src/flows/deploy.py`
  - `README.md`
  - `tests/`
- **Acceptance Criteria**:
  - One `just deploy-bento` command builds the webapp container and starts the webapp, Bento, and Quack services from the same Compose file; it does not require a second web deployment command.
  - The webapp production container builds from `web/`, serves the compiled frontend, exposes a documented host port, and has a healthcheck.
  - Browser requests use the Compose-resolvable Bento service name or a same-origin reverse-proxy path; no production bundle points at `127.0.0.1` for the API.
  - Compose service dependencies and healthchecks ensure webapp starts only after its required API/database services are ready.
  - `just deploy-bento --force` still passes `--force` to deployment and rebuilds the local webapp image as well as bypassing Bento caches.
  - Webapp, Bento, and Quack use one Compose network and the production DuckDB remains exclusively owned by Quack; the webapp never mounts or packages the database.
  - Offline frontend build/tests and Compose config validation pass; actual deployment remains subject to Docker Hub credentials, model artifacts, `QUACK_TOKEN`, and a running Docker daemon.

### [x] Task 8: Rewrite documentation and prove the local path

- **Description**: Document the authoritative raw CSV scope, the default deterministic minimal-seed workflow used by every test/validation command, DuckDB/dbt flow, local k3d responsibilities, Quack-backed production database, and Docker Hub/Compose Bento deployment. Mention `seed.py --all` only as an explicit production load option and do not execute it as proof of the workflow. Remove obsolete streaming platform language.
- **Files**:
  - `README.md`
  - `AGENTS.md`
  - `justfile`
- **Acceptance Criteria**:
  - An operator can run the minimal seed, dbt validation, training, and Compose deployment using documented commands without loading the full raw corpus.
  - All documented tests and validation use the minimal seed; `--all` is clearly optional and never part of setup, QA, or completion checks.
  - The normal workflow contains no implicit Wikipedia/network call. Documentation marks `just db-etl --enrich` as an explicit ETL-only opt-in operation, explains its `Playing style`-then-lead extraction order, and notes that only final QA invokes it.
  - Repeating the ingestion commands is idempotent.
  - The documented backup/recovery path covers the raw CSVs, DuckDB file, Prefect state, MLflow state, and Compose deployment env.
  - No documentation claims that a distributed ingestion layer, object storage, or an image registry is required for local operation.

## Dependencies

1. Raw CSV schema validation must pass before ingestion.
2. Validated CSV rows must load into DuckDB before dbt runs.
3. dbt tests must pass before training.
4. A promoted MLflow champion and serving artifacts must exist before Bento deployment.
5. Docker Hub image credentials must be configured before Compose deployment; Bento deployment is independent of k3d readiness.
6. Task 4's Bento Compose path must work before the Task 5 Quack companion is added.
7. The complete production DuckDB file must exist at the configured bind-mount path and `QUACK_TOKEN` must be set before production Compose starts.
8. Task 5's `ENVIRONMENT`-selected embedded/Quack database boundary must pass parity checks before Task 6 changes the persisted schemas and feature contract.
9. Task 7's webapp container depends on the Task 5 Compose stack and is deployed through the same `just deploy-bento` command.

## QA scenarios

- Run the default minimal seed and verify invalid rows are reported and valid rows reach DuckDB bronze; focused unit tests cover all-file discovery/ordering without executing a full `seed.py --all` load.
- Run default seed and ETL with network calls blocked and verify both complete without attempting Wikipedia enrichment.
- At the end of all implementation work, run the minimal seed followed by `just db-etl --enrich`; verify enriched profiles use the first `Playing style` paragraph when present and fall back to the lead paragraph only when absent.
- Introduce a header mismatch and verify ingestion fails before any bronze rows are written.
- Run the seed twice and verify bronze row count and `match_id` uniqueness are unchanged on the second run.
- Run dbt twice with no source changes and verify identical silver/gold results.
- Run training after dbt success, promote a champion, push the Bento image to Docker Hub (secure token login, `latest` tag), and boot Compose from a clean local image cache.
- Verify Compose `/healthz`, `/predict`, restart recovery, and force redeploy.
- Stop k3d and verify host-local Compose Bento remains independently operable when its required serving artifacts are present.
- Run representative queries with `ENVIRONMENT=dev` and `ENVIRONMENT=production`; verify equal results, parameter binding, local `bentoml serve` embedded behavior, Compose Bento remote behavior, restart recovery, and exclusive production-file ownership by the Quack container.
- Verify the production DuckDB mount survives Bento replacement and Quack container recreation, while the Bento image no longer contains `tennis.duckdb`.
- Apply Task 6 and verify `silver.player_rankings` has no remaining consumers, reduced silver/rolling schemas contain only their finalized fields, and `gold.match_features`/inference both emit the exact 36-column contract in the documented order.
- Verify profile/rank-history/dashboard endpoints still work by reading retained tables or deriving removed values from bronze, and verify no old-shape model can start serving.
- Build the webapp container, validate the production bundle contains no localhost API target, and boot the single Compose stack with webapp, Bento, and Quack healthchecks.

## Completion criteria

- All regular ATP CSVs under `data/raw` are the only match source of truth; `data/raw/2026.csv` is the current deterministic seed subset.
- Direct authoritative-file validation/ingestion: schema validation, invalid-row reporting, direct bronze load, and re-ingestion idempotency by `match_id`.
- Every setup, QA, and validation command uses the deterministic minimal seed; full-corpus `seed.py --all` remains an explicit production operation and is not run by this plan.
- Seed and default ETL make no online Wikipedia enrichment calls. ETL-only `--enrich` uses the first `Playing style` paragraph with lead-paragraph fallback, and receives one final live-network validation against the seeded miniset.
- Validated rows load directly into DuckDB bronze; dbt produces baseline-compatible silver/gold tables.
- Training remains standalone and MLflow aliases remain the promotion mechanism.
- Production Bento is a Docker Hub image pushed with secure token login and the `latest` tag, booted by the tracked Docker Compose file beside the locally built webapp and Quack DuckDB companion; no k3d registry or Kubernetes Bento rollout exists.
- `ENVIRONMENT=dev|production` is the sole backend-mode switch: local `bentoml serve` uses embedded DuckDB and production Compose Bento uses the Quack companion; both execute the same DuckDB SQL and pass parity checks.
- The full production database lives only in the persistent Quack mount, is not baked into Bento, and remains intact across model/container replacement.
- README documents the complete verified workflow and recovery behavior.
- The webapp, Bento, and Quack services deploy together through `just deploy-bento` and share one Compose network without exposing the production database to the webapp.
- Schema reduction is complete when `silver.player_rankings` is replaced by bronze/player-history queries, silver/rolling tables match their retained contracts, all consumers are refactored, and training plus inference share the finalized 36-column feature contract. No feature-selection training comparison is required.
