# Plan: Phase 2 local CSV-to-DuckDB pipeline

## Goal

Keep the platform local and simple. The authoritative match sources are the raw ATP CSV files:

```text
data/raw/2026.csv                 ┐
data/raw/2026_challenger.csv      ├─ validate raw ATP rows -> DuckDB bronze
ongoing CSV exports -> dedupe ----┘                         -> dbt silver/gold
                                                               -> training -> MLflow
                                                               -> Docker Hub image -> Compose Bento
```

Phase 1 dbt migration is complete. Phase 2 keeps the existing single-node k3d services needed for Prefect and MLflow, while raw CSV ingestion writes directly to the local DuckDB bronze tables after validation. Production BentoML serving runs outside Kubernetes through Docker Compose and pulls an immutable image from Docker Hub.

## Deployment boundaries

- **Raw data**: `data/raw/2026.csv` is the regular-tour source of truth. `data/raw/2026_challenger.csv` is the Challenger source of truth.
- **Ongoing data**: `data/raw/ongoing_tourneys.csv` appends to the regular file; `data/raw/challenger_ongoing_tourneys.csv` appends to the Challenger file. Appends are schema-checked and deduplicated by `(tourney_date, tourney_id, match_num)` before either file is rewritten.
- **DuckDB**: validated CSV rows load directly into `bronze.match_events`; dbt owns silver/gold transformations. The legacy direct CSV path remains the production path.
- **k3d**: only the existing local Prefect/MLflow platform and supporting stateful services. Ingestion remains a host/local-file operation.
- **BentoML**: `src/flows/deploy.py` builds the promoted Bento, pushes `${DOCKER_IMAGE}:${DOCKER_TAG}` to Docker Hub, and runs `docker compose -f compose.production.yaml up -d --pull always`.
- **Bento access**: host-local `http://127.0.0.1:3000`; no Bento Kubernetes Deployment and no Bento Traefik route.
- **Credentials**: Docker Hub credentials come from the operator environment or an untracked local env file. Tracked templates contain names only, never tokens.

## Current baseline

- `src/flows/ingest.py` already maps ATP-format rows to the bronze contract and calls `run_ingestion_checks` before insertion.
- `insert_bronze_rows` uses `match_id` as the DuckDB primary key, so database re-ingestion is idempotent; the source CSV merge must also be idempotent.
- `data/raw/2026.csv`, `ongoing_tourneys.csv`, and `challenger_ongoing_tourneys.csv` exist. `2026_challenger.csv` may need to be created.
- Training is standalone in `src/flows/pipeline.py`. MLflow aliases remain `@best` and `@champion`.
- `bentofile.yaml` remains the Bento template; deploy generates a pinned Bento file.

## Tasks

### [ ] Task 1: Inventory the local platform and simplify environment commands

- **Description**: Remove obsolete distributed-ingestion assumptions from the Phase 2 architecture and document the compact local path: CSV -> validation -> DuckDB -> dbt -> training/MLflow. Keep only the k3d workloads that are still required for local orchestration and model tracking. Use one documented env template for DuckDB, Prefect, MLflow, Docker Hub image name/tag, and Compose settings.
- **Files**:
  - `infra/k3d/config.yaml`
  - `infra/k3d/start.sh`
  - `infra/manifests/default/config-map.yaml`
  - `justfile`
  - `.env.example` (new or updated)
  - `README.md`
- **Acceptance Criteria**:
  - No external ingestion dependency is required to run ETL or training.
  - `just setup`, `just db-init`, `just db-etl`, `just train`, and `just deploy-bento` have clear, minimal responsibilities.
  - Local DuckDB and pipeline artifacts persist across k3d restarts.
  - No k3d image registry or registry hostname remains in the deployment path.

### [ ] Task 2: Maintain authoritative 2026 regular and Challenger CSVs

- **Description**: Add a small, deterministic maintenance command that reads the existing target CSV and its ongoing export, verifies identical ATP headers, validates the raw identity fields, appends only unseen matches, and atomically rewrites the target. Run it for both regular and Challenger data. If `2026_challenger.csv` is absent, create it from the Challenger ongoing export with the canonical header.
- **Files**:
  - `src/flows/ingest.py` or `src/flows/merge_raw.py` (new helper/CLI)
  - `tests/` (focused merge tests)
  - `justfile`
  - `data/raw/2026.csv`
  - `data/raw/2026_challenger.csv`
- **Acceptance Criteria**:
  - Regular ongoing rows merge into `data/raw/2026.csv`.
  - Challenger ongoing rows merge into `data/raw/2026_challenger.csv`.
  - Duplicate identity is `(tourney_date, tourney_id, match_num)`; rerunning the command adds zero rows and does not reorder or duplicate existing data.
  - Headers and column order are preserved; mismatched schemas fail before any target is changed.
  - Invalid rows are rejected with a visible report containing input, added, skipped, and invalid counts.
  - Target writes are atomic, so an interrupted or invalid merge cannot leave a partial CSV.
  - A focused test covers first merge, rerun, duplicate rows inside the ongoing file, and schema mismatch.

### [ ] Task 3: Load validated ATP CSVs directly into DuckDB

- **Description**: Make the two 2026 authoritative files the default ingestion inputs. Validate raw rows before insertion, preserve the existing bronze contract, and keep database inserts idempotent by `match_id`. Do not introduce a message broker, object store, connector, or intermediate Parquet source.
- **Files**:
  - `src/flows/ingest.py`
  - `src/features/validate.py`
  - `src/features/columns.py`
  - `infra/duckdb/seed.py`
  - `src/flows/etl.py`
  - `tests/`
- **Acceptance Criteria**:
  - Regular and Challenger CSVs can be ingested directly after raw validation.
  - Invalid rows are dropped/reported before DuckDB insertion; valid rows remain available for dbt.
  - Re-ingesting either source does not create duplicate bronze rows.
  - `just db-etl` builds silver/gold from DuckDB bronze with no external ingestion service.
  - Existing Wikipedia enrichment remains best-effort and cannot damage durable match ingestion.

### [ ] Task 4: Verify dbt, training, and MLflow against the CSV source of truth

- **Description**: Remove any remaining source assumptions that point ETL away from local DuckDB bronze. Verify baseline-compatible silver/gold output for regular and Challenger matches, then keep training standalone and promotion alias-based.
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

### [ ] Task 5: Build and serve Bento through Docker Hub and Compose

- **Description**: Replace the old Kubernetes/k3d Bento rollout with a host-executed deployment. Build the promoted Bento, log in to Docker Hub using token input through stdin, push an immutable tag and the documented moving tag, then pull and boot one Compose service. Compose must not build locally and must not require k3d.
- **Files**:
  - `src/flows/deploy.py`
  - `compose.production.yaml` (new)
  - `.env.example`
  - `justfile`
  - `infra/manifests/deploy/bentoml.yaml` (remove)
  - `README.md`
- **Acceptance Criteria**:
  - `just deploy-bento` performs Bento build -> Docker Hub push -> Compose pull/up.
  - `just deploy-bento -- --force` (or the documented equivalent) bypasses build/image caches.
  - Compose has one service with `image: ${DOCKER_IMAGE}:${DOCKER_TAG}`, `pull_policy: always`, `3000:3000`, `/healthz` healthcheck, and `restart: unless-stopped`.
  - A clean Docker host can pull the immutable image and boot Bento without k3d or Kubernetes.
  - `/healthz` and `/predict` pass after boot and after a container restart.
  - No Docker Hub token is written to Git, image labels, Compose YAML, or logs.

### [ ] Task 6: Rewrite documentation and prove the local path

- **Description**: Document the authoritative CSV files, ongoing merge command, dedupe key, validation report, DuckDB/dbt flow, local k3d responsibilities, and Docker Hub/Compose Bento deployment. Remove obsolete streaming platform language.
- **Files**:
  - `README.md`
  - `AGENTS.md`
  - `justfile`
- **Acceptance Criteria**:
  - An operator can merge both ongoing files, ingest both 2026 CSVs, run dbt, train, and deploy Bento using documented commands.
  - Repeating the merge and ingestion commands is idempotent.
  - The documented backup/recovery path covers the raw CSVs, DuckDB file, Prefect state, MLflow state, and Compose deployment env.
  - No documentation claims that a distributed ingestion layer, object storage, or an image registry is required for local operation.

## Dependencies

1. Raw CSV merge/schema validation must pass before ingestion.
2. Validated CSV rows must load into DuckDB before dbt runs.
3. dbt tests must pass before training.
4. A promoted MLflow champion and serving artifacts must exist before Bento deployment.
5. Docker Hub image credentials must be configured before Compose deployment; Bento deployment is independent of k3d readiness.

## QA scenarios

- Merge regular ongoing data into `2026.csv` and Challenger ongoing data into `2026_challenger.csv`; record added/skipped/invalid counts.
- Run both merges twice and verify the second run adds zero rows and the identity key is unique in each file.
- Introduce a duplicate ongoing row and verify it is skipped.
- Introduce a header mismatch and verify the target remains byte-for-byte unchanged.
- Ingest both 2026 files and verify invalid rows are reported and valid rows reach DuckDB bronze.
- Run ingestion twice and verify bronze row count and `match_id` uniqueness are unchanged on the second run.
- Run dbt twice with no source changes and verify identical silver/gold results.
- Run training after dbt success, promote a champion, push the Bento image to Docker Hub, and boot Compose from a clean local image cache.
- Verify Compose `/healthz`, `/predict`, restart recovery, and force redeploy.
- Stop k3d and verify host-local Compose Bento remains independently operable when its required serving artifacts are present.

## Completion criteria

- Raw regular and Challenger ATP CSVs are the only match source of truth.
- Ongoing CSV data merges into the corresponding 2026 files with schema checks, validation, atomic writes, and stable-key deduplication.
- Validated rows load directly into DuckDB bronze; dbt produces baseline-compatible silver/gold tables.
- Training remains standalone and MLflow aliases remain the promotion mechanism.
- Production Bento is an immutable Docker Hub image booted by the tracked Docker Compose file; no k3d registry or Kubernetes Bento rollout exists.
- README documents the complete verified workflow and recovery behavior.
