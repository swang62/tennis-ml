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

The eventual ingestion scope is all raw ATP CSVs in `data/raw`. Today the deterministic seed subset is the two 2026 files — `data/raw/2026.csv` (regular tour) and `data/raw/2026_challenger.csv` (Challenger) — which `infra/duckdb/seed.py` loads first. There is no ongoing export/merge/append process; new CSVs are added as raw files and picked up by the seed run.

Phase 1 dbt migration is complete. Phase 2 keeps the existing single-node k3d services needed for Prefect and MLflow, while raw CSV ingestion writes directly to the local DuckDB bronze tables after validation. Production BentoML serving runs outside Kubernetes through Docker Compose and pulls the `latest` image from Docker Hub.

## Deployment boundaries

- **Raw data**: all raw ATP CSVs under `data/raw` are the source of truth. They are discovered in sorted filename order and loaded directly. The two 2026 files are the current deterministic seed subset; the scope is all raw ATP CSVs, not only 2026 files.
- **DuckDB**: validated CSV rows load directly into `bronze.match_events`; dbt owns silver/gold transformations. The direct CSV path remains the production path.
- **k3d**: only the existing local Prefect/MLflow platform and supporting stateful services. Ingestion remains a host/local-file operation.
- **BentoML**: `src/flows/deploy.py` builds the promoted Bento, securely logs in to Docker Hub, pushes the `${DOCKER_IMAGE}:latest` tag, and runs `docker compose -f compose.production.yaml up -d --pull always`.
- **Bento access**: host-local `http://127.0.0.1:3000`; no Bento Kubernetes Deployment and no Bento Traefik route.
- **Credentials**: Settings live in the existing untracked `.env` file. It sets `DOCKER_REPO=swang62` and `IMAGE_NAME=tennis-ml`; `DOCKER_TOKEN` is read securely via stdin by `deploy.py` and is never printed or committed. Secrets must never be written to Git. `.env` is untracked and there is no `.env.example` requirement.

## Current baseline

- `src/flows/ingest.py` already maps ATP-format rows to the bronze contract and calls `run_ingestion_checks` before insertion.
- `insert_bronze_rows` uses `match_id` as the DuckDB primary key, so database re-ingestion of any authoritative CSV is idempotent.
- `infra/duckdb/seed.py` supports `--all`: it discovers CSVs sorted under `data/raw`, loads all rows chronologically, is idempotent via `match_id`, and by default runs offline and skips Wikipedia enrichment; `--enrich` opts in.
- All raw ATP CSVs under `data/raw` are the authoritative ingestion scope and are ingested directly. The two 2026 files are the current deterministic seed subset. There are no ongoing CSV exports, no merge command, and no rewriting/append/dedupe maintenance phase.
- `src/flows/deploy.py` reads `DOCKER_TOKEN` securely via stdin to `docker login`, derives the username from `DOCKER_USERNAME` or the `DOCKER_REPO` owner, then pushes the `latest` tag.
- Training is standalone in `src/flows/pipeline.py`. MLflow aliases remain `@best` and `@champion`.
- `bentofile.yaml` remains the Bento template; deploy generates a pinned Bento file.

## Paused: schema reduction (design review required)

Phase 2 is **paused** pending a table/column redesign. No redesign task is complete, and this plan ships no code or data changes.

Approved schema decisions:

1. **Remove materialized `silver.player_rankings`.** Rank history and rank-point queries come from bronze/player-history sources instead. This is a redesign task in scope below; it is not complete.
2. **Trim `gold.match_features` current-match per-side stats.** Of the six current-match stat types (x two sides: player and opponent), remove only three types: `serve_win_pct`, `df_per_svc_game`, `break_points_saved_pct`. The other three types are retained current-match analysis fields (not profile stats): `first_serve_win_pct`, `second_serve_win_pct`, `aces_per_svc_game` (both sides). Only the three removed types are dropped; the three retained current-match analysis fields remain.
3. **Preserve the design review before any `rolling_features`/`player_matches` removal.** Completion of the `player_rankings` removal and the current-match column trim is gated on that review. The remaining redesign candidates additionally require time-split ablation to confirm; they are not marked complete.

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

### [x] Task 2: Load validated ATP CSVs directly into DuckDB

- **Description**: Seed all raw ATP CSVs under `data/raw` into DuckDB. `infra/duckdb/seed.py --all --offline` discovers CSVs in sorted filename order, loads all rows chronologically, and keeps inserts idempotent by `match_id`; `--enrich` opts in to best-effort Wikipedia enrichment. Validate raw rows before insertion, preserve the existing bronze contract, and keep database inserts idempotent by `match_id`. Do not introduce a message broker, object store, connector, or intermediate Parquet source.
- **Files**:
  - `src/flows/ingest.py`
  - `src/features/validate.py`
  - `src/features/columns.py`
  - `infra/duckdb/seed.py`
  - `src/flows/etl.py`
  - `tests/`
- **Acceptance Criteria**:
  - All raw ATP CSVs under `data/raw` can be ingested directly after raw validation; the two 2026 files are the current deterministic seed subset.
  - `seed.py --all --offline` loads all CSV rows chronologically without requiring network enrichment; `--enrich` opts in to Wikipedia enrichment.
  - Columns and headers are schema-validated before ingestion; mismatched schemas fail before any bronze rows are written.
  - Invalid rows are dropped/reported before DuckDB insertion; valid rows remain available for dbt.
  - Re-ingesting any source does not create duplicate bronze rows.
  - `just db-etl` builds silver/gold from DuckDB bronze with no external ingestion service.
  - Existing Wikipedia enrichment remains best-effort and cannot damage durable match ingestion.

### [x] Task 3: Verify dbt, training, and MLflow against the CSV source of truth

- **Description**: Remove any remaining source assumptions that point ETL away from local DuckDB bronze. Verify baseline-compatible silver/gold output for all raw ATP CSV matches, then keep training standalone and promotion alias-based.
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

### [ ] Task 4: Build and serve Bento through Docker Hub and Compose

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
  - Compose has one service with `image: ${DOCKER_IMAGE}:latest`, `pull_policy: always`, `3000:3000`, `/healthz` healthcheck, and `restart: unless-stopped`.
  - A clean Docker host can pull the `latest` image and boot Bento without k3d or Kubernetes.
  - `/healthz` and `/predict` pass after boot and after a container restart.
  - No Docker Hub token is written to Git, image labels, Compose YAML, or logs.

### [ ] Task 5: Schema reduction (paused, design review required)

- **Description**: Apply the approved schema decisions pending the table/column design review. Remove the materialized `silver.player_rankings` table so rank history and rank-point queries come from bronze/player-history sources instead. In `gold.match_features`, drop only the three current-match per-side stat types `serve_win_pct`, `df_per_svc_game`, `break_points_saved_pct` (both player and opponent); keep the retained current-match analysis fields `first_serve_win_pct`, `second_serve_win_pct`, and `aces_per_svc_game` (both sides). Preserve the design review before any `rolling_features`/`player_matches` removal. This task is paused and not complete; remaining redesign candidates require time-split ablation before they are confirmed.
- **Files**:
  - `dbt/models/silver/`
  - `dbt/models/gold/`
  - `src/features/columns.py`
  - `src/features/`
  - `tests/`
- **Acceptance Criteria**:
  - `silver.player_rankings` is removed and rank history/rank-point queries resolve from bronze/player-history sources.
  - `gold.match_features` retains `first_serve_win_pct`, `second_serve_win_pct`, `aces_per_svc_game` per side and drops only `serve_win_pct`, `df_per_svc_game`, `break_points_saved_pct` per side.
  - The design review is preserved before any `rolling_features`/`player_matches` removal; remaining candidates pass time-split ablation before being marked complete.

#### `gold.match_features` reduction candidate inventory (unconfirmed, requires time-split ablation)

The groups below are candidates for time-split ablation, not approved removals. Each is recorded so it can be tested on its own split; none may be dropped on the strength of this list alone.

- **5/10/20 window consolidation**: `win_rate_5`, `win_rate_10`, `win_rate_20`. Consider retaining one short and one medium/long horizon rather than all three.
- **5/10 pair consolidation**: `ace_rate`, `first_serve_pct`, `break_points_saved_pct`, `first_serve_win_pct`, `second_serve_win_pct`, `serve_win_pct`, `df_rate`, `aces_per_svc_game`. Candidate to collapse the separate 5 and 10 window variants.
- **10/20 consolidation**: `rank_trend_10`, `rank_trend_20`.
- **5/10 consolidation**: `avg_rank_faced_5`, `avg_rank_faced_10`.
- **Streak representation**: `win_streak` + `loss_streak` could become one signed streak, but this is a representation change, not blind deletion.
- **Exact H2H redundancy**: `opponent_h2h_matches`/`wins`/`win_rate` are deterministic complements of the `player_h2h_*` columns and can potentially be removed from the model contract after parity tests.
- **Differential redundancy**: `diff` columns are mathematically derived from the side columns; do not remove without comparing side-only vs diff-only vs both.
- **Surface one-hot compression**: `is_clay`/`is_grass`/`is_hard`/`is_carpet` could become one categorical encoding, but preserve carpet support and test model/inference compatibility.
- **Metadata/non-feature columns**: `match_id`, `match_date`, `player_id`, `opponent_id`, `tournament`, `round`, `surface`, `match_won` must be classified separately; `match_won` is the label and metadata is retained for evaluation/debugging, not proposed for blind deletion.

**Currently approved trim (this is the only approved reduction):** remove both-side current-match `serve_win_pct`, `df_per_svc_game`, `break_points_saved_pct`; retain both-side `first_serve_win_pct`, `second_serve_win_pct`, `aces_per_svc_game`. All groups above remain unconfirmed pending time-split ablation.

### [ ] Task 6: Rewrite documentation and prove the local path

- **Description**: Document the authoritative CSV files (all raw ATP CSVs, with the two 2026 files as the current deterministic seed subset), the direct ingestion path via `seed.py --all --offline` (with optional `--enrich`), validation report, DuckDB/dbt flow, local k3d responsibilities, and Docker Hub/Compose Bento deployment. Remove obsolete streaming platform language.
- **Files**:
  - `README.md`
  - `AGENTS.md`
  - `justfile`
- **Acceptance Criteria**:
  - An operator can seed all raw ATP CSVs, run dbt, train, and deploy Bento using documented commands.
  - Repeating the ingestion commands is idempotent.
  - The documented backup/recovery path covers the raw CSVs, DuckDB file, Prefect state, MLflow state, and Compose deployment env.
  - No documentation claims that a distributed ingestion layer, object storage, or an image registry is required for local operation.

## Dependencies

1. Raw CSV schema validation must pass before ingestion.
2. Validated CSV rows must load into DuckDB before dbt runs.
3. dbt tests must pass before training.
4. A promoted MLflow champion and serving artifacts must exist before Bento deployment.
5. Docker Hub image credentials must be configured before Compose deployment; Bento deployment is independent of k3d readiness.
6. The schema-reduction design review must pass before `silver.player_rankings` or `gold.match_features` current-match columns are removed; remaining redesign candidates additionally require time-split ablation before completion.

## QA scenarios

- Run `seed.py --all --offline` and verify invalid rows are reported and valid rows reach DuckDB bronze; repeat with `--enrich` and verify enrichment is optional and does not damage durable ingestion.
- Introduce a header mismatch and verify ingestion fails before any bronze rows are written.
- Run the seed twice and verify bronze row count and `match_id` uniqueness are unchanged on the second run.
- Run dbt twice with no source changes and verify identical silver/gold results.
- Run training after dbt success, promote a champion, push the Bento image to Docker Hub (secure token login, `latest` tag), and boot Compose from a clean local image cache.
- Verify Compose `/healthz`, `/predict`, restart recovery, and force redeploy.
- Stop k3d and verify host-local Compose Bento remains independently operable when its required serving artifacts are present.
- After the schema-reduction design review, run the redesign and verify `silver.player_rankings` removal keeps rank history/rank-point queries working from bronze/player-history sources, and `gold.match_features` retains `first_serve_win_pct`, `second_serve_win_pct`, `aces_per_svc_game` per side while dropping only `serve_win_pct`, `df_per_svc_game`, `break_points_saved_pct` per side.

## Completion criteria

- All raw ATP CSVs under `data/raw` are the only match source of truth; the two 2026 files are the current deterministic seed subset.
- Direct authoritative-file validation/ingestion: schema validation, invalid-row reporting, direct bronze load, and re-ingestion idempotency by `match_id`.
- `seed.py --all --offline` seeds all raw CSVs chronologically; `--enrich` optionally enables Wikipedia enrichment.
- Validated rows load directly into DuckDB bronze; dbt produces baseline-compatible silver/gold tables.
- Training remains standalone and MLflow aliases remain the promotion mechanism.
- Production Bento is a Docker Hub image pushed with secure token login and the `latest` tag, booted by the tracked Docker Compose file; no k3d registry or Kubernetes Bento rollout exists.
- README documents the complete verified workflow and recovery behavior.
- Schema reduction is complete only after the design review: `silver.player_rankings` is removed in favor of bronze/player-history rank sources, and `gold.match_features` retains `first_serve_win_pct`, `second_serve_win_pct`, `aces_per_svc_game` per side while dropping only `serve_win_pct`, `df_per_svc_game`, `break_points_saved_pct` per side; remaining redesign candidates pass time-split ablation.