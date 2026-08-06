# tennis-ml

Production-grade MLOps pipeline for tennis match prediction. Prefect, PostgreSQL, MLflow, BentoML — the full stack.

## Stack

| Layer               | Tool                                      |
| ------------------- | ----------------------------------------- |
| Orchestration       | Prefect (retries, ETL triggers)           |
| Experiment tracking | MLflow (model registry, trial comparison) |
| Model serving       | BentoML                                   |
| Data warehouse      | PostgreSQL (pg_duckdb)                    |
| Development         | Jupyter + Papermill                       |

## Project Structure

```
infra/           — k3d config, static K8s manifests, PostgreSQL init SQL
notebooks/       — EDA + parameterized Papermill notebooks
src/
  features/      — Feature column definitions (shared)
  flows/         — ETL Prefect flow + standalone training pipeline (src/flows/pipeline.py)
  models/        — Player similarity index (FAISS)
  serving/       — BentoML service
  db/            — PostgreSQL client + DuckDB training snapshot
web/             — React + TanStack dashboard (Vite, local dev, HMR)
```

## Quick Start

```bash
# 1. Full local dev setup (deps + k3d cluster for Prefect/MLflow + PostgreSQL init)
just setup

# 2. Seed the deterministic minimal match set into PostgreSQL bronze
just db-seed

# 3. Start the Prefect worker (must run on the host, see below)
just worker
```

## Prefect worker

Start the local worker from the repo root:

```bash
just worker
```

## Data Flow

```
 Raw match data → seed.py (validate + load direct to PostgreSQL)
                      ↓
            ┌──────────────────────┐
            │        BRONZE        │
            │ (match_events)       │
            └──────────┬───────────┘
                       │ dbt build
                       ↓
            ┌──────────────────────┐
            │        SILVER        │
            │  player_matches +    │
            │  rolling_features    │
            └──────────┬───────────┘
                       │ dbt build
                       ↓
            ┌──────────────────────┐
            │        GOLD          │
            │  match_features      │
            └──────────┬───────────┘
                       ↓
         Standalone training pipeline.py (features, tuning, evaluation, promotion)
                       ↓
                MLflow registry → BentoML serving
```

The regular ATP CSVs under `data/raw` are the authoritative match source. They
are loaded directly into PostgreSQL bronze after validation; no external
ingestion service, object storage, or image registry is required for ETL,
training, or tests. k3d only runs Prefect and MLflow for local orchestration
and model tracking; the Bento/web serving stack is host-local via Docker
Compose (see Deployment).

`just db-seed` loads a deterministic minimal seed: the most recent matches of
the best-ranked players in `data/raw/2026.csv` (top 10 players by latest rank,
10 recent matches each, deduped to 28 distinct matches / 35 players) — not the
full corpus. It is permanently offline: it
never performs Wikipedia enrichment and cannot be made to. Re-running it is
idempotent — its own rows are replaced in place. The full historical corpus is
only loaded by the explicit `just db-seed -- --all` production load option,
which is never part of setup, tests, or QA.

## Trigger Model

| Event          | Action                                 | Method                                                  |
| -------------- | -------------------------------------- | ------------------------------------------------------- |
| Manual ingest  | Load CSV → bronze                      | `just db-seed` (deterministic seed subset)              |
| Manual trigger | Training pipeline                      | `just train`                                            |
| Model promoted | Web + BentoML rebuild + deploy   | `just deploy-bento` (reads `@champion`) |
| Force redeploy | Rebuild + redeploy regardless of cache | `just deploy-bento --force`                          |

## Pipelines

- `ingest` — validate raw ATP CSV → bronze
- `seed.py` (default, `just db-seed`) — the deterministic minimal seed from `data/raw/2026.csv` (top 10 players by latest rank, recent matches, deduped); permanently offline, idempotent
- `seed.py --all` (`just db-seed --all`) — explicit production load option: discover every regular ATP CSV under `data/raw`, load chronologically → bronze (idempotent by `match_id`); still permanently offline, but never part of setup/tests/QA
- `etl` — bronze → silver → gold: player_matches + rolling_features → match_features, plus feature enrichment and sanitization. Offline by default; `just db-etl --enrich` is the explicit opt-in to Wikipedia bio enrichment (first `Playing style` paragraph, article-lead fallback).
- `pipeline.py` — standalone training runner: features → tune 3 models → pick best → train final → evaluate → promote

## Inspecting the data

Inspect each stage directly in PostgreSQL: `bronze.match_events` (raw), `silver.player_matches` (per-player rows), `silver.rolling_features` (post-match snapshots), `gold.match_features` (final canonical training rows).

## Serving & Inference

### Endpoints

Two BentoML endpoints exposed by `src/serving/service.py`:

| Endpoint            | Method | Purpose                                                      |
| ------------------- | ------ | ------------------------------------------------------------ |
| `/healthz`          | GET    | Liveness probe (alias `/livez`, `/readyz`)                   |
| `/predict`          | POST   | Stacked-ensemble prediction for one match                    |
| `/predict_from_ids` | POST   | On-demand prediction from minimal inputs (two ids + surface) |

### Deployment: one command, one Compose stack

`just deploy-bento` (optionally `--force`) is the single deployment path. It
builds the Bento image locally from the promoted `ensemble_lr_model@champion`
artifacts, pushes it to Docker Hub as `latest`, builds the webapp image
locally from `web/`, and boots one Compose stack (`compose.yaml`) containing
three services: PostgreSQL, the Bento, and the webapp. There is no separate
web deployment step.

| Service    | Image source                 | Host port | Healthcheck                                 |
| ---------- | ---------------------------- | --------- | ------------------------------------------- |
| `postgres` | `pgduckdb/pgduckdb` (pinned) | 6543      | `pg_isready` + `pg_duckdb` extension check  |
| `bento`    | Docker Hub `latest`          | 3000      | authenticated `SELECT 1` against PostgreSQL |
| `web`      | built locally                | 8187      | `wget` of the SPA root inside the container |

- PostgreSQL runs on the pinned `pgduckdb/pgduckdb` image, mapped to host port
  **6543:5432**; the Bento reaches it over the Compose network at
  `postgres:5432`. It owns a named volume and applies
  `infra/postgres/init.sql` (extension + schemas + base tables — structure
  only) on first start.
- `bento` starts after PostgreSQL is healthy; `web` starts after `bento` is
  healthy.
- The webapp serves the compiled SPA via nginx and proxies `/api/*` to the
  Bento over the Compose network (`bento:3000`); the browser only talks to
  port 8187.
- PostgreSQL is a third-party pinned image, never tagged or pushed by
  `deploy.py`; only the Bento image is pushed to Docker Hub.

Deploy-time credentials live in the git-ignored `.env`. Values are never
printed or committed:

- `DATABASE_URL` (or `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB`) —
  the shared PostgreSQL connection contract. Host commands, dbt, and the
  Compose stack derive the same URL from these components, differing only in
  host/port (`127.0.0.1:6543` on the host, `postgres:5432` on the Compose
  network).
- `DOCKER_TOKEN` (optional) — Docker Hub auth, passed to `docker login` via
  stdin; when unset, deploy relies on an already-authenticated Docker CLI.
- `DOCKER_USERNAME` / `DOCKER_REPO` / `IMAGE_NAME` — Docker Hub identity and
  image naming (`${DOCKER_REPO}/${IMAGE_NAME}:latest`).

Local ETL, training, and tests never touch Docker Hub — the registry is
involved only in `just deploy-bento` / production serving.

### Operational database and training snapshot

PostgreSQL is the only operational backend. Host commands (`just db-init`,
`db-seed`, `db-etl`, `db-dbt`), dbt, the Bento, and the dashboard all connect
through the shared `DATABASE_URL` contract in `.env`. `just deploy-bento`
requires the PostgreSQL credential (passed to Compose, never printed).
`just db-reset` (destructive) drops and recreates the bronze/silver/gold
schemas; it refuses to run against any target other than the expected local
dev database (`127.0.0.1:6543` + configured `POSTGRES_DB`), so a stray
environment name can never reset a non-local database.

Training is the only DuckDB consumer: `just db-snapshot` pulls an atomic,
validated two-table snapshot (`gold.match_features` + `gold.player_profiles`)
from PostgreSQL into the ignored `data/processed/training_snapshot.duckdb`,
and `just train` refreshes it automatically before the notebooks run. No seed
or ETL command touches a DuckDB file.

### Input schema

The `/predict_from_ids` endpoint accepts a JSON object with the following fields:

| Input         | Required | Default | Valid values                                                   |
| ------------- | -------- | ------- | -------------------------------------------------------------- |
| `player_id`   | yes      | —       | non-empty str                                                  |
| `opponent_id` | yes      | —       | non-empty str                                                  |
| `surface`     | yes      | —       | `clay` / `grass` / `hard` / `carpet`                           |
| `tournament`  | no       | 0       | `grand_slam` / `masters` / `atp_500` / `atp_250` / `davis_cup` |
| `round`       | no       | 0       | `r128` / `r64` / `r32` / `r16` / `qf` / `sf` / `f`             |
| `as_of_date`  | no       | today   | `datetime.date`                                                |

## Extra Notes

- **Canonicalization** — balanced symmetric features, the lower lexicographic player id becomes the `player_*` side, so `(A, B)` and `(B, A)` produce identical rows.
- **Rolling form lookup** — live inference reads each player's newest `silver.rolling_features` snapshot strictly before `as_of_date` (dbt computes the snapshots from `silver.player_matches`).
- **Cold-start imputation** — missing players (no eligible snapshot) get on-demand global aggregates.
- **Unranked players** — ATP rank 0 is the missing marker; it maps to NULL in `silver.player_matches`, so matches are never dropped for missing rankings and rolling rank averages skip unranked matches. Training imputes the NULL (median) along with every other missing cell.
- **Bento image is fully self-contained** — Bento loads 4 artifacts from the MLflow registry by alias, ensemble model uses best/promoted `[p_linear, p_gbdt, p_nn]` → `p_win`. The database is deliberately NOT in the image: production serving reads PostgreSQL live through psycopg.
