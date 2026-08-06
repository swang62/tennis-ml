# tennis-ml

Production-grade MLOps pipeline for tennis match prediction. Prefect, DuckDB, MLflow, BentoML — the full stack.

## Stack

| Layer               | Tool                                      |
| ------------------- | ----------------------------------------- |
| Orchestration       | Prefect (retries, ETL triggers)           |
| Experiment tracking | MLflow (model registry, trial comparison) |
| Model serving       | BentoML                                   |
| Data warehouse      | DuckDB (embedded dev / Quack-served prod) |
| Development         | Jupyter + Papermill                       |

## Project Structure

```
infra/           — k3d config, static K8s manifests, DuckDB init SQL
notebooks/       — EDA + parameterized Papermill notebooks
src/
  features/      — Feature column definitions (shared)
  flows/         — ETL Prefect flow + standalone training pipeline (src/flows/pipeline.py)
  models/        — Player similarity index (FAISS)
  serving/       — BentoML service
  db/            — DuckDB client
web/             — React + TanStack dashboard (Vite, local dev, HMR)
```

## Quick Start

```bash
# 1. Full local dev setup (deps + k3d cluster for Prefect/MLflow + DuckDB init)
just setup

# 2. Seed the deterministic minimal match set into DuckDB bronze
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
 Raw match data → seed.py (validate + load direct to DuckDB)
                      ↓
            ┌──────────────────────┐
            │        BRONZE        │
            │ (match_events)       │
            └──────────┬───────────┘
                       │ dbt build
                       ↓
            ┌──────────────────────┐
            │        SILVER        │
            │   player_matches     │
            └──────────┬───────────┘
                       │ dbt build
                       ↓
            ┌──────────────────────┐
            │        GOLD          │
            │  rolling_features + │
            │  match_features      │
            └──────────┬───────────┘
                       ↓
         Standalone training pipeline.py (features, tuning, evaluation, promotion)
                       ↓
                MLflow registry → BentoML serving
```

The regular ATP CSVs under `data/raw` are the authoritative match source. They
are loaded directly into DuckDB bronze after validation; no external ingestion
service, object storage, or image registry is required for ETL, training, or
tests. k3d only runs Prefect and MLflow for local orchestration and model
tracking; the Bento/web/Quack serving stack is host-local via Docker Compose
(see Deployment).

`just db-seed` loads a deterministic minimal seed: the most recent matches of
the best-ranked players in `data/raw/2026.csv` (top 10 players by latest rank,
~100 matches, deduped) — not the full corpus. It is permanently offline: it
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
- `etl` — bronze → silver → gold: player_matches → rolling_features → match_features, plus feature enrichment and sanitization. Offline by default; `just db-etl --enrich` is the explicit opt-in to Wikipedia bio enrichment (first `Playing style` paragraph, article-lead fallback).
- `pipeline.py` — standalone training runner: features → tune 3 models → pick best → train final → evaluate → promote

## Inspecting the data

Inspect each stage directly in DuckDB: `bronze.match_events` (raw), `silver.player_matches` (per-player rows), `gold.rolling_features` (post-match snapshots), `gold.match_features` (final canonical training rows).

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
locally from `web/`, and boots one Compose stack (`compose.production.yaml`)
containing three services: Quack DuckDB, the Bento, and the webapp. There is
no separate web deployment step.

| Service    | Image source        | Host port | Healthcheck                                   |
| ---------- | ------------------- | --------- | --------------------------------------------- |
| `quack-db` | Docker Hub `latest` | internal  | real `SELECT 1` over the Quack protocol       |
| `bento`    | Docker Hub `latest` | 3000      | GET `http://127.0.0.1:3000/healthz`           |
| `web`      | built locally       | 8187      | `wget` of the SPA root inside the container   |

- `bento` starts after `quack-db` is healthy; `web` starts after `bento` is
  healthy.
- Set `ENVIRONMENT=production` in the untracked `.env` for the Compose stack;
  set it to `dev` when running local Python/dbt/Bento development against the
  embedded database. No CLI environment switch is required.
- The webapp serves the compiled SPA via nginx and proxies `/api/*` to the
  Bento over the Compose network (`bento:3000`); the browser only talks to
  port 8187.
- Quack serves `/data/tennis.duckdb` on the Compose network only (its native
  9494 port is not published by default); the Bento attaches it as a remote
  catalog.

Deploy-time credentials live in the git-ignored `.env` (and the shell for
`QUACK_TOKEN`). Values are never printed or committed:

- `QUACK_TOKEN` (required for deploy, min 4 chars) — the runtime secret shared
  by Quack and the Bento; passed to Compose via the environment and never
  written to logs or argv.
- `DOCKER_TOKEN` (optional) — Docker Hub auth, passed to `docker login` via
  stdin; when unset, deploy relies on an already-authenticated Docker CLI.
- `DOCKER_USERNAME` / `DOCKER_REPO` / `IMAGE_NAME` — Docker Hub identity and
  image naming (`${DOCKER_REPO}/${IMAGE_NAME}:latest`).

Local ETL, training, and tests never touch Docker Hub or Quack — the registry
and remote DB are involved only in `just deploy-bento` / production serving.

### Serving modes

The DuckDB client (`src/db/client.py`) uses a single explicit `ENVIRONMENT` switch:

- `dev` (default) — an embedded DuckDB at `TENNIS_DB_PATH` (or `data/tennis.duckdb`). Used by ETL, training, and local `bentoml serve` (`just deploy-local`).
- `production` — the production DuckDB is served remotely by the official DuckDB Quack companion image (`infra/duckdb/`), which owns `/data/tennis.duckdb` (a Compose named volume, independent of the Bento) and serves it on `0.0.0.0:9494` with an explicit `QUACK_TOKEN`. The Bento opens a local session, loads `quack`, ATTACHes the remote URI with the token, and makes it the default catalog, so the existing `bronze.*`/`silver.*`/`gold.*` SQL resolves verbatim. The DB is never baked into or fingerprinted by the Bento image. A missing/invalid `ENVIRONMENT` or missing Quack config fails fast — it never falls back to the dev DB.

`just deploy-bento` requires `QUACK_TOKEN` in the environment (passed to Compose, never printed). Local ETL/dbt keep the embedded `data/tennis.duckdb`; `dbt/profiles.yml` selects its target by the same `ENVIRONMENT` switch.

The `quack-db` service also publishes `:9494` to the host, so operators can drive
the running server manually without opening the DB file or entering the
container. Schemas are already applied by the container on startup, so only
data-focused commands are needed:

```bash
just db-seed-prod --all   # seed every ATP CSV under data/raw/ into the running server
just db-etl-prod          # bronze -> silver -> gold against the running server
```

These run with `ENVIRONMENT=production`, attach the server over the Quack
protocol at `quack:127.0.0.1:9494`, and reuse `QUACK_TOKEN` from `.env` (no
secret is written into any command or config file). `db-seed` / `db-etl` remain
the local embedded-DB (`dev`) defaults.

`QUACK_TOKEN` is the built-in Quack shared token, required for every remote
attach (host-side operators, the Bento, and deploy). It is bearer/full database
access — a single shared secret, not per-user authorization — so anyone holding
it can read and modify the served DB. Keep it secret, never commit or print it,
and rotate it if it leaks.

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
- **Rolling form lookup** — each player's newest `gold.rolling_features` snapshot strictly before `as_of_date` and are computed on-demand from `silver.player_matches`.
- **Cold-start imputation** — missing players (no eligible snapshot) get on-demand global aggregates.
- **Unranked players** — ATP rank 0 is the missing marker; it maps to NULL in `silver.player_matches`, so matches are never dropped for missing rankings and rolling rank averages skip unranked matches. Training imputes the NULL (median) along with every other missing cell.
- **Bento image is fully self-contained** — Bento loads 4 artifacts from the MLflow registry by alias, ensemble model uses best/promoted `[p_linear, p_gbdt, p_nn]` → `p_win`. The DuckDB is deliberately NOT in the image: production queries it through the Quack companion.
