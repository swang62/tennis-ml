# tennis-ml

Production-grade MLOps pipeline for tennis match prediction. Prefect, DuckDB, MLflow, BentoML — the full stack.

## Stack

| Layer               | Tool                                      |
| ------------------- | ----------------------------------------- |
| Orchestration       | Prefect (retries, ETL triggers)           |
| Experiment tracking | MLflow (model registry, trial comparison) |
| Model serving       | BentoML                                   |
| Data warehouse      | DuckDB (embedded)                         |
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
# 1. Full local dev setup (deps + k3d cluster for Prefect/MLflow + DuckDB init + Seed data)
just setup

# 2. Start the Prefect worker (must run on the host, see below)
just worker
```

## Prefect worker

Start the local worker from the repo root:

```bash
just worker
``0

## Data Flow

```
 Raw match data → ingest.py (pre-validation)
                      ↓
            ┌──────────────────────┐
            │        BRONZE        │
            │ (match_events)       │
            └──────────┬───────────┘
                       │ Prefect etl (dbt build)
                       ↓
            ┌──────────────────────┐
            │        SILVER        │
            │ player_matches +     │
            │ player_rankings      │
            └──────────┬───────────┘
                       │ dbt build
                       ↓
            ┌──────────────────────┐
            │        GOLD          │
            │  rolling_features + │
            │  match_features      │
            └──────────┬───────────┘
                       ↓
         Jupyter notebooks (training, evaluation, promotion)
                       ↓
                MLflow registry → BentoML serving
```

## Trigger Model

| Event          | Action                                 | Method                                                  |
| -------------- | -------------------------------------- | ------------------------------------------------------- |
| Manual ingest  | Load CSV → bronze                      | `uv run python -m src.flows.ingest data/matches.csv`    |
| Manual trigger | Training pipeline                      | `just pipeline`                                         |
| Model promoted | BentoML rebuild + deploy               | `uv run python src/flows/deploy.py` (reads `@champion`) |
| Force redeploy | Rebuild + redeploy regardless of cache | `uv run python src/flows/deploy.py --force`             |

## Pipelines

- `ingest` — CSV → bronze
- `etl` — bronze → silver → gold: player_matches/player_rankings → rolling_features → match_features, plus feature enrichment and sanitization
- `pipeline.py` — standalone training runner: features → tune 3 models → pick best → train final → evaluate → promote

## Inspecting the data

Inspect each stage directly in DuckDB: `bronze.match_events` (raw), `silver.player_matches` (per-player rows), `silver.player_rankings` (ranking series), `gold.rolling_features` (post-match snapshots), `gold.match_features` (final canonical training rows).

## Serving & Inference

### Endpoints

Two BentoML endpoints exposed by `src/serving/service.py`:

| Endpoint            | Method | Purpose                                                      |
| ------------------- | ------ | ------------------------------------------------------------ |
| `/healthz`          | GET    | Liveness probe (alias `/livez`, `/readyz`)                   |
| `/predict`          | POST   | Stacked-ensemble prediction for one match                    |
| `/predict_from_ids` | POST   | On-demand prediction from minimal inputs (two ids + surface) |

The service runs on port 3000 via Docker Compose (`just deploy-bento`), pulling an immutable image from Docker Hub.

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
- **Bento image is fully self-contained** —Bento loads 4 artifacts from the MLflow registry by alias, ensemble model uses best/promoted `[p_linear, p_gbdt, p_nn]` → `p_win`.
