# tennis-ml

Production-grade MLOps pipeline for tennis match prediction. Prefect, PostgreSQL, MLflow, BentoML — the full stack.

## Stack

| Layer               | Tool                                      |
| ------------------- | ----------------------------------------- |
| Orchestration       | Prefect (retries, ETL triggers)           |
| Experiment tracking | MLflow (model registry, trial comparison) |
| Model serving       | BentoML                                   |
| Data warehouse      | PostgreSQL                                |
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

# 3. Start the Prefect worker on host
just worker

# 4. Local dev: Bento API (:3000) + Vite dashboard (:5173)
just dev
```

## Data Flow

```
 Raw match data → seed.py (validated PostgreSQL)
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
Training pipeline.py (features, tuning, evaluation, promotion)
                       ↓
                MLflow registry
                       ↓
          BentoML production endpoints
```

## Trigger Model

| Event          | Action                             | Method                                     |
| -------------- | ---------------------------------- | ------------------------------------------ |
| Manual ingest  | Load CSV → bronze                  | `just db-seed` (deterministic seed subset) |
| Manual trigger | Training pipeline                  | `just train`                               |
| Model promoted | Push promoted Bento image          | `just deploy-bento` (reads `@champion`)    |
| Force redeploy | Rebuild + push regardless of cache | `just deploy-bento --force`                |

## Pipelines

- `ingest.py` — validate raw ATP CSV → bronze
- `seed.py` — the deterministic minimal seed from `data/raw/2026.csv`
- `etl.py` — bronze → silver → gold: player_matches + rolling_features → match_features, plus feature enrichment and sanitization.
- `pipeline.py` — training runner: features → tune 3 models → pick best → train final → evaluate → promote

## Serving & Inference

### Deployment

| Service    | Image source                | Host port | Healthcheck                                 |
| ---------- | --------------------------- | --------- | ------------------------------------------- |
| `postgres` | `postgres:18.4`             | 6543      | `pg_isready` (database readiness)           |
| `bento`    | `swang62/tennis-ml:latest`  | none      | authenticated `SELECT 1` against PostgreSQL |
| `web`      | `swang62/tennis-web:latest` | 8187      | `wget` of the SPA root inside the container |

### Standalone Compose deployment

The stack runs from published images only; the only host requirement is Docker:

```bash
cp .env.example .env   # or export the vars below
docker compose pull
docker compose up -d
```

Required env vars (compose reads them from `.env` or the shell):

| Var                 | Used by         | Purpose                                                      |
| ------------------- | --------------- | ------------------------------------------------------------ |
| `POSTGRES_PASSWORD` | postgres, bento | PostgreSQL password; bento's `DATABASE_URL` derives from it  |
| `DRIFT_API_KEY`     | web             | Authenticates the `/api/internal/*` nginx operational routes |

### Input schema for inference

The `/predict_from_ids` endpoint accepts a JSON object with the following fields:

| Input         | Required | Default | Valid values                                       |
| ------------- | -------- | ------- | -------------------------------------------------- |
| `player_id`   | yes      | —       | non-empty str                                      |
| `opponent_id` | yes      | —       | non-empty str                                      |
| `surface`     | yes      | —       | `clay` / `grass` / `hard` / `carpet`               |
| `tournament`  | no       | 0       | `grand_slam` / `masters` / `atp_500` / `atp_250`   |
| `round`       | no       | 0       | `r128` / `r64` / `r32` / `r16` / `qf` / `sf` / `f` |
| `as_of_date`  | no       | today   | `datetime.date`                                    |

## Extra Notes

- **Canonicalization** — balanced symmetric features, the lower lexicographic player id becomes the `player_*` side
- **Rolling form lookup** — live inference reads each player's newest snapshot strictly before `as_of_date`
- **Cold-start imputation** — missing players use the materialized `gold.tour_averages` singleton (pre-computed full-pool defaults + weighted tour benchmarks), never on-demand aggregates.
- **Bento image data sources** — Bento loads 4 artifacts from the MLflow registry pinned to the champion's exact lineage tags (no base aliases), ensemble model uses promoted `[p_linear, p_gbdt, p_nn]` → `p_win`. Production serving reads PostgreSQL live through sidecar.
