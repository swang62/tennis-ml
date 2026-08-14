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
  db/            — PostgreSQL related functions
  evaluate/      — Evaluate model performance
  features/      — Feature column definitions
  flows/         — ETL Prefect flow + pipelines
  models/        — Player similarity and neural networks
  serving/       — BentoML service
web/             — React + TanStack dashboard (Vite, local dev, HMR)
```

## Quick Start

```bash
# 1. Full local dev setup (deps + k3d cluster for Prefect + PostgreSQL init)
just setup

# 2. Seed the deterministic minimal match set into PostgreSQL bronze
just db-seed

# 3. Generate silver/gold tables
just db-etl

# 4. Local dev: Bento API (:3000) + Vite dashboard (:5173)
just dev

# 5. Optional use serviceman to run worker as daemon
serviceman <start|restart> tennis-prefect-worker
serviceman logs tennis-prefect-worker
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

| Event          | Action                             | Method                                  |
| -------------- | ---------------------------------- | --------------------------------------- |
| Manual ingest  | Load CSV → bronze → gold           | `just db-seed && just db-etl`           |
| Manual trigger | Training pipeline                  | `just train`                            |
| Model promoted | Push promoted Bento image          | `just deploy-bento` (reads `@champion`) |
| Force redeploy | Rebuild + push regardless of cache | `just deploy-bento --force`             |

## Pipelines

- `seed.py` — seed either miniset or full dataset into bronze, rankings + enrichment
- `etl.py` — bronze → silver → gold: player_matches + rolling_features → match_features.
- `pipeline.py` — training runner: features → tune 3 model categories → pick best → train final → evaluate → promote

## Serving & Inference

### Deployment

| Service    | Image source                  | Host port | Healthcheck                                 |
| ---------- | ----------------------------- | --------- | ------------------------------------------- |
| `postgres` | `postgres:18.4`               | 6543      | `pg_isready` (database readiness)           |
| `bento`    | `swang62/tennis-bento:latest` | none      | authenticated `SELECT 1` against PostgreSQL |
| `web`      | `swang62/tennis-web:latest`   | 8187      | `wget` of the SPA root inside the container |

### Standalone Compose deployment

The stack runs from published images only; the only host requirement is Docker. For optional SSL, you will need to generate CA keys in `infra/postgres/tls/`.

```bash
cp .env.example .env
docker compose up -d
```

Required env vars (compose reads them from `.env` or the shell):

| Var                 | Used by         | Purpose                                                      |
| ------------------- | --------------- | ------------------------------------------------------------ |
| `POSTGRES_PASSWORD` | postgres, bento | PostgreSQL password; bento's `DATABASE_URL` derives from it  |
| `BENTO_API_KEY`     | web             | Authenticates the `/api/internal/*` nginx operational routes |

### Inference schema

The `/predict_from_ids` endpoint accepts a JSON object with the following fields:

| Input         | Required | Default | Valid values                                       |
| ------------- | -------- | ------- | -------------------------------------------------- |
| `player_id`   | yes      | —       | non-empty str                                      |
| `opponent_id` | yes      | —       | non-empty str                                      |
| `surface`     | yes      | —       | `clay` / `grass` / `hard` / `carpet`               |
| `tournament`  | no       | 0       | `grand_slam` / `masters` / `atp_500` / `atp_250`   |
| `round`       | no       | 0       | `r128` / `r64` / `r32` / `r16` / `qf` / `sf` / `f` |
| `as_of_date`  | no       | today   | `datetime.date`                                    |

## Dependency Groups

The project uses uv's nested dependency groups. The `inference` group is the strict runtime subset:

```bash
# Full local development (all notebooks, ETL, linting, tests)
uv sync

# Inference-only — the exact packages in the production Bento image
uv sync --group inference
```

## Extra Notes

- **Canonicalization** — balanced symmetric features, the lower lexicographic player id becomes the `player_*` side
- **Rolling form lookup** — live inference reads each player's newest snapshot strictly before `as_of_date`
- **Cold-start imputation** — missing players use the materialized `gold.tour_averages` singleton (pre-computed full-pool defaults + weighted tour benchmarks), never on-demand aggregates.
- **Bento image data sources** — Bento loads native sklearn/XGBoost/LightGBM models materialized at deploy time from the champion's exact lineage tags; no MLflow at serving time. NN is ONNX Runtime. Bio embeddings are compressed NumPy `.npz`, not Parquet. Production serving reads PostgreSQL live through sidecar.
