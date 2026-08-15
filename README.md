# tennis-ml

Tennis match predictions end to end. I made this repo with the goal of learning modern MLOps, deploying a full stack pipeline with automated ingestion, training, evaluation, promotion, deployment, and a nice UI to boot. I love tennis, and so I built a simple and fun way to look up stats and match predictions, and avoiding the ATP website as much as possible (which is a true eyesore).

Use this site to predict the next Wimbledon match betwen your two favorite players, and maybe gain an edge over your local bookie while you're at it ;)

## Features

- **Weekly ATP scrape** — a Prefect flow pulls weekly rankings from ATP Tour site through a stealth browser. Self-healing backfills.
- **Prefect automation** — automatic ingest/ETL triggers with drift detection to track model performance and evolving feature distributions
- **dbt warehouse** — bronze (validated matches + rankings) → silver (player-perspective matches + post-match rolling snapshots) → gold (canonical match rows, player profiles, tour averages)
- **Deterministic training** — Papermill notebooks tune linear, GBDT, and NN model families with Optuna, ensemble logistic regression as final model, and gated promotion.
- **Exact lineage, one alias** — full lineage tracking in MLFlow, name, version, run, and artifact tags; final  `@champion` alias in production
- **Drift monitoring** — Evidently PSI on feature and prediction distributions, current-window performance vs. the champion's metrics, and retraining recommendations.

## Stack

| Layer               | Tool                                      |
| ------------------- | ----------------------------------------- |
| Orchestration       | Prefect (cron, retries, ETL triggers)     |
| Experiment tracking | MLflow (model registry, trial comparison) |
| Model serving       | BentoML, Nginx, TanStack React Vite       |
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

## Web App

React 19 + TypeScript dashboard on Vite, Tailwind CSS 4, and the TanStack suite, with ECharts for visualizations. All model endpoints are proxied through at `/api`

- **Player directory search** — MiniSearch fuzzy search over the full player list
- **Player similarity** — find similar-playstyle players (FAISS) built with bio embeddings and service/return stats dims
- **Head-to-head** — cumulative charts and the ensemble's win probability per model
- **SEO** — canonical / OG / JSON-LD tags, robots.txt and sitemap

## Quick Start for Local Development

Only requirement is a free account on [Dagshub](https://dagshub.com), [Docker](https://www.docker.com/), and [Justfile](https://just.systems/).

```bash
# 1. Full local dev setup (deps + k3d cluster for Prefect + PostgreSQL init)
just setup

# 2. Apply schema migrations, then seed the matches into PostgreSQL bronze
just db-migrate
just db-seed

# 3. Generate silver/gold tables
just db-etl

# 4. Local dev: Bento API (:3000) + Vite dashboard (:5173)
just dev

# 5. Optional use serviceman to run worker as daemon on MacOS
serviceman <start|restart> tennis-prefect-worker
serviceman logs tennis-prefect-worker
```

## Data Architecture

```
 Weekly ATP scrape (rankings) → seed.py (validated PostgreSQL)
                      ↓
            ┌──────────────────────┐
            │        BRONZE        │
            │  (match_events)      │
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
Training (features, tuning, evaluation, promotion)
                       ↓
                MLflow registry
                       ↓
          BentoML production endpoints
                       ↓
          Drift monitor (Evidently, weekly)
```

## Pipelines / Flows

- `scrape.py` — weekly ATP rankings backfill. Fetches every missing Monday from the ATP Tour site via stealth-Chromium
- `seed.py` — seed either miniset or full dataset into bronze, rankings + Wikipedia enrichment
- `etl.py` —  bronze → silver → gold: player_matches + rolling_features → match_features + player_profiles
- `pipeline.py` — duckdb snapshot of all gold features → training → feature engineering → tune 3 model categories → pick best → train final → evaluate → promote
- `drift.py` — Evidently PSI on current vs. reference windows scored through the production Bento, performance vs. the champion's pinned metrics
- `deploy.py` — build the champion's Bento image (NN exported to ONNX) and push to Docker Hub

## Production Serving & Inference

### Docker Overview

| Service    | Image source                  | Host port | Healthcheck                                 |
| ---------- | ----------------------------- | --------- | ------------------------------------------- |
| `postgres` | `postgres:18.4`               | 6543      | `pg_isready` (database readiness)           |
| `bento`    | `swang62/tennis-bento:latest` | none      | authenticated `SELECT 1` against PostgreSQL |
| `web`      | `swang62/tennis-web:latest`   | 8187      | `wget` of the SPA root inside the container |

### Standalone Compose deployment

The stack runs from published images only; the only host requirement is Docker. For optional SSL, you can generate SSL certs in `infra/postgres/tls/` and upload them to your server.

```bash
cp .env.example .env
docker compose up -d
```

Required env vars:

| Var                 | Purpose                                                      |
| ------------------- | ------------------------------------------------------------ |
| `POSTGRES_PASSWORD` | PostgreSQL password; bento's `DATABASE_URL` derives from it  |
| `BENTO_API_KEY`     | Authenticates the `/api/model_info` and `/api/predict_from_ids_bulk` routes |

### Inference schema

The `/predict_from_ids` endpoint accepts a JSON object with the following fields:

| Input         | Required | Default | Valid values                                       |
| ------------- | -------- | ------- | -------------------------------------------------- |
| `player_id`   | yes      | —       | non-empty str                                      |
| `opponent_id` | yes      | —       | non-empty str                                      |
| `surface`     | yes      | —       | `clay` / `grass` / `hard` / `carpet`               |
| `tournament`  | no       | —       | `grand_slam` / `masters` / `atp_500` / `atp_250`   |
| `round`       | no       | —       | `r128` / `r64` / `r32` / `r16` / `qf` / `sf` / `f` |
| `as_of_date`  | no       | today   | `datetime.date`                                    |

## Dependency Groups

The project uses uv's nested dependency groups. The `inference` group is the strict runtime subset:

```bash
# Full local development (all notebooks, ETL, linting, tests)
uv sync

# Inference-only — the exact packages in the production Bento image
uv sync --group inference
```

## Acknowledgements

Big shoutout to [TennisMyLife](https://stats.tennismylife.org/) for providing me with the inspiration.

- [Jeff Sackmann](https://github.com/JeffSackmann) — decades of match-level results, despite removing them from GitHub :(
- [Prefect](https://www.prefect.io/) — workflow orchestration for the weekly scrape, ETL, and drift monitoring
- [MLflow](https://mlflow.org/) — experiment tracking and the model registry behind the `@champion` alias
- [BentoML](https://www.bentoml.com/) — model serving

## License

MIT — see [LICENSE](LICENSE).
