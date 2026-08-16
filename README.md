# tennis-ml

End-to-end tennis match prediction with automated ingestion, feature building, model training, evaluation, promotion, deployment, and a web dashboard.

## Features

- **Weekly ATP scrape** — Prefect rankings ingestion with self-healing backfills.
- **Warehouse pipeline** — PostgreSQL bronze → dbt silver → dbt gold.
- **Model training** — Optuna tunes linear, GBDT, and neural-network models before an OOF-trained logistic stacker combines them.
- **Reproducible deployment** — MLflow lineage, a single `@champion` alias, and BentoML serving.
- **Monitoring** — Evidently drift checks and current-window performance checks.

## Stack

| Layer               | Tool                                      |
| ------------------- | ----------------------------------------- |
| Orchestration       | Prefect (cron, retries, ETL triggers)     |
| Experiment tracking | MLflow (model registry, trial comparison) |
| Model serving       | BentoML, Docker Compose                  |
| Web app             | React, TypeScript, Vite, TanStack, ECharts |
| Data warehouse      | PostgreSQL                                |
| Development         | Jupyter + Papermill                       |


## Web App

React 19 + TypeScript dashboard built with Vite, Tailwind CSS 4, TanStack, and ECharts.

- **Player directory** — MiniSearch fuzzy search over the player list.
- **Player similarity** — FAISS search using bio embeddings and service/return statistics.
- **Head-to-head** — cumulative charts and per-model win probabilities.
- **SEO** — canonical, Open Graph, JSON-LD, robots, and sitemap metadata.

## Quick Start for Local Development

Requirements: [DagsHub](https://dagshub.com), [Docker](https://www.docker.com/), [Just](https://just.systems/), and `uv`.

```bash
# 1. Install dependencies and create the local cluster
just cluster-setup

# 2. Apply migrations and seed PostgreSQL bronze
just migrate && just seed

# 3. Build silver and gold
just etl

# 4. Start local Bento and Vite services
just dev

# 5. Optional use serviceman to run worker as daemon on MacOS
serviceman <start|restart> tennis-prefect-worker
serviceman logs tennis-prefect-worker
```

## Data Architecture

```
Rankings and match data → PostgreSQL bronze
                         → dbt silver: player-perspective matches and rolling features
                         → dbt gold: training rows, player profiles, and tour averages
                         → training and evaluation
                         → MLflow `@champion`
                         → BentoML serving
                         → weekly drift monitoring
```

## Modeling Techniques

Training uses player-perspective rows: each physical match produces both orientations with complementary binary labels. Features are built from information available before the match.

### Validation and tuning

- **Five time-forward folds** — validation uses later date bands while training uses strictly earlier matches.
- **Match-safe grouping** — both orientations of a physical match stay in the same fold, preventing leakage.
- **Stratification** — every validation fold preserves the one-positive/one-negative directional label balance.
- **Optuna tuning** — each model family is tuned independently against validation log loss.
- **Early stopping** — GBDT models stop when validation loss stops improving; neural networks use validation BCE with patience and pruning.

### Base models

- **Linear family:** logistic regression, SVM, and Gaussian Naive Bayes.
- **Gradient boosting:** XGBoost and LightGBM, with native early stopping.
- **Neural network:** a tabular/bio-feature MLP trained with Adam and binary cross-entropy with logits (`BCEWithLogitsLoss`).

Log loss is binary cross-entropy evaluated on predicted probabilities. The neural network uses the numerically stable logits form of the same objective.

### OOF ensemble

Each selected base model produces out-of-fold (OOF) predictions for training matches and separate predictions for the untouched test set. OOF predictions are converted to symmetric match evidence and used to fit a zero-intercept logistic-regression stacker.

This prevents the ensemble from training on predictions made by models that saw those rows during fitting. Base models are ranked by OOF ROC-AUC; the final ensemble is judged on the held-out test set.

## Evaluation and Promotion

Candidate and production models are compared on held-out test predictions using:

- **Log loss** — primary probability-quality metric.
- **ROC-AUC** — ranking quality.
- **Brier score** — probability calibration.
- **Accuracy** — thresholded classification quality.

A candidate is promoted when its test log loss is strictly lower than the incumbent's and its ROC-AUC does not fall by more than the configured tolerance. The first candidate is promoted automatically. `@champion` stores the exact base-model versions and artifact lineage used by the ensemble.

## Drift Monitoring

The weekly drift flow scores new production data through the champion Bento and compares it with a size-matched historical reference window. It checks:

- PSI for monitored input features and prediction probabilities.
- The share of features with significant PSI.
- Current-window ROC-AUC and calibration against the champion's pinned metrics when enough matches are available.

PSI, calibration, and performance use configured thresholds and minimum sample sizes. Crossing a feature or prediction PSI threshold, losing calibration, or dropping in ROC-AUC produces a drift report and retraining recommendation. Drift monitoring does not silently replace the champion.

## Pipelines / Flows

- `scrape.py` — weekly ATP rankings backfill.
- `seed.py` — load matches and rankings into bronze, with optional enrichment.
- `etl.py` — build bronze → silver → gold with dbt.
- `pipeline.py` — snapshot data, train models, build the ensemble, evaluate, and promote.
- `drift.py` — compare current production data and performance with the champion.
- `deploy.py` — materialize the champion's serving artifacts and publish the Bento image.

## Production Serving & Inference

### Docker Overview

| Service    | Image source                  | Host port | Purpose                  |
| ---------- | ----------------------------- | --------- | ------------------------ |
| `postgres` | `postgres:18.4`               | 6543      | Feature and serving data |
| `bento`    | `swang62/tennis-bento:latest` | internal  | Model API                |
| `web`      | `swang62/tennis-web:latest`   | 8187      | Dashboard                |

### Standalone Compose deployment

The stack runs from published images. The only host requirement is Docker.

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

The `inference` group is the strict runtime subset used by the production Bento:

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
