# tennis-ml — AGENTS.md

Commands run via `just` (not make). `uv` is the package manager (not pip/poetry).

## Quick reference

| Action | Command |
|---|---|
| Install deps | `uv sync` |
| Full local setup | `just setup` |
| Lint | `ruff check src/` |
| Format | `ruff format src/` |
| Lint + format all | `uv run pre-commit run --all-files` |
| Ingest CSV | `just db-ingest` |
| Run ETL | `just db-etl` |
| Train pipeline | `just pipeline` |
| Run training only | `just train` |
| Reset DB | `just db-reset` |
| Dashboard (local) | `just dashboard-local` |
| BentoML dev server | `just bento-local` |
| BentoML build + deploy | `just bento-build` |
| Teardown | `just destroy` |

## Key facts

- **DuckDB** is the data warehouse — embedded, file-based local database.
- **Prefect** orchestrates ETL and training flows.
- **Papermill** all runs via parameterized Jupyter notebooks under, logging all artifiacts to MLflow.
- **Model serving** is BentoML, pulls models/features from MLflow registry, serves via FastAPI.
- **Container runtime** K3d/K3s single-node kubernetes cluster to handle all deployments
- **Visualization** is via custom Streamlit dashboard, also deployed as docker image

## Architecture overview

**Data flow:** CSV → bronze.match_events (ingest) → gold.match_features (DuckDB SQL rolling transforms in ETL) → training notebooks → MLflow model registry → BentoML serving.

**Model strategy:** Three model classes compete independently via Optuna (linear, GBDT, neural net). The best from each is stacked via a simple logistic regression meta-model on their probability outputs. Architecturally designed for 80k match samples.

**Two-tower match sequence predictor:** Each player tower merges a sequence pathway (LSTM/GRU/TCN encoding match history) with a static pathway (rank, age, style, bio embeddings). The pairwise head uses `concat(a, b, a-b, a*b)` followed by a small MLP classifier.

**Player similarity (src/models/similarity.py):** Separate unsupervised FAISS index built from player bios via fastembed + one-hot encoded categoricals. Not part of match prediction — purely a content-based retrieval system.

## Prefect flows (src/flows/)

- `ingest.py` — validate and insert CSV match data into DuckDB bronze layer
- `etl.py` — bronze-to-gold transforms (DuckDB SQL), player profile enrichment
- `training.py` — Optuna hyperparameter search across model classes, train final model, evaluate, promote
- `pipeline.py` — orchestrates ETL + training as a single Prefect flow
