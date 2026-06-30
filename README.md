# tennis-ml

Production-grade MLOps pipeline for tennis match prediction. Prefect, DuckDB, MLflow, BentoML — the full stack.

## Stack

| Layer | Tool |
|---|---|
| Orchestration | Prefect (retries, ETL triggers) |
| Experiment tracking | MLflow (model registry, trial comparison) |
| Model serving | BentoML |
| Data warehouse | DuckDB (embedded) |
| Development | Jupyter + Papermill |

## Project Structure

```
infra/           — k3d config, static K8s manifests, DuckDB init SQL
notebooks/       — EDA + parameterized Papermill notebooks
src/
  features/      — Feature column definitions (shared)
  flows/         — Prefect pipelines (training, ETL, pipeline, ingest)
  models/        — Player similarity index (FAISS)
  serving/       — BentoML service
  dashboard/     — Streamlit dashboard
  db/            — DuckDB client
```

## Quick Start

```bash
# 1. Full local dev setup (deps + k3d cluster + DuckDB init)
just setup
```

## Data Flow

```
        CSV match data → ingest.py (validate + insert)
                      ↓
            ┌──────────────────────┐
            │ DuckDB bronze        │ ← persistent DuckDB file
            │ (match_events)       │
            └──────────┬───────────┘
                       │ Prefect etl_flow
                       ↓
            ┌──────────────────────┐
            │ DuckDB gold          │ ← derived features
            │ (match_features)     │
            └──────────┬───────────┘
                       │
                       ↓
         Papermill notebooks (training, evaluation)
                       ↓
         MLflow registry → BentoML serving
```

## Trigger Model

| Event | Action | Method |
|---|---|---|
| Manual ingest | Load CSV → bronze | `just db-ingest` |
| Manual trigger | Training pipeline | `just pipeline` |
| Model promoted | BentoML rebuild | Prefect task + `just bento-build` |

## Pipelines

- `etl_flow` — bronze → gold transforms (DuckDB SQL), player profile enrichment
- `training_flow` — on demand: features → tune 3 models → pick best → train final → evaluate → promote

## Access

All services are exposed via Traefik ingress on k3d's loadbalancer (port 8080).

| Service | URL |
|---|---|
| MLflow | `mlflow.macsteve.lan` 
| Prefect | `prefect.macsteve.lan` 
| BentoML | `bento.macsteve.lan` 
| Dashboard | `dashboard.macsteve.lan` 
