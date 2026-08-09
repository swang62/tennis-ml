# tennis-ml — AGENTS.md

Commands run via `just` (not make). `uv` is the package manager (not pip/poetry).

## Overall approach

End-to-end MLOps pipeline for tennis match prediction. Data flows CSV → PostgreSQL bronze → dbt silver (player-perspective expansion + post-match rolling snapshots; preprocessing: rank 0 → NULL, non-draw rounds → ordinal 0) → dbt gold (canonical match training rows) → Papermill notebooks (Optuna tuning across 3 model classes) → MLflow registry (exact lineage tags, no stages; single `@champion` alias) → BentoML serving via Docker Hub + Docker Compose.

**Model strategy:** three model classes (linear, GBDT, neural net) compete independently via Optuna, then a logistic-regression meta-model stacks their probability outputs. Architecturally designed for ~80k match samples.

**Separation of concerns:**
- **Training** is standalone (no Prefect) — `pipeline.py` runs all notebooks in order.
- **ETL** is a Prefect flow that runs `dbt build` (one command, dependency-ordered). All promoted models are registered in MLFlow.
- **Deploy** is a separate Prefect flow gated on the latest promoted model — it only redeploys when the `@champion` alias or serving artifacts actually changed (or when run with `--force`).

## Directory layout

```
infra/           — k3d config, static K8s manifests, PostgreSQL init SQL
notebooks/       — EDA + parameterized Papermill notebooks (00–05)
src/
  features/      — shared feature column definitions + PostgreSQL-backed inference builder
  flows/         — ETL (Prefect), standalone training runner, deploy flow
  models/        — Player similarity index (FAISS), NN architecture
  serving/       — BentoML service (model-only — no feature derivation)
  db/            — PostgreSQL client + DuckDB training snapshot
web/             — React + TanStack dashboard (Vite, local dev, HMR)
dbt/             — silver→gold SQL models + tests (bronze is the PostgreSQL source; the feature single source of truth)
```

## Big gotchas & unique aspects

**PostgreSQL is the feature single source of truth.** Per-match player rolling snapshots (`silver.rolling_features`) drive both the canonical training rows (`gold.match_features`) and the on-demand inference feature builder. There is no materialized "latest" table — inference queries the snapshots directly, as-of-dated. Rolling data is always synced to match data with dbt and is always up-to-date.

**Canonical orientation by lower ATP id.** Every match row (training and inference) puts the lexicographically-lower player id on the `player_*` side so `(A, B)` and `(B, A)` produce identical rows. Predicted `p_win` is P(canonical player wins) exactly as the row is sent.

**Bento has two endpoints — model-only and ids-only.** `/predict` takes a finalized feature row + two ids (model-only, upstream-built). `/predict_from_ids` takes the minimal human inputs (two ids + surface + optional tournament/round/date) and calls `build_inference_features` internally against the live PostgreSQL gold tables; training data is pulled separately into a local DuckDB snapshot (`just db-snapshot`).

**Exact lineage, one alias.** Stages are deprecated and base models carry no aliases — each 02 notebook registers a numbered version and records its exact registered name, version, run ID, and model URI (plus immutable scaler/embedding/feature artifact URIs and content hashes) in the handoffs. Promotion in `05_evaluate` tags the new ensemble version with those exact pins and then assigns `@champion`; deploy resolves `@champion` and reads every base pin from those immutable model-version tags. `@champion` is the only alias in the registry. No `copy_model_version`, no staging registered model — single production environment.

**NN is served via ONNX Runtime, not torch.** The deploy flow exports the pinned `nn_best` PyTorch model to a single-file ONNX at deploy time and packages that into the Bento image. torch is not a serving dependency. The GBDT path may pick CatBoost / XGBoost / LightGBM at Optuna time — the serving image pins all three so whichever wins loads cleanly.

**Ingress is single-entrypoint only.** Host access to cluster services goes through `*.macsteve.lan` only. Caddy routes `*.macsteve.lan` to the k3d load balancer on `localhost:8080`, so do not add host port-forwards or ad hoc tunnels. The local Caddy TLS cert is self-signed. All services work over `https://*.macsteve.lan` — MLflow/Prefect clients when told to skip TLS verification (`MLFLOW_TRACKING_INSECURE_TLS=true`, `PREFECT_API_TLS_INSECURE_SKIP_VERIFY=True`). Inside the cluster, services still use Kubernetes DNS names (`mlflow`, `prefect-server`, etc.). BentoML serving is host-local via Docker Compose at `http://127.0.0.1:3000` — it is not a cluster service.

**Canonical service DNS names.** The `*.macsteve.lan` hostnames (recorded in `infra/manifests/default/ingress.yaml`) map to cluster services:

| Hostname                    | Service                    | Port |
| --------------------------- | -------------------------- | ---- |
| `mlflow.macsteve.lan`       | `mlflow`                   | 5000 |
| `prefect.macsteve.lan`      | `prefect-server`           | 4200 |

DNS/TLS for `*.macsteve.lan` itself is served by the host

**MLflow is the registry of record.** Training runs against the local store or k8s service. Deploy resolves `@champion` (and the exact base pins tagged on it) from whatever `MLFLOW_TRACKING_URI` points at.
