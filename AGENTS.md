# tennis-ml — AGENTS.md

Commands run via `just` (not make). `uv` is the package manager (not pip/poetry).

## Overall approach

End-to-end MLOps pipeline for tennis match prediction. Data flows CSV → DuckDB bronze → dbt silver (player-perspective expansion; preprocessing: rank 0 → NULL, non-draw rounds → ordinal 0) → dbt gold (player-centric rolling snapshots → canonical match training rows) → Papermill notebooks (Optuna tuning across 3 model classes) → MLflow registry (alias-based, no stages) → BentoML serving on a local k3d cluster.

**Model strategy:** three model classes (linear, GBDT, neural net) compete independently via Optuna, then a logistic-regression meta-model stacks their probability outputs. Architecturally designed for ~80k match samples.

**Separation of concerns:**
- **Training** is standalone (no Prefect) — `pipeline.py` runs all notebooks in order.
- **ETL** is a Prefect flow that runs `dbt build` (one command, dependency-ordered). All promoted models are registered in MLFlow.
- **Deploy** is a separate Prefect flow gated on the latest promoted model — it only redeploys when the `@champion` alias or serving artifacts actually changed (or when run with `--force`).

## Directory layout

```
infra/           — k3d config, static K8s manifests, DuckDB init SQL
notebooks/       — EDA + parameterized Papermill notebooks (00–05)
src/
  features/      — shared feature column definitions + DuckDB-backed inference builder
  flows/         — ETL (Prefect), standalone training runner, deploy flow
  models/        — Player similarity index (FAISS), NN architecture
  serving/       — BentoML service (model-only — no feature derivation)
  db/            — DuckDB client
web/             — React + TanStack dashboard (Vite, local dev, HMR)
dbt/             — silver→gold SQL models + tests (bronze is the DuckDB source; the feature single source of truth)
```

## Big gotchas & unique aspects

**DuckDB is the feature single source of truth.** Per-match player rolling snapshots (`gold.rolling_features`) drive both the canonical training rows (`gold.match_features`) and the on-demand inference feature builder. There is no materialized "latest" table — inference queries the snapshots directly, as-of-dated. Rolling data is always synced to match data with dbt and is always up-to-date.

**Canonical orientation by lower ATP id.** Every match row (training and inference) puts the lexicographically-lower player id on the `player_*` side so `(A, B)` and `(B, A)` produce identical rows. Predicted `p_win` is P(canonical player wins) exactly as the row is sent.

**Bento has two endpoints — model-only and ids-only.** `/predict` takes a finalized feature row + two ids (model-only, upstream-built). `/predict_from_ids` takes the minimal human inputs (two ids + surface + optional tournament/round/date) and calls `build_inference_features` internally against the DuckDB gold tables snapshotted into the image at deploy time.

**MLflow aliases, not versions or stages.** Stages are deprecated; the registry uses `@best` (each base model) and `@champion` (the ensemble LR head). Promotion in `05_evaluate` sets `@champion` on the version it just registered; deploy resolves via `get_model_version_by_alias`. No `copy_model_version`, no staging registered model — single production environment.

**NN is served via ONNX Runtime, not torch.** The deploy flow exports the pinned `nn_best` PyTorch model to a single-file ONNX at deploy time and packages that into the Bento image. torch is not a serving dependency. The GBDT path may pick CatBoost / XGBoost / LightGBM at Optuna time — the serving image pins all three so whichever wins loads cleanly.

**Ingress is single-entrypoint only.** Host access to cluster services goes through `*.macsteve.lan` only. Caddy routes `*.macsteve.lan` to the k3d load balancer on `localhost:8080`, so do not add host port-forwards or ad hoc tunnels. The local Caddy TLS cert is self-signed. All services work over `https://*.macsteve.lan` — Bento out of the box, and MLflow/Prefect clients when told to skip TLS verification (`MLFLOW_TRACKING_INSECURE_TLS=true`, `PREFECT_API_TLS_INSECURE_SKIP_VERIFY=True`). Plain `http://*.macsteve.lan:8080` (no TLS) also works for any HTTP client. Inside the cluster, services still use Kubernetes DNS names (`mlflow`, `prefect-server`, `bento-serving`, `tennis-ml-registry`, etc.).

**Canonical service DNS names.** The `*.macsteve.lan` hostnames (recorded in `infra/manifests/default/ingress.yaml`) map to cluster services:

| Hostname                    | Service                    | Port |
| --------------------------- | -------------------------- | ---- |
| `bento.macsteve.lan`        | `bento-serving`            | 3000 |
| `mlflow.macsteve.lan`       | `mlflow`                   | 5000 |
| `prefect.macsteve.lan`      | `prefect-server`           | 4200 |
| `registry.macsteve.lan`     | `tennis-ml-registry`       | 5000 |

DNS/TLS for `*.macsteve.lan` itself is served by the host

**Local MLflow is the registry of record.** Training runs against the local store. Deploy resolves `@champion`/`@best` from whatever `MLFLOW_TRACKING_URI` points at.
