# tennis-ml — AGENTS.md

Commands run via `just` (not make). `uv` is the package manager (not pip/poetry).

## Overall approach

End-to-end MLOps pipeline for tennis match prediction. Data flows CSV → PostgreSQL bronze → dbt silver (player-perspective expansion + post-match rolling snapshots; preprocessing: rank 0 → NULL, non-draw rounds → ordinal 0) → dbt gold (canonical match training rows) → Papermill notebooks (Optuna tuning across 3 model classes) → MLflow registry (exact lineage tags, no stages; single `@champion` alias) → BentoML serving via Docker Hub + Docker Compose.

**Model strategy:** three model classes (linear, GBDT, neural net) compete independently via Optuna, then a logistic-regression meta-model stacks their probability outputs. Architecturally designed for ~80k match samples.

**Separation of concerns:**
- **Training** is standalone (no Prefect) — `just train` runs all notebooks in order.
- **ETL** is a Prefect flow that runs `dbt build` (one command, dependency-ordered). All promoted models are registered in MLFlow.
- **Deploy** is a separate Prefect flow gated on the latest promoted model — it only redeploys when the `@champion` alias or serving artifacts actually changed (or when run with `--force`).

**Just recipes invoke project `.py` files directly.** Recipes run the exact file via `uv run python path/to/file.py` (e.g. `just migrate` → `uv run python src/db/migrate_db.py migrate`); `*args` recipes pass CLI args through unchanged. Do not add console scripts or generic CLI wrappers just to serve the justfile.

**The host Prefect worker is a serviceman user agent.** Its name is `tennis-prefect-worker`; it runs `uv run python infra/prefect/worker.py` from the repository root. That entrypoint loads the root `.env` before registering the rankings deployment and starting `tennis-pool`. Do not add a `just worker` recipe.

**Rankings browser state is persistent.** The Prefect flow uses one headed CloakBrowser profile and one page per run, navigating that page across weeks. Its profile is `~/.local/share/tennis-prefect-worker/cloakbrowser`; retain it so Cloudflare clearance cookies survive later runs.

## Directory layout

```
infra/           — k3d config, static K8s manifests, PostgreSQL init SQL
notebooks/       — EDA + parameterized Papermill notebooks (00–05)
src/
  features/      — shared feature column definitions, inference builder, and tour_averages singleton loader
  flows/         — ETL (Prefect), standalone training runner, deploy flow
  models/        — Player similarity index (FAISS), NN architecture
  serving/       — BentoML service (model-only — no feature derivation)
  db/            — PostgreSQL client + DuckDB training snapshot
web/             — React + TanStack dashboard (Vite, local dev, HMR)
dbt/             — silver→gold SQL models + tests (bronze is the PostgreSQL source; the feature single source of truth)
```

## Big gotchas & unique aspects

**PostgreSQL is the feature single source of truth.** Per-match player rolling snapshots (`silver.rolling_features`) drive both the canonical training rows (`gold.match_features`) and the as-of-dated inference feature builder. Cold-start fallbacks and tour-wide comparisons come from the materialized `gold.tour_averages` singleton (always exactly one row), never on-demand AVG/PERCENTILE queries. Rolling data is always synced to match data with dbt and is always up-to-date.

**Exact lineage, one alias.** Stages are deprecated and base models carry no aliases — each 02 notebook registers a numbered version and records its exact registered name, version, run ID, and model URI (plus immutable scaler/embedding/feature artifact URIs and content hashes) in the handoffs. Promotion in `05_evaluate` tags the new ensemble version with those exact pins and then assigns `@champion`; deploy resolves `@champion` and reads every base pin from those immutable model-version tags. `@champion` is the only alias in the registry. No `copy_model_version`, no staging registered model — single production environment.

**NN is served via ONNX Runtime, not torch.** The deploy flow exports the pinned `nn_best` PyTorch model to a single-file ONNX at deploy time and packages that into the Bento image. torch is not a serving dependency. The GBDT path may pick XGBoost or LightGBM at Optuna time — the serving image pins both so whichever wins loads cleanly.

**No ingress — a single node port.** Host access to the cluster is one port only: the k3d config maps host `localhost:4200` directly to the `prefect-server` NodePort service (cluster nodePort `30420`), so do not add ingress objects, host port-forwards, or ad hoc tunnels. There is no Caddy/TLS layer. Inside the cluster, services still use Kubernetes DNS names (`prefect-server:4200`, etc.). MLflow is DagsHub-hosted. BentoML serving is host-local via Docker Compose at `http://127.0.0.1:3000` — it is not a cluster service.

**MLflow is the registry of record, hosted on DagsHub.** Training logs runs and models to the DagsHub MLflow backend, and deploy resolves `@champion` (and the exact base pins tagged on it) from whatever `MLFLOW_TRACKING_URI` points at.

**Ranking identity map** — `data/ranking_player_map.csv` is the authoritative reviewed mapping from ranking-source player id (`ranking_player_id`, the id in `data/raw/rankings/atp_rankings_*.csv`) to the canonical player id (`player_id`, the ATP_Database id space used by matches and profiles); `ranking_name` is an audit/review field only, never a production match key. Ingestion validates the map (structure, duplicate source ids, conflicting targets, unknown canonical ids) before any write and rejects it otherwise; unmapped top-200 rows are skipped and reported with source id, name, and count, never silently name-matched. `ranking_name_candidates()` in `src/db/ingest.py` is a deterministic normalized-name review aid for maintainers extending the map.

**Tests are self-contained — no live data, ever.** Tests must never read or use `DATABASE_URL`, open database clients/connections (`get_conn`, `psycopg.connect`, `refresh_snapshot` against a real source), call `migrate_db`/`seed`/dbt against a DB, or include the deleted live-db fixture names (`postgres_ready`, `gold_ready`, `seeded_test_db`, `_postgres_reachable`). The suite must pass with `DATABASE_URL` unset or blocked, with no PostgreSQL server, gold tables, or any pre-built external state. External database behavior is asserted hermetically at the boundary: mock `execute_df` where the code under test imports it (string-patched module attributes), or run the same SQL against an in-memory DuckDB fixture (see `test_inference_features.py`). A test that depends on a live database is a contract violation — convert it to a boundary mock, never skip it. `tests/test_no_live_db.py` is a static guard that fails CI if these patterns are reintroduced.

**Official rankings** — ingested ranks are strictly official ATP top-200 values per week; downstream rank/current-rank values are actual official ranks only, never estimated or interpolated.

## Extra Notes

- **Symmetric, order-preserving** — the ensemble satisfies `p_win(player, opponent) = 1 - p_win(opponent, player)`; the first-supplied id is the player side, so `p_win` always means "first-supplied id wins". H2H and all endpoints report ids in the order supplied — no sorting or canonicalization.
- **Rolling form lookup** — live inference reads each player's newest snapshot strictly before `as_of_date`
- **Cold-start imputation** — missing players use the materialized `gold.tour_averages` singleton (pre-computed full-pool defaults + weighted tour benchmarks), never on-demand aggregates.
- **Bento image data sources** — Bento loads native sklearn/XGBoost/LightGBM models materialized at deploy time from the champion's exact lineage tags; no MLflow at serving time. NN is ONNX Runtime. Bio embeddings are compressed NumPy `.npz`, not Parquet. Production serving reads PostgreSQL live through sidecar.
