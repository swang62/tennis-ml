# tennis-ml — AGENTS.md

Commands run via `just` (not make). `uv` is the package manager (not pip/poetry).

## Overall approach

End-to-end MLOps pipeline for tennis match prediction.

Data flow: CSV → PostgreSQL bronze → dbt silver (player-perspective matches + post-match rolling snapshots) → dbt gold (canonical training rows) → Papermill notebooks (Optuna tuning across all model classes) → MLflow registry → BentoML serving → Web dashboard.

**Model strategy:** the three model classes compete independently via Optuna, then a logistic-regression meta-model stacks their probability outputs. Architected for ~200k match samples.

**Separation of concerns:**
- **Training** is standalone — `just train` runs all notebooks in order.
- **ETL** is a Prefect flow running `dbt build` (one command, dependency-ordered).
- **Deploy** is gated on the latest promoted model — redeploys only when `@champion` or serving artifacts improve the model (or with `--force`).

**Just recipes invoke project `.py` files directly** via `uv run python …`; `*args` recipes pass CLI args through unchanged. Do not add console scripts or generic CLI wrappers.

**The host Prefect worker is a serviceman user agent.** Its entrypoint loads the root `.env`, registers deployment, and connects to the pool.

**Rankings browser state is persistent.** The Prefect flow uses one headed browser profile and one page per run, navigating across weeks. Retain the profile so Cloudflare clearance cookies survive later runs.

## Directory layout

```
artifacts/       — drift reports, model logs, notebook outputs
data/            — raw rankings/matches, processed snapshots, identity map, seed data
dbt/             — silver→gold SQL models + tests (bronze is the PostgreSQL source; the feature single source of truth)
infra/           — k3d config, static K8s manifests, PostgreSQL init SQL, Prefect worker
notebooks/
  eda/           — exploratory analysis notebooks
  parameters/    — parameterized Papermill training notebooks (00–04)
scripts/         — dev, test, and web-check helpers (not pipeline code)
src/
  db/            — PostgreSQL client, DuckDB training snapshot, seeding/ingestion
  evaluate/      — promotion gating, model symmetry checks
  features/      — shared feature definitions, inference builder, tour averages
  flows/         — ETL (Prefect), standalone training runner, deploy flow
  models/        — player similarity index, NN architecture
  serving/       — BentoML service (model-only — no feature derivation)
tests/           — self-contained pytest suite (no live DB)
web/             — React + TanStack dashboard (pages, lib, components)
```

## Big gotchas

**PostgreSQL is the feature single source of truth.** Per-match rolling snapshots drive the training rows and the as-of-dated inference builder. Cold-start fallbacks come from the tour-averages singleton (always one row, full-pool defaults + weighted tour benchmarks), never on-demand aggregates. All data is synced with dbt.

**Exact lineage, one alias.** Base models carry no aliases — each registers a numbered version and records exact name, version, run ID, model URI, and immutable artifact URIs/hashes in the handoffs. Promotion tags the ensemble with those pins, then assigns `@champion`; deploy resolves `@champion` and reads every pin from those tags.

**Artifact boundary.** Champion pins cover only artifacts that affect match probabilities. After an operator refreshes the DuckDB snapshot for the selected `DATABASE_URL`, deploy rebuilds the directory/MiniSearch payload and FAISS similarity assets from that snapshot; it consumes the cluster assignments fitted by `just train`. These navigation artifacts are never MLflow/champion lineage and are packaged only into the web/Bento outputs.

**NN servers via ONNX Runtime, not torch.** Deploy exports the pinned PyTorch model to single-file ONNX at deploy time; torch isn't a serving dependency. GBDT may pick XGBoost or LightGBM at Optuna time — the image pins both so whichever wins loads.

**No ingress — a single node port.** Host access is one port mapped to a NodePort service. No ingress objects, port-forwards, or ad hoc tunnels. No TLS. Inside the cluster, services use Kubernetes DNS names. MLflow is DagsHub-hosted. BentoML is via Docker Compose — not a cluster service.

**Ranking identity map** — a reviewed CSV maps ranking-source player id to canonical player id; the name field is audit-only, never a match key. Ingestion validates the map before any write; unmapped rows are skipped and reported, auto name-matched with normalization.

**Tests are self-contained — no live data, no external calls, ever.** Use `just lint` for all lint/typechecks. Tests must never open DB connections or depend on external state — the suite must pass with no PostgreSQL server and no pre-built tables. Tests must also never make external API or network calls, including to MLflow, DagsHub, or Prefect. Any such behavior is asserted hermetically via mocks/fakes or local fixtures (e.g. an in-memory DuckDB fixture, mocked MLflow/Prefect clients).

## Data manipulation

- **Symmetric, order-preserving** — the ensemble satisfies `p_win(a, b) = 1 - p_win(b, a)`; the first-supplied id is the player side. Endpoints report ids in the order supplied — no sorting or canonicalization.
- **Rolling form lookup** — live inference reads each player's newest snapshot strictly before `as_of_date`.
- **Bento image data sources** — Bento loads native models materialized at deploy time from the champion's lineage tags; no MLflow at serving time. NN is ONNX Runtime. Bio embeddings are compressed NumPy, not Parquet.
