# tennis-ml agent map

## Commands and boundaries

- Use `just` recipes, not `make`; use `uv`, not pip or poetry.
- Recipes call project Python files directly with `uv run python`; preserve
  passthrough `*args`. Do not add console-script wrappers.
- Keep training, ETL, and deployment separate: `just train`, the Prefect/dbt
  ETL flow, and champion-gated deployment are distinct paths.
- The Prefect worker is a serviceman user agent. Its entrypoint loads the root
  `.env`, registers deployments, and connects to the pool.
- Keep the headed rankings browser profile and its single page across weekly
  navigation so Cloudflare clearance survives runs.

## System map

CSV -> PostgreSQL bronze -> dbt silver/gold -> Papermill/Optuna -> MLflow
registry -> BentoML -> React dashboard.

- `db/`: PostgreSQL client, ingestion, seeding, and DuckDB snapshots.
- `features/`: feature definitions, as-of inference, and tour fallbacks.
- `models/`: grouped CV, neural network, and similarity index.
- `evaluate/`: calibration, symmetry, metrics, and promotion.
- `flows/`: ETL, training, monitoring, and deployment orchestration.
- `serving/`: model-only Bento service; feature derivation belongs in
  `features/`.
- `dbt/`: the feature source of truth. `web/`: React/TanStack UI.

## Non-negotiable data and deployment rules

- PostgreSQL rolling snapshots are the feature source of truth. Inference
  selects each player's newest snapshot strictly before `as_of_date`.
- Cold starts use the single tour-averages row, never ad-hoc aggregates.
- Preserve player perspective and order: `p(a,b) = 1 - p(b,a)`; never sort or
  canonicalize endpoint ids.
- The reviewed ranking identity map uses source ids to canonical ids; names are
  audit-only. Validate it before writes and report unmapped rows.
- Base models get numbered versions, not aliases. Promotion pins exact model,
  run, URI, and immutable artifact hashes on the ensemble, then assigns only
  `@champion`. Deployment resolves those pins.
- Similarity assets are outside champion lineage: rebuild FAISS after the
  selected `DATABASE_URL` snapshot refresh and package them only in Bento.
- Serving uses ONNX Runtime for neural models, not torch. Bento loads
  materialized native models and never contacts MLflow at runtime.
- Keep the single NodePort topology: no ingress, tunnels, or TLS. MLflow is
  DagsHub-hosted; Bento runs through Docker Compose.
- Every rankings and match scrape/fetch, including dry runs and validation, must
  use the existing persistent CloakBrowser session and its single page. Direct
  `curl`, `requests`, or other HTTP fetches are not an acceptable scrape path.

## Minimalism and testing

- Inspect the affected flow and callers before editing. Fix shared causes once;
  make the smallest correct change and avoid speculative abstractions.
- Tests are allowed only to verify observable behavior, data correctness,
  persistence/error safety, security, or external contracts. Do not add
  regression tests that merely freeze an implementation, constants, call order,
  mock interactions, exact SQL, fake Prefect/MLflow/browser wiring, or synthetic
  pin self-equality. Prefer real or local fixture data and test databases for
  data manipulation; delete tests that only protect an implementation choice.
- Never add `assert CONSTANT == value` merely to freeze a configurable or
  internal constant. Assert the observable behavior it influences; retain a
  literal assertion only for a genuine public/persisted contract.
- Tests are hermetic: no live DB, network, MLflow, DagsHub, Prefect, or
  pre-built tables. Use fakes, mocks at external boundaries, and local fixtures.
- Keep docstrings and comments concise: at most one short paragraph, usually one
  line. Remove essay-like narration; describe only public purpose, contract,
  invariant, or non-obvious rationale. Use comments for why, not what; do not
  add comments that merely repeat constants, settings, parameter values, or
  immediately visible code. Source/config is the authority for values. Do not
  add inline comments to every new line.
- Run `just lint` and the narrowest relevant tests, then the full suite before
  declaring completion. Never commit unless asked.

## Change safety

- Preserve unrelated worktree changes.
- Do not delete user data, files, databases, or persisted artifacts without
  explicit approval.
- Surface uncertainty and verification gaps instead of weakening checks.
