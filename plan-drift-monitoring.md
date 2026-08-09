# Plan: Production Bento drift check

## Goal

Add a no-argument `just check-drift` recipe that refreshes dbt, resolves the
current `ensemble_lr_model@champion`, verifies the running production Bento
image serves that exact champion, and asks Bento to batch-generate predictions
for every completed match dated after the champion's UTC MLflow creation date.
Compare those predictions with actual outcomes and log aggregate feature,
prediction, calibration, and performance drift results to MLflow.

## Validated operating model

### Champion cutoff

The current promotion flow already provides the desired idempotent cutoff:

- `notebooks/parameters/05_evaluate.ipynb` registers
  `ensemble_lr_model` only inside the successful promotion branch.
- The newly registered version is immediately assigned `@champion`.
- MLflow automatically records that model version's immutable
  `creation_timestamp`.
- Failed training, rejected candidates, and repeated evaluation do not create a
  production version or move `@champion`.
- `deploy-bento` and `check-drift` are downstream, read-only consumers of this
  decision. Neither may edit evaluation runs, registered versions, model tags,
  or aliases.

Therefore the monitoring population is always:

```text
gold.match_features.match_date > UTC-date(champion.creation_timestamp)
```

A new promotion resets the cutoff. Failed or rejected retraining leaves the
existing champion and cumulative monitoring window unchanged. No custom
`data_cutoff_date` is needed.

This is an accepted date-level approximation, not exact training-row lineage.
Before registration, evaluation records the maximum match date used anywhere in
training, tuning, selection, or evaluation and rejects promotion if it is later
than the current UTC date captured immediately before registration. The MLflow
tracking server and evaluator are assumed to share the same UTC date. Matches on
the resulting model-version creation date remain excluded.

### Prediction authority

The drift process must not load MLflow model artifacts, scalers, embeddings, or
ONNX files and must not recreate ensemble inference. The **running production
Bento image** is the only prediction authority. The drift process uses MLflow
only to:

- resolve champion identity and cutoff;
- retrieve promotion baseline metrics;
- store aggregate monitoring history and reports.

Evaluation also uses production Bento as the authority for the incumbent side
of promotion: candidate head + candidate base predictions are compared with
deployed champion head + its frozen base models on the same ordered held-out
match contexts and labels. The incumbent uses its baked feature contract and the
candidate uses its current training contract. Contract hashes are audit/routing
metadata, not promotion criteria.

### Immutable ensemble lineage

Base-model `@best` aliases are removed. They are harmful here because a champion
head must remain paired with the exact bases that generated its stack inputs.

- Base models remain registered as immutable numbered versions under the
  existing registered-model names.
- Base training handoffs record registered name, numeric version, run ID, and
  immutable `runs:/...` model URI; no alias field.
- The candidate stack run records all three exact base pins.
- On promotion, the new `ensemble_lr_model` version receives model-version tags
  for the exact linear, GBDT, and NN names/versions/run URIs before `@champion`
  is assigned.
- Those champion model-version tags are the single source of truth for immutable
  prediction lineage. They also pin every prediction-critical scaler, embedding,
  and feature-metadata artifact by run URI and content hash. Deployment, serving,
  evaluation, and drift read this lineage; none reconstructs it from aliases or
  mutable local files.
- `@champion` is the only model alias used by this model-serving path.
- Deploy resolves bases only from the champion's immutable pins.

### Non-circular deployment identity

Deployment generates one canonical manifest directly from the champion
model-version tags. The manifest contains champion identity and creation time,
exact base-model and auxiliary-artifact pins, the feature-contract version/hash,
and a deterministic `build_input_fingerprint`.

`build_input_fingerprint` hashes canonical serialized champion lineage plus the
content hashes of every source and artifact included in the Bento build. It
explicitly excludes the generated manifest, Bento tag, Docker tag, Docker
digest, timestamps, and deployment-state files. The generated manifest therefore
cannot affect the fingerprint stored inside itself.

The Bento tag and pushed Docker image identity are build outputs known only after
the build. Deployment records them in successful local deployment state keyed by
`(champion version, build_input_fingerprint)`; they are not lineage authorities
and are never copied back into the baked manifest. `/model_info` returns the
baked manifest. Drift may record the current Bento/Docker output identity when
available, but validates model lineage against the manifest derived from the
MLflow champion tags.

### Finalized gold feature contract

`gold.match_features` becomes the single source of truth for model-ready values:

- every `FEATURE_COLS` cell is non-null and finite;
- imputation is date-aware and uses only rolling snapshots strictly before each
  row's match date;
- each player side is imputed before matchup differences are calculated;
- dbt materializes reusable date-keyed feature defaults, including a row for the
  dbt run date;
- inference reads the newest materialized defaults at or before `as_of_date`
  (or the latest available row for a future date) rather than calculating
  averages/percentiles on demand;
- training consumes finalized gold values directly and performs no independent
  `SimpleImputer` fit.

The contract has an explicit version/schema hash. It is included in candidate
lineage, deployment manifests, build-input fingerprints, evaluation reports, and
drift baselines so a serving-preprocessing change cannot look like an unchanged
deployment.

### Fact-checked serving distinction

- Local `bentoml serve src/serving/service.py:TennisPredictor` is not guaranteed
  to represent the champion. It reads local `*:latest` Bento models and mutable
  `data/processed` files that later failed training can overwrite.
- The Docker image built by `just deploy-bento` packages model artifacts for the
  champion resolved at build time. It is the correct inference target.
- Promotion and deployment are separate, so the deployed image may temporarily
  lag MLflow `@champion`. Drift must detect this mismatch and fail with a deploy
  instruction rather than score the wrong model.

## Accepted date limitation

Gold stores `match_date` as a date while MLflow records a timestamp. Bronze has
no ingestion timestamp. Consequently:

- matches on the UTC champion-creation date are excluded because within-day
  training visibility is unknowable;
- later backfills dated on or before the cutoff are not monitored;
- true insertion-time monitoring would require a separate `ingested_at` schema
  change and is out of scope.

## Scope

### In scope

- `just check-drift` with no arguments.
- `dbt build` before analysis so silver rolling snapshots and gold rows are
  current.
- A non-null, finite, model-ready gold contract shared by snapshots, training,
  scalar inference, bulk inference, evaluation, and drift.
- MLflow champion creation timestamp as the sole cutoff.
- Production Docker Bento as the sole prediction generator.
- Full candidate-versus-production evaluation using candidate artifacts for the
  new stack and production Bento for the incumbent stack.
- A production model-identity endpoint and chunked batch-prediction endpoint.
- API-key-protected Nginx routes for those two operational endpoints; Bento
  remains reachable only inside the Compose network.
- Cumulative post-promotion labeled evaluation.
- Feature-distribution, prediction-distribution, calibration, and performance
  measurements using existing SciPy/scikit-learn dependencies.
- Terminal summary and idempotent aggregate MLflow report.
- Separate MLflow experiments for immutable champion baselines and cumulative
  drift checks.
- Measurements, deltas, and sample-sufficiency metadata only; no interpreted
  health/drift/investigate/retraining status and no automatic action.

### Out of scope

- `just drift`, `--since`, or any manual cutoff input.
- Loading or executing MLflow model artifacts in the drift script.
- Local development Bento as a drift target.
- DriftWatch or another monitoring service.
- Prefect scheduling, notifications, or daemons.
- Direct host exposure of Bento port 3000.
- Frontend application access to the operational endpoints.
- Bento request logging or row-level prediction persistence.
- Automatic retraining, promotion, rollback, or deployment.
- Production recommendation thresholds or user-configurable runtime policy.
- Ingestion-time/backfill detection.
- A second train-time or inference-time statistical imputation policy.

### Authorized one-time destructive reset

The user explicitly authorized deleting all current generated/test state because no
production environment has been deployed and both development databases contain
reproducible seeded data. The reset includes:

- the Kubernetes MLflow `mlflow-data` PVC contents (tracking SQLite database,
  registry metadata, runs, and artifacts);
- any repo-local MLflow store if one exists at execution time;
- generated `artifacts/` and `data/processed/` training/deployment outputs;
- project-owned local Bento models/Bentos and exact local Docker image refs;
- the configured Homebrew PostgreSQL development schemas/data;
- the Compose PostgreSQL development volume/data.

The PostgreSQL reset may remove every application schema/table/row, including
bronze source rows, player profiles, silver snapshots, and gold features. These
are recreated from file-backed inputs and the new ETL contract.

Preserve source code, `.env` except for adding the authorized drift key, the
entire true `data/raw/` directory, other file-backed source datasets, dependency
caches/virtual environments, unrelated
Bento/Docker objects, and unrelated Kubernetes/PVC/database resources. Docker
Hub is not deleted; the first clean deploy overwrites its single `latest` tag.

## Data and baseline contract

### Current population

After dbt succeeds, query `gold.match_features` where
`match_date > cutoff_date`, ordered by `(match_date, match_id)`. Send only the
minimal match context required by the production inference builder:

- canonical player/opponent IDs;
- surface, tournament level, round encoding, indoor flag, and match date as
  `as_of_date`;
- never `match_won`.

Bento's bulk inference builder retrieves strictly-prior rolling state, joins the
precomputed dbt defaults for missing player state, constructs finalized
`FEATURE_COLS`, and runs one model batch. It does not compute imputation
statistics. It returns finalized features plus `p_linear`, `p_gbdt`, `p_nn`,
and `p_win`. The drift process uses returned features for distribution checks
and joins predictions in request order to `match_won` and report segments.
`p_win` remains P(the canonical lower-ID `player_*` side wins).

Unknown/`0` surface is a supported explicit input and produces the existing
all-zero surface one-hot representation, so it does not make a completed match
unscoreable.

### Reference population

- Feature drift: gold rows on or before the cutoff, using the same finalized
  `FEATURE_COLS` representation. Under the accepted project assumption, these
  rows represent the population available to the champion pipeline.
- Prediction drift: production Bento scores a deterministic reference sample
  from the pre-cutoff population through the same batch endpoint. This is only
  a probability-distribution baseline, not an unbiased accuracy estimate.
- Performance baseline: retrieve the champion's held-out candidate metrics from
  the MLflow evaluation run that promoted `champion.run_id`. Existing champions
  are resolved read-only by searching the `evaluation` experiment for
  `candidate_run_id=<champion.run_id>` and `promoted=1`. Drift does not add tags
  to the model version or modify the evaluation run.
- Held-out ROC AUC, accuracy, Brier, log loss, and calibration values from the
  clean promotion evaluation are comparative baselines when their calculation
  specification matches the drift report. Any unavailable or incompatible
  baseline delta is reported as `unavailable`, never inferred.

Reference sampling is deterministic, capped at 10,000 rows, stratified by date
and surface where possible, and seeded from champion version. This limits Bento
load and prevents huge reference size from making trivial distribution changes
statistically significant.

### Cached baseline lifecycle

The first `check-drift` with a non-empty post-cutoff population for a distinct
`(champion version, feature-contract hash)` creates one immutable run in
the `drift_baselines` MLflow experiment:

- resolve and record the existing promoted evaluation run and its baseline
  metrics;
- profile pre-cutoff `FEATURE_COLS` into fixed numeric quantile-bin edges/counts
  and discrete value counts;
- ask production Bento to score the deterministic pre-cutoff reference sample;
- store fixed probability-bin counts for `p_win` and each base probability;
- record reference date range, sample count, ordered match-ID hash, feature
  schema hash, statistical configuration, champion identity, and Bento image
  build-input fingerprint used to create the baseline;
- store aggregate JSON profiles only—no match IDs, player IDs, labels, raw
  feature rows, or per-match predictions.

Later checks reuse that completed baseline run without querying or rescoring the
pre-cutoff population. A new champion or feature-contract hash creates a new
baseline. A different validated build-input fingerprint for the same champion
is compared with the existing baseline so serving changes are not normalized
away. Later database backfills do not mutate an existing baseline.

## Reporting policy

Monitoring reports fixed-bin distribution measurements, statistical test
outputs, calibration/performance metrics, baseline deltas, sample counts, and
whether each metric is computable. It does not convert those values into
`healthy`, `drift`, `investigate`, or retraining statuses.

The implementation must freeze and document its formulas, bin construction,
smoothing, correction families, calibration definition, deterministic seeds,
and undefined-data behavior before accepting calculation tests. Production has
no threshold configuration. Unit tests may pass explicit thresholds to pure
calculation helpers when testing boundary behavior, but those thresholds do not
enter `just check-drift` or its MLflow reports.

## Tasks

### [x] Task 1: Finalize and validate gold features in dbt

- **Description**: Move all statistical/default imputation into dbt. Add a gold
  defaults model keyed by available as-of dates, calculated only from strictly
  prior rolling snapshots. Refactor `gold.match_features` to join those defaults,
  source rank, rank points, age, and rolling state strictly before the match date,
  impute each player perspective, and only then calculate canonical matchup
  features. Multiple matches on one date intentionally share the same pre-day
  state. Remove notebook `SimpleImputer` usage and replace inference-time
  aggregate calculations with a lookup of materialized defaults.
- **Files**: `dbt/models/gold/feature_defaults.sql` (new),
  `dbt/models/gold/feature_defaults.yml` (new),
  `dbt/models/gold/match_features.sql`,
  `dbt/models/gold/match_features.yml`, `dbt/dbt_project.yml`,
  `dbt/tests/gold/match_features_no_null_model_features.sql` (new),
  `notebooks/parameters/01_train_test_split.ipynb`, `src/features/columns.py`,
  `src/features/inference.py`, `src/db/snapshot.py`, `src/flows/etl.py`,
  `tests/test_inference_features.py`, `tests/test_e2e_ingest_to_inference.py`,
  `tests/test_snapshot.py`, `tests/test_rolling_contract.py`.
- **Acceptance Criteria**:
  - `feature_defaults` contains deterministic mean/median/default values for
    every historical match date plus the dbt run date, based only on snapshots
    with `snapshot_date < as_of_date`.
  - Median is used for rank/rank-points/streak-like values; mean is used for
    rates, age, years-pro, handedness rate, and other continuous values; indoor
    defaults to `0`; explicit constants cover an empty prior pool.
  - `match_features` imputes side-level ranking, profile, rolling, and context
    values before calculating differences.
  - Gold and scalar/bulk inference use the same strictly-prior date semantics for
    rank, rank points, age, and rolling state; no same-day snapshot contributes.
  - Every `FEATURE_COLS` value is non-null and finite; `dbt build` fails if the
    contract is violated.
  - Similarity-only appended columns may retain their separately documented
    nullability and are not confused with model features.
  - Training split reads finalized values, asserts the contract, and contains no
    `SimpleImputer` or fitted imputation artifact.
  - Snapshot validation rejects NULL/non-finite model features.
  - Scalar inference performs no AVG/PERCENTILE imputation queries. It uses the
    newest defaults row at or before `as_of_date`, falling back to the latest
    available row for future dates.
  - For historical match fixtures, scalar inference output equals the
    corresponding finalized gold `FEATURE_COLS` row.
  - Unknown/`0` surface is accepted by scalar and bulk inference and maps to all
    zero surface indicator columns, matching gold.
- **Guardrails**: No target leakage: current-match outcomes/stats and future
  snapshots cannot contribute to defaults. Keep PostgreSQL/dbt as feature source
  of truth; do not move imputation constants into a second Python policy.

### [x] Task 2: Remove base aliases and freeze ensemble lineage

- **Description**: Stop creating/resolving base `@best` aliases. Persist exact
  numbered base versions and immutable run URIs through each base handoff and
  candidate stack run. When evaluation promotes a candidate, tag the new
  champion version with those exact base and prediction-critical auxiliary
  artifact pins. Update deployment to generate its canonical manifest and
  non-circular build-input fingerprint from those tags.
- **Files**: `notebooks/parameters/02_tune_linear.ipynb`,
  `notebooks/parameters/02_tune_gbdt.ipynb`,
  `notebooks/parameters/02_tune_nn.ipynb`,
  `notebooks/parameters/00_embeddings.ipynb`,
  `notebooks/parameters/03_ensemble_split.ipynb`,
  `notebooks/parameters/04_ensemble_stack.ipynb`,
  `notebooks/parameters/05_evaluate.ipynb`, `src/flows/deploy.py`,
  `src/constants.py`, `AGENTS.md`, `README.md`, `tests/test_deploy.py`, `tests/`.
- **Acceptance Criteria**:
  - Base notebooks register numbered versions but never call
    `set_registered_model_alias(..., "best", ...)`.
  - Candidate lineage records exact base name/version/run ID/model URI for all
    three classes.
  - Scaler, embedding, and feature-metadata artifacts are logged immutably and
    recorded in candidate/champion lineage by run URI and content hash; deploy
    never substitutes mutable `data/processed` copies.
  - A promoted ensemble model version is tagged with those exact pins and then
    assigned `@champion`; rejected candidates are never registered as ensemble
    versions.
  - The champion model-version tags are the only lineage authority and include
    every exact base version/run URI plus immutable scaler, embedding, and
    feature-metadata artifact URIs and hashes.
  - `build_input_fingerprint` includes canonical champion lineage and hashes of
    every source/artifact build input, but excludes the generated manifest and
    all post-build Bento/Docker identities.
  - No legacy champion/base-pin fallback is implemented; the registry starts
    empty and every new champion satisfies the exact-pin contract.
  - Repository guidance and README no longer claim that base `@best` aliases are
    part of training, deployment, or serving.
- **Guardrails**: Keep existing registered-model names to avoid an unrelated
  registry naming change. `@champion` is the only alias created in the clean
  registry. Deploy never mutates lineage tags or aliases.

### [x] Task 3: Add bulk IDs-based production inference and remove `/predict`

- **Description**: Remove unused model-only `/predict`. Add a private bulk form
  of `/predict_from_ids` that accepts minimal match contexts, builds all
  finalized feature rows with a set-oriented batch inference builder, and calls
  `_predict_proba` once. Preserve scalar `/predict_from_ids` for the frontend and
  make scalar/bulk builders share the same feature semantics.
- **Files**: `src/features/inference.py`, `src/serving/service.py`,
  `web/nginx.conf`, `README.md`, `tests/test_inference_features.py`,
  `tests/` (new batch API and parity tests).
- **Acceptance Criteria**:
  - Frontend continues to use only `/predict_from_ids`; `/predict` is removed
    from the Bento service, endpoint docs, and direct service tests. The explicit
    public Nginx `/api/predict` rejection remains as a defense-in-depth tombstone.
  - Bulk input uses the same fields/defaults/validation as repeated
    `/predict_from_ids`, including each row's historical `as_of_date`.
  - Batch feature construction uses set-oriented PostgreSQL queries rather than
    issuing the scalar builder's full query sequence for every row.
  - Every bulk-built finalized row matches scalar
    `_build_inference_features_with_meta` output for the same input, including
    cold-start and NULL imputation behavior.
  - Bento rejects batches above 1,000 rows. Nginx limits the authenticated batch
    request body to 10 MiB, permits at most one active batch per client, and uses
    a 120-second upstream timeout; the drift client chunks below these limits.
  - Output preserves input order and returns IDs, finalized `FEATURE_COLS`,
    `p_linear`, `p_gbdt`, `p_nn`, and `p_win` for every row.
  - One-row bulk output matches `/predict_from_ids`; multi-row output matches
    repeated scalar calls within numeric tolerance.
  - Batch logging is aggregate-only and does not print every player or feature
    row.
- **Guardrails**: Reuse `_predict_proba`; do not implement ensemble math in the
  drift/evaluation scripts. Keep the bulk route internal; Nginx owns host-facing
  authentication. Do not create a second imputation policy.

### [x] Task 4: Add private Nginx operational routes

- **Description**: Package the deployment manifest produced from immutable
  champion lineage and expose it through Bento `/model_info`. Keep Bento port
  3000 internal to Compose. Convert the Nginx config to an official image
  startup template and add two operational routes:
  `/api/internal/model-info` and `/api/internal/predict-batch`. Require an
  `X-Drift-API-Key` header matching `DRIFT_API_KEY`, supplied from the operator's
  untracked root `.env` to the Nginx container. Proxy the authenticated routes
  to Bento's internal endpoints with the bounded operational limits defined in
  Task 3.
- **Files**: `src/flows/deploy.py`, `src/serving/service.py`, `bentofile.yaml`,
  `web/nginx.conf`, `web/Dockerfile`, `compose.yaml`, `.env` (local untracked
  setup only), `scripts/dev.sh`, `README.md`,
  `tests/test_deploy.py`, `tests/`
  (Nginx/config contract tests where supported).
- **Acceptance Criteria**:
  - `compose.yaml` has no host port mapping for Bento.
  - The web/Nginx host port is bound to `127.0.0.1:8187`, matching the accepted
    loopback-only plain-HTTP operational threat model for the API key.
  - Built image contains an immutable canonical manifest with champion
    name/version/run ID/creation timestamp, exact base and auxiliary-artifact
    pins, feature-contract version/schema hash, and
    `build_input_fingerprint`; it contains no Bento tag, Docker identity, or
    hash that includes the generated manifest itself.
  - Bento `/model_info` returns the baked manifest and production deployment
    mode plus non-secret PostgreSQL connection metadata; source-mode local
    serving does not claim production mode.
  - Compose startup requires a non-empty `DRIFT_API_KEY` and passes it only to
    the Nginx/web service, not Bento or the browser build.
  - Local setup preserves all existing `.env` entries and, when the key is
    absent, adds one generated high-entropy `DRIFT_API_KEY`; its value is never
    committed, displayed, or copied into documentation/example files.
  - Nginx startup renders the template with the key; unresolved or missing key
    prevents a usable production configuration. Template substitution is
    allowlisted to `DRIFT_API_KEY` so native Nginx `$...` variables are preserved.
  - Missing or incorrect key returns 401/403 without proxying to Bento; correct
    key permits only the intended method and JSON content type.
  - The global 10 KB public payload limit remains, while the authenticated batch
    location sets an explicit 10 MiB body limit, 120-second upstream timeout,
    and one active batch per client; Bento rejects more than 1,000 rows.
  - Nginx access/error logs and application output never contain the supplied
    key; frontend source and built assets contain neither the key nor calls to
    internal routes.
  - Existing public API allowlist and explicit `/api/predict` rejection remain
    unchanged.
  - `just deploy-bento` resolves champion `C`, builds directly as the single
    `${DOCKER_REPO}/${IMAGE_NAME}:latest` image reference, and pushes only that
    tag.
  - Docker login, build, containerization, or push failure is visible, exits
    nonzero, and does not record deployment success.
  - Deployment failure never changes MLflow: champion `C`, its alias, tags, and
    all registered versions remain untouched so the operator can retry until
    push succeeds.
  - A successful push records champion `C` as the pushed version. The command
    does not pull or recreate Compose services; the operator owns that step.
  - Before reporting success, deploy re-reads `@champion`; if it changed during
    the build/push, the command reports the race and exits nonzero without
    modifying either version or alias.
  - The accepted single-`latest` limitation is explicit: if `@champion` changes
    during build/push, stale `latest` may already have been published before the
    final check fails; the operator reruns deployment for the new champion.
  - Successful deployment state is keyed by champion version and deterministic
    `build_input_fingerprint` and records post-build Bento/Docker output
    identities separately. Re-running unchanged deployment is a no-op with exit
    0; `--force` explicitly rebuilds and re-pushes.
  - Failed build/push never writes successful state, so an ordinary rerun retries
    the same champion without requiring cleanup.
  - No deploy path starts or resumes an evaluation run or calls any MLflow
    model-version/tag/alias mutation API.
  - Idempotency trusts local successful-deployment state as requested; it does
    not contact Docker Hub to revalidate the remote `latest` digest on an
    unchanged run.
- **Guardrails**: Never commit the key, bake it into an image, pass it as a build
  argument, expose it to Vite, put it in a URL/query string, or print it in an
  error. Authentication uses the request header only.

### [x] Task 5: Compare complete candidate and incumbent ensembles

- **Description**: Correct the promotion gate so the candidate is evaluated as
  new head + new base predictions, while the incumbent is evaluated as the
  currently deployed production Bento's old head + old bases. The incumbent uses
  its baked contract and the candidate uses its current training contract. Use
  the same ordered held-out raw match
  contexts and labels for both, and decide promotion only from outcome metrics,
  never contract-hash equality or feature-row equality. When
  a champion exists, require production `/model-info` to match it before
  requesting incumbent predictions from the private bulk endpoint.
- **Files**: `notebooks/parameters/01_train_test_split.ipynb`,
  `notebooks/parameters/02_tune_linear.ipynb`,
  `notebooks/parameters/02_tune_gbdt.ipynb`,
  `notebooks/parameters/02_tune_nn.ipynb`,
  `notebooks/parameters/03_ensemble_split.ipynb`,
  `notebooks/parameters/04_ensemble_stack.ipynb`,
  `notebooks/parameters/05_evaluate.ipynb`, `src/constants.py`, `tests/`
  (focused promotion-gate tests or extracted pure decision tests).
- **Acceptance Criteria**:
  - Held-out metadata includes every minimal inference input required to rebuild
    each historical row, including match date, canonical IDs, surface,
    tournament/round context, and indoor state.
  - Tuning/early stopping/model selection use a chronological validation split;
    the final evaluation split used for promotion and monitoring baselines is
    never consumed by Optuna, early stopping, or base-model selection.
  - Candidate probabilities remain candidate head over the exact candidate
    `[p_linear, p_gbdt, p_nn]` test matrix.
  - Candidate lineage records the maximum match date used by every training,
    tuning, selection, and evaluation split. Evaluation rejects the candidate
    before registration if that date is later than the current UTC date.
  - With an existing champion, evaluation resolves MLflow `@champion`, verifies
    production Bento reports the same champion version/run, and obtains
    incumbent probabilities by sending the held-out contexts through production
    bulk inference.
  - Production and candidate contract hashes are recorded for audit and route
    each stack through its own preprocessing. A mismatch does not block metric
    comparison or affect the promotion decision.
  - A contract migration retains the database inputs or versioned compatibility
    views required by the deployed incumbent until evaluation completes; each
    stack must successfully score its own contract before metrics are compared.
  - Candidate and incumbent metrics are computed against the exact same
    `y_test` rows and order.
  - Stale/unavailable production Bento, rejected API key, row-count/order
    mismatch, or non-finite prediction fails evaluation before promotion.
  - First-ever promotion skips incumbent Bento scoring because no champion
    exists and retains the existing first-promotion behavior.
  - Existing weighted metric/composite policy remains unchanged unless a
    separately approved task changes it; only incumbent prediction generation
    is corrected.
  - Evaluation logs candidate/production metrics and decision as before. On
    promotion it logs ROC AUC, accuracy, Brier, log loss, and the frozen
    calibration metric/specification, registers the head, writes exact base and
    auxiliary-artifact model-version tags, records the resulting MLflow creation
    timestamp, and only then assigns `@champion`.
- **Guardrails**: Do not compare the old head against candidate base
  probabilities. Do not load/reconstruct incumbent MLflow artifacts in the
  notebook. Do not promote when production identity cannot be proven. Never
  feed candidate-contract feature rows directly to an incompatible incumbent.

### [ ] Task 6: Reset all generated state and bootstrap a clean baseline

- **Description**: After Tasks 1-5 are implemented and verified, perform the
  explicitly authorized destructive reset of all project-owned generated
  state: MLflow tracking/registry/artifacts, local generated training/deploy
  outputs, project Bento models/Bentos and exact Docker image refs, and both
  seeded development databases.
  Recreate/reseed both databases under the new dbt contract, recreate empty
  MLflow, retrain from a fresh finalized-gold snapshot, allow the first clean
  candidate to become the first champion, then force-build and push the clean
  Bento image. The operator manually pulls/restarts production.
- **Files**: `infra/manifests/default/mlflow.yaml` (reference for reset target),
  `infra/postgres/init.sql`, `compose.yaml`, `src/db/init_db.py`,
  `src/db/seed.py`, `src/flows/pipeline.py`, `src/flows/deploy.py`,
  `README.md` (one-time clean-reset runbook), generated MLflow/PVC/database/local
  output state (destructive operational step only).
- **Acceptance Criteria**:
  - Before deletion, the runbook produces an allowlisted inventory with exact
    filesystem paths, Kubernetes context/namespace/PVC name, Compose project and
    volume name, Docker/Bento names/tags, and both PostgreSQL
    host/port/database/user targets; the operator confirms that exact inventory.
  - Reset targets only the `mlflow-data` PVC contents (`/data/mlflow.db` and
    `/data/artifacts`) used by the MLflow deployment.
  - Only repo-local MLflow paths present in the confirmed inventory are removed;
    no discovery-based machine-wide or unrelated tracking store is touched.
  - Generated `artifacts/` and `data/processed/` outputs are removed and
    regenerated; `data/raw` and source-controlled inputs remain intact.
  - Project-owned local Bento models/Bentos and local Docker images are removed
    by exact confirmed names/tags; generic Docker build-cache deletion and
    unrelated Bento/Docker objects are excluded.
  - Both the configured Homebrew development database and Compose PostgreSQL
    development volume are fully reset—including bronze, profiles, silver, and
    gold—then initialized, deterministically reseeded from preserved file inputs,
    and rebuilt with dbt under the new finalized contract.
  - Post-reset row counts/keys and finalized feature outputs match across the two
    seeded database environments.
  - MLflow is stopped/quiesced before reset, recreated from manifests, healthy,
    and verified to contain only MLflow's empty default experiment, with no runs,
    non-default experiments, registered models, versions, aliases, or artifacts
    before retraining.
  - No backup is required because the user explicitly approved deleting all
    MLflow state; the runbook still prints the exact target and requires a final
    scope confirmation immediately before deletion.
  - Fresh `dbt build` and training snapshot validation pass before `just train`.
  - Fresh training creates new base versions without `@best`; first evaluation
    sees no incumbent and registers one `ensemble_lr_model@champion` with
    exact base-version, auxiliary-artifact, and feature-contract tags.
  - The recreated registry contains no historical versions; correctness does not
    depend on a particular implementation-assigned starting version number.
  - `just deploy-bento --force` builds and pushes only Docker Hub `latest` from
    the new champion, overwriting the remote old/test `latest` without a separate
    delete; failures exit nonzero and leave MLflow unchanged.
  - After the operator manually pulls/restarts Compose, authenticated
    `/model-info` reports the clean champion, exact bases, and new feature-contract
    hash.
  - A non-mutating promotion-gate integration check sends the persisted ordered
    held-out contexts through the deployed champion, verifies identity/order and
    computes incumbent metrics without registering a model or moving an alias.
    This exercises the incumbent path after first-promotion bootstrap; the next
    real training cycle uses the same verified path.
- **Guardrails**: Database deletion is limited to the two explicitly approved
  seeded development targets. Never delete or modify `data/raw/`. Do not delete
  other file-backed source datasets, source files, Docker
  Hub repositories/tags as a separate step, dependency environments, or
  unrelated Bento/Docker/MLflow/Kubernetes resources. Do not run training
  against a partially reset database or unhealthy MLflow server.

### [x] Task 7: Add `just check-drift`

- **Description**: Add a no-argument recipe that invokes the existing dbt build
  path and then runs `src/flows/check_drift.py`. The script resolves MLflow
  champion/cutoff, calls host `/api/internal/model-info`, requires exact champion
  identity and production mode, queries reference/current gold populations,
  sends bounded requests to host `/api/internal/predict-batch`, and computes the
  report. It never calls Bento's internal route names directly.
- **Files**: `justfile`, `src/flows/check_drift.py` (new), `src/constants.py`,
  `tests/test_drift_monitor.py` (new).
- **Acceptance Criteria**:
  - Command is exactly `just check-drift`, accepts no arguments, and runs dbt
    before selecting data.
  - Default target is the production web/Nginx URL at
    `http://127.0.0.1:8187/api/internal`; direct Bento URLs are unsupported.
  - Script reads `DRIFT_API_KEY` from the existing root `.env` environment,
    sends it only as `X-Drift-API-Key`, and fails without revealing it when
    absent or rejected.
  - `/model-info` must report production mode; local dev mode is rejected.
  - Drift queries non-secret PostgreSQL connection metadata from its configured
    `DATABASE_URL` and requires an exact match with Bento's reported database
    name, server address, and server port before dbt/scoring results are joined.
    Credentials and connection URLs are never returned or logged. The accepted
    limitation is that indistinguishable clones/proxies are not detected.
  - No MLflow champion, unreachable Bento, unhealthy Bento, or identity mismatch
    fails before scoring or monitoring-run creation.
  - Identity mismatch tells the operator to run `just deploy-bento` and restart
    or repull the Compose service.
  - Report records MLflow champion identity, the manifest
    `build_input_fingerprint`, any separately available Bento/Docker output
    identity, exact UTC creation timestamp, cutoff date, date ranges, and row
    counts.
  - Client requests contain at most 1,000 rows and remain below 10 MiB; requests
    are sequential and preserve global deterministic order across chunks.
  - The command acquires a project-local single-writer process lock before any
    MLflow baseline/check lookup and fails visibly if another check is active.
  - Same-day rows are excluded; empty post-cutoff population succeeds as
    `insufficient_data` without calling batch inference or creating a baseline.
- **Guardrails**: Drift script must not import/load BentoML, MLflow model flavors,
  ONNX Runtime, torch, scaler files, embeddings, or ensemble code.
  It must not log the API key or include it in MLflow params/tags/artifacts.

### [x] Task 8: Compute drift and labeled performance

- **Description**: Compare pre/post-cutoff feature populations, Bento reference
  and current probability distributions, and current labeled metrics against
  the held-out MLflow promotion baseline. On the first check with a non-empty
  post-cutoff population for a champion and feature contract, create the
  immutable aggregate baseline described above; on
  later checks, including a changed build-input fingerprint for the same
  champion/feature contract, load it without recomputing reference predictions.
  Report feature, prediction, calibration, and performance measurements without
  interpreting them as statuses or recommendations.
- **Files**: `src/flows/check_drift.py`, `src/constants.py`,
  `tests/test_drift_monitor.py`.
- **Acceptance Criteria**:
  - Current metrics include log loss, Brier, accuracy at 0.5, calibration error,
    ROC AUC when both classes exist, and sample count.
  - Drift locates the existing promoted evaluation run read-only through its
    `params.candidate_run_id` and `metrics.promoted = 1`, requires exactly one
    completed match, and consumes its compatible evaluation metrics without
    changing that run.
  - Exactly one completed `drift_baselines` run exists per champion and
    feature-contract hash; retries reuse it or safely complete a previously
    failed attempt without duplicating the baseline.
  - Baseline artifacts contain only aggregate bins/counts, identities, hashes,
    the frozen calculation specification, and scalar metrics—not thresholds,
    raw reference records, or predictions.
  - Current distributions are evaluated against baseline-fixed bins so later
    checks do not shift their own reference boundaries.
  - Log loss and calibration deltas are reported only when promotion evaluation
    logged them under the same frozen calculation specification; otherwise the
    deltas are `unavailable`.
  - Feature and prediction output reports the frozen calculation results and
    effect measurements without threshold-derived labels.
  - Undefined or undersized metrics are explicitly `unavailable` or
    `insufficient_data`, never coerced to a health/drift status.
  - Global and eligible surface/tournament segments are reported.
  - No production output field expresses healthy/drift/investigate/retraining or
    accepts recommendation thresholds.
- **Guardrails**: Pre-cutoff Bento scores are distribution reference only; do
  not report their in-sample accuracy as champion baseline. Do not call
  prediction movement “accuracy” or claim proven concept drift.

### [x] Task 9: Log idempotent aggregate results to MLflow

- **Description**: Use `drift_baselines` for immutable references and
  `drift_checks` for cumulative monitoring results. Build a stable check key from
  champion version/creation timestamp, current `build_input_fingerprint`, current
  maximum `(match_date, match_id)`, row count, and a digest of the ordered
  selected match contexts/labels plus returned finalized features/predictions.
  The client therefore scores before deduplicating the current report. Reuse an
  existing completed run for the same key; create a new cumulative run only when
  champion, deployment, or current population changes.
- **Files**: `src/flows/check_drift.py`, `src/constants.py`,
  `tests/test_drift_monitor.py`.
- **Acceptance Criteria**:
  - Repeated unchanged checks do not create duplicate MLflow runs.
  - Idempotency relies on the Task 7 single-writer lock rather than assuming
    MLflow enforces uniqueness.
  - Repeated checks never recompute or mutate the completed champion baseline.
  - Failed/rejected training does not change the key or window.
  - Later matches create one new cumulative report.
  - A corrected context or label for an existing match changes the selected-row
    digest and creates a new report. A rolling/default correction that changes
    returned features or predictions also changes the digest; no raw row is
    stored in the key.
  - Empty populations use explicit null maximum-key fields and SHA-256 digests
    of empty ordered input and output populations.
  - New champion plus matching deployment starts a new lineage.
  - Same champion with a different validated `build_input_fingerprint` reuses
    the frozen champion baseline but creates a separate auditable check report.
  - MLflow stores aggregate parameters/metrics/tags and one JSON report, never
    raw rows, player IDs, labels, or per-match predictions.
  - Operational failure exits nonzero; successfully calculated measurements are
    recorded regardless of their values.
- **Guardrails**: No row-level monitoring database and no automatic retraining
  or deployment.

## Dependencies and order

1. Task 1 makes gold the finalized non-null feature contract and removes
   duplicate imputation.
2. Task 2 removes mutable base aliases and establishes exact ensemble lineage.
3. Tasks 3 and 4 add bulk production inference, identity, authenticated routing,
   and local secret setup.
4. Task 5 corrects promotion to compare complete candidate and incumbent
   ensembles.
5. Task 6 destructively resets all generated state and seeded databases, then
   retrains, promotes, and deploys the first clean champion.
6. Task 7 adds the drift command and champion/deployment gate.
7. Task 8 adds measurements and cached baselines.
8. Task 9 adds idempotent MLflow history.

There is no migration path or retained lineage. Tasks 1-5 are completed first,
then Task 6 recreates both seeded databases and MLflow from empty state and
establishes the first clean champion/image. Complete-ensemble incumbent
comparison begins with the next training cycle.

No new runtime dependency is required.

## QA/testing scenarios

- No MLflow champion: fail before contacting inference.
- Clean reset: MLflow SQLite/artifact PVC state, seeded PostgreSQL match
  state plus explicitly approved seeded dev databases/local generated outputs are
  removed; raw/source data remains intact.
- Homebrew and Compose databases after reseed/dbt: deterministic keys/counts and
  finalized gold contract agree.
- First training after reset: first-promotion path creates a fully tagged
  champion without requiring an incumbent Bento call.
- First clean deploy: forced build/push; operator manually pulls/restarts and
  verifies model info.
- Gold contract: every current `FEATURE_COLS` value (currently 36 columns) is
  non-null/finite after dbt.
- First-match/cold-start gold row: values come from strictly-prior defaults or
  explicit empty-pool constants, never future snapshots.
- Training split: no imputer fit/transform occurs.
- Tuning and early stopping never consume the final chronological evaluation
  split used by promotion and monitoring baselines.
- Scalar and bulk inference: no aggregate imputation queries; finalized output
  matches gold for historical fixtures.
- Two matches for one player on the same date use identical pre-day rolling/rank
  state in gold and IDs inference.
- Unknown/`0` surface is scored with all-zero surface indicators.
- Future inference date: uses the latest materialized dbt defaults.
- Direct host access to Bento port 3000: unavailable.
- Missing/wrong drift API key: Nginx rejects before Bento; no monitoring run.
- Correct key through `/api/internal/*`: Nginx proxies to production Bento.
- Frontend bundle: contains no key and makes no internal endpoint calls.
- Production Bento matches champion: continue.
- Drift and Bento PostgreSQL name/address/port metadata differ: fail before
  joining predictions to gold rows or labels.
- Promotion with an incumbent champion: candidate full stack is compared with
  production Bento's incumbent full stack on identical ordered raw held-out
  contexts and labels; the incumbent uses its baked contract and the candidate
  uses its current training contract.
- Promotion with stale/unavailable production Bento: fail before changing
  `@champion`.
- First promotion with no champion: no production Bento comparison required.
- Old head applied to candidate base probabilities: regression test rejects this
  former behavior.
- Champion promoted but image not redeployed: fail with deploy instruction.
- Docker push failure: `just deploy-bento` exits nonzero, records no successful
  deployment, and leaves MLflow champion/version/alias/tag state unchanged.
- Docker push succeeds: only Docker Hub `latest` is updated; Compose is not
  pulled or restarted automatically.
- Repeated unchanged successful deploy: no rebuild or push unless `--force` is
  supplied.
- Repeated deploy after failure: retries because no success state was written.
- Evaluation immutability: deploy and drift make no changes to the evaluation
  notebook output, evaluation MLflow run, champion alias, model tags, or model
  versions.
- Repointed base alias: production image and predictions remain unchanged.
- Failed/rejected retraining: champion cutoff and production predictions remain
  unchanged.
- Candidate data extends beyond the UTC registration date: fail before assigning
  `@champion`.
- dbt failure: stop before reading stale gold data.
- Match on champion model-version creation date: excluded.
- Later match: sent once in deterministic batch order.
- A request above 1,000 rows or 10 MiB is rejected without partial scoring.
- Client chunks the population within the fixed limits; concatenated results
  preserve deterministic order and the same metrics as an equivalent allowed
  single request.
- Batch endpoint output equals one-row production prediction output.
- Empty/small current set: successful `insufficient_data` report.
- First check for a champion with a non-empty post-cutoff population: create one
  aggregate baseline run, then evaluate the current population. Empty checks
  defer baseline creation.
- Later check for the same champion/feature contract, including a changed
  validated build-input fingerprint: reuse the baseline without sending
  pre-cutoff rows to Bento again and record a distinct check report.
- Pre-cutoff backfill after baseline creation: existing baseline remains
  immutable and subsequent checks remain comparable.
- Injected feature/probability shifts change the reported distribution
  measurements without creating a threshold-derived status.
- Degraded outcomes change the reported labeled metrics and baseline deltas but
  do not emit a retraining recommendation.
- Repeated unchanged check: existing MLflow report is reused.
- New completed matches: one new cumulative report is logged.

## Success criteria

`just check-drift` answers:

1. Which MLflow champion is active and when was its model version created?
2. Is production Docker Bento serving that exact immutable champion?
3. Which completed matches occurred after its cutoff?
4. What predictions did the actual production Bento produce for those matches?
5. What feature, probability, labeled-performance, and calibration measurements
   and baseline deltas are available for human interpretation?

It does so without manual timestamps, local dev inference, manual model artifact
loading, duplicated prediction code, request logging, row-level storage, or an
automatic retraining side effect.
