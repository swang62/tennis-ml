# Plan — Pinned Stacked-Ensemble Serving & Promotion

Goal: make the stacked ensemble (3 base models + LR meta) the deployable unit,
with every candidate run pinned to exact base-model versions, promotion gated on
test evaluation beating the current production model, and a Bento that composes
all 4 models plus decoupled aux artifacts. Base versions come from the recorded
pins on the promoted run (no `latest` fallback); the single deploy target
(`just deploy-bento`) resolves the current `production_model` via
`models:/production_model/latest` and no-ops when it has not advanced past the
last deployed version.

## Deployment stance — manual, single-path, decoupled from training

Bento builds are intentionally out of the training critical path: a build takes
~30 min and is not worth running on every promotion. Training (pipeline/train
flows) runs the full notebook chain through 05, the evaluation/promotion
stage. 05 only registers `production_model` in MLflow; it never triggers a
build. Deployment is a separate, manual final stage: `src/flows/deploy.py` is
deploy-only — it runs no notebook and just invokes `just deploy-bento`, the
single deployment path from registry to running image.

`just deploy-bento` gates on promotion progress: it resolves the latest
`production_model` version and no-ops ("nothing to deploy") when it is not
newer than the last deployed version recorded in
`data/processed/bento_build_state.json` (`deployed_version`). Only a newer
promotion triggers work: import the pinned base + production model versions,
build the Bento, containerize it, and `k3d image import` it into the cluster
when the cluster is running. If the cluster is not running, the build and
containerize still happen and a later re-run reuses the built Bento and
completes the import. The gate keeps re-runs cheap: a `just deploy-bento` run
with no newer promotion is a no-op.

The build is additionally cached by a state fingerprint (production version,
base versions recorded on its run, and the contents of the template bentofile,
service code, and aux artifacts), so a retry after a skipped or failed cluster
import skips the ~30 min build. It performs no strict pin validation — the
recorded versions are used as-is; a missing pin or missing aux file fails with
a clear message, and an unresolvable version fails naturally at import time.

## Key discovery — surface context is already in the features

`src/features/rolling.py` defines:

```python
CONTEXT_COLS = ["is_clay", "is_grass", "is_hard", "tournament_level", "round_encoded"]
FEATURE_COLS = PLAYER_COLS + OPPONENT_COLS + DIFF_COLS + CONTEXT_COLS   # rolling.py:74
```

So the one-hot surface context (`is_clay`/`is_grass`/`is_hard`) is already part
of the training feature schema, and `01_train_test_split.ipynb` keeps raw
`surface` in `info_cols` (`["match_id", "match_date", "player_id", "opponent_id",
"tournament", "round", "surface"]`) for diagnostics only.

The user's surface requirement is therefore already structurally satisfied. No
feature-schema expansion is needed. The only action is to preserve this behavior
(do not drop the surface one-hots; do not fold raw `surface` into the model).

## Current state (exact notebook names + wiring)

```
notebooks/parameters/00_embeddings.ipynb        → bio_embeddings.parquet (player_id -> bio_*)
notebooks/parameters/01_train_test_split.ipynb  → X_*/y_*/info_*.parquet, feature_cols.json
notebooks/parameters/02_tune_gbdt.ipynb         → gbdt_{oof,test}.parquet, gbdt_score.json
notebooks/parameters/02_tune_linear.ipynb       → linear_{oof,test}.parquet, linear_score.json
notebooks/parameters/02_tune_nn.ipynb           → nn_{oof,test}.parquet, nn_score.json
notebooks/parameters/03_ensemble_split.ipynb → oof_preds/test_preds.parquet + base_pins.json
notebooks/parameters/04_ensemble_stack.ipynb    → LR meta, logged as "stacked_ensemble" run
notebooks/parameters/05_evaluate.ipynb          → gated promotion (manifest load, AUC gate, register only; deploy is manual)
src/flows/deploy.py                             → deploy-only manual flow (runs no notebook; calls `deploy_bento()`, behind `just deploy-bento`)
```

Orchestration wiring (both must stay in sync after the rename):

- `src/flows/pipeline.py` — `NOTEBOOK_PARAMS` + `NB_ORDER` (standalone runner;
  runs the full chain through 05)
- `src/flows/training.py` — `NOTEBOOK_PARAMS` + `NB_ORDER` (Prefect flow;
  runs the full chain through 05)
- `src/flows/deploy.py` — deploy-only manual flow: runs no notebook, just
  invokes the single deployment path (`deploy_bento()`, behind `just
  deploy-bento`)
- `00_embeddings.ipynb` is wired into `NOTEBOOK_PARAMS` and runs first in
  `NB_ORDER` (ahead of 01) in **both** flow files — it is now part of every
  training pipeline path (standalone and Prefect).
- `src/serving/` — serving code only (`service.py`); the misplaced deploy
  hook was removed
- `src/serving/service.py` — loads one `BentoModel("tennis_prediction:latest")`
- `bentofile.yaml` — single `models: ["tennis_prediction:latest"]`
- `justfile` — `deploy-bento` / `pipeline` / `train` targets

Known gaps this plan closes: 02 notebooks log unregistered artifacts (no
version, unpinnable); 05 loads the meta-model via `runs[0]` (unpinned) and only
prints a recommendation; service serves a single model; scaler is computed
inline in 02_tune_linear and never persisted; bio embeddings are not packaged
into the Bento; the build had no way to know which versions to import (now the
recorded pins on the promoted run; used as-is, not validated).

## Work items (implement in order)

### 02 — Base-model registration

Files: `notebooks/parameters/02_tune_gbdt.ipynb`, `02_tune_linear.ipynb`, `02_tune_nn.ipynb`

- After the existing `mlflow.<flavor>.log_model(...)`, register each best model:
  `mlflow.register_model(f"runs:/{run_id}/<artifact>", f"base_{name}")` →
  `base_gbdt`, `base_linear`, `base_nn`.
- Write the pinned identity to disk for downstream notebooks:
  `data/processed/{name}_model_version.json` =
  `{"name": "base_linear", "version": 3, "uri": "models:/base_linear/3"}`.
- Log the version as a run param (`base_version`) on the 02 run.

Verify: rerun the three notebooks; `MlflowClient().get_model_version("base_linear", 3)`
resolves; the version jsons exist with matching numbers.

### 03 — Rename notebook to `03_ensemble_split`

Files: `notebooks/parameters/03_ensemble_split.ipynb`,
`src/flows/pipeline.py`, `src/flows/training.py`

- The notebook was renamed (git mv) to `03_ensemble_split.ipynb` — content
  unchanged: still picks best per class and builds OOF/test prediction matrices.
- Update the notebook key in `NOTEBOOK_PARAMS` and `NB_ORDER` in both
  `src/flows/pipeline.py` and `src/flows/training.py`.
- Add pin propagation: 03 reads the three `{name}_model_version.json` files from
  02 and writes `data/processed/base_pins.json` (names + versions + URIs) that 04
  logs and the manual build step uses.

Verify: `python -c "from src.flows.pipeline import NB_ORDER; print(NB_ORDER)"` shows the
new name; rerun 03 produces `base_pins.json`.

### 04 — Candidate-only LR logging with pinned base versions

File: `notebooks/parameters/04_ensemble_stack.ipynb`

- Keep training the LR meta-model on `oof_preds.parquet` (unchanged).
- Log the run as a candidate only — no `mlflow.register_model` in this notebook.
- Log pins as run params: `base_linear_version`, `base_gbdt_version`,
  `base_nn_version` from `base_pins.json`, plus the registered names and
  run ids/URIs (`base_{class}_registered_name`, `base_{class}_model_uri`).
- Write `data/processed/candidate_manifest.json` =
  `{"candidate_run_id": <run_id>, "model_uri": "runs:/<run_id>/stacked_ensemble",
  "artifact_path": "stacked_ensemble"}` — the exact handoff 05 loads, so no
  latest-run lookup anywhere.
- Do not resolve or overwrite `tennis_prediction` / `production_model` here.

Verify: rerun 04; the run shows the three pinned version params; no registered
model was created or transitioned; `candidate_manifest.json` exists with a
`model_uri` that resolves via `MlflowClient().get_run(run_id)`.

### 05 — Gated promotion only after better evaluation

File: `notebooks/parameters/05_evaluate.ipynb`, `src/flows/pipeline.py`,
`src/flows/training.py` (wired into `NOTEBOOK_PARAMS` + `NB_ORDER`)

- Load the candidate meta-model from `data/processed/candidate_manifest.json`
  (written by 04): `model_uri` → `mlflow.sklearn.load_model` — never `runs[0]`.
- Load production via `mlflow.sklearn.load_model("models:/{production_model_name}/latest")`
  (first run with no production → promote by default, as today).
- Compare both on the SAME `test_preds.parquet` matrix (never `X_test`).
- Idempotency guard: skip promotion when the current production version's
  `run_id` already equals the manifest's `candidate_run_id`.
- Gate: register the candidate under `production_model_name` only if
  `candidate_test_auc > production_test_auc` (optionally + small epsilon param).
- On promotion only: `mlflow.register_model(runs:/<run>/stacked_ensemble,
  production_model_name)`. Deployment is NOT triggered here — it is decoupled
  and manual (see "Deployment stance"). 05 may still read an optional
  `rebuild_cmd` param for a manual deploy hook, but it defaults to empty/off
  and no pipeline wiring passes it in.
- Log decision metrics: `promoted` (0/1), `promotion_auc_delta`.

Verify: with no production model the run promotes; rerun the same manifest must
skip (idempotent, `promoted=0`, no re-register); a worse candidate must log
`promoted=0` and not register. No build is ever triggered from 05.

### Bento composition of 4 artifacts

Files: `bentofile.yaml`, `src/serving/service.py`

- `bentofile.yaml` `models:` lists 4 pinned BentoML model refs (one per base +
  the LR meta), each imported from its pinned MLflow version
  (`bentoml.mlflow.import_model("base_linear", "models:/base_linear/3")`).
- `service.py`: replace the single `self.model` with the 3 base models + meta;
  `predict` = canonical row → per-base `predict_proba` (handle SVM/NN proba
  paths) → 3-vector → LR meta `predict_proba` → final probability in [0, 1].
  Keep the `FEATURE_COLS` missing-column guard.

Verify: `uv run bentoml build --bentofile bentofile.yaml` succeeds; local
`bentoml serve` smoke test returns probabilities and rejects missing columns.

### Decoupled aux artifacts (scaler + bio embeddings)

Files: `notebooks/parameters/02_tune_linear.ipynb` (scaler persistence),
`bentofile.yaml` / build step (packaging), `src/serving/service.py` (loading)

- Persist the scaler: 02_tune_linear currently fits `StandardScaler` inline and
  throws it away — `joblib.dump` to `data/processed/linear_scaler.pkl` (fit on
  train only, alongside the 02 logging).
- Package both aux files into the Bento as plain data artifacts
  (`data/processed/linear_scaler.pkl`, `data/processed/bio_embeddings.parquet`
  via `include` — both are outside the current `include: ["src/"]`), loaded at
  service init, not baked into any model.
- They are versioned by the pinned build step, not by MLflow.

Verify: built Bento contains both files (`bentoml get` / build output); serve
from a clean workdir without the repo's `data/processed` on disk.

### Single deploy path — `just deploy-bento` (manual, opt-in, no strict pin validation)

Deliberately not wired into promotion: builds take ~30 min, so this runs only
when a human decides to deploy (see "Deployment stance"). Cheap on repeats:
a re-run with no newer promoted model no-ops without building.

Files: `justfile` (`deploy-bento` target), `src/flows/deploy.py` (`deploy_bento()`)

- `deploy_bento()` resolves `models:/production_model/latest` and no-ops when
  its version is not newer than the last deployed version recorded in
  `data/processed/bento_build_state.json` (`deployed_version`). Otherwise it
  reads the base pins recorded as params on the production run, imports the
  exact referenced MLflow versions into the BentoML store (reusing existing
  imports), writes `data/processed/bentofile.pinned.yaml` from the
  `bentofile.yaml` template, and builds the Bento. Recorded versions are used
  as-is — no validation that they match some expected schema, no `latest`
  fallback.
- The target then containerizes the Bento and imports the image into the k3d
  cluster when it is running. `deployed_version` is only recorded after a
  successful import, so a skipped or failed import is retried by re-running
  `just deploy-bento`, which reuses the built Bento via the fingerprint.
- Caching: a state fingerprint (production version, base versions, hashes of
  the template bentofile, `service.py`, and the aux files) is stored in
  `data/processed/bento_build_state.json` alongside the built tag. If the
  fingerprint is unchanged and the built Bento still exists, the build is
  skipped ("no rebuild needed").
- Missing recorded pins or missing aux files fail with a clear message; an
  unresolvable model version fails naturally at import time.

Verify: first run builds; re-running with no new promotion no-ops ("no newer
promoted production model"); re-running after a new promotion (new versions)
builds again.

## Verification checklist (end-to-end)

1. `uv run python src/flows/pipeline.py` runs all 8 notebooks (00_embeddings
   first in `NB_ORDER`, new 03 name) clean, through 05 (gated
   evaluation/promotion; no Bento build is ever triggered); same for
   `uv run python src/flows/training.py`.
2. `base_pins.json` matches the params on the promoted 04 run.
3. `uv run python src/flows/deploy.py` runs the deploy only — no notebook; it
   invokes `deploy_bento()` (equivalent to `just deploy-bento`). 05's
   `promoted=1` is logged by the pipeline run when candidate AUC beats
   production.
4. `just deploy-bento` with no newer promoted production model no-ops ("no
   newer promoted production model"); a new promotion (new pinned versions)
   triggers a rebuild and cluster import.
5. Deployed Bento serves probabilities via 4-model composition + aux artifacts.
