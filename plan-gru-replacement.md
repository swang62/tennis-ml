# Plan: GRU replaces tabular NN

## Goal

Replace the production tabular neural-network base model with the existing GRU while preserving the current public and registry compatibility names: `nn`, `nn_best`, `nn_model`, `nn_best.onnx`, and `p_nn`.

Keep `TabularMLP` and `02_tune_nn.ipynb` as unused historical reference code. Do not use Optuna for the GRU; declare fixed parameters at the top of the new production notebook.

## Agreed decisions

- Use a new production notebook: `02_tune_gru.ipynb`.
- Keep the old `02_tune_nn.ipynb` and `TabularMLP` class.
- Preserve the logical `nn` slot throughout MLflow, the ensemble, deployment, serving responses, and the web API.
- Build GRU histories on demand from PostgreSQL, strictly before the requested `as_of_date`; do not package a stale history-store artifact.
- Use the discovery configuration: history length 10, hidden dimension 32, dropout 0.2, learning rate 0.001, weight decay `1e-5`, 3 epochs, batch size 4096, 5 grouped forward OOF folds.
- Retain Optuna dependencies because other model families still use them, but do not import or run Optuna in GRU training.
- Enforce exact paired prediction symmetry: `p(a,b) = 1 - p(b,a)`.
- After implementation, run training and evaluation only; do not promote or deploy a candidate automatically.

## Scope

### In

- GRU training, OOF artifacts, MLflow handoff, ensemble compatibility, ONNX export, deploy-time preprocessing materialization, and Bento runtime inference.
- Runtime construction of both current-match context and each player's causal history.
- Persisted preprocessing for the final refit model.
- Explicit non-promotion path for the requested validation run.

### Out

- Removing `TabularMLP`, `02_tune_nn.ipynb`, or Optuna packages.
- Renaming `nn` model/response contracts to `gru`.
- Changing the linear or GBDT bases.
- Changing current external request fields or adding a new serving endpoint.
- Adding the proposed SVM unless separately approved.

## Tasks

### [ ] Task 1: Define the production GRU contract

- **Files**: `src/training/nn.py`, `src/training/gru_history.py`, `notebooks/parameters/02_tune_gru.ipynb` (new)
- **Description**:
  - Keep `TabularMLP` untouched.
  - Promote `SymmetricGRU` as the model used by the new notebook.
  - Place all fixed GRU parameters in the notebook's parameter/configuration cell; no Optuna import, study, pruning, or search path.
  - Make the GRU input contract explicit: 18 raw history features, 12 current-match context features, two history lengths/masks, and the exact ordered names from `GRU_RAW_NAMES` and `GRU_CONTEXT_NAMES`.
  - After fit-band imputation, fit one mean/scale pair per history feature on valid timesteps from strictly earlier fit data; standardize valid history entries only and preserve zero padding plus its separate mask/length.
  - Refactor only pure history transform/helpers as needed so offline training and online feature derivation use identical calculations.
  - Ensure historical sequence construction excludes rows not strictly before a target's `as_of_date`, including same-date rows, so training matches the serving time boundary.
- **Acceptance criteria**:
  - The notebook contains the agreed fixed values and contains no Optuna code.
  - Offline training and online preparation share the same raw-history transformations, fit-band scaling rules, and ordered names.
  - A target match never appears in either side's sequence, and same-date rows are excluded for an as-of-date request.
- **Guardrails**:
  - Do not modify or delete `TabularMLP`.
  - Do not change the existing 18 raw history features or 12 context features without a separate request.

### [ ] Task 2: Build runtime GRU inputs from PostgreSQL

- **Files**: `src/features/gru_inference.py` (new), `src/features/inference.py`, `src/training/gru_history.py`
- **Description**:
  - Add feature-layer functions that take the validated directional match context already produced by `build_inference_features` / `build_inference_features_bulk` and construct GRU inputs for both requested players.
  - Query `silver.player_matches` set-wise for the latest 10 player-perspective rows per requested player strictly before each `as_of_date`.
  - Transform those rows into right-justified, padded sequences plus lengths/masks; use the persisted final-model fill values and per-feature scaling statistics for valid sequence values.
  - Select GRU current-match context by name from the existing inference row, so surface, best-of, tournament/round, Elo, age, and H2H values exactly match the tabular inference contract.
  - Provide a batched path for the existing bulk endpoint; avoid per-row database round trips.
- **Acceptance criteria**:
  - Single and bulk inference can construct finite tensors for established players and cold starts.
  - Every history query is strictly before its request date and preserves request player order.
  - Per-player history values and the current context match offline feature preparation for an equivalent historical fixture.
- **Guardrails**:
  - Do not source serving history from `data/training_snapshot.duckdb` or `models/gru_history.parquet`.
  - Do not sort/canonicalize player and opponent IDs.

### [ ] Task 3: Train and log the GRU in the compatible NN slot

- **Files**: `notebooks/parameters/02_tune_gru.ipynb` (new), `src/flows/pipeline.py`, `notebooks/parameters/03_train_ensemble.ipynb`, `notebooks/parameters/04_evaluate.ipynb`, `src/constants.py`
- **Description**:
  - Replace `02_tune_nn.ipynb` with `02_tune_gru.ipynb` in `NB_ORDER`.
  - Recreate the current tabular-NN artifact interface using GRU predictions: `nn_oof.parquet`, `nn_test.parquet`, validation score, run handoff/version metadata, and MLflow model artifact under the existing `nn_model` / `nn_best` identity.
  - Fit preprocessing separately per OOF training fold, then fit final preprocessing on train plus validation for the refit model and test prediction.
  - Persist and log the final GRU preprocessing artifact: raw-history fill statistics, raw-history mean/scale statistics, context scaler/statistics, ordered raw/context names, history length, and schema version/hash.
  - Extend candidate/lineage metadata so deployment can verify and retrieve the exact GRU preprocessing artifact alongside its pinned raw model.
  - Keep `STACK_ORDER = ("linear", "gbdt", "nn")`; the ensemble should need only the compatible GRU-produced NN artifacts.
  - Add an explicit no-promotion parameter/path through pipeline evaluation so the required training/evaluation run cannot register or assign a champion.
- **Acceptance criteria**:
  - A full non-promoting pipeline run creates valid NN-compatible OOF/test artifacts from the GRU.
  - OOF preprocessing never uses future-fold data; final preprocessing excludes the untouched test band.
  - Evaluation consumes the GRU artifacts without changing the stack order or public model names.
  - Non-promoting mode records candidate metrics but makes no registry/champion mutation.
- **Guardrails**:
  - Do not rename `p_nn`, `nn_best`, `nn_model`, or the `nn` stack key.
  - Do not remove the old Optuna notebook or package dependencies.

### [ ] Task 4: Export, package, and serve the GRU with ONNX Runtime

- **Files**: `src/flows/deploy.py`, `src/serving/service.py`, `src/constants.py`, `bentofile.yaml` (only if its explicit artifact list requires it)
- **Description**:
  - Replace the tabular-only ONNX materializer with a GRU export wrapper around the pinned `nn_best` raw model.
  - Export named ONNX inputs for player and opponent histories, their lengths/masks, and current context, all with dynamic batch dimension; retain a single-file ONNX artifact.
  - At deployment, download the pinned final preprocessing artifact, validate its schema/names/hash, and package it with `nn_best.onnx`.
  - At runtime, load the GRU ONNX session and preprocessing artifact, build paired directional GRU inputs from PostgreSQL, and score them with ONNX Runtime.
  - Enforce the symmetry rule at the paired prediction boundary: combine the AB and BA logits into one antisymmetric logit, then expose complementary `p_nn` values. Use the same paired calculation in training/evaluation artifacts.
  - Preserve native linear/GBDT loading, the current ensemble interface, response schema, and `p_nn` observability fields.
- **Acceptance criteria**:
  - Deploy-time ONNX output matches PyTorch GRU logits within a documented numerical tolerance for single-row and multi-row batches.
  - Runtime validates all GRU preprocessing metadata before accepting requests.
  - For every pair, `p_nn_ab + p_nn_ba == 1` within floating-point tolerance, and the final endpoint continues to preserve player order.
  - Runtime needs no Torch, MLflow, or Optuna dependency; it uses the existing ONNX Runtime path.
- **Guardrails**:
  - Bento must not contact MLflow at runtime.
  - Do not package the training DuckDB snapshot or history parquet into Bento.

### [ ] Task 5: Add behavior-focused coverage and perform the requested validation run

- **Files**: `tests/test_gru_history.py`, `tests/test_gru_discovery.py`, `tests/test_inference_features.py`, `tests/test_inference_units.py`, `tests/test_pipeline.py`, `tests/test_deploy.py`, `tests/test_service_probabilities.py`, `tests/test_service_symmetry.py`, `tests/test_service_data_endpoints.py`
- **Description**:
  - Update existing GRU tests for the 12-context contract and production preprocessing artifacts.
  - Add hermetic fixtures covering strict-as-of history construction, cold starts, batched player histories, and offline/online transformation equivalence.
  - Verify ONNX/PyTorch parity with a local model and test that paired GRU probabilities are exact complements.
  - Verify pipeline artifact/lineage compatibility under the existing `nn` names and assert no registry mutation in non-promoting mode.
  - Run the narrow GRU, inference, deployment, service, and pipeline tests; then run `just lint` and the full suite.
  - Refresh dbt and the DuckDB snapshot, then run the non-promoting training/evaluation pipeline. Report candidate metrics and leave deployment untouched.
- **Acceptance criteria**:
  - All added tests are hermetic and do not contact live PostgreSQL, MLflow, Prefect, or Bento services.
  - `just lint` and the full test suite pass.
  - dbt data tests and snapshot refresh succeed before training.
  - The completed run reports GRU/ensemble candidate metrics and does not promote or deploy.
- **Guardrails**:
  - Do not add tests that freeze SQL strings, model internals, mock call ordering, or synthetic pin self-equality.
  - Do not deploy or promote without a separate explicit request after metric review.

## Dependencies

1. Task 1 establishes a stable model and preprocessing contract.
2. Task 2 must share Task 1 transformations before training/runtime parity is possible.
3. Task 3 produces the pinned model and preprocessing artifacts consumed by Task 4.
4. Task 4 must pass ONNX parity before the full non-promoting validation run in Task 5.

## QA scenarios

- Known-player single prediction and reversed orientation return complementary GRU probabilities.
- Cold-start player receives padded history, persisted fill values, tour-average-backed current features, and a finite prediction.
- Bulk requests preserve input order and use set-oriented history reads.
- A historical request does not read same-day, future, or target-match statistics.
- A deployed GRU artifact uses only pinned ONNX/preprocessing assets and PostgreSQL as-of feature data.
- A candidate GRU training run writes `nn`-named artifacts, evaluates the ensemble, and cannot mutate `@champion` in non-promoting mode.
