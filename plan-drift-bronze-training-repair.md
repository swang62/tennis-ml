# Plan: Bronze Drift Monitoring and Training Repair

## Goal

Simplify drift monitoring to use one physical match from `bronze.match_events` as the source observation, derive two symmetric prediction rows from it, and send the raw bronze context directly to Bento. This removes the gold numeric-context decode/re-encode path while retaining match-context PSI, prediction-distribution PSI, and performance monitoring.

Repair `just train` so a new PostgreSQL-to-DuckDB snapshot creates a fresh grouped-CV assignment, while the GBDT, linear, and NN tuners still prove they use the identical assignment within that training run.

## Scope

### Included

- Replace drift window reads from `gold.match_features` with physical-match reads from `bronze.match_events`.
- Derive orientation labels from `winner_id`; score both player orders so each physical match contributes one positive and one negative observation.
- Send bronze string fields (`tournament`, `round`, `surface`, `is_indoor`, `match_date`) directly to the existing bulk Bento schema.
- Retain PSI only for the selected match-context attributes (`surface`, `tournament`, `round`, `is_indoor`) and `p_win`; retain performance metrics against the derived labels. Do not monitor player profile attributes.
- Make grouped-fold persistence safe across a newly refreshed training snapshot and strict across the three tuning notebooks in one pipeline run.
- Add hermetic regression tests; no live database in tests.

### Excluded

- Changing Bento's public API or feature-building implementation.
- Replacing dbt/gold training features or retraining a model as part of the repair.
- Using seasonal match composition, rankings, rolling form, schedule volume, H2H, surface, tournament tier, or round as PSI retrain signals.
- Altering MLflow promotion policy beyond ensuring a repaired training run can complete.

## Tasks

### [x] Task 1: Define the bronze-backed drift observation contract

- **Description**: In `src/flows/drift.py`, define the single physical-match projection from `bronze.match_events`: `match_id`, `match_date`, `player1_id`, `player2_id`, `winner_id`, `surface`, `tournament`, `round`, and normalized `is_indoor`. Keep one source row per physical match, then expand it into two requested orientations for scoring: player1→player2 with label `winner_id == player1_id`, and player2→player1 with label `winner_id == player2_id`.
- **Files**: `src/flows/drift.py`, `tests/test_drift_monitor.py`
- **Acceptance Criteria**:
  - Current and reference source windows each have one observation per `match_id`; each scored window has two symmetric rows per match.
  - The current window is `match_date > champion cutoff`; reference remains newest-first, size-matched, and restored to chronological order.
  - The derived labels are boolean/integer and agree with the requested orientation for every test fixture; five physical matches produce ten valid scored rows with balanced labels.
  - Test data represents bronze columns rather than prebuilt gold feature rows.
- **Guardrails**: Do not read `gold.match_features` for drift labels or prediction context. Do not sort/canonicalize player IDs; preserve both bronze orientations when generating requests.

### [x] Task 2: Send bronze context directly through the Bento request model

- **Description**: Build two raw inference contexts per bronze row using both player orders, raw `surface`, `tournament`, `round`, `match_date` as `as_of_date`, and normalized `is_indoor`. Validate each generated context with the real `PredictFromIdsRow` Pydantic model before posting the bulk envelope.
- **Files**: `src/flows/drift.py`, `src/evaluate/promotion.py` (only if the generic incumbent-context helper must support or be separated from the bronze contract), `tests/test_drift_monitor.py`, relevant promotion tests
- **Acceptance Criteria**:
  - Drift never emits internal `tournament_level` or `round_encoded` fields to the endpoint.
  - A payload generated from a bronze fixture round-trips through `PredictFromIdsRow.model_validate(...).model_dump(mode="json")` unchanged.
  - `p_win` maps to the first-supplied side and is evaluated against the matching orientation label; the two predictions for a physical match are complementary by construction of the labels.
  - The production bulk endpoint remains the only code that converts public strings into its internal feature encodings.
- **Guardrails**: Do not duplicate the public endpoint schema in tests. Reuse Pydantic validation as the contract boundary.

### [x] Task 3: Limit drift analysis to stable signals and predictions

- **Description**: Keep PSI for `p_win` and only the bronze match-context attributes `surface`, `tournament`, `round`, and `is_indoor`. Remove player profile attributes (`age_diff`, years-pro, and handedness) from the drift analysis. Continue to skip the report below `DRIFT_MIN_N_FOR_CHECK`; retain the loose retrain thresholds already selected.
- **Files**: `src/flows/drift.py`, `src/constants.py`, `tests/test_drift_monitor.py`, `tests/test_drift_recommendation.py`
- **Acceptance Criteria**:
  - Evidently receives only the selected match-context attributes, the orientation-level label, and `p_win`; it does not receive player profile attributes.
  - The report cannot contain duplicated columns.
  - Fewer than 10 current physical matches logs `insufficient_data` and does not calculate PSI, correlation, or a retrain verdict; symmetric expansion is still tested with a five-match fixture producing ten scored rows.
  - Tests verify exact analysis-frame column uniqueness and the small-window guard.
- **Guardrails**: Do not use PSI as the primary detector for malformed scraped rows; existing validation/schema/range checks remain the data-quality boundary.

### [x] Task 4: Make grouped CV assignments lifecycle-aware

- **Description**: Separate "start a new training snapshot/run" from "reuse the same fold assignment across model tuners." The split notebook should create/replace the fold assignment immediately after writing the new split artifacts. Each `02_*` notebook should load and verify that current-run assignment instead of asserting a stale file from a prior snapshot matches. `pipeline.py` only needs to preserve the existing ordering: refresh the snapshot, run `01`, then run all three `02` notebooks.
- **Files**: `src/flows/pipeline.py`, `src/models/grouped_cv.py`, `notebooks/parameters/01_train_test_split.ipynb`, `notebooks/parameters/02_tune_gbdt.ipynb`, `notebooks/parameters/02_tune_linear.ipynb`, `notebooks/parameters/02_tune_nn.ipynb`, `tests/test_grouped_cv.py`, focused pipeline tests if present
- **Acceptance Criteria**:
  - A full `just train` after the snapshot grows from 240 to 245 matches replaces the old 240-row `fold_assignment.parquet` once, then does not fail in any tuner.
  - All three tuning notebooks consume one identical assignment for the current split.
  - Loading an assignment whose match IDs, fold count, or canonical current-split ordering do not match fails clearly before tuning.
  - The assignment artifact remains logged for every model run as today.
  - **Guardrails**: Replace the assignment only at the start of a new split/run; do not overwrite it from any `02_*` notebook. Do not weaken grouped-CV guarantees or split the two player orientations of a physical match across folds.

### [x] Task 5: Verify end-to-end boundaries

- **Description**: Run the focused hermetic drift and grouped-CV tests, then run `just train` against the current Docker PostgreSQL data. If credentials/model availability block a full training run, run through the repaired GBDT notebook phase and report the exact blocker rather than masking it.
- **Files**: no production source changes beyond prior tasks
- **Acceptance Criteria**:
  - Drift tests cover bronze-to-Pydantic-context conversion, label derivation, minimum sample behavior, column uniqueness, and payload response alignment.
  - Grouped-CV tests cover fresh-run replacement, same-run reuse, and stale/mismatched assignment rejection.
  - `just train` passes the previously failing 02 GBDT fold-assignment step.
  - No test opens a live database connection.
- **Guardrails**: Do not add live PostgreSQL fixtures to the test suite. Do not delete user data or existing MLflow versions during verification.

## Dependencies

1. Tasks 1 and 2 define the replacement observation/prediction contract before Task 3 configures its report.
2. Task 4 is independent of drift implementation but must complete before the full training verification in Task 5.
3. Task 5 follows all implementation tasks.

## QA/Testing Scenarios

1. A bronze fixture produces two valid symmetric Bento requests per physical match, derives complementary orientation labels, and aligns response `p_win` with those labels.
2. A post-cutoff window with fewer than 10 physical matches records `insufficient_data` without calling Bento or Evidently.
3. A normal window produces a unique analysis frame, match-context PSI, prediction PSI, and performance metrics.
4. A five-match fixture produces ten scored rows; a new 245-match split replaces the former 240-match fold artifact once; GBDT, linear, and NN each use the same new folds.
5. A corrupted or stale fold artifact fails at the explicit validation boundary rather than deep inside an Optuna notebook.
