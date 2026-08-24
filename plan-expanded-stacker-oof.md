# Plan: expanded stacker OOF

## Goal

Train the ensemble stacker and select its calibration temperature from strictly time-forward, cross-fitted base-model evidence spanning the existing train and validation bands. Keep the test band untouched and continue to generate test predictions from one final train-plus-validation refit per base model.

## Scope

- **In:** a separate train-plus-validation fold assignment; per-base cross-fitted evidence artifacts; ensemble evidence and strictly forward stacker predictions; calibration input replacement; notebook validation and an end-to-end development training run.
- **Out:** split boundaries, base hyperparameter-selection procedure, selected GBDT round/NN epoch policy, final base refits, test evaluation, promotion thresholds, serving, deployment, and the existing train-only `*_oof.parquet` contract.

## Decided approach

- Reuse the already selected base parameters, GBDT round count, and NN epoch count while generating expanded evidence. This avoids nested tuning; the validation portion has accepted mild model-selection contamination.
- Build a second `CV_FOLDS + 1` grouped time-forward assignment over concatenated train and validation rows.
- Do not emit synthetic zero predictions for the grow-in band. Expanded evidence begins with the first band that can be predicted by a model trained on prior dates.
- Train stacker cross-fit models only on **earlier** expanded evidence folds. Omit the first base-predicted fold from stacker-CV output because it has no earlier stacker evidence.
- Fit the final stacker on all valid expanded evidence; select/finalize calibration only from the strictly forward stacker-CV output.
- Persist the new artifacts as local training diagnostics. Do not add them to candidate manifest hashes or champion lineage.

## Tasks

### [x] Task 1: Add a distinct train-plus-validation fold assignment

- **Description:** In the split notebook, concatenate train and validation feature metadata/labels in chronological source order and create `train_val_fold_assignment.parquet` with the existing grouped, match-safe fold utility. Keep `fold_assignment.parquet` train-only and unchanged.
- **Files:** `notebooks/parameters/01_train_test_split.ipynb`
- **Acceptance Criteria:**
  - The new assignment contains exactly one row per physical train-or-validation match, with `match_id`, `match_date`, and `fold`.
  - It validates directional label balance and never splits two orientations of a match.
  - Existing split artifacts and train-only assignment retain their current paths and semantics.
- **Guardrails:** Do not move any split cutoff or include test rows.

### [x] Task 2: Produce expanded linear cross-fitted evidence

- **Description:** After linear selection, construct train-plus-validation rows and load the new assignment. For each non-grow-in fold, fit a fold-local scaler and the selected linear family/parameters only on earlier rows, then predict the held-out fold. Write only scored rows to `models/linear_train_val_oof.parquet` with `match_id`, `player_id`, `opponent_id`, `fold`, and `pred`.
- **Files:** `notebooks/parameters/02_tune_linear.ipynb`
- **Acceptance Criteria:**
  - Every expanded OOF prediction comes from a model/scaler that excluded its match and all rows from its date band onward.
  - There are no zero-filled grow-in predictions.
  - The existing `linear_oof.parquet`, validation metrics, final train-plus-validation refit, test artifact, and logged model remain unchanged in purpose.
- **Guardrails:** Do not use test data or reuse the final-refit scaler/model for expanded OOF predictions.

### [x] Task 3: Produce expanded GBDT cross-fitted evidence with frozen selection budget

- **Description:** Generate `models/gbdt_train_val_oof.parquet` from train-plus-validation time-forward folds, using the already selected framework, parameters, and `selection_best_rounds`. Fit each fold model on prior rows for exactly that fixed round count, without a fold validation/evaluation set or early stopping.
- **Files:** `notebooks/parameters/02_tune_gbdt.ipynb`
- **Acceptance Criteria:**
  - Each output row is held out from its base-fold fit.
  - Every fold uses a positive, explicitly validated selected round count.
  - No test data, final-refit model, fold future data, arbitrary fallback round count, or early-stopping evaluation data contributes to expanded OOF output.
  - Existing train-only GBDT OOF and final-refit/test behavior remain separate.
- **Guardrails:** Do not retune per fold or alter the selection and final-refit paths.

### [x] Task 4: Produce expanded NN cross-fitted evidence with fixed epochs

- **Description:** Generate `models/nn_train_val_oof.parquet` from the new assignment. For each non-grow-in fold, use a fresh fold-local scaler and a fresh NN with selected architecture/hyperparameters, trained on prior rows for exactly `selection_best_epochs`, with no validation callbacks or checkpoint selection.
- **Files:** `notebooks/parameters/02_tune_nn.ipynb`
- **Acceptance Criteria:**
  - Every output row is held out from that fold’s scaler and model fit.
  - The epoch count is the deterministic selected checkpoint epoch already used by the final refit and is positive.
  - Expanded OOF fitting contains no validation callbacks, test rows, or final-refit model reuse.
  - Existing train-only NN OOF and final-refit/test behavior remain separate.
- **Guardrails:** Do not derive epochs from the trainer’s final stopped epoch or introduce nested hyperparameter tuning.

### [x] Task 5: Build expanded evidence and train the final stacker

- **Description:** In the ensemble notebook, load the three expanded base artifacts, verify they cover the same match/orientation/fold keys, and transform paired directional probabilities into deterministic antisymmetric one-row-per-match evidence. Persist `models/train_val_evidence.parquet` with `match_id`, `match_date`, `fold`, `match_won`, and one column per base model. Fit the deployable zero-intercept logistic stacker on all valid expanded evidence and continue to apply it to final-refit test evidence.
- **Files:** `notebooks/parameters/03_train_ensemble.ipynb`
- **Acceptance Criteria:**
  - The evidence merge fails clearly on missing, duplicate, mismatched, or non-complementary directional base predictions.
  - No grow-in rows or zero-filled placeholder predictions reach the final stacker.
  - The candidate test evidence still comes only from the existing final-refit `*_test.parquet` artifacts.
  - Candidate manifest lineage remains restricted to the deployed base artifact pins; expanded diagnostics are not added as manifest pins/hashes.
- **Guardrails:** Do not modify the stacker architecture, intercept setting, test labels, model registration flow, or existing train-only OOF files.

### [x] Task 6: Generate strictly forward stacker-CV calibration evidence

- **Description:** Replace the leave-one-fold-out stacker-CV generation with a chronological procedure over `train_val_evidence.parquet`: for each fold after the first base-predicted fold, fit a temporary stacker only on earlier evidence folds and predict its current fold. Persist `models/train_val_stack_cv.parquet` with `match_id`, `match_date`, `fold`, `stack_pred_cv`, and `match_won`.
- **Files:** `notebooks/parameters/03_train_ensemble.ipynb`
- **Acceptance Criteria:**
  - Every stack-CV prediction is produced without future evidence labels/features.
  - The first stacker-eligible fold is excluded because no prior stacker training evidence exists.
  - Every persisted stack-CV fold contains complete match-safe evidence and both directional label classes at the match level.
- **Guardrails:** Do not retain the current “all other folds” cross-fit behavior for calibration, and do not use test evidence.

### [x] Task 7: Switch calibration to expanded strictly forward evidence

- **Description:** Load `train_val_stack_cv.parquet` in evaluation and pass its ordered fold predictions/labels to the existing temperature-selection logic. Preserve candidate test metrics as evaluation-only and keep the final temperature fit limited to accepted expanded stack-CV evidence.
- **Files:** `notebooks/parameters/04_evaluate.ipynb`
- **Acceptance Criteria:**
  - Calibration selection and final temperature fit never inspect test labels/predictions.
  - The evaluation notebook fails clearly if expanded stack-CV data is missing, empty, unordered, or incompatible with its label/fold contract.
  - Promotion still compares the same untouched candidate test metrics with the incumbent.
- **Guardrails:** Do not change `select_temperature` thresholds or promotion logic.

### [ ] Task 8: Validate the expanded evidence contract

- **Description:** Parse edited notebooks and run repository checks. Execute a full development training run, then inspect persisted artifacts to verify fold coverage, no grow-in placeholders, chronological base/stacker fit boundaries, test-artifact lineage, and calibration source.
- **Files:** Changed notebooks; `tests/test_pipeline.py` only if its notebook artifact contract needs a behavior-level update.
- **Acceptance Criteria:**
  - Edited notebooks parse as JSON and all code cells parse syntactically.
  - `just lint` and `just test` pass.
  - A development `just train` produces all five new artifacts and a candidate whose test evidence uses final-refit base artifacts.
  - Expanded OOF and stack-CV artifacts are internally keyed/aligned, contain no grow-in placeholders, and obey their strict forward boundaries.
  - Calibration reads the new stack-CV artifact; the existing train-only OOF artifacts remain present and unchanged in schema.
- **Guardrails:** Keep tests hermetic; do not need a production run to validate this change and do not weaken existing checks.

## Dependencies

1. Task 1 precedes Tasks 2-4.
2. Tasks 2-4 must complete before Tasks 5-6.
3. Task 6 precedes Task 7.
4. Task 8 follows all implementation tasks.

## QA scenarios

1. A train-plus-validation base fold cannot fit either orientation of the match it predicts, or any rows dated inside/later than its held-out band.
2. The expanded grow-in band is absent from all expanded base/evidence/stack-CV artifacts.
3. A stack-CV prediction for fold N is trained only from evidence folds earlier than N.
4. The final stacker sees all valid expanded evidence, but its test inputs come only from final-refit base predictions.
5. Temperature selection consumes only chronological stack-CV predictions; the test set remains unused until candidate evaluation.
6. A stale/misaligned base artifact, missing expanded fold, or directional-pair mismatch fails before stacker fitting.
