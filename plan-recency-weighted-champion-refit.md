# Plan: recency-weighted champion refit

## Goal

Train candidate model families with 4-year exponential recency weighting. Keep the existing chronological selection and promotion metrics unweighted. After a candidate passes promotion, refit only the winning ensemble configuration on all data through the full-snapshot `train_data_max_match_date`, then assign that refit artifact the `@champion` alias.

The champion's pinned metrics remain the pre-refit test metrics that justified promotion.

## Scope

### In

- One adjustable 4-year recency half-life constant.
- Recency-weighted fitting for every candidate model family and the selection stacker.
- A new post-promotion notebook that refits only the selected winning ensemble configuration.
- Refit of selected base models, stacker, and temperature calibration.
- Champion lineage that pins the half-life and distinguishes selection from refit provenance.

### Out

- Changing the 96/2/2 chronological split.
- Weighting validation, test, early-stopping, or promotion metrics.
- Rerunning Optuna, trying losing families, or reevaluating the promotion gate during refit.
- Changing feature derivation, serving APIs, or the meaning of `train_data_max_match_date`.

## Decisions

- The combined final holdout remains 4%: 2% validation and 2% test.
- The recency half-life starts at four years and is a single adjustable constant.
- All weight calculations use the one full-snapshot cutoff already persisted in `split_meta.max_match_date`. This makes all folds deterministic, including the earlier CV folds.
- Metrics remain unweighted so candidate promotion remains comparable with historical champions.
- After promotion, refit only the model families and hyperparameters selected by the winning candidate:
  - selected linear family: Logistic Regression or GaussianNB;
  - selected GBDT family: XGBoost or LightGBM;
  - selected NN architecture/hyperparameters;
  - the resulting ensemble stacker and temperature calibrator.
- Refit does not run tuning or fit discarded trials/families.
- GBDT best iteration and NN best epoch are frozen from selection. Refit consumes all rows through the cutoff without reserving another validation slice.
- The current `train_data_max_match_date` tag retains its current full-snapshot meaning, including the former validation/test dates after promotion.

## Tasks

### [ ] Task 1: Define and test the shared recency-weight contract

- **Files:** `src/constants.py`, new `src/training/recency.py`, new `tests/test_recency.py`
- **Description:** Add one named half-life constant, set to four years in days. Add a pure helper that accepts row match dates and an explicit cutoff, computes exponential age decay, and normalizes the output to mean 1.0.
- **Acceptance Criteria:**
  - A row exactly four years old has relative weight 0.5.
  - Weights are finite, positive, monotonic by age, deterministic, and mean-normalized.
  - Both directional rows of a match receive the same weight.
- **Guardrails:** Never use wall-clock time. Do not reuse the unrelated similarity weighting helper.

### [ ] Task 2: Weight every candidate base-model training path

- **Files:** `notebooks/parameters/02_tune_linear.ipynb`, `notebooks/parameters/02_tune_gbdt.ipynb`, `notebooks/parameters/02_tune_nn.ipynb`, `src/training/nn.py`, `tests/test_nn.py`
- **Description:** Derive date-aligned weights from `info_train.match_date` and the persisted full-snapshot cutoff. Pass them into every trial, selected candidate fit, and forward-CV OOF fit for Logistic Regression, GaussianNB, XGBoost, LightGBM, and the NN.
- **Acceptance Criteria:**
  - All base fitting paths receive correctly aligned weights.
  - NN datasets include weights and its training loss is normalized weighted BCE-with-logits.
  - NN validation loss stays unweighted.
  - GBDT early-stopping validation metrics stay unweighted.
- **Guardrails:** Do not apply sample weights to validation/test scoring or alter grouped time-forward folds.

### [ ] Task 3: Weight the candidate stacker and selection calibration

- **Files:** `notebooks/parameters/03_train_ensemble.ipynb`
- **Description:** Retain the selected physical match date with each OOF evidence row. Generate one weight per physical match and pass it to stacker fitting and cross-fitted selection calibration.
- **Acceptance Criteria:**
  - Stacker and selection calibration use one weight per physical match, not two directional-row weights.
  - Date and weight alignment survives the deterministic orientation selection.
  - Antisymmetric evidence and unweighted test evaluation remain unchanged.
- **Guardrails:** Do not alter the candidate test evidence or promotion metrics.

### [ ] Task 4: Pin recency configuration in selection lineage

- **Files:** `src/constants.py`, `notebooks/parameters/02_tune_linear.ipynb`, `notebooks/parameters/02_tune_gbdt.ipynb`, `notebooks/parameters/02_tune_nn.ipynb`, `notebooks/parameters/03_train_ensemble.ipynb`, `notebooks/parameters/04_evaluate.ipynb`, `tests/test_promotion.py`
- **Description:** Add a stable lineage key for the half-life and persist it on candidate runs, the candidate manifest, selection records, refit artifacts, and the champion model version.
- **Acceptance Criteria:** A champion's lineage exposes the configured half-life, full cutoff, immutable selection metrics, and selection artifact identifiers.
- **Guardrails:** Never replace selection metrics with refit training metrics.

### [ ] Task 5: Add a promoted-candidate-only refit notebook

- **Files:** new `notebooks/parameters/05_refit_promoted_candidate.ipynb`, `src/flows/pipeline.py`, `tests/test_pipeline.py`
- **Description:** Create a notebook invoked after `04_evaluate.ipynb` only when the evaluation step reports promotion success. It reads the accepted candidate manifest and its frozen selected configuration.
- **Flow:**
  1. Load all split rows through the persisted `train_data_max_match_date` and derive recency weights.
  2. Rebuild forward OOF predictions for only the selected base configurations; do not invoke Optuna or the rejected alternatives.
  3. Fit a weighted refit stacker from those full-history OOF predictions.
  4. Cross-fit the refit stacker and fit a weighted refit temperature calibrator from its OOF predictions.
  5. Refit only the selected base models on all rows using frozen hyperparameters, selected GBDT rounds, and selected NN epochs.
  6. Log and register the refit bases, stacker, and calibration artifacts.
  7. Return exact refit pins for champion registration.
- **Acceptance Criteria:**
  - A rejected candidate never runs this notebook or produces refit artifacts.
  - The notebook does not tune or fit discarded candidate models.
  - The selected refit bases train on all rows through the cutoff.
  - The earliest forward-CV grow-in band is excluded only where honest OOF predictions are impossible.
- **Guardrails:** No post-refit test score may replace the immutable selection score.

### [ ] Task 6: Register the refit ensemble as champion while preserving selection records

- **Files:** `notebooks/parameters/04_evaluate.ipynb`, `notebooks/parameters/05_refit_promoted_candidate.ipynb`, `src/constants.py`, `tests/test_promotion.py`, `tests/test_deploy.py`
- **Description:** Change `04_evaluate.ipynb` so it decides promotion and persists immutable selection lineage, but does not register/alias pre-refit artifacts as champion. After the refit notebook succeeds, register the refit ensemble and bases, attach refit pins to the champion, attach selection provenance/metrics and recency metadata, then set `ensemble_lr_model@champion`.
- **Acceptance Criteria:**
  - Champion base pins resolve to refit artifacts.
  - Champion selection metric tags resolve to the pre-refit evaluated candidate.
  - Deployment needs no behavior change and continues to resolve only champion pins.
  - A refit or registration failure never moves `@champion`.
- **Guardrails:** Preserve the existing model-only deployment lineage contract and immutable artifact hashes.

### [ ] Task 7: Verify end-to-end behavior hermetically

- **Files:** `tests/test_recency.py`, `tests/test_nn.py`, `tests/test_promotion.py`, `tests/test_pipeline.py`, existing deployment lineage tests
- **Description:** Add behavior-focused tests at pure weighting, NN loss, promotion/refit lineage, and notebook orchestration seams using local fixtures and fake MLflow boundaries.
- **Acceptance Criteria:**
  - A fake accepted promotion triggers one refit flow using only manifest-selected configurations.
  - A fake rejected promotion triggers no refit.
  - Champion metrics remain the selection metrics while pins identify refit artifacts.
  - No test reaches live DB, MLflow, network, or notebooks.

## Dependencies

1. Task 1 before Tasks 2 and 3.
2. Tasks 2 and 3 before the refit notebook.
3. Task 4 before champion registration changes.
4. Task 5 before Task 6.
5. Task 7 validates each completed step and the final flow.

## QA scenarios

- Current-date samples have the highest weight; four-year-old samples have half the relative weight.
- Candidate tuning is weighted, but validation/test log loss and promotion decisions are unweighted.
- A rejected candidate leaves current champion aliases and artifacts unchanged.
- An accepted candidate triggers only its winning base configurations, stacker, and calibrator in the refit notebook.
- The registered champion uses refit pins but exposes pre-refit test metrics and the selected half-life.
- The existing deployment resolves and packages the refit pins without reaching MLflow at serving time.
- Run `just lint`, narrow affected tests, then the full suite.
