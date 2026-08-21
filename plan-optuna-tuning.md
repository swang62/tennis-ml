# Plan: Optuna Tuning Corrections

## Goal

Correct neural-network validation-loss selection, make GBDT Optuna pruning act on boosting-round validation metrics, independently compare competing model families, and enable LightGBM row subsampling.

## Scope

### In

- NN: correct the probability-to-logit conversion used for the Optuna objective and change the trial budget from 20 to 30.
- GBDT: replace the one conditional 80-trial study with independent 40-trial XGBoost and 40-trial LightGBM studies; report validation log loss per boosting iteration so `MedianPruner` can terminate weak trials.
- LightGBM: ensure its sampled row fraction is active by setting a nonzero bagging frequency.
- Linear: replace the conditional model-family study with independent 40-trial Logistic Regression and 15-trial GaussianNB studies; neither one-shot fit has an intermediate state for pruning.
- Add hermetic regression tests for the NN metric conversion and the GBDT reporting/pruning bridge.
- Run `just lint` and the focused pytest tests.

### Out

- No change to the 94/4/2 chronological train/validation/test split.
- No change to model classes, feature definitions, final test evaluation, ensemble training, or promotion criteria.

## Decisions Recorded

- Run independent 40-trial Logistic Regression and 15-trial GaussianNB studies, then select the lower validation-log-loss winner.
- Set NN tuning to 30 trials.
- Run separate 40-trial XGBoost and LightGBM studies, then select the lower validation-log-loss winner.
- Retain the existing NN Lightning pruning callback: it already reports `val_loss` every epoch. Remove only redundant post-fit pruning logic if it adds no value.
- Use a fixed LightGBM bagging frequency of 1 with the existing sampled-row fraction search range. Do not add a separate bagging-frequency dimension.
- Tests must be self-contained and must not need a database, MLflow, network access, or a full notebook training run.

## Tasks

### [x] Task 1: Add a testable GBDT Optuna reporting bridge

- **Description**: Introduce the smallest shared, plain-Python helper needed to translate XGBoost and LightGBM per-iteration validation log-loss callbacks into `trial.report(value, step)` and raise `optuna.TrialPruned` when `trial.should_prune()` returns true. Keep framework-specific callback adapters thin and retain current early-stopping behavior.
- **Files**: `src/models/optuna_pruning.py` (new), `notebooks/parameters/02_tune_gbdt.ipynb`.
- **Acceptance Criteria**:
  - Both framework-specific studies report chronological-validation binary log loss with a monotonically increasing boosting-round step.
  - A fake trial that requests pruning stops the training callback by raising `optuna.TrialPruned`.
  - Existing early stopping remains at 25 rounds and still selects against the same validation band.
- **Guardrails**: Do not make tree count an additional tuning loop, do not use the test split, and do not add an Optuna integration dependency beyond the existing packages.

### [x] Task 2: Correct search configuration and objective values

- **Description**: In the NN notebook, replace the invalid `log(probability)` input to `binary_cross_entropy_with_logits` with a numerically clipped logit conversion, or compute log loss directly from probabilities. Set `n_trials` to 30. In the GBDT notebook, create separate 40-trial studies with their own framework-specific spaces, select the lower-validation-loss winner, and configure LightGBM bagging frequency as 1 so its current `subsample` range becomes effective. In the linear notebook, create independent 40-trial Logistic Regression and 15-trial GaussianNB studies and select the lower-validation-loss winner; remove the nonfunctional MedianPruner.
- **Files**: `notebooks/parameters/02_tune_nn.ipynb`, `notebooks/parameters/02_tune_gbdt.ipynb`, `notebooks/parameters/02_tune_linear.ipynb`.
- **Acceptance Criteria**:
  - NN objective equals binary cross-entropy / log loss for the same predicted probabilities within numeric tolerance.
  - NN, GBDT, and linear selection still minimize validation log loss.
  - XGBoost and LightGBM each receive 40 trials, use only their own parameter spaces, and the final GBDT winner is the lower validation-log-loss study winner.
  - Logistic Regression receives 40 trials and GaussianNB receives 15 trials in separate studies; the final linear winner is the lower validation-log-loss study winner.
  - LightGBM receives both sampled-row fraction and a nonzero bagging frequency.
  - NN per-epoch pruning remains supplied by `PyTorchLightningPruningCallback`; GBDT uses the new per-iteration reporting bridge; linear has no inert pruner.
- **Guardrails**: Preserve seed 42, model search-space bounds, batch size, early-stopping settings, validation data ownership, and all MLflow artifact contracts unless a changed line is directly required by this task.

### [x] Task 3: Add hermetic regression coverage

- **Description**: Test the shared reporting helper with fake trial and callback environments. Add a numeric regression test covering probability-to-logit / binary-cross-entropy equivalence. If notebook configuration cannot be imported directly, assert its values through a narrow notebook-source test rather than executing the notebook.
- **Files**: `tests/test_optuna_pruning.py` (new), optionally `tests/test_tuning_notebooks.py` (new).
- **Acceptance Criteria**:
  - Tests verify `report()` receives the validation value and boosting step.
  - Tests verify pruning propagates as `TrialPruned`.
  - Tests catch regression to `log(probability)` as a BCE-with-logits input.
  - Tests verify the agreed NN, XGBoost, LightGBM, Logistic Regression, and GaussianNB trial budgets; separate-study setup; and active LightGBM bagging setting if configuration is source-asserted.
- **Guardrails**: No live databases, no external MLflow/Prefect calls, no downloaded data, GPU dependency, or full training in tests.

### [x] Task 4: Validate the changed tuning path

- **Description**: Run focused tests, then project lint/type checks. Inspect the notebook diff to ensure only the planned tuning behavior changed.
- **Files**: all files above.
- **Acceptance Criteria**:
  - Focused pytest tests pass.
  - `just lint` passes.
  - No unrelated notebook output cells, generated artifacts, dependency lock updates, or data files are changed.
- **Guardrails**: Do not run the full pipeline or any command requiring PostgreSQL, a network service, or existing artifacts solely for this change.

## Dependencies

- Task 1 precedes Tasks 2 and 3.
- Tasks 2 and 3 precede Task 4.
- The repository already provides Optuna, Optuna's PyTorch Lightning integration, XGBoost, and LightGBM; no new package is required.

## QA Scenarios

1. NN candidate produces probabilities near 0 and 1: the selected objective remains finite due to clipping and matches probability-space log loss.
2. XGBoost / LightGBM callback receives validation metrics over several boosting rounds: Optuna records each intermediate loss with the correct step.
3. A weak trial in either GBDT study crosses the configured median threshold: it is marked pruned rather than continuing to its estimator cap.
4. A competitive GBDT trial is not pruned: early stopping retains its normal control over the best iteration.
5. LightGBM draws rows according to the tuned fraction because bagging frequency is nonzero.
6. The selected GBDT artifact identifies its winning framework and comes only from the corresponding independent study.
7. The selected linear artifact identifies its winning algorithm and comes only from the corresponding independent study.
