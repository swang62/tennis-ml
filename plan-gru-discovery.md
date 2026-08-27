# Plan: GRU Discovery Notebook

## Goal

Create a standalone notebook that determines whether a compact GRU over each
player's ten causal prior matches can match the existing tabular NN's log loss.
It uses the existing full dataset and chronological train/validation/test split,
but does not affect the ensemble, MLflow registry, deployment, serving, or the
normal training pipeline.

## Scope

### In

- A new independently runnable tuning notebook, patterned after
  `notebooks/parameters/02_tune_nn.ipynb`.
- The exact existing chronological train/validation/test split and grouped,
  time-forward CV assignments; no newly sampled split or random CV.
- In-memory, causal historical-match tensors sourced from
  `silver.player_matches`.
- One shared GRU encoder for player and opponent histories, plus a small
  context-conditioned MLP scorer.
- Ten Optuna trials over a deliberately narrow search space.
- Validation and untouched-test log-loss comparison with the current tabular
  MLP.
- A local JSON results artifact containing configuration and metrics.

### Out

- No changes to `src/flows/pipeline.py`, `03_train_ensemble.ipynb`,
  `04_evaluate.ipynb`, MLflow registration, champion promotion, ONNX export,
  Bento service, or inference feature construction.
- No persistent sequence-cache format or database schema/dbt-model changes.
- No requirement that the GRU improve calibration, AUC, or ensemble score for
  this discovery decision; the comparison criterion is log loss.

## Fixed Model Contract

- **History length:** ten player-perspective rows, left-padded to length ten.
- **Causality:** training histories contain only rows ordered strictly before
  the target by `(match_date, match_num, match_id)`; same-day earlier matches
  are valid. The notebook has no serving-time behavior.
- **History storage:** materialize one `float32` history store in RAM, keyed by
  `(player_id, match_id)`, rather than duplicate player and opponent histories
  for every directional target row. Dataloader batches gather the two histories
  through integer row indices. Keep a boolean valid-step mask in RAM.
- **Raw timestep values (14):** match result; clay/grass/hard/carpet flags;
  log opponent ranking; ace-per-service-game; double-fault-per-service-point;
  first-serve-in percentage; first-serve-win percentage; second-serve-win
  percentage; return-points-won percentage; break-points-saved percentage;
  log gap since the preceding match; and log total service points.
- **Missingness:** use train-band statistics to impute missing raw values and
  append availability indicators for ranking, each rate group, match gap, and
  match volume. The exact history width is therefore the 14 values plus these
  explicit flags (expected roughly 22-23), not silently zero-filled values.
- **Current context (12):** surface one-hot (four), indoor flag, best-of,
  tournament level, round, Elo difference, H2H exposure, H2H advantage, and
  surface-H2H advantage. It excludes all profile, ranking, age, and manual
  rolling/difference inputs.
- **Model:** one shared one-layer GRU with hidden size 32 encodes each history;
  a 32-unit ReLU/dropout MLP scores `[player_embedding, opponent_embedding,
  context]`. The directional logit is
  `score(player, opponent, context) - score(opponent, player, context)` so
  swapped player/opponent inputs produce complementary probabilities exactly.
- **Training:** retain the existing Lightning BCE-with-logits loss, batch size,
  single-process dataloader default, train-band/chronological-validation
  selection split, early stopping, and Optuna pruning behavior. Reuse the
  existing match-grouped, time-forward fold assignments for OOF evaluation.
- **Search:** exactly ten trials. Keep GRU and MLP widths/layer counts fixed;
  tune only learning rate, weight decay, and dropout across narrow ranges:
  `3e-4..3e-3`, `1e-6..1e-4`, and `0.0..0.2`, respectively.

## Tasks

### [x] Task 1: Add reusable sequence-history preparation utilities

- **Description:** Add a focused training module that reads the needed columns
  from `silver.player_matches` in one ordered query, computes safe rate and log
  transforms, and creates a causal fixed-length history plus mask for every
  player-match row. Map the existing split's `(player_id, match_id)` rows to
  integer history-store indices and construct the selected 12-column context
  tensor from the existing `X_*` frames.
- **Files:** `src/training/gru_history.py` (new);
  `src/features/columns.py` (new GRU-specific column constants only if the
  constants improve one-source-of-truth clarity).
- **Acceptance Criteria:**
  - No per-target database queries or Python dataframe self-joins.
  - Every split row maps to both a player and opponent history index.
  - A target's own row never appears in either history.
  - Same-day order uses `match_num` then `match_id`.
  - Sequence tensors are `float32`, masks distinguish padding from a real
    zero-valued record, and all values are finite after fold-safe imputation.
  - The reusable store is built once per notebook execution and shared by all
    ten trials.
- **Guardrails:** Do not add a dbt model, a persisted cache, rolling aggregates,
  or a history query inside a Dataset `__getitem__` call.

### [x] Task 2: Add the compact symmetric GRU predictor

- **Description:** Add a Lightning module that accepts player histories,
  opponent histories, valid lengths/masks, and current context. Pack or select
  each sequence's final valid GRU state so left padding cannot influence the
  embedding. Reuse the same GRU for both sides, calculate two swapped scorer
  values, subtract them, and retain the existing weighted/unweighted BCE batch
  behavior.
- **Files:** `src/training/nn.py`.
- **Acceptance Criteria:**
  - Forward output is one logit per row with shape `[batch]`.
  - The module accepts empty histories without error and maps them to a learned
    or explicit zero-history representation without treating padding as matches.
  - Swapping player/opponent histories while retaining invariant context negates
    the logit within floating-point tolerance.
  - The GRU and scoring head both receive gradients from one BCE loss.
- **Guardrails:** Keep existing `TabularMLP` behavior and its public saved-model
  contract unchanged. Add the GRU model as a separate class in the same module;
  do not add player-ID embeddings, attention, multilayer recurrence, or separate
  player/opponent encoders.

### [x] Task 3: Create the standalone GRU discovery notebook

- **Description:** Add a notebook alongside `02_tune_nn.ipynb`. Load the same
  existing `X_train`, `X_val`, `X_test`, labels, info frames, and split/fold
  assignments as the tabular NN notebook. Build histories once, fit feature
  normalization only on each fit band, and run ten narrow Optuna trials on the
  existing chronological validation band. Then run the selected configuration
  through the same grouped time-forward CV to calculate OOF log loss, use the
  tabular NN's selection/refit protocol, and score the untouched test band.
  Print tabular-NN versus GRU validation, OOF, and test log loss in one
  comparison table.
- **Files:** `notebooks/parameters/02_tune_gru_discovery.ipynb` (new).
- **Acceptance Criteria:**
  - Runs directly without adding itself to `NB_ORDER`.
  - Executes exactly ten trials; fixed `hidden_dim=32` and one GRU layer.
  - Uses the full existing split data and exact match-grouped time-forward CV
    assignments, not a separate date range or random CV.
  - Never reads the test labels during hyperparameter selection, pruning, early
    stopping, or OOF fitting.
  - Writes `gru_discovery_results.json` under a notebook-local/discovery output
    directory with best parameters, validation/OOF/test log loss, baseline
    tabular-NN metrics when available, tensor shapes, and elapsed preprocessing,
    tuning, and OOF time.
  - Does not write `nn_oof.parquet`, `nn_test.parquet`, `nn_score.json`,
    `nn_model_version.json`, a scaler intended for serving, or any MLflow run.
- **Guardrails:** Do not modify the existing `02_tune_nn.ipynb`; this must be a
  peer experiment. Do not register, promote, or package the selected GRU.

### [x] Task 4: Add narrow behavioral coverage

- **Description:** Add hermetic tests for history causality/padding/missingness
  and the GRU's directional symmetry.
- **Files:** `tests/test_gru_history.py` (new);
  `tests/test_gru_discovery.py` (new).
- **Acceptance Criteria:**
  - A local fixture demonstrates that no current or future match can enter a
    target history, including same-day ordered matches.
  - A short-history fixture demonstrates correct padding and mask/length.
  - Missing raw stats produce finite values and availability indicators.
  - Swapped sequence sides negate the model logit.
  - Tests are hermetic: no database, network, MLflow, Prefect, or notebook
    execution.
-  - Do not modify `tests/test_nn.py` or `tests/test_pipeline.py`.
- **Guardrails:** Do not assert implementation-specific tensor internals,
  exact SQL, callback ordering, or constant values.

## Dependencies

1. Task 1 defines the data contract used by Tasks 2 and 3.
2. Task 2 defines the model and batch contract consumed by Task 3.
3. Task 4 verifies Tasks 1-3 and must be complete before interpreting notebook
   results.

## QA Scenarios

- Player with zero, one, and fewer than ten prior matches.
- Multiple matches for one player on the same date with different `match_num`.
- Missing rank, denominators, or raw serving fields.
- A pair of opposite directional rows for the same target match.
- Train/validation/test rows all map to player-match histories.
- The notebook performs ten trials, reproduces the existing time-forward CV,
  and reports validation/OOF/test log loss without creating production-model,
  ensemble, registry, ONNX, or serving artifacts.

## Decision Rule

Treat the GRU as comparable only if its untouched-test log loss is no worse
than the current tabular MLP under the same split. If it loses, retain the
tabular MLP and stop; no production integration follows from this notebook.
