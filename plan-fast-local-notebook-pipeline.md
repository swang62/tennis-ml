# Plan: Fast local notebook pipeline with test and production profiles

## Goal

Reduce local end-to-end notebook pipeline startup and tuning overhead while preserving the current artifacts and model behavior. The local runner will reuse one Jupyter kernel, the smoke profile will use cheaper NN settings, plotting imports will be deferred, and bio embeddings will be limited to players present in the match dataset.

All notebook parameters, tuning ranges, model constants, and test/production runtime choices will live in one configuration module. Switching profiles will require commenting one assignment and uncommenting the adjacent assignment.

## Scope

### In scope

- Shared-kernel execution for `src/flows/pipeline.py` only.
- One central test/production parameter source used by both local and Prefect runners.
- Explicit test and production values for all notebook parameters and tunable search-space constants.
- Faster NN smoke tuning with production values retained.
- Conditional Optuna pruning so smoke runs avoid importing the Lightning integration.
- Lazy plotting imports and a profile-controlled plotting switch.
- Bio embeddings restricted to players referenced by `gold.match_features`.
- Process-local caching of FastEmbed model instances.
- Per-notebook and total runtime reporting.

### Out of scope

- Changing `src/flows/training.py` from Papermill to shared-kernel execution.
- Parallel notebook execution.
- Persisting a warm kernel between separate pipeline invocations.
- Changing model feature columns, output schemas, MLflow experiment names, ensemble behavior, or promotion rules.
- Adding dependencies; `nbclient`, `nbformat`, and `jupyter_client` are already installed.
- Guaranteeing a sub-30-second run on every machine or cold model cache. Runtime improvement is measured best effort.

## Profile values

The implementation should make these values explicit in the central configuration. Existing CLI/environment overrides remain higher priority than the selected profile.

### Test profile — active by default

- GBDT: 10 trials, 3 CV folds, estimator/iteration range 100–300.
- Linear: 10 trials, 3 CV folds.
- NN: 5 trials, 10 epochs, patience 2, batch size 256, 3 CV folds, Optuna Lightning pruning disabled.
- Ensemble split: 3 CV folds.
- Evaluation plots: disabled.
- Existing test split, seed, paths, experiment names, model names, and search ranges remain unchanged unless explicitly listed above.

### Production profile — present but inactive

- GBDT: 100 trials, 5 CV folds, estimator/iteration range 100–2000.
- Linear: 50 trials, 5 CV folds.
- NN: 50 trials, 100 epochs, patience 10, batch size 128, 5 CV folds, Optuna Lightning pruning enabled.
- Ensemble split: 5 CV folds.
- Evaluation plots: enabled.
- Existing production-compatible paths, seeds, experiment names, model names, and full search ranges remain explicit.

The switch should be visually obvious:

```python
NOTEBOOK_PARAMS = TEST_NOTEBOOK_PARAMS
# NOTEBOOK_PARAMS = PRODUCTION_NOTEBOOK_PARAMS
```

Switching to production means commenting the first assignment and uncommenting the second; no notebook edits are required.

## Tasks

### [ ] Task 1: Centralize notebook parameters and tuning constants

- **Description**: Add one configuration module containing complete test and production parameter maps keyed by notebook filename. Move active runtime settings and Optuna search-space constants out of notebook bodies into these maps. Keep the maps explicit rather than adding a configuration framework.
- **Files**:
  - `src/flows/notebook_profiles.py` (new)
  - `src/flows/pipeline.py`
  - `src/flows/training.py`
  - `notebooks/parameters/00_embeddings.ipynb`
  - `notebooks/parameters/01_train_test_split.ipynb`
  - `notebooks/parameters/02_tune_gbdt.ipynb`
  - `notebooks/parameters/02_tune_linear.ipynb`
  - `notebooks/parameters/02_tune_nn.ipynb`
  - `notebooks/parameters/03_ensemble_split.ipynb`
  - `notebooks/parameters/04_ensemble_stack.ipynb`
  - `notebooks/parameters/05_evaluate.ipynb`
- **Implementation details**:
  - Define `TEST_NOTEBOOK_PARAMS`, `PRODUCTION_NOTEBOOK_PARAMS`, and the adjacent comment/uncomment `NOTEBOOK_PARAMS` assignment.
  - Store all current paths, table names, split sizes, seeds, experiment names, model names, CV folds, trial counts, training limits, plotting flags, and embedding model/table inputs in both profiles.
  - Store GBDT model choices and numeric search bounds, including estimators/iterations, depth, leaves, learning rate, sampling fractions, and regularization.
  - Store linear model choices and numeric/categorical search bounds, including `C`, penalties, kernels, gamma, variance smoothing, and maximum iterations.
  - Store NN architecture/search constants, including hidden-dimension bounds, fusion width, dropout bounds, learning-rate bounds, dataloader workers, epochs, patience, batch size, CV folds, and pruning enablement.
  - Make `pipeline.py` and `training.py` import the same selected map. `training.py` retains Papermill execution unchanged.
  - Preserve `--params` and `PIPELINE_PARAMS` as final per-notebook overrides in the local runner.
  - Make standalone notebook parameter cells source their defaults from the central test profile, while remaining compatible with Papermill-injected assignments.
  - Add a small startup validation that both profile maps have exactly the notebook keys expected by `NB_ORDER` and each required tuning parameter is present.
- **Acceptance Criteria**:
  - One adjacent comment/uncomment change selects test or production settings.
  - No active Optuna search bound, NN architecture constant, CV count, trial count, epoch/patience/batch setting, or plotting toggle remains hidden in notebook logic.
  - Local and Prefect runners use the same selected parameter source.
  - Existing JSON/environment overrides still win over profile values.
- **Guardrails**:
  - Do not introduce YAML, Pydantic, environment-variable matrices, or a configuration dependency.
  - Do not change model candidate families or feature columns while parameterizing their existing settings.

### [ ] Task 2: Reuse one kernel in the local pipeline runner

- **Description**: Replace per-notebook Papermill execution in the local runner with nbclient execution against one externally owned kernel. Preserve parameter injection and timestamped executed-notebook outputs.
- **Files**:
  - `src/flows/pipeline.py`
- **Implementation details**:
  - Keep `ensure_kernel()` and the repo-local `tennis-ml` kernelspec contract.
  - Start one `jupyter_client.KernelManager` before iterating through `NB_ORDER`.
  - Load each notebook with `nbformat`, replace its first `parameters`-tagged cell with valid Python assignments generated from the merged profile/override map, and execute it using `NotebookClient(notebook, km=shared_kernel_manager)`.
  - Support the current parameter value types: strings, numbers, booleans, `None`, lists, and nested dictionaries.
  - Write each executed notebook to the existing timestamped path under `artifacts/notebooks/`.
  - Time and print each notebook plus the total pipeline duration.
  - If a notebook fails, shut down the shared kernel, start a clean one, retry that notebook once, and then fail normally if the retry also fails.
  - Always shut down the kernel in a `finally` path.
- **Acceptance Criteria**:
  - One successful local pipeline invocation starts one kernel in the normal path.
  - All notebooks still execute in `NB_ORDER` and produce timestamped output notebooks.
  - The injected values visible in executed notebook parameter cells match the active profile plus any overrides.
  - A failed/terminated run does not leave the runner-owned kernel alive.
  - `src/flows/training.py` continues using Papermill.
- **Guardrails**:
  - Do not reset the kernel namespace between successful notebooks, because that would discard cached imports.
  - Do not make notebook-to-notebook Python variables part of the data contract; notebooks must continue reading their required artifacts from disk.
  - Do not implement a custom Papermill engine when nbclient's supported external `KernelManager` path is sufficient.

### [ ] Task 3: Apply the fast NN smoke profile and conditional pruning

- **Description**: Reduce Lightning/Optuna overhead in test mode while retaining full production values in the central profile.
- **Files**:
  - `src/flows/notebook_profiles.py`
  - `notebooks/parameters/02_tune_nn.ipynb`
- **Implementation details**:
  - Use the test/production values specified above.
  - Add an `enable_pruning` parameter.
  - Import `PyTorchLightningPruningCallback` only inside the enabled branch so the smoke profile avoids the integration import and callback overhead.
  - Build the trainer callback list from EarlyStopping plus the optional pruning callback.
  - Parameterize the existing two-branch tabular/bio MLP search ranges and fusion width without changing the architecture.
  - Preserve `nn_oof.parquet`, `nn_test.parquet`, and `nn_score.json` contracts.
- **Acceptance Criteria**:
  - Test execution uses 5 trials, at most 10 epochs, patience 2, batch size 256, 3 folds, and no pruning callback.
  - Production execution uses the listed production values and enables pruning.
  - The NN continues consuming the same tabular columns as the other models plus canonical player/opponent bio embeddings.
  - OOF and test prediction row counts remain aligned with the feature artifacts.
- **Guardrails**:
  - Do not remove early stopping.
  - Do not alter the canonical player/opponent ordering or reintroduce sequence models.

### [ ] Task 4: Defer evaluation plotting imports

- **Description**: Avoid importing matplotlib and seaborn in smoke runs unless plots are requested.
- **Files**:
  - `src/flows/notebook_profiles.py`
  - `notebooks/parameters/05_evaluate.ipynb`
- **Implementation details**:
  - Add `render_plots=False` to the test profile and `render_plots=True` to production.
  - Move matplotlib and seaborn imports into the plotting cell and execute plotting code only when `render_plots` is true.
  - Keep metric calculation, MLflow comparison, and model promotion independent of plotting.
- **Acceptance Criteria**:
  - Test runs do not import matplotlib or seaborn.
  - Production runs still generate the same evaluation plots.
  - Evaluation metrics and promotion decisions are identical whether plotting is enabled or disabled.
- **Guardrails**:
  - Do not skip evaluation or validation metrics in test mode.

### [ ] Task 5: Embed only players used by match features and cache the model

- **Description**: Limit embedding work to relevant player IDs and avoid reconstructing the same FastEmbed model more than once per process/model name.
- **Files**:
  - `src/flows/notebook_profiles.py`
  - `notebooks/parameters/00_embeddings.ipynb`
  - `src/models/similarity.py`
- **Implementation details**:
  - Add the canonical match feature table name to the embeddings notebook parameters.
  - Query distinct `player_id` and `opponent_id` values from the match feature table and retrieve only matching rows from `gold.player_profiles`.
  - Preserve one embedding row per relevant player ID and existing empty-summary behavior.
  - Add a standard-library, process-local cache keyed by embedding model name around `TextEmbedding` construction; keep embedding outputs unchanged.
  - Update the function documentation so it no longer claims implementation-level purity while retaining deterministic output semantics.
- **Acceptance Criteria**:
  - `bio_embeddings.parquet` contains exactly the distinct non-empty player IDs referenced by the current match feature table that exist in `gold.player_profiles`.
  - Duplicate match appearances do not produce duplicate embedding rows.
  - Missing summaries still produce the existing empty-string embedding.
  - Repeated embedding calls with the same model in one process reuse one `TextEmbedding` instance.
  - NN handling for players without profile rows remains unchanged (zero-filled embedding lookup).
- **Guardrails**:
  - Do not store embeddings in DuckDB.
  - Do not change the embedding model or vector dimensions between profiles unless explicitly configured later.
  - Do not remove the shared embedding path used by `PlayerSimilarity`.

### [ ] Task 6: Verify behavior, isolation, and runtime

- **Description**: Validate the central profiles, shared-kernel execution, artifact contracts, and measured speed without claiming an unmeasured target.
- **Files**:
  - `src/flows/pipeline.py`
  - `src/flows/notebook_profiles.py`
  - generated files under `artifacts/` and `data/processed/` (not committed)
- **Checks**:
  - Run focused lint/type checks on touched Python files.
  - Validate every notebook parses as JSON and retains exactly one `parameters`-tagged cell.
  - Run `just pipeline` with the test profile and record per-notebook plus total wall time.
  - Confirm all eight notebooks complete in order.
  - Confirm expected files exist and row counts align: training/test labels, each model's OOF/test predictions, assembled ensemble inputs, NN score JSON, and final evaluation outputs.
  - Confirm the executed notebooks record test values, especially NN trial/epoch/pruning settings and `render_plots=False`.
  - Run the pipeline a second time in the same configuration to detect accidental reliance on stale shared-kernel variables and report warm-cache timing separately.
  - Temporarily select the production profile only for a parameter-injection dry check; do not run full production tuning as part of routine verification.
- **Acceptance Criteria**:
  - The test pipeline completes end to end with no artifact-contract regressions.
  - Both consecutive runs produce valid outputs and the same row alignment.
  - Runtime reporting distinguishes cold model-cache and warm-cache conditions.
  - Actual runtime and any remaining bottleneck are reported; sub-30 seconds is treated as a measured outcome, not guaranteed.
- **Guardrails**:
  - Do not commit generated notebooks, model files, MLflow data, or processed artifacts.
  - Do not weaken feature validation, evaluation metrics, or promotion safeguards to improve timing.

## Dependencies

1. Task 1 must precede Tasks 3–5 because those tasks consume the centralized profile values.
2. Task 2 can be implemented after Task 1 and must precede runtime measurement.
3. Tasks 3–5 are independent after Task 1.
4. Task 6 runs after all implementation tasks.

## QA / Testing Scenarios

1. **Default test profile**: run with no overrides; verify smoke NN settings, plots disabled, all artifacts produced.
2. **CLI override**: override one notebook parameter via `--params`; verify it supersedes the selected profile in the executed notebook.
3. **Environment override**: repeat with `PIPELINE_PARAMS`; verify precedence remains correct.
4. **Production injection check**: switch the two adjacent assignments and inspect merged values without launching full production tuning.
5. **Relevant-player embeddings**: include repeated player IDs and a profile with a missing summary; verify deduplication and empty-summary handling.
6. **Missing profile**: reference a player absent from `gold.player_profiles`; verify NN lookup retains its existing zero-vector fallback.
7. **Kernel retry**: introduce a controlled failing notebook/cell in a temporary copy; verify one clean-kernel retry and final shutdown.
8. **Shared-state isolation**: execute the pipeline twice and verify downstream artifacts do not depend on variables left by prior notebooks.
9. **Plotting profiles**: verify smoke mode omits plotting imports while a production-profile dry execution reaches the plotting branch when enabled.
10. **Runtime comparison**: compare the new measured total against the recorded 72.76-second baseline and identify the remaining slowest notebook.
