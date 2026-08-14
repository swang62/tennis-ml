# Plan: Post-promotion probability calibration (temperature scaling) + plots folder

## Goal

Calibrate the final ensemble `p_win` so it matches observed win frequency, by
applying a post-promotion **temperature scaling** (zero-intercept Platt on the
logit) fit on the OOF holdout set. The calibration is a serving-time artifact
applied to the stacker output; it never touches the promotion decision, never
changes hard predictions, and preserves the antisymmetry contract
(`p_win(A,B) + p_win(B,A) = 1`, `f(0.5) = 0.5`). Also: save every evaluation
figure to a flat `artifacts/plots/` folder (papermill does not keep notebook
cell outputs).

## Why temperature scaling (and not isotonic)

The stacker is a zero-intercept logistic regression over antisymmetric
evidence, so reversing a match negates every input and `p_win → 1 − p_win`.
A post-hoc calibrator must preserve that: `f(p) + f(1−p) = 1`, which forces
`f(0.5) = 0.5`. Plain isotonic regression has no such guarantee and would
break `test_service_symmetry.py` and the served complement property.

Temperature scaling `p' = sigmoid(t · logit(p))` satisfies all constraints:

- `logit(1−p) = −logit(p)` → complement preserved exactly
- `logit(0.5) = 0 → p' = 0.5` fixed point
- monotone in logit → AUC/accuracy/rankings unchanged; only Brier and
  calibration move
- one scalar `t`, fit on OOF by minimizing log loss

Caveat: one scalar fixes systematic over/underconfidence but not a
non-monotone bow. If the calibrated test reliability curve still bows badly,
the next rung is symmetric isotonic in logit space (more code, still
contract-safe). Try temperature first.

## Scope

In scope:

- Persist OOF evidence from 03 so 04 can fit the calibrator
- Fit temperature on OOF in 04, after the promotion decision
- Freeze `calibration_t.json` into `data/deploy/` on promotion; package + fingerprint it
- Apply temperature at the serving chokepoint (`_stack_evidence`)
- Reliability diagram shows raw vs calibrated candidate
- All evaluation figures saved to `artifacts/plots/` (flat, overwritten)
- Hermetic test for the calibrated symmetry contract

Out of scope:

- Isotonic / per-bucket calibration
- Changing the promotion gate metrics (they stay raw)
- Retraining the stacker or touching base models
- Changing `/predict` request/response schema

## Tasks

### [ ] Task 1: Add `PLOTS` constant

- **Description**: Add `PLOTS = ARTIFACTS / "plots"` to the shared constants so
  notebooks reference one path.
- **Files**: `src/constants.py`
- **Acceptance Criteria**: `PLOTS` exists and points at `artifacts/plots`.

### [ ] Task 2: Persist OOF evidence from 03

- **Description**: In `03_train_ensemble.ipynb`, the `kind == "oof"` branch
  currently keeps `oof_evidence`/`oof_eval` in memory only. Persist both,
  mirroring the existing test writes: `oof_evidence.parquet` (columns exactly
  `["linear", "gbdt", "nn"]`) and `oof_eval.parquet` (columns exactly
  `["match_id", "match_won"]`). Keep the existing column/identity validation.
- **Files**: `notebooks/parameters/03_train_ensemble.ipynb`
- **Acceptance Criteria**: Running 03 produces `data/processed/oof_evidence.parquet`
  and `data/processed/oof_eval.parquet` with the same one-row-per-match contract
  as the test files.
- **Guardrails**: Do not change the stacker fit, MLflow logging, or manifest.

### [ ] Task 3: Fit temperature on OOF and gate the calibrated candidate

- **Description**: In `04_evaluate.ipynb`, insert a calibration cell BEFORE the
  Metrics cell (gate evaluates the candidate as it will be served — calibrated;
  the decision is provably identical to raw since temperature is monotone and
  0.5-fixed, so Brier weight 0.00 is the only differing metric and it does not
  contribute to the composite). The cell must:
  - Load `oof_evidence.parquet` + `oof_eval.parquet`
  - Compute `p_oof = candidate.predict_proba(oof_evidence[STACK_ORDER])[:, 1]`
  - Fit `t` minimizing OOF log loss over `p' = sigmoid(t · logit(p_oof))` via
    `scipy.optimize.minimize_scalar`, bounded (e.g. `(0.05, 20)`); assert `t > 0`
  - Compute `y_proba_cal = sigmoid(t · logit(y_proba))` on test evidence
  - Compute the gate metrics (`cand_metrics`, `prod_metrics` where available)
    on **calibrated candidate** `y_proba_cal` (and calibrated incumbent
    `prod_proba` from the Bento), so the promotion gate compares served
    behavior; assert the composite is unchanged vs raw (sanity check only)
  - Write `data/processed/calibration_t.json` as `{"temperature": t}`
  - If `promoted`, copy it to `data/deploy/calibration_t.json`
  - Log `calibration_t` to the evaluation MLflow run
- **Files**: `notebooks/parameters/04_evaluate.ipynb`
- **Acceptance Criteria**: After the notebook runs, `calibration_t.json` exists;
  gate metrics are computed from `y_proba_cal`; the promotion decision equals
  the decision from raw metrics (verifiable via the sanity assert).
- **Guardrails**: `t` must be positive. Do not modify the registered model or
  its lineage tags.

### [ ] Task 4: Reliability diagram — calibrated candidate vs calibrated incumbent

- **Description**: In the existing calibration-curves cell of `04_evaluate.ipynb`,
  make the **final comparison calibrated vs calibrated** (Task 6 makes the live
  Bento return the incumbent's own calibrated p_win, so `prod_proba` needs NO
  local transform — applying the candidate's `t` to it would double-calibrate):
  - Calibrated candidate: `calibration_curve(y_eval, y_proba_cal)` where
    `y_proba_cal = sigmoid(t · logit(y_proba))` with the candidate's fresh `t`.
  - Calibrated incumbent: `prod_proba` from the Bento as-is (its own frozen `t`).
  - Raw candidate kept as a faint diagnostic line (shows the OOF fix), perfect
    diagonal, each curve labeled with its Brier.
  - Display-only calibrated candidate Brier computed here and logged to MLflow
    (gate Brier is weight 0.00; the figure is where the comparable Brier
    comparison lives).
  - If the production Bento is unavailable, plot only the candidate curves.
  - **Remove the pointless printed metrics-comparison table** from the Metrics
    cell (cell 8): the candidate-vs-production metric/delta print compares raw
    candidate against calibrated incumbent and adds no decision value. Keep the
    candidate-only headline diagnostics (ROC-AUC/Accuracy/etc. + classification
    report) and keep the `cand_metrics`/`prod_metrics` dict computations — the
    promotion gate reads them (now computed on calibrated candidate per Task 3).
    Removing only the print leaves the gate intact.
- **Files**: `notebooks/parameters/04_evaluate.ipynb`
- **Acceptance Criteria**: The figure shows calibrated candidate vs calibrated
  incumbent on the same scale (plus raw candidate for diagnostics) with Brier in
  the legend; `artifacts/plots/calibration_curves.png` is written (flat); the
  printed metric/delta table is gone from cell 8 while `cand_metrics` and
  `prod_metrics` are still computed and passed to `decide_promotion`.
- **Guardrails**: Do not transform `prod_proba` locally. Do not change the
  promotion gate or remove the `cand_metrics`/`prod_metrics` computation.

### [ ] Task 5: Save all evaluation figures to `artifacts/plots/`

- **Description**: In every figure cell of `04_evaluate.ipynb` (ROC, PR,
  calibration, confusion matrices, SHAP beeswarm + up to 3 waterfalls), add
  `PLOTS.mkdir(parents=True, exist_ok=True)` once and
  `fig.savefig(PLOTS / "<same name>.png")` next to the existing
  `mlflow.log_figure` calls. Use identical filenames as the MLflow figure paths
  (`roc_curves.png`, `pr_curves.png`, `calibration_curves.png`,
  `confusion_matrices.png`, `shap_beeswarm.png`,
  `shap_waterfall_misclassified_{i}.png`). Flat overwrite per run.
- **Files**: `notebooks/parameters/04_evaluate.ipynb`, `src/constants.py`
- **Acceptance Criteria**: All figure cells leave PNGs in `artifacts/plots/`;
  each run overwrites the previous images.
- **Guardrails**: No timestamped subdirectories; do not remove the MLflow
  logging of the same figures.

### [ ] Task 6: Apply temperature in serving

- **Description**: In `src/serving/service.py`:
  - `_stack_evidence` gains keyword-only `temperature: float = 1.0`; after
    computing `p_win`, apply `p_win = sigmoid(temperature * logit(p_win))` using
    the existing `src.evaluate.symmetry` helpers. Default 1.0 keeps existing
    callers and tests behavior-identical.
  - Service init reads `AUX_DIR / "calibration_t.json"` (default `1.0` when the
    file is absent) and stores it as `self.temperature`. Graceful fallback: if
    the file is present but malformed (unparsable JSON, missing/invalid
    `temperature`, or `temperature <= 0`), log a warning and fall back to `1.0`
    — never crash, never silently collapse to `0.5`.
  - The `_predict_proba` call site passes `self.temperature` into
    `_stack_evidence` — this is the single chokepoint used by both
    `predict_from_ids` and the bulk endpoint.
- **Files**: `src/serving/service.py`
- **Acceptance Criteria**: `p_win` returned by the API equals
  `sigmoid(t · logit(p_win_raw))`; with `t = 1.0` output is unchanged; a
  missing, malformed, or non-positive-temperature file falls back to `1.0`
  with a warning and serves normally.
- **Guardrails**: Do not calibrate the base probabilities (`p_linear`, `p_gbdt`,
  `p_nn`) — only `p_win`. Keep `_stack_evidence` pure (no module-level state).

### [ ] Task 7: Package and fingerprint the calibration artifact

- **Description**: In `src/flows/deploy.py`:
  - Add `CALIBRATION_FILE = DEPLOY_ARTIFACTS / "calibration_t.json"` and include
    it in `AUX_FILES` and `SOURCE_FINGERPRINT_FILES` (a retuned temperature must
    change the build fingerprint and trigger a rebuild).
  - Before `_check_aux_files()`, write a default `{"temperature": 1.0}` when the
    file is missing, so an existing champion promoted before this change still
    builds (no-op calibration).
- **Files**: `src/flows/deploy.py`, `bentofile.yaml` (add
  `data/deploy/calibration_t.json` to `include`)
- **Acceptance Criteria**: Deploy succeeds with and without the file present;
  fingerprint changes when the file content changes.
- **Guardrails**: Do not change the champion lineage tag schema.

### [ ] Task 8: Hermetic test for calibrated symmetry

- **Description**: In `tests/test_service_symmetry.py`, add a test that calls
  `_stack_evidence` with `temperature=2.0` (and `0.5`) on a toy stacker:
  reverse pairs still produce complementary `p_win` (`|p + p_rev − 1| < 1e-6`),
  and `p=0.5` maps to `0.5`. Pure numpy/sklearn, no DB/MLflow/Bento — passes the
  `test_no_live_db.py` guard.
- **Files**: `tests/test_service_symmetry.py`
- **Acceptance Criteria**: New test passes; existing symmetry tests still pass
  with the default `temperature=1.0`.
- **Guardrails**: No live database or network access.

## Dependencies

- Task 2 before Task 3 (03 must persist OOF evidence).
- Task 3 before Task 4/5 (calibration fit, `y_proba_cal`, and
  `calibration_t.json` must exist before the gate and report cells run).
- Task 6 before Task 8 (test exercises the new parameter).
- Task 7 independent of 3–6 but must ship with them (serving reads the file).

## QA/Testing scenarios

1. `just train` runs 00–04 end to end; `data/processed/calibration_t.json`
   exists; promotion decision equals the decision from a no-calibration run on
   the same candidate (sanity assert in Task 3 passes), while gate metrics are
   computed on the calibrated candidate.
2. `artifacts/plots/*.png` present after training, overwritten on the next run.
3. `uv run pytest` passes — including the new temperature symmetry test and the
   `test_no_live_db.py` static guard.
4. Local serve (`just dev`): `/predict_from_ids` returns
   `p_win = sigmoid(t·logit(raw))`; with `calibration_t.json` deleted, API
   returns the raw `p_win` (t = 1.0) — proving the default path.
5. Reliability diagram (Task 4): calibrated candidate vs calibrated incumbent
   curves on the same scale; the calibrated candidate hugs the diagonal closer
   than the raw diagnostic line. If it does not (non-monotone bow), flag for
   symmetric-isotonic follow-up, do not ship a cosmetic fix.
6. Metrics table (cell 8): the printed candidate-vs-production metric/delta
   table is removed (raw candidate vs calibrated incumbent was not
   apples-to-apples and added no decision value). Candidate-only headline
   diagnostics and the `cand_metrics`/`prod_metrics` dicts remain — the latter
   still drive `decide_promotion`. The comparable Brier comparison lives in the
   Task 4 figure.
