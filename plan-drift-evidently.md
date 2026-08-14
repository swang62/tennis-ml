# Plan: Drift monitoring migration to Evidently (on-demand size-matched reference)

## Goal

Replace the hand-rolled drift computation in `src/flows/drift.py` with
**Evidently** (`evidently`, the industry-standard ML monitoring library), while
honoring one hard constraint: **never keep a full training snapshot, and never
recompute the champion's performance on the training set at drift time.**

Performance metrics are pinned on the champion at promotion (tags). The
distribution reference is **pulled on demand** from `gold.match_features` as a
window size-matched to the current new-match count, so the comparison is
apples-to-apples (same model, same data volume, adjacent time windows).

Output is a proper comparison (champion-at-promotion metrics vs current window,
plus reference-vs-current distribution drift) and a recommendation, with
feature-distribution drift, prediction drift, and target drift all checked
against thresholds.

## Current state

- `src/flows/drift.py` hand-rolls: `_compute_metrics` (8 sklearn metrics),
  `_psi` (`pd.qcut` on predictions), KS (`scipy.stats.ks_2samp`), and compares
  against a **self-referential moving baseline** (the previous drift run's
  `baseline.json`), not against the champion's promotion-time numbers.
- The champion model version stores lineage tags (`base_*`/`aux_*`) and
  `train_data_max_match_date`, but **no performance metrics**.
- OOF evidence is already persisted by 03: `data/processed/oof_evidence.parquet`
  (columns `linear`, `gbdt`, `nn`) and `oof_eval.parquet`
  (columns `match_id`, `match_won`) — see `plan-calibration.md`.
- `src/db/snapshot.py` already proves the DuckDB + `ATTACH … TYPE postgres`
  pattern (used by `just db-snapshot`), and `gold.match_features` is the
  up-to-date, deterministic source of historical feature rows.
- No drift library is installed (verified against `pyproject.toml`/`uv.lock`).

## Design decisions

1. **Performance metrics are pinned, never recomputed.** At promotion, 04 stamps
   the 8 gate metrics (`METRIC_NAMES`) plus `promotion_composite`, the eval
   split size, and the eval max date onto the champion model version as tags —
   the same mechanism already used for `train_data_max_match_date`. Drift reads
   them via `client.get_model_version(...).tags`; it never re-scores the
   training set.

2. **The distribution reference is pulled on demand and size-matched
   (apples-to-apples).** Drift pulls two adjacent windows from
   `gold.match_features` and scores **both** through the deployed champion
   Bento:
   - `reference` = the most recent `N` matches strictly before the cutoff
     (`match_date < train_data_max_match_date`),
   - `current` = the new matches after the cutoff (`match_date > cutoff`),
   where `N` matches the current count, floored and capped (e.g. min 50,
   max 2000). Equal-size, adjacent windows from the same model remove the
   "2 new matches vs 80k training rows" asymmetry that saturates metrics.
   Evidently is pure compute — it does **not** pull data — so the flow owns the
   pull (`execute_df` against gold, up-to-date after `run_dbt_build`) and hands
   Evidently two pandas frames. No frozen snapshot to maintain; only the cutoff
   (already a tag) and champion identity (already resolvable) are required.

3. **Evidently does the drift math + report.** Drift runs
   `Report(metrics=[DataDriftPreset(...), ...])` and a `TestSuite` with
   `reference_data=reference, current_data=current`, producing per-feature
   drift (PSI/KS/Wasserstein), prediction drift, target drift, and a JSON +
   HTML report.

4. **Recommendations come from thresholds** (below), mapped to
   `healthy` / `investigate` / `retrain`, printed prominently and logged as
   MLflow metrics/tags so a Prefect automation can notify on `retrain`.

## Contract after change

| Signal | Reference source | Method |
| --- | --- | --- |
| Performance (roc_auc, f1, …) | pinned tags | delta vs current-window metrics |
| Feature distribution (39 `FEATURE_COLS`) | on-demand pre-cutoff window (size-matched) | Evidently `DataDriftPreset` (PSI/KS/Wasserstein) |
| Prediction distribution (`p_win`) | pre-cutoff window `p_win` (same champion) | Evidently `DataDriftPreset` on prediction |
| Target / label rate (`match_won`) | pre-cutoff window labels | Evidently target drift + `calibration_rate` delta |

## Thresholds (reasonable defaults, single source of truth in one constants block)

| Metric | Threshold | Meaning |
| --- | --- | --- |
| per-feature PSI | < 0.1 / 0.1–0.2 / ≥ 0.2 | no drift / moderate / significant |
| drift share (share of drifted features) | ≥ 0.5 | feature drift |
| prediction `p_win` PSI | ≥ 0.2 | prediction distribution shift |
| `|Δcalibration_rate|` | > 0.05 | target shift |
| roc_auc drop vs pinned | > 0.05, only when n ≥ 30 | degradation (guards the n=2 artifact) |

### Recommendation mapping

- **retrain** — any of: drift share ≥ 0.5, prediction PSI ≥ 0.2,
  `|Δcalibration_rate|` > 0.05, or (n ≥ 30 and roc_auc drop > 0.05).
- **investigate** — no retrain trigger, but any feature PSI in 0.1–0.2.
- **healthy** — otherwise.

The verdict is written to the drift summary (`recommendation`,
`retrain_required`) and as a flow-run tag/result so a Prefect automation can
send a notification on `retrain`. (Notification block wiring is a follow-up;
this plan exposes the signal, it does not build the email/Slack block.)

## Tasks

### [ ] Task 1: Add Evidently to dependencies

- **Files**: `pyproject.toml` (`training` and `local` groups), `uv.lock`.
- Evidently runs in the drift flow (host worker), never in the serving image,
  so it goes in `training` (and transitively `local`), not `inference`.
- **Acceptance**: `uv sync` resolves; `import evidently` works in the worker env.

### [ ] Task 2: Pin performance metrics at promotion

- **Files**: `notebooks/parameters/04_evaluate.ipynb`.
- On promotion, after `compute_metrics`/`decide_promotion`:
  - Stamp the 8 `candidate_*` metrics + `promotion_composite` + eval split size
    + eval max date as champion model-version tags (mirror
    `TRAIN_DATA_MAX_DATE_KEY`). Tag keys are a single shared constant set (e.g.
    `METRIC_PREFIX = "metric_"`), so drift reads them uniformly.
- **Acceptance**: a promoted champion carries the metric tags; re-promoting
  refreshes them.
- **Guardrails**: do not change the promotion decision or lineage tags. The
  distribution reference is NOT frozen here — it is pulled on demand at drift
  time (Task 3).

### [ ] Task 3: Rewrite drift computation with Evidently

- **Files**: `src/flows/drift.py`, `src/constants.py` (thresholds + tag keys +
  window bounds).
- Keep the flow scaffolding (dbt build → champion validation → MLflow logging).
  Replace `_compute_metrics`/`_psi`/`_drift_summary` internals with:
  - Read the pinned metric tags from the champion version (Task 2).
  - Pull two size-matched windows from `gold.match_features` and score **both**
    through the champion Bento (`/api/predict_from_ids_bulk`, `{"rows": ...}`):
    `current` = matches after cutoff; `reference` = the most recent `N` matches
    before cutoff, `N` matched to the current count (floored/capped constants).
  - Build each frame as `FEATURE_COLS` + `match_won` + `p_win`.
  - Run `Report`/`TestSuite` with `reference_data=reference,
    current_data=current`, `ColumnMapping(target="match_won", prediction="p_win")`.
  - Emit JSON (for `drift_summary.json` + MLflow) and an HTML report artifact.
- **Acceptance**: drift produces an Evidently report with per-feature drift,
  prediction drift, target drift, and a comparison of current metrics vs pinned
  metrics; `artifacts/`/MLflow contain the JSON + HTML.
- **Guardrails**: keep the `_post_batch` `{"rows": ...}` envelope and the two
  route constants unchanged; keep `_validate_production` identity check.

### [ ] Task 4: Recommendation + Prefect signal

- **Files**: `src/flows/drift.py`, `src/constants.py`.
- Implement the threshold table above as a single constants block. Compute the
  verdict, add `recommendation`/`retrain_required` to the summary, log them as
  MLflow tags/metrics, and print a concise comparison table + verdict to stdout.
- **Acceptance**: the summary contains a human-readable recommendation; a
  `retrain_required=True` run is distinguishable by tags so a Prefect
  automation can notify on it.
- **Guardrails**: do not build the notification block itself (follow-up).

### [ ] Task 5: Hermetic tests

- **Files**: `tests/test_drift_monitor.py` (extend), new
  `tests/test_drift_recommendation.py`.
- Evidently runs on in-memory pandas frames — no live DB. Tests cover:
  - recommendation mapping for each threshold branch,
  - metric-tag parsing (champion tags → pinned metrics),
  - the windowing helper: `reference` = the most recent `N` pre-cutoff matches,
    `N` matched to the current count with the floor/cap applied.
- **Acceptance**: `uv run pytest` passes, including the `test_no_live_db.py`
  static guard (no `get_conn`/live DB in the new code).

## Dependencies

- Task 2 before Task 3 (the pinned metric tags must exist).
- Task 1 before Task 3 (evidently import).
- Task 4 depends on Task 3's report shape.
- Task 5 alongside 3–4.

## QA / testing scenarios

1. `just train` promotes a candidate; the champion version shows the 8
   `metric_*` tags.
2. `just drift` pulls the size-matched pre-cutoff reference + post-cutoff
   current windows, scores both through the champion, and prints a comparison
   table + a `recommendation` (healthy/investigate/retrain).
3. Synthetic drift: shifting a subset of features raises the drift share and
   the recommendation escalates to `retrain`; a shifted `p_win` distribution
   raises prediction PSI.
4. Hermetic tests pass with `DATABASE_URL` unset (no live DB).
5. The existing drift route contract (POST `{"rows": [...]}` to
   `/api/predict_from_ids_bulk`, GET `/api/model_info`) is unchanged.

## Out of scope

- The Prefect notification block (email/Slack) — this plan only emits the
  `retrain_required` signal for a later automation.
- `alibi-detect`/MMD detectors — Evidently's PSI/KS/Wasserstein is sufficient
  for the first version.
- Changing the serving image or the `/predict_from_ids*` schemas.
- Streaming/online drift (this remains a weekly batch check).
