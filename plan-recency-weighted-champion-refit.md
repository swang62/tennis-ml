# Plan: recency-weighted champion refit

## Goal

Redefine rolling `_10` features as a player's literal last ten matches, ordered
by `(match_date, match_num)`, then train candidates with eight-year exponential
recency weighting. Keep chronological selection and promotion metrics
unweighted. After promotion, refit only the winning configuration on all data
through `train_data_max_match_date` and assign that refit artifact `@champion`.

## Scope

### In

- Seed source `match_num` into bronze, then carry it through silver, gold, H2H,
  and date-only inference.
- Use same-date earlier match numbers as predecessors; use the final match on an
  `as_of_date` as the inference marker.
- One adjustable eight-year recency half-life for candidate and refit fitting.
- Promoted-candidate-only refit, with selection and refit lineage pinned.

### Out

- Changing the 96/2/2 split, weighting validation/test/promotion metrics,
  retuning during refit, adding model features, or changing serving APIs.

## Decisions

- `_10` is literal player-match recency, not date-strict previous-tournament
  form.
- A historical target sees earlier dates and smaller same-date `match_num`
  values, never itself or later matches. Missing match numbers retain
  date-strict behavior.
- H2H follows the same predecessor ordering and remains capped at five matches.
- Date-only inference already has the player ID, so it selects that player's
  latest recorded state through `as_of_date` with `(match_date DESC,
  match_num DESC)`. It needs no public `match_num` input.
- Gold rows remain pre-match to prevent leakage; inference is post-last-known-
  match state for the next prediction. Both use the same literal player-history
  ordering.
- Player history does not need tournament ID as a tie-breaker because a player
  cannot play multiple tournaments on one date.
- The recency half-life is eight years. Weights use the persisted full-snapshot
  cutoff, normalize to mean 1.0, and
  remain unweighted outside fit calls.
- Champion metrics stay the pre-refit selection metrics. Champion pins identify
  refit artifacts.

## Tasks

### [ ] Task 1: seed and persist source match sequence

- **Description:** Add nullable integer `match_num` to the PostgreSQL schema,
  dbt bronze source declaration, and canonical bronze contract. Map raw CSV
  `match_num` in `atp_rows_to_bronze()` and populate it in the scrape-to-raw
  path. Reseed bronze from the historical raw CSV corpus after migration so all
  source-backed rows carry the field.
- **Files:** `infra/postgres/schema.sql`, `dbt/models/sources.yml`,
  `src/features/columns.py`, `src/features/validate.py`, `src/db/ingest.py`,
  `src/db/seed.py`, `src/flows/matches.py`,
  `dbt/models/silver/player_matches.sql`.
- **Acceptance Criteria:** Raw `match_num` seeds to the bronze row as an integer,
  survives idempotent writes, and is available to silver after the reseed.
- **Guardrails:** Do not parse opaque `match_id` or alter physical-match identity
  or player orientation.

### [ ] Task 2: make rolling state literal last-ten history

- **Description:** Rework `silver.rolling_features` to compute trailing player
  windows in `(match_date, match_num)` order. For missing sequence values,
  preserve date-strict predecessor semantics rather than inventing same-day
  ordering.
- **Files:** `dbt/models/silver/rolling_features.sql`.
- **Acceptance Criteria:** Same-date matches 1, 2, 3 build post-match state in
  that order; match 11 retains only matches 2-11.
- **Guardrails:** Do not use `match_id` as a recency surrogate.

### [ ] Task 3: align gold and inference match ordering

- **Description:** Use date/match-number causal ordering in gold prior snapshot
  and H2H lookups. For date-only inference, select each requested player's
  latest post-match snapshot and the pair's recent meetings through the
  requested date with `(match_date DESC, match_num DESC)`.
- **Files:** `dbt/models/gold/match_features.sql`, `src/features/inference.py`.
- **Acceptance Criteria:** Later same-tournament matches see earlier numbered
  form and H2H. Training rows exclude their own and future match state.
  Inference uses the literal latest known player state, including the highest
  match number on the requested date, without a new API parameter.
- **Guardrails:** Preserve feature order, smoothing, recent-five H2H cap,
  cold-start behavior, and directional symmetry.

### [ ] Task 4: define the shared recency-weight contract

- **Files:** `src/constants.py`, new `src/training/recency.py`.
- **Description:** Add a named eight-year half-life and pure mean-normalized
  exponential date weights derived from the explicit full-snapshot cutoff.
- **Acceptance Criteria:** Weights are finite, positive, monotonic by age,
  deterministic, and identical for both orientations of a match.
- **Guardrails:** Never use wall-clock time or the similarity weighting helper.

### [ ] Task 5: weight every candidate base-model fit

- **Files:** `notebooks/parameters/02_tune_linear.ipynb`,
  `notebooks/parameters/02_tune_gbdt.ipynb`,
  `notebooks/parameters/02_tune_nn.ipynb`, `src/training/nn.py`.
- **Description:** Pass date-aligned weights to linear, GBDT, and NN trial,
  selected-candidate, and forward-CV OOF training fits. Use normalized weighted
  BCE for NN training only.
- **Acceptance Criteria:** All fitting paths receive aligned weights; NN
  validation loss and GBDT early-stopping validation remain unweighted.
- **Guardrails:** Do not change folds, validation/test scoring, or promotion.

### [ ] Task 6: weight the candidate stacker and calibration

- **Files:** `notebooks/parameters/03_train_ensemble.ipynb`.
- **Description:** Keep physical match dates alongside OOF evidence, derive one
  weight per physical match, and use it for stacker fitting and cross-fitted
  selection calibration.
- **Acceptance Criteria:** Dates and weights survive deterministic orientation
  selection; candidate test evidence remains unweighted and antisymmetric.
- **Guardrails:** Do not duplicate directional-row weights at this boundary.

### [ ] Task 7: pin selection recency lineage

- **Files:** `src/constants.py`, `notebooks/parameters/02_tune_linear.ipynb`,
  `notebooks/parameters/02_tune_gbdt.ipynb`,
  `notebooks/parameters/02_tune_nn.ipynb`,
  `notebooks/parameters/03_train_ensemble.ipynb`,
  `notebooks/parameters/04_evaluate.ipynb`.
- **Description:** Record half-life and full cutoff on candidate runs, manifests,
  selection records, refit artifacts, and champion versions.
- **Acceptance Criteria:** Champion lineage exposes half-life, cutoff, immutable
  selection metrics, and selection artifact IDs.
- **Guardrails:** Never replace selection metrics with refit metrics.

### [ ] Task 8: add promoted-candidate-only refit

- **Files:** new `notebooks/parameters/05_refit_promoted_candidate.ipynb`,
  `src/flows/pipeline.py`.
- **Description:** After a successful promotion decision, refit only selected
  base configurations, stacker, and calibration on all rows through the cutoff.
  Rebuild required OOF evidence without Optuna or losing families.
- **Acceptance Criteria:** A rejected candidate produces no refit artifact; a
  refit consumes frozen hyperparameters, GBDT rounds, and NN epochs.
- **Guardrails:** Do not reserve another validation slice or recalculate a test
  score for promotion.

### [ ] Task 9: register refit artifacts as champion

- **Files:** `notebooks/parameters/04_evaluate.ipynb`,
  `notebooks/parameters/05_refit_promoted_candidate.ipynb`,
  `src/constants.py`.
- **Description:** Defer alias movement until the refit and registration succeed,
  then pin refit bases, stacker, calibration, and immutable selection provenance
  before assigning `@champion`.
- **Acceptance Criteria:** Refit failure never moves `@champion`; deployment
  keeps resolving champion pins without MLflow at serving time.
- **Guardrails:** Preserve immutable artifact hashes and materialized model-only
  serving.

### [ ] Task 10: full ETL refresh, validation, and testing

- **Files:** `tests/test_ingest.py`, `tests/test_matches_csv.py`,
  `tests/test_matches_upsert.py`, `tests/test_inference_features.py`, new
  `tests/test_recency.py`, `tests/test_nn.py`, `tests/test_promotion.py`,
  `tests/test_pipeline.py`, `tests/test_deploy.py`, and relevant `dbt/tests/`
  files.
- **Description:** After all implementation, apply the idempotent migration,
  deliberately reseed bronze from raw CSV with `just seed --reset`, then run a
  non-incremental `just etl` full refresh. Add and run behavior tests for
  same-date ordering, literal trailing ten, H2H/inference recency, shared
  ordering semantics, weight behavior, refit lineage, and alias failure safety.
- **Acceptance Criteria:** `just migrate`, `just seed --reset`, `just etl`, dbt
  data tests, `just lint`, and `just test` pass.
- **Guardrails:** Tests stay hermetic and use no live DB, network, MLflow,
  DagsHub, Prefect, or notebook execution.

## Dependencies

1. Tasks 1-3 establish correct chronological features before recency-weighting.
2. Task 4 precedes Tasks 5-6.
3. Tasks 5-7 precede Task 8; Task 8 precedes Task 9.
4. Task 10 validates the complete integrated change at the end.

## QA scenarios

- Eleven same-date matches yield exactly the trailing ten by `match_num`.
- Match 4 sees matches 1-3 but not itself or later matches in rolling and H2H
  features.
- Date-only inference for a date selects that player's final post-match state by
  descending `match_num`, with no match-number API input.
- A missing sequence, if any remains after reseeding, uses strictly earlier
  dates only.
- A four-year-old row has half the relative pre-normalization weight of the
  cutoff-date row.
- Candidate fits are weighted while validation, test, and promotion remain
  unweighted.
- Rejected candidates do not alter the champion; accepted candidates expose
  pre-refit selection metrics and refit artifact pins.
