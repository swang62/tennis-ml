# Plan: symmetric matchup model

## Goal

Replace lower-ATP-ID canonical model orientation with a symmetric player-perspective training and serving contract. Every request must return `p_win = P(supplied player_id wins)`, and reversing the same request must return exactly the complementary probability for each base model and the ensemble.

Repair the identified training-pipeline integrity problems, add the missing return-strength signal and low-sample safeguards, remove frontend probability remapping, then retrain from scratch. Existing registry artifacts are incompatible and must not be deployed under this contract.

## Scope

### In

- Two directional gold rows per physical match, keyed by `(match_id, player_id)` and grouped by immutable `match_id` throughout splitting, OOF, evaluation, and artifacts.
- Direction-preserving inference features: supplied player is `player_*`; supplied opponent is `opponent_*`.
- Exact base and ensemble antisymmetry through two-direction scoring and logit-space projection.
- Symmetric zero-intercept logistic stacker over base antisymmetric evidence.
- Training split/OOF/NN test leakage fixes and artifact alignment/lineage verification.
- Feature-contract updates: return-strength differential, fixed empirical-Bayes smoothing with exposure features, H2H exposure plus smoothed directional advantage, same-day parity repair, carpet fallback repair, aggregate-contract coverage, and binary indoor validation.
- Frontend removal of canonical probability remapping; H2H display orientation remains a presentation-only concern unless its API is also changed.
- Hermetic regression tests and static dbt/snapshot contract tests.

### Out

- A weak/inverted-base-model candidate-promotion gate.
- Time-safe replacement of full-history fallback defaults or current biography embeddings; their historical leakage is accepted by design.
- A new rank-as-of data architecture, time-versioned bios, new model families, or pairwise-ranking replacement.
- Backward compatibility for canonical lower-ID registered models.

## Fixed Design Decisions

- Keep signed differences, paired side-specific state features, and invariant match context. Do not use difference-only features.
- `match_id` identifies one physical match. Mirrored rows share it and must never cross a split or fold.
- Each physical training match has total weight `1`: each of its two orientations uses weight `0.5` where supported.
- Each base scores both directions at inference and OOF generation. For clipped probabilities `p_ab` and `p_ba`, use:

  `evidence_ab = (logit(p_ab) - logit(p_ba)) / 2`

  `symmetric_p_ab = sigmoid(evidence_ab)`

  Use symmetric clipping at `1e-6`, reject NaN/infinite predictions, and centralize these operations so training and serving use identical code.
- Train the meta-model on named base evidence columns in fixed order `linear`, `gbdt`, `nn`, with `LogisticRegression(fit_intercept=False)`. A reversed request negates every input and therefore returns the exact complement.
- API returns only the supplied orientation. It may internally canonicalize an unordered pair solely to share H2H lookup/work, but must not alter model-side player order.
- Use fixed empirical-Bayes smoothing with explicit opportunity/exposure features: neutral 50% Beta(1,1) prior, `(successes + 1) / (opportunities + 2)`. Do not tune this constant against the tiny current corpus.
- Replace raw `player_h2h_matches` / `player_h2h_wins` features with `h2h_exposure` and `h2h_advantage = (wins + 1) / (meetings + 2) - 0.5`.

## Tasks

### [ ] Task 1: Define the new directional feature and row-identity contract

- **Description**: Update the shared feature definitions to replace canonical H2H fields, add return-strength differential and rate-exposure fields, and define a single immutable row identity contract: `match_id` is the group; `(match_id, player_id)` identifies a directional row. Document each feature’s swap behavior: signed features negate, paired features exchange, invariant context remains equal.
- **Files**:
  - `src/features/columns.py`
  - `dbt/models/gold/match_features.yml`
  - `tests/test_rolling_contract.py`
  - `src/db/snapshot.py`
  - `tests/test_snapshot.py`
- **Acceptance criteria**:
  - Feature order is explicit and shared by training, snapshot validation, inference, and serving.
  - Snapshot accepts exactly two rows per `match_id`, rejects duplicate `(match_id, player_id)`, and checks two distinct opponents/labels per group.
  - No feature name or documentation describes a model side as canonical/lower-ID.
- **Guardrails**: Do not add current-match statistics; every added feature remains strictly pre-match and has an inference equivalent.

### [ ] Task 2: Produce two gold player-perspective rows per match

- **Description**: Change `gold.match_features` from one lower-ID row to both enriched player perspectives. Replace the gold incremental/primary key and dbt tests accordingly. Compute feature pairs in requested/player perspective, including complementary H2H advantage. Ensure each side is fully imputed before differences are calculated.
- **Files**:
  - `dbt/models/gold/match_features.sql`
  - `dbt/models/gold/match_features.yml`
  - `dbt/dbt_project.yml`
  - `dbt/tests/gold/match_features_keeps_all_bronze_matches.sql`
  - `dbt/tests/gold/player_matches_two_rows_per_match.sql`
  - `tests/test_dbt_incremental.py`
  - `tests/fixtures/incremental_demo.sql`
- **Acceptance criteria**:
  - Every bronze match yields exactly two gold rows, one per player, with complementary labels.
  - Gold’s unique key and physical DB primary key are `(match_id, player_id)`.
  - Mirrored rows share feature context, exchange side fields, negate directional differences, and carry complementary directional H2H values.
  - Incremental reruns remain idempotent and changed/new rows do not leave only one perspective materialized.
- **Guardrails**: Preserve `silver.player_matches` as the existing two-perspective source; do not duplicate it again upstream.

### [ ] Task 3: Repair rolling feature derivation and fallbacks

- **Description**: Add return-strength and rate-exposure calculations to silver rolling features. Apply fixed smoothing consistently for sparse rates; carry needed exposure columns to gold. Make gold’s prior snapshot join strictly before `match_date`, matching inference. Give carpet surface form a neutral/overall fallback rather than a constant zero. Correct tour-average contract checks so every fallback column is validated.
- **Files**:
  - `dbt/models/silver/rolling_features.sql`
  - `dbt/models/silver/rolling_features.yml`
  - `dbt/models/gold/match_features.sql`
  - `dbt/models/gold/tour_averages.sql`
  - `dbt/tests/gold/tour_averages_contract.sql`
  - `dbt/tests/gold/tour_averages_rate_bounds.sql`
  - relevant existing dbt model tests under `dbt/tests/gold/` and `dbt/tests/silver/`
- **Acceptance criteria**:
  - `return_points_won_pct_diff` is train/serve parity-tested and based only on prior matches.
  - Sparse rate features use `(successes + 1) / (opportunities + 2)` and expose their history/opportunity counts.
  - Same-date matches cannot supply training snapshots unavailable to inference.
  - Carpet produces finite, non-degenerate surface-form values.
  - Aggregate tests fail for invalid values in any fallback column, not only the first checked column.
- **Guardrails**: Do not redesign full-history fallback/default architecture; it is intentionally out of scope.

### [ ] Task 4: Make inference directional and paired

- **Description**: Remove lower-ID sorting from feature orientation. Build one row for supplied `player_id` versus supplied `opponent_id`; build the reverse row from the same context for paired scoring. Keep unordered pair normalization only in H2H SQL lookup and convert its canonical winner count to the requested player’s direction. Apply the same behavior to scalar and bulk builders. Validate `is_indoor` as exactly `0` or `1` when supplied.
- **Files**:
  - `src/features/inference.py`
  - `tests/test_inference_features.py`
  - `tests/test_inference_units.py`
- **Acceptance criteria**:
  - `build_inference_features(A, B)` returns IDs `(A, B)`; reverse input returns `(B, A)`.
  - Swapping inputs exchanges all side-paired fields, negates signed fields, preserves context, complements H2H advantage, and leaves no NaN/inf values.
  - Scalar and bulk inference agree for both orientations.
  - Historical train/inference parity covers both gold perspectives of one match.
- **Guardrails**: Preserve existing parameter validation and as-of-date semantics.

### [ ] Task 5: Update snapshot and chronological split artifacts for groups

- **Description**: Change snapshot assumptions from one row per `match_id` to directional row identity. Have notebook 01 preserve `match_id` and directional row IDs in all feature/label/info artifacts, order deterministically by date/match/player, and make chronological partitions at physical-match granularity.
- **Files**:
  - `src/db/snapshot.py`
  - `tests/test_snapshot.py`
  - `notebooks/parameters/01_train_test_split.ipynb`
  - `notebooks/parameters/01_train_test_split.ipynb` parameter metadata as applicable
- **Acceptance criteria**:
  - Both orientations of every `match_id` are always in one top-level band.
  - Artifacts retain `match_id` and a directional-row identifier for alignment without treating metadata as model input.
  - No physical match appears in more than one of train, validation, or test.
  - Split metadata records match counts and directional-row counts.
- **Guardrails**: Keep the test split untouched by tuning, early stopping, feature selection, and calibration.

### [ ] Task 6: Correct base-model tuning, grouped OOF, and exact model lineage

- **Description**: Refactor each tuning notebook around a shared persisted fold assignment keyed by `match_id`. Fit linear/NN scalers inside every OOF fold. Repair NN validation to use `X_val/y_val`, repair its test predictions to use test tensors, and prevent any test use during Optuna/pruning/early stopping. Generate and persist paired directional base OOF/test probabilities with row IDs and fold IDs. Ensure registered base pins identify the exact fitted artifacts that produced those predictions; remove/replace any old-version retention behavior that violates this lineage.
- **Files**:
  - `notebooks/parameters/02_tune_linear.ipynb`
  - `notebooks/parameters/02_tune_gbdt.ipynb`
  - `notebooks/parameters/02_tune_nn.ipynb`
  - `src/models/nn.py`
  - `src/flows/pipeline.py`
  - any existing shared notebook helper/module identified during implementation
- **Acceptance criteria**:
  - Every OOF validation match is absent from that fold’s fit rows in both directions.
  - The same persisted fold map is used by linear, GBDT, and NN; artifacts assert the exact row/fold IDs match.
  - Linear and NN OOF scalers fit only their fold training rows.
  - NN Optuna, early stopping, and version evaluation use validation data, never test data.
  - `nn_test` predicts actual test tensors; lengths and row IDs match `info_test`.
  - Each base’s OOF/test artifact records model run/version/hash used to generate it, and those exact pins become ensemble lineage.
- **Guardrails**: Do not add a quality/promotion guard for weak or inverted base AUC; user explicitly excluded it.

### [ ] Task 7: Centralize exact antisymmetry and rebuild ensemble stacking

- **Description**: Add a shared numerical helper for finite-probability validation, symmetric clipping, logit/sigmoid conversion, and paired antisymmetric evidence. Use it when converting each base’s directional OOF and test predictions. Assemble named DataFrames, validate row/group/fold/model lineage, train a no-intercept named stacker, and log both coefficients and intercept (asserted zero/no-intercept) as model metadata.
- **Files**:
  - new focused module under `src/models/` or `src/evaluate/` for symmetric score projection
  - `notebooks/parameters/03_ensemble_split.ipynb`
  - `notebooks/parameters/04_ensemble_stack.ipynb`
  - `notebooks/parameters/05_evaluate.ipynb`
  - `src/evaluate/promotion.py`
  - focused new hermetic tests under `tests/`
- **Acceptance criteria**:
  - Stack inputs are read by names and explicitly reordered to `linear`, `gbdt`, `nn`; missing, extra, duplicate, or reordered columns fail clearly.
  - The stacker fits one physical-match orientation per group using antisymmetric base evidence and no intercept.
  - Evaluation uses one physical-match observation per match or documented 0.5 weights, never two independent mirrors.
  - Candidate/production comparison accepts requested orientation rather than sorted IDs.
  - The exact base pins that generated OOF/test predictions are copied to the promoted ensemble tags.
- **Guardrails**: Do not average raw `P(A|B)` and `1-P(B|A)` as the primary score path. Keep it only as a measured alternative if future calibration evidence requires it.

### [ ] Task 8: Score both directions in Bento and simplify response semantics

- **Description**: Update serving to build and batch both orientation rows per request; run each base for both, antisymmetrize each base score, stack the three evidence values, and return the supplied player’s result. Reverse calls must be constructed through the same pair calculation and be complementary. Update service diagnostics to call sides requested/directional, not canonical.
- **Files**:
  - `src/serving/service.py`
  - `src/features/inference.py`
  - `tests/test_service_*.py` relevant to prediction
  - new focused serving symmetry test(s)
- **Acceptance criteria**:
  - For every finite request, all four outputs (`p_linear`, `p_gbdt`, `p_nn`, `p_win`) satisfy reverse-request sum `= 1` within the stated API precision contract.
  - `player_id` and `opponent_id` in the response exactly match request order.
  - `predicted_winner` derives from the requested orientation’s probability.
  - Bulk prediction preserves input order and provides the same scalar-equivalent directional result.
- **Guardrails**: Do not expose canonical IDs or frontend-only remapping fields in the public prediction response.

### [ ] Task 9: Remove frontend lower-ID probability hacks

- **Description**: Treat backend prediction values as Player A / requested player directly. Remove `probabilityForPlayer` and canonical probability edge remapping; calculate Player A/Player B bars and odds directly from `p` and `1-p`. Keep `orientH2H` only if `/head_to_head` remains unordered/canonical; it is independent from model prediction orientation.
- **Files**:
  - `web/src/pages/H2H.tsx`
  - `web/src/lib/h2hOrientation.ts`
  - `web/tests/h2hOrientation.test.mjs`
  - `web/src/api.ts` if response comments/types mention canonical behavior
- **Acceptance criteria**:
  - Player A’s displayed probability is exactly `pred.p_win`.
  - Player B’s displayed probability is exactly `1 - pred.p_win`.
  - Base bars use each direct base response with no player-ID comparison.
  - Canonical probability helpers and their tests are removed; remaining H2H orientation tests cover only H2H response display.
- **Guardrails**: Do not change picker UX, historical H2H semantics, or unrelated visual design.

### [ ] Task 10: Enforce migration and retrain safely

- **Description**: Version the feature/model contract in candidate manifest and serving/deploy validation. Reject or clearly fail deployment when a champion’s contract does not match the symmetric feature and scorer contract. Rebuild dbt gold, refresh local training snapshot, train all notebooks, promote only through the existing promotion process, and deploy only the retrained champion.
- **Files**:
  - `src/flows/pipeline.py`
  - `src/flows/deploy.py` or the existing deploy-flow module
  - `src/serving/service.py`
  - `src/constants.py`
  - model/deploy manifest tests
- **Acceptance criteria**:
  - A pre-migration champion cannot be packaged or served as the symmetric contract.
  - New manifests record feature contract version, clip epsilon, scorer type, stack column order, and exact base pins.
  - Retraining produces valid directional train/validation/test artifacts and a serveable champion.
- **Guardrails**: Do not modify aliases/stages strategy; retain the single `@champion` model alias.

## Dependencies

1. Tasks 1-4 establish the data/feature contract before any training notebook changes.
2. Task 5 establishes physical-match identity and must precede Task 6.
3. Task 6 must produce trustworthy paired artifacts before Task 7 can train the stacker.
4. Task 7 provides shared scoring semantics required by Task 8.
5. Task 8 must be complete before Task 9 removes UI remapping.
6. Tasks 1-10 must complete before full retraining/deployment in Task 10.

## QA / Testing Scenarios

### Data and features

- One physical match creates two gold rows with reciprocal IDs, labels, differences, paired values, and H2H advantage.
- Same-date prior matches cannot leak into train-only snapshots when inference excludes them.
- Return-strength, smoothing, exposures, carpet values, and all fallbacks are finite and parity-tested.
- Snapshot rejects incomplete mirror pairs, duplicate directional rows, invalid labels, and feature-order drift.

### Splits and OOF

- No `match_id` overlaps a fold’s fit/validation sets or top-level bands.
- Every base uses exactly the same persisted match-to-fold mapping.
- Test tensors and IDs in NN exactly match test artifacts.
- Prediction artifacts fail when model pin, row ID, fold ID, or named column contract differs.

### Symmetry

- Feature builder: double swap restores original; signed values negate; paired values exchange; context is unchanged.
- For clipped values at `0`, `1`, `0.5`, and clip boundaries, projection is finite and reversed evidence negates.
- Every base satisfies `P(A,B) + P(B,A) = 1` after projection.
- Ensemble logit negates and ensemble API probabilities complement for separate reverse HTTP/API calls.
- Response IDs preserve request order; `predicted_winner` agrees with that orientation.

### Evaluation and frontend

- Stacker consumes columns by required names/order and has no fitted intercept.
- Candidate/production evaluation aligns by physical match identity and counts each match once.
- UI shows Player A as `p_win`, Player B as `1-p_win`, and no probability behavior depends on lexicographic player IDs.

## Validation Commands

Run only after implementation, using project recipes and self-contained tests:

- Focused Python tests for inference, snapshot, serving, symmetry helper, and static dbt contracts.
- Frontend orientation tests and production build.
- `just` recipes for lint/type/test/build as defined in the repository’s `justfile`.
- dbt build and snapshot/retrain only against the intended environment after confirming migration effects and model-artifact invalidation.

## Risks and Explicit Tradeoffs

- Mirroring doubles directional rows but not independent evidence; grouped splitting and weighted training prevent leakage/double-counting.
- Logit antisymmetrization is structurally exact but can sharpen poor base outputs; use fixed symmetric clipping and evaluate calibration after retraining. Arithmetic probability averaging is a future alternative, not a concurrent production pathway.
- Feature smoothing and H2H changes alter the feature/model contract, so all old models must be rejected.
