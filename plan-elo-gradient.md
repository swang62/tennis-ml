# Plan: elo-gradient

## Goal

Add causal `player_elo_gradient_10` and `opponent_elo_gradient_10` features across gold, training, and serving. Simplify form features into two Laplace-smoothed directional differences, remove all weighted-form features, remove the H2H history cap, and retain Bento Buildx output only in deployment logs.

## Scope

### In

- Overall (not surface-specific) raw Elo-gradient pairs, calculated as OLS slope per match from up to ten prior `post_elo` observations.
- Form-feature contract changes:
  - Rename `win_rate_diff` to `form_diff`.
  - Retain `surface_form_diff`.
  - Remove `weighted_form_10` from silver, tour-average fallbacks, gold, inference, and the model feature contract.
  - Laplace-smooth both side-level form rates before subtracting opponent from player.
- Complete causal H2H history for all model H2H features.
- Gold, scalar inference, bulk inference, schema/docs, and hermetic coverage.
- Quiet Bento Buildx console output with complete timestamped deployment logs.

### Out

- Changes to Elo rating calculation, bronze ingestion, or new dependencies/tables.
- Surface-specific Elo gradients.
- Backward compatibility for prior snapshots or model artifacts, snapshot regeneration, model retraining, champion promotion, and deployment.
- The UI-only `last5_player1_win_rate` API field and the separate web-image Buildx recipe.

## Tasks

### [ ] Task 1: Remove weighted form and rename overall form

- **Description**: Remove the recency-weighted `weighted_form_10` calculation and every downstream fallback, selected column, documentation entry, and inference-side value that supports it. Rename gold `win_rate_diff` to `form_diff`; retain the existing smoothed target-surface rate as `surface_form_diff`.
- **Files**: `dbt/models/silver/rolling_features.sql`, `dbt/models/silver/rolling_features.yml`, `dbt/models/gold/tour_averages.sql`, `dbt/models/gold/tour_averages.yml`, `dbt/models/gold/match_features.sql`, `dbt/models/gold/match_features.yml`, `src/features/columns.py`, `src/features/inference.py`, `tests/test_inference_features.py`
- **Acceptance Criteria**:
  - `form_diff` and `surface_form_diff` are calculated by subtracting independently Laplace-smoothed player and opponent rates.
  - No target or future match contributes to any form value.
  - No database model, tour-average row, inference-side value, feature column, or documentation refers to `weighted_form_10`, `player_weighted_form_10`, `opponent_weighted_form_10`, or `weighted_form_diff`.
- **Guardrails**: Do not replace weighted form with opponent-rank, rank-gap, or Elo-surprise weighting.

### [ ] Task 2: Derive causal overall Elo gradients in gold

- **Description**: In `match_features.sql`, retrieve each side's ten most recent `silver.elo_snapshots.post_elo` observations strictly preceding the target causal key. Order them chronologically, calculate OLS slope against consecutive match indexes, and expose `player_elo_gradient_10` and `opponent_elo_gradient_10`. Return `0` for fewer than two observations. Retain the existing exact-current-snapshot `pre_elo` join for `elo_diff` unchanged.
- **Files**: `dbt/models/gold/match_features.sql`, `dbt/models/gold/match_features.yml`
- **Acceptance Criteria**:
  - Neither gradient includes the target match's `post_elo` or any later match.
  - A monotonic increase yields a positive slope; a decrease yields a negative slope.
  - Windows of 2-9 observations use their OLS slope; windows of 0-1 return exactly `0`.
  - Both orientations retain each participant's own value rather than negating it.
- **Guardrails**: Do not add this to `rolling_features`; it runs before Python Elo materialization. Do not alter `elo_diff` or Elo snapshots.

### [ ] Task 3: Publish the revised shared model schema

- **Description**: Update the ordered feature contract to remove the raw weighted-form pair, rename the overall form difference, and add the two raw gradient columns. Update snapshot-schema validation expectations and feature documentation to exactly match gold and serving.
- **Files**: `src/features/columns.py`, `dbt/models/gold/match_features.yml`, `tests/test_snapshot.py`
- **Acceptance Criteria**:
  - `FEATURE_COLS` contains `form_diff`, `surface_form_diff`, and both gradient names exactly once in deterministic order.
  - It contains none of `win_rate_diff`, `weighted_form_10`, `player_weighted_form_10`, `opponent_weighted_form_10`, or `weighted_form_diff`.
  - Snapshot validation accepts the revised ordered schema and rejects missing or reordered features.
- **Guardrails**: Do not add a compatibility layer. Existing snapshots and model artifacts intentionally remain old-contract artifacts and must not be deployed with the new code.

### [ ] Task 4: Mirror revised form and Elo features in online inference

- **Description**: Update scalar and set-oriented bulk inference to emit the two form differences and both gradients from the same side-level snapshot values and formulas as gold. Retrieve each player's latest up-to-ten completed post-match Elo values strictly before the request `as_of_date`, calculate the OLS slope, and default to `0` with fewer than two values. Ensure same-day inference uses only data earlier than the date because the public request contains no target match tuple.
- **Files**: `src/features/inference.py`, `tests/test_inference_features.py`
- **Acceptance Criteria**:
  - Scalar and bulk builders emit the revised `FEATURE_COLS` order without nulls.
  - Both use the same Laplace-smoothed form formulas as gold.
  - As-of inference excludes same-day and future snapshots, matching the repository's strict-before-as-of contract.
  - Swapping players negates all form differences and exchanges the two raw gradient values.
  - Scalar/bulk parity holds, along with gold/inference parity for historical requests that have an unambiguous prior-date boundary.
- **Guardrails**: Reuse snapshot table/query conventions; do not query MLflow or add per-request database loops to the bulk path. Do not claim same-day match-tuple parity: the inference API does not accept that tuple.

### [ ] Task 5: Validate causal feature correctness

- **Description**: Update dbt checks and focused Python tests to independently validate form, gradient, schema, inference parity, and causal boundaries.
- **Files**: `dbt/tests/gold/match_features_no_current_match_leakage.sql`, `dbt/tests/gold/match_features_no_null_model_features.sql`, `tests/test_inference_features.py`, `tests/test_snapshot.py`
- **Acceptance Criteria**:
  - Leakage checks fail if a target outcome is included in form or gradient history.
  - No-null checks cover all revised fields.
  - Tests cover form smoothing, absence of weighted-form fields, 0/1/partial/full gradient history, player swap, strict date boundary, and scalar/bulk parity.
  - Run narrow dbt/Python feature tests, then `just lint` and the full test suite successfully.
- **Guardrails**: Keep tests hermetic with existing fixtures; do not add live DB, network, MLflow, Prefect, or browser dependencies.

### [ ] Task 6: Use complete causal H2H history

- **Description**: Remove the five-meeting limit from gold and scalar/bulk inference. Make `h2h_exposure`, `h2h_advantage`, and `h2h_surface_advantage` use every causally prior meeting while preserving smoothing and directional behavior.
- **Files**: `dbt/models/gold/match_features.sql`, `dbt/models/gold/match_features.yml`, `dbt/tests/gold/match_features_h2h_no_current_match.sql`, `src/features/inference.py`, `tests/test_inference_features.py`
- **Acceptance Criteria**:
  - More than five prior meetings contribute to all three H2H fields in gold and inference.
  - Gold remains strict-before-target; inference remains strict-before-`as_of_date`.
  - Reversing player and opponent preserves exposure and negates both advantages.
  - Scalar/bulk parity and leakage coverage include complete and surface-restricted history.
- **Guardrails**: Leave `last5_player1_win_rate` unchanged and remove any unused model-H2H cap constant.

### [ ] Task 7: Quiet Bento container build output

- **Description**: Stop routine Docker Buildx output from streaming to the console during `deploy_bento`, while continuing to write it to the timestamped deployment log. Keep deployment progress messages visible and retain child output plus failure diagnostics on failure.
- **Files**: `src/flows/deploy.py`, `tests/test_deploy.py`
- **Acceptance Criteria**:
  - Successful Bento Buildx output is absent from `deploy_bento` console output and complete in `logs/deploy_<timestamp>.log`.
  - Failure still raises and the log contains child output plus the failure diagnostic.
  - Direct non-deploy callers preserve current console behavior unless explicitly quiet.
- **Guardrails**: Limit this to Bento Buildx; do not hide Docker login failures or alter the web-image recipe.

## Dependencies

1. Tasks 1 and 2 establish gold semantics independently.
2. Task 3 publishes the combined feature-contract change after Tasks 1-2.
3. Task 4 mirrors the contract in serving after Task 3.
4. Task 5 validates Tasks 1-4.
5. Tasks 6 and 7 are independent and can be implemented separately.

## QA Scenarios

- Standard and surface form: no history yields neutral Laplace rates before subtraction; player swap negates each difference.
- Weighted-form fields are absent from silver, gold, tour-average fallbacks, inference, and `FEATURE_COLS`.
- Elo gradients: no/one history returns 0; partial history uses OLS; only newest ten strictly prior observations count.
- Inference: an `as_of_date` never incorporates same-day or later snapshots.
- H2H: six or more prior meetings affect all model H2H fields; UI last-five output is unchanged.
- Bento: routine Buildx output stays in the deployment log but out of the console; failures retain diagnostics.
