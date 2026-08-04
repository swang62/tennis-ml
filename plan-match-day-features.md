# Plan: Leakage-Safe Match-Day Features

## Goal

Improve pre-match prediction features while guaranteeing that training and inference use only completed historical matches, static player data, and known match context. Add dashboard-only all-time surface statistics, expose strength of schedule as `avg_rank_faced`, add fatigue/profile/form features, and implement perspective-explicit H2H features.

## Scope

### In scope

- Dashboard all-time clay/grass/hard win rates calculated on demand.
- `player_avg_rank_faced_10/20` and `opponent_avg_rank_faced_10/20` backed by the existing snapshot averages.
- Historical double-fault rates, losing streak, age, and prior 7/90-day match counts.
- Prior H2H meetings, wins, and win rates from both canonical perspectives.
- Training/inference parity, schema documentation, regression tests, retraining, and serving-artifact refresh.

### Out of scope

- Current-match or in-play statistics.
- Elo ratings.
- Persisted all-time surface rates in `gold.player_profiles`.

## Tasks

### [ ] Task 1: Add dashboard-only all-time surface rates

- **Description**:
  - Add a dashboard query/helper that orients all historical rows to the selected player's perspective and aggregates `match_won` by surface.
  - Return both all-time win rate and match count for clay, grass, and hard; represent an unplayed surface as `n/a (n=0)` rather than zero performance.
  - Replace the Player Comparison profile card's three last-10 surface display rows with `Clay/Grass/Hard win rate (all-time)` rows. Keep the rolling last-10 surface data and model feature unchanged elsewhere.
  - Reuse the Player Explorer's existing oriented-history/grouping behavior rather than creating a persisted table or duplicated calculation pipeline.
- **Files**:
  - `src/dashboard/app.py`
  - `tests/test_dashboard.py` (add only if dashboard helpers are made import-safe and the repository has an appropriate test seam; otherwise validate through an existing focused unit-test location)
- **Acceptance Criteria**:
  - Selecting a player calculates each surface rate from all completed matches currently in DuckDB.
  - Displayed counts and rates agree with a direct grouped query over that player's oriented match history.
  - A player with no matches on one surface displays `n/a`, not `0%`.
  - No database schema or training feature changes are introduced by this task.
- **Guardrails**:
  - Do not add all-time rate columns to `gold.player_profiles` or `gold.match_features`.
  - Do not replace `surface_win_rate_10` in `FEATURE_COLS`.

### [ ] Task 2: Add leakage-safe rolling form and strength-of-schedule fields

- **Description**:
  - Add `double_faults` to the rolling snapshot source and compute `df_rate_5`/`df_rate_10` as rolling double faults divided by rolling total serve points, with `NULLIF` denominator handling consistent with existing serve rates.
  - Compute `loss_streak` as consecutive losses ending at the snapshot match, mirroring `win_streak`; a win resets it to zero.
  - Reuse existing `avg_opponent_rank_10/20` snapshot values but expose them downstream as:
    - `player_avg_rank_faced_10`, `player_avg_rank_faced_20`
    - `opponent_avg_rank_faced_10`, `opponent_avg_rank_faced_20`
  - Add per-side double-fault/loss-streak fields and differentials `df_rate_diff`, `loss_streak_diff`, and `avg_rank_faced_diff` (based on the 10-match value).
  - Mirror all snapshot values and fallback aggregates in inference.
- **Files**:
  - `dbt/models/gold/rolling_features.sql`
  - `dbt/models/gold/rolling_features.yml`
  - `dbt/models/gold/match_features.sql`
  - `dbt/models/gold/match_features.yml`
  - `src/features/columns.py`
  - `src/features/inference.py`
  - `tests/test_rolling_contract.py`
  - `tests/test_inference_features.py`
  - `tests/test_inference_units.py`
- **Acceptance Criteria**:
  - Every training row receives these values only from snapshot N-1; first-match rows remain NULL where no history exists.
  - Inference reads the latest snapshot strictly before `as_of_date` and outputs the exact new `FEATURE_COLS` contract without NaNs after fallback handling.
  - `df_rate_5/10` are NULL when historical total serve points are zero and otherwise non-negative.
  - `win_streak` and `loss_streak` cannot both be positive on one snapshot.
  - Downstream names contain `avg_rank_faced`; no new public feature is named `avg_opponent_rank` or misspelled `oppoent`.
- **Guardrails**:
  - Do not recompute rolling formulas in Python; SQL remains the feature source of truth.
  - Do not rename the internal `rolling_features.avg_opponent_rank_10/20` columns unless required; alias at the match-feature boundary to minimize churn.

### [ ] Task 4: Implement perspective-explicit H2H features

- **Description**:
  - Add leakage-safe prior-meeting aggregates for the canonical pair, considering only matches with `match_date` strictly before the target date.
  - Expose:
    - `player_h2h_matches`, `player_h2h_wins`, `player_h2h_alltime_win_rate`, `player_h2h_last_5_win_rate`
  - `player_h2h_*` always describes the canonical `player_id` perspective;
  - Use zero meetings/wins and a neutral fallback policy for the undefined 0/0 win rate; select one explicit convention and apply it identically in SQL and inference. Recommended: `0.5` win rate when there are no prior meetings, while the zero meeting count lets models distinguish uncertainty.
- **Files**:
  - `dbt/models/gold/match_features.sql`
  - `dbt/models/gold/match_features.yml`
  - `src/features/columns.py`
  - `src/features/inference.py`
  - New `dbt/tests/gold/match_features_h2h_no_current_match.sql`
  - `tests/test_inference_features.py`
  - `tests/test_inference_units.py`
  - `tests/test_e2e_ingest_to_inference.py`
- **Acceptance Criteria**:
  - The first historical meeting has zero prior meetings and neutral H2H rates.
  - The second meeting sees exactly one prior result; the current meeting is never counted.
  - Same-date meetings are excluded consistently from both training and date-granularity inference.
  - `player_h2h_matches == opponent_h2h_matches`.
  - With prior meetings, `player_h2h_wins + opponent_h2h_wins == player_h2h_matches` and the two win rates sum to 1 within floating-point tolerance.
  - Reversing raw input ids still produces the same canonical feature row.
- **Guardrails**:
  - Do not use current-match `match_won` in H2H aggregates.
  - Do not add redundant H2H differential columns; per-side wins/rates already encode the comparison.
  - Use prepared parameters in inference SQL; do not interpolate player ids or dates.

### [ ] Task 5: Strengthen leakage and feature-contract tests

- **Description**:
  - Extend the existing N-1 leakage dbt test beyond `win_rate_10` to cover all snapshot-backed fields added in this plan.
  - Add explicit regression coverage that no current-match aces, double faults, first-serve totals, break-point totals, or outcome values enter `FEATURE_COLS`.
  - Update rolling/feature column contract expectations and assertions for per-side ordering, differential ordering, no-NaN inference, canonical id symmetry, cold starts, missing profiles, and missing ranks.
  - Add a train/inference parity fixture that compares one historical gold row's features to an inference row built at that match date from only earlier history.
- **Files**:
  - `dbt/tests/gold/match_features_no_current_match_leakage.sql`
  - New focused dbt tests from Tasks 3 and 4
  - `tests/test_rolling_contract.py`
  - `tests/test_inference_features.py`
  - `tests/test_inference_units.py`
  - `tests/test_e2e_ingest_to_inference.py`
- **Acceptance Criteria**:
  - Tests fail if a gold row uses snapshot N rather than N-1.
  - Tests fail if a current-match raw stat is added to the model feature contract.
  - Tests fail on any training/inference mismatch for the parity fixture.
  - Existing canonicalization, cold-start, and imputation tests remain green.
- **Guardrails**:
  - Prefer behavior-based assertions over brittle total-column counts alone.
  - Do not weaken existing leakage or range-window tests to accommodate new fields.

### [ ] Task 6: Rebuild, retrain, and validate serving compatibility

- **Description**:
  - Run the dbt build through the repository's `just` workflow and inspect representative rows for formula, date-boundary, H2H, and cold-start correctness.
  - Run focused Python tests, then the complete relevant test suite.
  - Run the training pipeline because `FEATURE_COLS` changes invalidate existing model artifacts; confirm the generated `feature_cols.json` matches serving input order.
  - Rebuild/redeploy only after a newly trained champion is promoted, following the existing alias-based deployment flow.
  - Smoke-test `/predict_from_ids` for known players, cold-start players, no-H2H pairs, and previously-met pairs.
- **Files**:
  - Generated training artifacts under the repository's existing ignored artifact paths only
  - No manually edited generated dbt `target/` files
- **Acceptance Criteria**:
  - `just etl` succeeds with all dbt model/tests green.
  - Relevant `pytest` suite succeeds.
  - `just pipeline` completes and logs the expanded feature contract.
  - A newly promoted model accepts the exact inference feature order and returns predictions from `/predict_from_ids` without missing/extra-column errors.
  - Dashboard profile cards display correct all-time surface statistics.
- **Guardrails**:
  - Do not reuse or deploy a model trained against the old feature contract.
  - Do not edit generated `dbt/target` output or committed model artifacts manually.

## Dependencies

1. Task 1 is independent and may be completed first.
2. Tasks 2-4 modify the shared feature contract and should be implemented before final contract tests.
3. Task 5 depends on the final SQL/Python feature names from Tasks 2-4.
4. Task 6 depends on all implementation and tests being complete.

## QA/Testing Scenarios

- Player with no match history: honest activity/H2H counts, valid inference fallback values, no NaNs.
- Player with history but none on a dashboard surface: `n/a (n=0)` display.
- Match exactly on 7/30/90-day boundaries.
- Multiple historical matches on the prediction date: excluded under date-granularity inference semantics.
- First and second meeting between a pair, followed by split and one-sided H2H records.
- Missing opponent ranks inside rolling windows: average rank faced skips NULL ranks consistently.
- Zero historical serve points: double-fault rates remain NULL in gold and are imputed consistently for inference.
- Win-to-loss and loss-to-win transitions: only one streak direction is positive.
- Missing birthdate/profile: age uses the historical profile-pool fallback.
- Raw player input order reversed: canonical feature row and prediction remain unchanged.
