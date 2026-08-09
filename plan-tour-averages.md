# Plan: Consolidate Defaults and Tour Averages

## Goal

Replace the date-keyed `gold.feature_defaults` relation and the partial
`gold.tour_averages` draft with one dbt-materialized, single-row
`gold.tour_averages` table. The singleton will be the only shared source for:

- difficult model-feature fallbacks used by dbt and inference; and
- weighted tour-wide rates used for player-profile comparisons.

Silver continues preserving observed NULLs. `gold.match_features` continues
performing final model-feature imputation during ETL, so DuckDB training data is
already complete. Bento reads the materialized singleton directly from
PostgreSQL when building inference or profile responses and never computes or
caches fallback aggregates.

## Scope

### In scope

- Consolidate the two current dbt models into one singleton relation.
- Preserve all existing gold imputation behavior while intentionally replacing
  date-keyed defaults with one full-pool row.
- Make scalar and bulk inference share one database-backed singleton loader.
- Add reusable weighted tour-rate benchmarks.
- Return profile comparison values and render signed percentage-point deltas.
- Update dbt, Python, API, frontend, tests, and current architecture docs.
- Provide a safe one-time cleanup path for the obsolete PostgreSQL relation.
- Remove the redundant MLflow feature-column artifact/pin lineage while keeping
  the internal `FEATURE_COLS` ordering contract used by training and serving.

### Out of scope

- Changing silver's observed-data/null-preservation contract.
- Moving imputation into notebooks or the DuckDB snapshot.
- Adding `gold.tour_averages` to the training snapshot.
- Computing averages in request handlers, inference SQL, or duplicated CTEs.
- Adding overall tour win rate or surface win rates: player-perspective expansion
  makes those mechanically 50% and therefore not useful benchmarks.
- Redesigning unrelated profile sections.

## Final `gold.tour_averages` Schema

The model must always materialize exactly one row.

### Singleton identity and observability

| Column | Definition |
|---|---|
| `singleton_id` | Constant `1`; non-null, unique, accepted value `[1]`, primary key. |
| `pool_as_of_date` | One day after the latest rolling snapshot; anchors activity and years-pro calculations to source data rather than wall-clock time. |
| `snapshot_pool_rows` | Count of rolling snapshot rows used by fallback aggregates. |
| `snapshot_pool_players` | Distinct players represented in the snapshot pool. |
| `profile_rows` | Player-profile rows represented in profile aggregates. |
| `player_match_rows` | Player-perspective rows represented in weighted tour rates. |

### Model fallback columns

Retain the current column names to minimize consumer churn and preserve the
existing `COALESCE` contract:

- Median-like defaults: `latest_player_ranking`,
  `latest_player_rank_points`, `streak`, `avg_player_rank_10`,
  `avg_rank_faced_10`.
- Mean defaults: `latest_player_age`, `weighted_form_10`, `win_rate_10`,
  `ace_rate_10`, `first_serve_pct_10`, `break_points_saved_pct_10`,
  `first_serve_win_pct_10`, `second_serve_win_pct_10`, `serve_win_pct_10`,
  `return_points_won_pct_10`, `df_rate_10`, `aces_per_svc_game_10`,
  `clay_win_rate_10`, `grass_win_rate_10`, `hard_win_rate_10`.
- Activity/profile defaults: `days_since_default`, `matches_30d_default`,
  `rate_default`, `left_handed_rate`, `avg_years_pro`.

Fallback semantics:

- Aggregate all available `silver.rolling_features` rows, matching the current
  latest date-keyed pool's player-match weighting as closely as possible.
- Use medians for rank/streak-like values and means for continuous rates.
- Calculate `days_since_default` from each player's latest snapshot relative to
  `pool_as_of_date`.
- Calculate `matches_30d_default` from the 30-day window ending at
  `pool_as_of_date`.
- Calculate `avg_years_pro` using the year of `pool_as_of_date`.
- Preserve current deterministic empty-pool constants so every fallback column
  is finite and non-null: ranking `100`, rank points `500`, age `26`, streak
  `0`, rolling rates/forms `0`, average ranks `100`, days since `365`, matches
  in 30 days `0`, unknown-surface rate `0`, left-handed rate `0`, and years pro
  `8`.
- Limited historical leakage is intentional and documented: old cold-start or
  otherwise missing cells use the same full-pool singleton as current rows.

### Weighted tour comparison columns

Compute ratios from summed numerators and denominators in
`silver.player_matches`; do not average player percentages:

| Column | Formula |
|---|---|
| `tour_ace_rate` | `SUM(aces) / SUM(first_serves_made)` |
| `tour_first_serve_pct` | `SUM(first_serves_made) / SUM(total_serve_points)` |
| `tour_break_points_saved_pct` | `SUM(break_points_saved) / SUM(break_points_faced)` |
| `tour_first_serve_win_pct` | `SUM(first_serve_points_won) / SUM(first_serves_made)` |
| `tour_second_serve_win_pct` | `SUM(second_serve_points_won) / SUM(total_serve_points - first_serves_made)` |
| `tour_serve_win_pct` | `SUM(first_serve_points_won + second_serve_points_won) / SUM(total_serve_points)` |
| `tour_return_points_won_pct` | `SUM(return_points_won) / SUM(return_points_available)` |
| `tour_df_rate` | `SUM(double_faults) / SUM(total_serve_points)` |
| `tour_aces_per_svc_game` | `SUM(aces) / SUM(service_games)` |

Use `DOUBLE PRECISION` numerators and `NULLIF(denominator, 0)`. These benchmark
columns may be NULL only when their source denominator is zero; default columns
must never be NULL. Normal seeded/full datasets must satisfy positive-denominator
tests for profile-exposed benchmarks.

## Tasks

### [ ] Task 1: Consolidate the dbt singleton model

- **Description**:
  - Rewrite the existing `tour_averages.sql` draft as the canonical singleton.
  - Merge the fallback calculations from `feature_defaults.sql` into a single
    source-anchored aggregate pipeline.
  - Use internal CTEs only inside this model to calculate snapshot, latest-player
    activity, profile, and weighted player-match aggregates once.
  - Add `singleton_id`, `pool_as_of_date`, observability counts, all existing
    fallback columns, and all weighted tour comparison columns defined above.
  - Delete the superseded `feature_defaults` model definition after its schema
    and tests have been transferred.
- **Files**:
  - `dbt/models/gold/tour_averages.sql`
  - `dbt/models/gold/tour_averages.yml`
  - `dbt/models/gold/feature_defaults.sql` (remove)
  - `dbt/models/gold/feature_defaults.yml` (remove)
  - `dbt/dbt_project.yml`
- **Acceptance Criteria**:
  - `gold.tour_averages` has exactly one row and the schema above.
  - Every fallback column is finite and non-null.
  - Weighted benchmark formulas match direct recomputation from
    `silver.player_matches`.
  - The model depends only on dbt silver models and `gold.player_profiles`; no
    runtime service computes the same aggregates.
  - `singleton_id = 1` is enforced by dbt schema tests and a post-build primary
    key.
- **Guardrails**:
  - Do not retain `as_of_date` rows or date-expansion joins.
  - Do not use unweighted `AVG(per-player-rate)` for tour benchmarks.
  - Do not add speculative benchmarks without an existing numerator/denominator
    contract in `silver.player_matches`.

### [ ] Task 2: Keep all finalized training imputation in dbt gold

- **Description**:
  - Replace the date equality join to `feature_defaults` with one explicit
    `CROSS JOIN {{ ref('tour_averages') }}`.
  - Preserve the existing `COALESCE` and `CASE` behavior for ranking, rank
    points, age, activity, rolling rates, unseen surfaces, handedness, and
    years-pro.
  - Preserve direct dbt handling for indoor context, H2H zero state, tournament
    encoding, and round encoding.
  - Rewrite comments and schema descriptions to distinguish strict-prior player
    snapshots from intentionally global fallback values.
  - Update the leakage regression test: prior player state must remain strictly
    prior, while fallback values must equal the singleton rather than a
    date-keyed row.
- **Files**:
  - `dbt/models/gold/match_features.sql`
  - `dbt/models/gold/match_features.yml`
  - `dbt/tests/gold/match_features_no_current_match_leakage.sql`
  - `dbt/tests/gold/match_features_no_null_model_features.sql`
  - `dbt/tests/gold/match_features_keeps_all_bronze_matches.sql`
- **Acceptance Criteria**:
  - `gold.match_features` retains one row per bronze match.
  - Every `FEATURE_COLS` value remains non-null and finite.
  - A first player match receives singleton fallbacks because no prior snapshot
    exists.
  - A player with history but a NULL denominator-derived metric or unseen
    surface receives the matching singleton fallback.
  - Missing handedness or `turned_pro` produces finite model representations
    without altering nullable source biography fields.
  - DuckDB receives already-imputed `gold.match_features`; no training notebook
    gains new imputation logic.
- **Guardrails**:
  - Do not replace observed non-null values with tour averages.
  - Do not impute raw nullable biography values inside `gold.player_profiles`.
  - Do not weaken the strict-prior snapshot or finite-feature tests.

### [ ] Task 3: Centralize validated singleton loading for Bento

- **Description**:
  - Add one shared loader for `gold.tour_averages`, used by inference and profile
    routes.
  - Read the one-row table through the existing PostgreSQL client; do not
    calculate or cache its values in Bento.
  - Validate exactly one returned row and all required fallback columns; fail
    clearly with a `run dbt build` message when absent or invalid.
- **Files**:
  - `src/features/tour_averages.py` (new shared loader)
  - `src/constants.py`
  - `src/features/columns.py`
- **Acceptance Criteria**:
  - `TOUR_AVERAGES_TABLE = "gold.tour_averages"` replaces
    `FEATURE_DEFAULTS_TABLE`.
  - Default-column and benchmark-column lists are clearly separated.
  - Each scalar request, bulk request, or profile request performs at most one
    one-row singleton lookup.
  - Scalar and bulk inference share the same query and schema validation.
- **Guardrails**:
  - No TTL, background refresh, process cache, or request-time aggregation.
  - Do not duplicate singleton query or validation logic across consumers.

### [ ] Task 4: Simplify scalar and bulk inference around the singleton

- **Description**:
  - Remove date-keyed default SQL, oldest-row fallback, and per-date bulk default
    resolution.
  - Load the database singleton once per scalar or bulk build.
  - Keep `as_of_date` only for strictly-prior player snapshot/profile/activity/H2H
    lookups; it no longer selects a defaults row.
  - Preserve cell-level fallback behavior: use observed snapshot/profile values
    when finite, otherwise use the corresponding singleton value.
  - Preserve inference/training parity for unseen surfaces, missing profile
    fields, and no-prior-snapshot requests.
  - Keep observability metadata, sourced from the singleton counts and
    `pool_as_of_date`.
- **Files**:
  - `src/features/inference.py`
  - `tests/test_inference_units.py`
  - `tests/test_inference_features.py`
  - `tests/test_e2e_ingest_to_inference.py`
- **Acceptance Criteria**:
  - `_load_defaults_oldest`, date-parameterized `_load_defaults`,
    `_load_defaults_bulk`, and their SQL constants are removed.
  - Scalar and bulk paths use the same singleton contract and produce identical feature
    contracts.
  - One-player and two-player cold-start tests remain finite; identical
    singleton fallbacks yield neutral difference features where appropriate.
  - Existing snapshot-backed inference values remain unchanged except where the
    old date-keyed fallback actually applied.
- **Guardrails**:
  - Do not reject valid players solely because a request date precedes their
    first snapshot; the singleton remains the cold-start fallback.
  - Do not add AVG/PERCENTILE queries to inference.

### [ ] Task 5: Expose tour benchmarks in player profiles

- **Description**:
  - Keep per-player career rates computed by the indexed player-specific query.
  - Read tour comparison rates from the shared database singleton loader.
  - Add a dedicated `tour_averages` object to the profile response rather than
    mixing tour values into the player's `career` object.
  - Initially expose the four rates matching the existing career contract:
    first-serve win, second-serve win, overall serve win, and break-points saved.
    The remaining benchmark columns stay available in PostgreSQL for future UI
    use without changing the model again.
- **Files**:
  - `src/serving/service.py`
  - `web/src/api.ts`
  - relevant service/profile API tests under `tests/`
- **Acceptance Criteria**:
  - Profile responses return player career values and matching tour values from
    the current materialized singleton row.
  - No profile request performs a full-table aggregate.
  - Missing/undefined tour denominators serialize as `null`, not a fabricated
    percentage.
- **Guardrails**:
  - Do not expose model fallback constants as profile benchmarks.
  - Do not calculate deltas in the backend; return source rates and let the UI
    format them.

### [ ] Task 6: Render signed percentage-point comparisons

- **Description**:
  - For each displayed profile rate with a tour counterpart, calculate
    `player_rate - tour_rate` in the frontend.
  - Render two decimals and an explicit sign inside parentheses, e.g.
    `65.2% (+5.13%)` or `58.6% (-3.45%)`.
  - Use the existing grass/green theme above average, clay/red below average,
    and neutral text at exactly zero.
  - Provide accessible text identifying the delta as percentage points above or
    below tour average.
- **Files**:
  - `web/src/pages/Profile.tsx`
  - `web/src/index.css`
  - `web/src/api.ts`
- **Acceptance Criteria**:
  - Positive, negative, zero, and null benchmark states render correctly.
  - The main player percentage remains visually primary; comparison text is
    secondary but readable.
  - Colors use existing theme tokens and remain distinguishable without relying
    on color alone because the sign is always shown.
- **Guardrails**:
  - Use percentage-point differences, not relative percentage change.
  - Do not add a charting or table dependency for two inline comparisons.

### [ ] Task 7: Add singleton and migration regression coverage

- **Description**:
  - Add dbt singular tests for exactly one row, required fallback finiteness,
    rate bounds, non-negative counts, and weighted formula parity.
  - Update integration readiness checks so tests cannot run against stale gold
    outputs missing `tour_averages`.
  - Preserve exact DuckDB snapshot scope: only `match_features` and
    `player_profiles`.
  - Define a one-time local migration procedure: build and validate
    `gold.tour_averages`, update all consumers, then remove
    `gold.feature_defaults`. Do not leave a permanent legacy-drop hook.
- **Files**:
  - `dbt/tests/gold/tour_averages_singleton.sql` (new)
  - `dbt/tests/gold/tour_averages_no_invalid_defaults.sql` (new)
  - `dbt/tests/gold/tour_averages_weighted_rates.sql` (new)
  - `tests/conftest.py`
  - `tests/test_snapshot.py`
  - `src/db/snapshot.py` (verification only; expected to remain behaviorally unchanged)
- **Acceptance Criteria**:
  - Fresh builds contain `gold.tour_averages` and not a dbt-managed
    `gold.feature_defaults` model.
  - Existing databases have a documented, explicitly approved cleanup step for
    the obsolete physical table.
  - Snapshot tests still enforce exactly two DuckDB tables.
  - All dbt and Python contracts fail loudly on missing/duplicate/invalid
    singleton rows.
- **Guardrails**:
  - Do not automatically drop PostgreSQL relations from normal dbt runs.
  - Do not add `tour_averages` to `SNAPSHOT_TABLES`.

### [ ] Task 8: Remove redundant MLflow feature-column lineage

- **Description**:
  - Remove the `train_test_split` MLflow experiment run; it registers no model
    and its split metadata is not consumed downstream.
  - Stop writing `feature_cols.json`, logging `features.txt`, and writing
    `feature_pins.json` in the split notebook.
  - Remove `feature_pins` from the ensemble candidate manifest, promotion tag
    construction, champion version tags, deploy fingerprint inputs, and Bento
    `/model_info` manifest.
  - Update evaluation/SHAP code to import and use `FEATURE_COLS` directly rather
    than reopening `feature_cols.json`.
  - Remove tests and docs that expect the deleted artifacts/tags.
  - Preserve `FEATURE_COLS` itself as the single internal ordered model-input
    list used to select training matrices and reorder internally built inference
    rows before the scaler/models.
- **Files**:
  - `notebooks/parameters/01_train_test_split.ipynb`
  - `notebooks/parameters/04_ensemble_stack.ipynb`
  - `notebooks/parameters/05_evaluate.ipynb`
  - `src/constants.py`
  - `src/flows/deploy.py`
  - `tests/test_deploy.py`
  - `tests/test_rolling_contract.py`
  - current lineage documentation in `AGENTS.md` and `README.md`
- **Acceptance Criteria**:
  - No repository reference remains to `features.txt`, `feature_cols.json`,
    `feature_pins.json`, `features_uri`, `feature_cols_hash`,
    `aux_features_uri`, or `aux_feature_cols_hash`.
  - `01_train_test_split` performs the chronological split and writes training
    datasets without contacting MLflow.
  - Candidate manifests and champion tags retain model, scaler, embedding, and
    bio-feature lineage but contain no tabular feature-list pins.
  - Add `src/features/columns.py` to `SOURCE_FINGERPRINT_FILES` so removing the
    MLflow feature hash does not allow a changed internal feature order to reuse
    an old Bento image.
  - Deploy fingerprints still change when the shared internal feature order
    changes, without an MLflow feature artifact.
  - Training, inference construction, snapshot validation, and `_predict_proba`
    continue importing the same `FEATURE_COLS` list and preserve exact ordering.
- **Guardrails**:
  - Do not remove or dynamically infer `FEATURE_COLS`; model/scaler inputs remain
    positional even though public endpoints accept IDs rather than raw features.
  - Do not remove `bio_feature_cols.json` or its immutable pins; those columns
    describe the separately materialized embedding matrix and are still loaded
    by training and serving.
  - Do not delete existing generated files under `data/processed` without
    separate approval; stop producing and referencing them instead.

### [ ] Task 9: Align current architecture documentation

- **Description**:
  - Update current docs to describe the singleton and the division of labor:
    silver preserves NULLs, dbt gold finalizes training features, and Bento
    reads the materialized fallback/benchmark row without recomputing it.
  - Correct the README statement claiming cold starts use on-demand aggregates.
  - Mark the old date-keyed-default section in the historical drift plan as
    superseded rather than rewriting unrelated history.
- **Files**:
  - `README.md`
  - `AGENTS.md`
  - `infra/postgres/init.sql` (ownership comment only)
  - `plan-drift-monitoring.md` (minimal superseded note)
- **Acceptance Criteria**:
  - No current documentation claims defaults are date-keyed or computed on
    demand.
  - PostgreSQL and DuckDB table inventories remain accurate.
- **Guardrails**:
  - Minimal correctness edits only; no documentation expansion unrelated to the
    migration.

## Dependencies

1. Task 1 fixes the schema contract before any consumer changes.
2. Task 2 must land with Task 1 so dbt dependency resolution never references a
   removed model.
3. Task 3 establishes the shared runtime loader before Tasks 4 and 5.
4. Task 5 API changes precede Task 6 frontend formatting.
5. Task 7 validates Tasks 1–6 and controls removal of the obsolete relation.
6. Task 8 is independent of the database migration but must precede final
   lineage/deploy verification.
7. Task 9 follows the final verified behavior.

## QA/Testing Scenarios

1. **Singleton build**: dbt creates exactly one row with `singleton_id = 1`,
   source-anchored metadata, finite defaults, and valid benchmark bounds.
2. **Weighted-rate parity**: each `tour_*` rate equals a direct `SUM / SUM`
   recomputation from `silver.player_matches` within floating tolerance.
3. **First-match training**: both player perspectives lacking a prior snapshot
   receive singleton defaults and produce one finite canonical match row.
4. **Partial missingness**: a known player with missing ranking, zero-denominator
   serve metrics, unseen surface, unknown handedness, or missing `turned_pro`
   receives only the required fallback cells; observed cells remain unchanged.
5. **Strict-prior state**: current-match rolling results never enter that match's
   training features; accepted leakage is limited to the singleton fallback.
6. **Scalar inference**: each prediction performs one singleton lookup,
   preserves as-of snapshot/H2H behavior, and uses fallbacks only where needed.
7. **Bulk inference**: one batch performs one singleton lookup while retaining
   per-row snapshot and activity dates.
8. **Profile comparison**: player above average displays `(+N.NN%)` in green;
   below average displays `(-N.NN%)` in clay/red; equal is neutral; missing tour
   rate displays no delta.
9. **Freshness**: a completed dbt rebuild is visible to the next Bento request;
   no process restart or cache invalidation is required.
10. **Training snapshot**: DuckDB still contains exactly `gold.match_features`
    and `gold.player_profiles`, with all 36 model features finite.
11. **Migration**: after consumer verification and explicit cleanup, the old
    `gold.feature_defaults` physical table is absent and no code or dbt reference
    remains.
12. **Lineage simplification**: the training pipeline creates no split MLflow
    run or feature-column artifacts, while every training/serving matrix still
    selects columns using the shared internal `FEATURE_COLS` order.

## Final Verification Commands

Use the repository's existing `just`/`uv` workflows during implementation:

- Run the focused dbt build/tests for `tour_averages` and downstream
  `match_features`, then the full dbt test suite.
- Run focused inference, profile-service, snapshot, and frontend checks.
- Run the full Python test suite.
- Run frontend typecheck/build.
- Refresh and validate the DuckDB snapshot only after PostgreSQL gold passes.

Do not run destructive database cleanup without explicit approval at execution
time.
