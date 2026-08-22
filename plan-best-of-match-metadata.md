# Plan: best-of match metadata

## Goal

Make `best_of` a required match context in the prediction UI and API, and add it to the shared training and inference feature contract.

## Scope

- Accept `best_of` values `1`, `3`, and `5`.
- Preserve a valid source value when it is present.
- For a missing or invalid source value, derive it in this order:
  1. `round == "rr"` becomes `1`.
  2. `tournament == "grand_slam"` becomes `5`.
  3. A score with four or more completed sets becomes `5`.
  4. All other matches become `3`.
- Keep `round` and `best_of` independent in the UI and API. Users may combine any allowed values.
- Keep player order and prediction symmetry unchanged.

## Out of scope

- Retaining other discarded raw CSV metadata.
- Changing the tournament or round encodings.
- Automatically coupling the Round Robin selector to `best_of`.
- Migrating existing production data in place. The operator will rebuild bronze with the existing forced seed path.
- Broad test expansion, test cases that repeat existing coverage, and explanatory comments that restate the code.

## Tasks

### [ ] Task 1: Persist and validate `best_of` in bronze

- **Description**: Add `best_of` to `bronze.match_events` with a database-level domain of `1`, `3`, or `5`. Make the schema migration safe for an existing database. Add the column to the canonical bronze-column contract and enforce the same domain in ingestion validation.
- **Files**: `infra/postgres/schema.sql`, `src/features/columns.py`, `src/features/validate.py`, `tests/test_validate.py`
- **Acceptance criteria**:
  - A migrated database accepts only `1`, `3`, and `5` for `best_of`.
  - Existing bronze insert and upsert paths include the new column.
  - Invalid `best_of` data fails the hermetic validation test before a database write.
- **Guardrails**: Do not add a separate migration system or alter unrelated bronze columns.

### [ ] Task 2: Populate source and fallback values during ingestion

- **Description**: Extend the raw ATP CSV input contract to retain `best_of`. Add one shared normalizer that accepts only `1`, `3`, and `5`; otherwise applies the agreed fallback precedence from the Goal. Use it in `atp_rows_to_bronze` and in the Hawkeye-to-bronze mapper. Ensure the raw CSV writer retains the normalized value after a scrape.
- **Files**: `src/db/ingest.py`, `src/flows/matches.py`, `tests/test_ingest.py`, `tests/test_matches_fetch.py`, `tests/test_matches_csv.py`, `tests/test_matches_upsert.py`
- **Acceptance criteria**:
  - A valid source `best_of` survives raw CSV to bronze unchanged.
  - Missing, zero, malformed, and unsupported source values follow the exact `rr`, Grand Slam, score-length, then `3` fallback order.
  - Hawkeye rows write the same normalized value to bronze and the appended raw CSV.
  - The bronze upsert includes `best_of` when updating an existing row.
- **Guardrails**: Do not infer a best-of-five value from a three-set score unless the Grand Slam rule applies. Do not use player profile data for this fallback. Test the fallback table once at the ingestion seam. Add only a concise comment where the precedence is non-obvious.

### [ ] Task 3: Add `best_of` to dbt and the model feature contract

- **Description**: Select bronze `best_of` into `gold.match_features`, add the numeric `best_of` feature to the context feature list, and update the dbt schema and non-null/finite checks. Let the existing snapshot validation and all model families consume the expanded `FEATURE_COLS` contract.
- **Files**: `dbt/models/gold/match_features.sql`, `dbt/models/gold/match_features.yml`, `dbt/tests/gold/match_features_no_null_model_features.sql`, `src/features/columns.py`, `src/db/snapshot.py`, `tests/test_snapshot.py`
- **Acceptance criteria**:
  - Each reciprocal gold row has the same `best_of` value for its physical match.
  - Gold has a finite numeric `best_of` feature with only `1`, `3`, or `5`.
  - Snapshot validation accepts the new ordered feature contract and rejects missing or invalid values.
  - Training inputs for linear, GBDT, and neural models all receive the same expanded feature list.
- **Guardrails**: Add one numeric feature only. Do not one-hot encode `best_of` or create model-specific feature lists. Extend existing contract tests instead of adding overlapping tests.

### [ ] Task 4: Match inference to gold feature construction

- **Description**: Add a shared inference codebook and validation for `best_of`, carry it through normalized request context, and emit the numeric feature in `_assemble_row`. Update scalar and bulk builders so they produce identical rows.
- **Files**: `src/features/inference.py`, `tests/test_inference_features.py`, `tests/test_inference_units.py`
- **Acceptance criteria**:
  - Inference accepts only `1`, `3`, or `5` and rejects booleans, missing values, and other integers.
  - The emitted column is named and ordered exactly as `FEATURE_COLS` requires.
  - Scalar and bulk inference agree for each allowed value.
  - Swapping player IDs preserves the existing complementary-probability invariant.
- **Guardrails**: Preserve the current strict-prior snapshot lookup and cold-start behavior. Do not sort player IDs. Cover the new value domain and existing parity invariant only.

### [ ] Task 5: Expose the required API and frontend control

- **Description**: Add a public serving enum and required `best_of` field to the Pydantic request schema. Thread the value to both inference orientations. Add the TypeScript union, request payload field, and a required selector beside Surface in the H2H prediction controls.
- **Files**: `src/serving/service.py`, `web/src/api.ts`, `web/src/pages/H2H.tsx`, `tests/test_service_data_endpoints.py`
- **Acceptance criteria**:
  - The API rejects omitted or unsupported `best_of` values and still forbids unknown request fields.
  - A valid `best_of` reaches both directional prediction rows unchanged.
  - The H2H form sends one of `1`, `3`, or `5` with every prediction request.
  - The selector does not change when the user chooses Round Robin.
- **Guardrails**: Do not change existing tournament, round, venue, or date defaults. Do not expose raw numeric tournament or round encodings. Extend existing API tests rather than adding UI test infrastructure.

### [ ] Task 6: Rebuild data and produce a compatible champion

- **Description**: After deploying the schema and ingestion code, run the existing force seed path so already-stored match IDs are replaced with normalized `best_of` values. Then rebuild dbt gold, create a new training snapshot, retrain, evaluate, promote only if the existing gate passes, and materialize a new Bento.
- **Files**: No source changes. Uses existing `just` recipes and flow commands.
- **Acceptance criteria**:
  - Bronze rows hold only `1`, `3`, or `5` after the forced seed.
  - dbt builds gold with the new feature and no null-feature failures.
  - The candidate manifest, champion lineage, and Bento all pin the new feature contract.
  - Deployment resolves the promoted pinned model and continues to serve predictions.
- **Guardrails**: `just etl` does not migrate or repopulate bronze. Run the existing schema migration and forced seed before ETL. Do not promote a model that fails the established evaluation gate.

## Dependencies

1. Task 1 before Task 2.
2. Tasks 1 and 2 before Task 3.
3. Task 3 before Tasks 4 and 5.
4. Tasks 4 and 5 before Task 6.

## QA scenarios

- A raw ATP row with `best_of=5` remains `5` through bronze, gold, snapshot, and inference.
- A missing `best_of` Round Robin row becomes `1`.
- A missing `best_of` Grand Slam row with a three-set score becomes `5`.
- A missing non-Grand-Slam row with four completed sets becomes `5`.
- A missing non-Grand-Slam row with three or fewer completed sets becomes `3`.
- The API rejects `best_of=0`, `best_of=2`, `best_of=true`, and an omitted `best_of`.
- The H2H request includes the selected value for `1`, `3`, and `5`.
- Run the narrow affected pytest/dbt tests during development, then `just lint` and the full test suite before handoff.
