# Plan: Cleanup, Test Suite, Bugfixes

## Goal

Three deliverables in the tennis-ml repo:

1. **Shared dbt helper** — dedupe the identical `dbt build` subprocess invocation
   in `src/flows/etl.py` and `tests/test_inference_features.py`.
2. **Bugfixes + cleanup** — remove dead code, fix SQL-parametrization and
   player-id bugs in the ingest path, delete an orphaned script, align stale
   docs with the current 54-feature contract.
3. **Full pytest suite** — everything reasonable that does NOT depend on a
   cluster / MLflow / Prefect / BentoML / Docker / k3d. The only external
   dependency allowed is a local `dbt build` (gold-layer DB-backed tests),
   gated behind a `gold` marker so the fast suite skips it.

## Scope Boundaries

- IN: `src/utils.py`, `src/flows/etl.py`, `src/flows/ingest.py`, `src/db/client.py`,
  `tests/`, `pyproject.toml` (pytest config only), `justfile` (test targets only),
  `infra/duckdb/make_samples.py` (delete), `README.md` (contract text only).
- OUT: `src/serving/`, `src/flows/deploy.py`, `src/flows/pipeline.py`,
  `notebooks/`, `dbt/`, `infra/k3d/`, `infra/manifests/`, MLflow/Prefect/Bento
  behavior. No new dependencies. No changes to `data/tennis.duckdb` by tests
  except the existing gold-bootstrap fixture.
- Working-tree caveat: the repo is mid-refactor (seed.sql/samples.csv removed,
  `infra/duckdb/seed.py` + new `ingest.py` API landed, README actively edited).
  Every task below targets the CURRENT on-disk state; re-verify each file's
  contents before editing.

## Tasks

### Task 1: Add `run_dbt_build()` to src/utils.py

- **Description**: Add a prefect-free helper that runs the gold `dbt build`.
  ```python
  import subprocess

  DBT_BUILD_CMD = ["uv", "run", "dbt", "build", "--project-dir", "dbt", "--profiles-dir", "dbt"]

  def run_dbt_build() -> subprocess.CompletedProcess:
      """Run `dbt build` (gold layer) from the repo root; raise on failure."""
      return subprocess.run(DBT_BUILD_CMD, cwd=ROOT, check=True)
  ```
  `ROOT` is already imported in `src/utils.py`. `subprocess` is added at the
  top. Module stays free of prefect/mlflow/bentoml imports.
- **Files**: `src/utils.py`
- **Acceptance Criteria**: `run_dbt_build()` importable without importing
  prefect/mlflow/bentoml; command string is exactly
  `["uv","run","dbt","build","--project-dir","dbt","--profiles-dir","dbt"]`;
  `check=True` and `cwd=ROOT`.
- **Guardrails**: Do not add a new module; reuse `src/utils.py` per decision.
  Do not import src.flows (prefect).

### Task 2: Use the helper in the ETL flow

- **Description**: `src/flows/etl.py` `bronze_to_gold` replaces its inline
  `subprocess.run(["uv","run","dbt","build",...], cwd=ROOT, check=True)` with
  `run_dbt_build()`. Keep the task decorator, retries, and the gold row-count
  check unchanged.
- **Files**: `src/flows/etl.py` (import + line ~22)
- **Acceptance Criteria**: `bronze_to_gold` no longer contains a subprocess
  call or the dbt command literal; behavior identical (raises on failure,
  streams dbt output, returns gold row count).
- **Guardrails**: Do not restructure the flow; do not touch `enrich_bios`/`etl_flow`.

### Task 3: Use the helper in the test bootstrap fixture

- **Description**: In `tests/test_inference_features.py` the session fixture's
  third bootstrap command (`["uv","run","dbt","build",...]` via subprocess) is
  replaced by `run_dbt_build()`. The init/seed steps remain subprocess
  commands via `run_init.py`. Update the fixture docstring if it mentions the
  raw command.
- **Files**: `tests/test_inference_features.py` (fixture ~lines 62-73)
- **Acceptance Criteria**: Fixture runs init, seed, then `run_dbt_build()`;
  a failure still surfaces with output visible to pytest.
- **Guardrails**: Keep capture-on-failure behavior reasonable (streaming is
  acceptable since pytest captures it).

### Task 4: Delete dead `query()` from src/db/client.py

- **Description**: `query()` has no callers (verified: only definition exists)
  and double-executes SQL (`fetchall()` then `.description` = two runs). Delete
  the function. `to_dataframe` / `execute_df` remain.
- **Files**: `src/db/client.py`
- **Acceptance Criteria**: No `query(` definition or callers remain; `rg
  "query\(" src/` shows no hits for the deleted function.
- **Guardrails**: Do not touch `get_conn`, `get_client`, `to_dataframe`,
  `execute_df`.

### Task 5: Fix `enrich_player` / `enrich_missing` SQL + player-id bug

- **Description**: In `src/flows/ingest.py`:
  - Rewrite `enrich_player(name, player_id=None)` to default
    `player_id = name` (fixes `enrich_missing()` inserting the literal string
    `'None'` into `player_id`).
  - Replace the f-string INSERT with a prepared statement using `?` params for
    `player_id`, title, summary, handedness, backhand, height, turned_pro
    (DuckDB does not backslash-escape quotes; the current
    `.replace("'", "\\'")` breaks on apostrophes and is injection-style).
    `None` values bind as NULL; drop the `height_value`/`turned_pro_value`
    string sentinels and the `safe_summary`/`safe_title` escaping.
  - Keep the `ON CONFLICT (player_id) DO UPDATE` enrichment-only semantics.
- **Files**: `src/flows/ingest.py` (`enrich_player` ~440-495, `enrich_missing`
  ~528-549)
- **Acceptance Criteria**: A title/summary containing an apostrophe inserts and
  round-trips correctly; `enrich_missing()` never writes `'None'` as
  `player_id`; existing base metadata survives re-enrichment. Covered by
  regression tests in Task 14.
- **Guardrails**: Preserve `enrich_player`'s signature `(name, player_id=None)`
  and return semantics (True written / False skipped). Do not touch
  `enrich_players`.

### Task 6: Delete orphaned infra/duckdb/make_samples.py

- **Description**: `make_samples.py` generated `data/test/samples.csv` for the
  deleted `seed.sql`. Nothing references it (justfile `db-samples` removed).
  Delete the file per decision.
- **Files**: `infra/duckdb/make_samples.py`
- **Acceptance Criteria**: File gone; `rg "make_samples"` returns no hits.
- **Guardrails**: Explicitly approved deletion. `data/test/atp_sample.csv`
  stays (used as a raw-ATP fixture).

### Task 7: Align stale README /predict contract (check-only if already removed)

- **Description**: The code contract is 54 FEATURE_COLS (player_* 19,
  opponent_* 19, *_diff 11, context 5) -> 56-column rows with the two ids. If
  the README still contains the stale "45 FEATURE_COLS / 47-column row"
  section, update it to 54/56 and the composition breakdown. The README is
  being actively edited; if the stale section is already gone, skip and note.
- **Files**: `README.md`
- **Acceptance Criteria**: No "45 FEATURE_COLS" / "47-column" text remains; any
  feature-count text agrees with `src/features/columns.py` (54).
- **Guardrails**: Docs text only; do not rewrite other README sections.

### Task 8: Add pytest configuration to pyproject.toml

- **Description**: Add:
  ```toml
  [tool.pytest.ini_options]
  testpaths = ["tests"]
  markers = ["gold: requires seeded DuckDB gold tables (dbt build)"]
  ```
- **Files**: `pyproject.toml`
- **Acceptance Criteria**: `uv run pytest` discovers only `tests/`; the `gold`
  marker is registered (no PytestUnknownMarkWarning).
- **Guardrails**: No other pyproject changes.

### Task 9: Add justfile test targets

- **Description**: Add:
  ```
  # Fast hermetic tests (no DuckDB gold / dbt build)
  test:
      uv run pytest -m "not gold"

  # Full suite incl. gold DB-backed tests (builds gold once)
  test-gold:
      uv run pytest
  ```
- **Files**: `justfile`
- **Acceptance Criteria**: `just test` runs the fast suite; `just test-gold`
  runs everything.
- **Guardrails**: Do not modify existing justfile recipes.

### Task 10: New tests — test_rolling_contract.py

- **Description**: Pure contract tests for `src/features/columns.py`:
  - `FEATURE_COLS` == `PLAYER_COLS + OPPONENT_COLS + DIFF_COLS + CONTEXT_COLS`
    (order preserved); count == 54; player_* == 19; opponent_* == 19;
    diff == 11; context == 5.
  - No duplicate column names across the 56-column row (incl. ids).
  - Naming conventions: every PLAYER_COLS/OPPONENT_COLS entry carries the
    `player_`/`opponent_` prefix (except `player_ranking`/`opponent_ranking`);
    DIFF_COLS entries end in `_diff`.
  - Conditional parity: if `data/processed/feature_cols.json` exists, its list
    equals `FEATURE_COLS`; otherwise `pytest.skip` (artifact is gitignored and
    regenerated by training).
- **Files**: `tests/test_rolling_contract.py` (new)
- **Acceptance Criteria**: All pass with no DB access.

### Task 11: New tests — test_validate.py

- **Description**: `src/features/validate.py` `run_ingestion_checks`:
  - Clean minimal df (match_id, player_ranking, opponent_ranking, match_won,
    one object col) -> `passed=True`, empty results.
  - Each failure mode detected: nulls, duplicate match_id, player_ranking<=0,
    opponent_ranking<=0, mixed object types (int+str in one object column),
    match_won with >2 unique values.
  - Returns dict with `passed` and `results`; input df is not mutated.
- **Files**: `tests/test_validate.py` (new)
- **Acceptance Criteria**: All pass; no DB.

### Task 12: New tests — test_db_client.py

- **Description**: `src/db/client.py` against an in-memory duckdb by setting
  `src.db.client._conn = duckdb.connect(":memory:")` in a fixture (restore
  after). Create a temp table; assert:
  - `to_dataframe(sql)` returns a DataFrame with expected columns/rows.
  - `execute_df(sql, params)` with `?` placeholders returns correct rows
    (prepared, no string interpolation — include a literal-quote value to prove
    param binding).
  - `execute_df(sql)` without params works.
- **Files**: `tests/test_db_client.py` (new)
- **Acceptance Criteria**: All pass; real `data/tennis.duckdb` untouched.
- **Guardrails**: Reset `_conn` to `None` after the module (teardown).

### Task 13: New tests — test_inference_units.py

- **Description**: Pure units of `src/features/inference.py` (no DB):
  - `_to_date`: pd.Timestamp, datetime, date, ISO str coerce to date;
    non-coercible raises TypeError.
  - `_agg_or`: value returned; None -> default; NaN -> default.
  - `build_inference_features` boundary validation raises BEFORE any DB access:
    empty/whitespace/non-str player_id and opponent_id (ValueError); invalid
    surface/tournament/round (ValueError); `as_of_date` given an int (TypeError).
- **Files**: `tests/test_inference_units.py` (new)
- **Acceptance Criteria**: All pass with no DuckDB file (validation paths never
  reach `execute_df`).

### Task 14: New tests — test_ingest.py

- **Description**: `src/flows/ingest.py`, hermetic (mock network and DB):
  - `_parse_int`: valid ints parsed; '' and '0' -> NA; non-integer within range
    and out-of-range values raise ValueError.
  - `_parse_birthdate`: valid YYYYMMDD parsed; '' -> NA; malformed / out-of-year
    (1900-2100) raises.
  - `extract_infobox_fields`: extracts plays/backhand/height/turned_pro from a
    sample summary; missing keys absent.
  - `atp_rows_to_bronze`: tiny synthetic raw rows -> correct bronze columns
    (ISO match_date, LEVEL_MAP tournament, lowercased round/surface,
    break_points_won = faced - saved, winner on player1 side); `selected_ids`
    filter; empty rows -> empty df; `_guard_bronze` raises on out-of-range
    UTINYINT, non-positive ranking, duplicate match_id.
  - `load_atp_csv`: missing raw column raises; ineligible rows (NULL
    rank/id) dropped; real fixture `data/test/atp_sample.csv` loads to a
    non-empty bronze df.
  - `search_wikipedia` / `fetch_summary`: monkeypatched `requests.get` returning
    canned JSON (no network); no-pages path returns None.
  - `load_atp_profiles`: monkeypatched `ingest.get_conn` -> in-memory conn with
    the gold.player_profiles schema; upsert inserts base columns and preserves
    existing enrichment columns on reload; `player_ids` filter.
  - `enrich_player`: monkeypatched `search_wikipedia`/`fetch_summary` (title
    with an apostrophe) + in-memory conn; asserts player_id equals the passed
    id (regression: never 'None') and apostrophe round-trips (regression for
    the param fix).
  - `enrich_missing`: monkeypatched `get_players_without_profiles` ->
    ['X'] and `enrich_player` -> True; asserts count.
  - `enrich_players`: in-memory conn with profile rows + monkeypatched
    `enrich_player`; asserts count and that ids lacking a name are skipped.
- **Files**: `tests/test_ingest.py` (new)
- **Acceptance Criteria**: All pass with no network and no real DB file.

### Task 15: New tests — test_nn.py

- **Description**: `src/models/nn.py` `TabularBioMLP` (torch/lightning only):
  - forward on batch (2, tab_dim), (2, bio_dim), (2, bio_dim) returns shape
    (2,).
  - logits -> sigmoid in (0,1).
  - deterministic identical outputs across two forwards with dropout=0.
  - `hparams["lr"]` persisted via `save_hyperparameters`.
- **Files**: `tests/test_nn.py` (new)
- **Acceptance Criteria**: All pass; no mlflow/bentoml imports.

### Task 16: New tests — test_similarity.py

- **Description**: `src/models/similarity.py`, no network (mock
  `TextEmbedding` so no model download):
  - `embed_bio_summaries`: monkeypatched `TextEmbedding` -> fixed dim vectors;
    output has `player_id` first then `bio_0..bio_N`; empty/missing summaries
    still produce a row.
  - `PlayerSimilarity.find_by_name`: exact + case-insensitive; None on miss.
  - `search`: hand-built in-memory `faiss.IndexFlatIP` + `players`/`player_ids`;
    returns top_k, excludes self, score formatted "0.xxx"; unknown query -> [];
    single-player index -> []; empty players -> [].
  - `build()`: monkeypatched `to_dataframe` (small df), `TextEmbedding`,
    `DEFAULT_INDEX`/`DEFAULT_METADATA` -> tmp_path; builds index and files,
    `load()` reads them back.
- **Files**: `tests/test_similarity.py` (new)
- **Acceptance Criteria**: All pass with no model download and no real DB.

### Task 17: New tests — test_utils.py

- **Description**: `src/utils.py`:
  - `ensure_kernel` with `src.utils.KERNEL_DIR` monkeypatched to tmp_path:
    writes kernel.json whose argv starts with `sys.executable`; returns
    KERNEL_NAME; JUPYTER_PATH gains the repo path entry.
  - `load_env` with `src.utils.ROOT` monkeypatched to a tmp dir containing
    `.env`: sets the var; idempotent on second call.
- **Files**: `tests/test_utils.py` (new)
- **Acceptance Criteria**: All pass; no writes outside tmp_path.

### Task 18: New tests — test_dbt_helper.py

- **Description**: `src.utils.run_dbt_build` with `subprocess.run`
  monkeypatched:
  - called with exactly `DBT_BUILD_CMD`, `cwd=ROOT`, `check=True`.
  - propagates `subprocess.CalledProcessError` when the fake run raises.
- **Files**: `tests/test_dbt_helper.py` (new)
- **Acceptance Criteria**: All pass; no real dbt executed.

### Task 19: New tests — test_seed.py (light)

- **Description**: `infra/duckdb/seed.py` pure logic:
  - `select_matches` on tiny synthetic matches: picks TOP_PLAYERS by latest
    rank (tie-break player_id), keeps RECENT most-recent per player, dedupes to
    distinct (tourney_id, match_num), deterministic ordering by
    (tourney_date, tourney_id, match_num).
- **Files**: `tests/test_seed.py` (new)
- **Acceptance Criteria**: All pass; no raw CSV required, no DB.
- **Guardrails**: Do not test `seed.main()` (needs raw 2026.csv + real DB).

### Task 20: Mark existing DB-backed tests as gold

- **Description**: In `tests/test_inference_features.py` add module-level
  `pytestmark = pytest.mark.gold` so the fast suite skips it; verify the
  bootstrap fixture still builds gold once via `run_dbt_build()` (Task 3).
- **Files**: `tests/test_inference_features.py`
- **Acceptance Criteria**: `uv run pytest -m "not gold"` skips this file;
  `just test-gold` runs it and passes against a freshly seeded DB.
- **Guardrails**: Do not change the test logic/assertions.

## Dependencies

- Tasks 1-3 (helper + callers) are the core dedupe; Task 18 (helper tests)
  depends on Task 1. Tasks 14/5 (ingest fix + regression tests) pair. Task 7
  (README) is independent. Tasks 8-9 (config) precede Task 20 and any full run.
- Task order: 1 -> 2,3,18 -> 4 -> 5 -> 14 -> 6 -> 7 -> 8,9 -> 10-17,19 -> 20 -> verify.

## QA / Verification

1. `uv run ruff check src tests` — no lint regressions on changed files.
2. `uv run pytest -m "not gold"` — full fast/hermetic suite green.
3. `just test-gold` (or `uv run pytest`) — gold bootstrap (init -> seed.py ->
   run_dbt_build) succeeds and the existing inference assertions pass.
4. Sanity: `uv run python -c "from src.utils import run_dbt_build; from src.features.columns import FEATURE_COLS; print(len(FEATURE_COLS))"` prints 54 and imports cleanly (no prefect/mlflow/bentoml in the chain).
5. Confirm deletions: `rg "query\(|make_samples|45 FEATURE_COLS|47-column" src/ infra/ README.md` returns no hits (README check only if section present).
