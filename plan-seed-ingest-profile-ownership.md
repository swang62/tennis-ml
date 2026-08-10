# Plan: bootstrap ingestion, scraping, and aggregate-only profiles

## Goal

Establish clear ownership and executable entry points:

- `src/db/ingest.py` is a non-runnable library for all database ingestion
  operations.
- `src/db/seed.py` is the initial-bootstrap CLI.
- `src/flows/scrape.py` is the Prefect flow for post-seed web scraping.
- `gold.player_profiles` is created only by `etl.py` through dbt and stores
  only `player_id` plus derived aggregates.
- `bronze.player_profiles` exclusively stores ATP profile metadata and
  Wikipedia enrichment.
- `justfile` contains no direct Python file/module invocations; it calls only
  installed project commands or external tools.

## Confirmed behavior

### Initial seed

- `db-seed`: deterministic miniset, bronze metadata, and all locally available
  official top-200 ranking history for those seeded canonical players.
- `db-seed --all`: all supported local regular-tour match CSVs, bronze
  metadata, and ranking history only for the resulting player set.
- `db-seed --enrich`: the only opt-in seed mode that calls Wikipedia;
  `--all --enrich` is valid.
- Missing approved ranking-map coverage is reported and skipped, never guessed.
- Initial seed is offline unless `--enrich` is explicitly supplied.

### Database lifecycle

1. `db-init` creates schemas and bronze-owned relations only.
2. `db-seed` writes bronze matches, profiles, and filtered official rankings.
3. `db-etl` invokes `src/flows/etl.py`, whose dbt build creates silver models,
   gold aggregates, and `gold.player_profiles`.
4. Before step 3, the Bento data APIs degrade to empty player results; prediction
   is unavailable because there are no selectable players.

### Post-seed operations

- `src/flows/scrape.py` owns Prefect orchestration for current/future web
  scraping: ATP rankings now; match scraping later when actually implemented.
- `src/db/ingest.py` exposes database ingestion operations for new supplied
  data; it has no CLI and performs no scraping orchestration.
- No automatic Wikipedia enrichment runs after bootstrap. The existing removed
  `db-enrich` justfile recipe is not restored by this work.

## Scope boundaries

### In scope

- Move/rename modules and repair all imports, tests, docs, and Prefect
  deployment entrypoints.
- Seed flag behavior and filtered historical ranking import.
- Gold/bronze profile ownership split and consuming SQL joins.
- Project command entrypoints and justfile cleanup.
- Regression coverage for fresh bootstrap, idempotency, and ownership.

### Out of scope

- Implementing future match scraping.
- Changing the approved ranking identity-map policy.
- Running dbt, seed, or enrichment automatically during Bento startup.
- Adding another persisted metadata projection or changing model features.

## Tasks

### [ ] Task 1: Move ingestion into a non-runnable database library

- **Files:** `src/flows/ingest.py` -> `src/db/ingest.py`, `src/db/seed.py`,
  `src/flows/rankings.py`, `src/countries.py`, `tests/test_ingest.py`,
  `tests/test_e2e_ingest_to_inference.py`, `tests/test_dbt_helper.py`,
  `dbt/models/sources.yml`, `AGENTS.md`
- **Description:**
  - Relocate the existing ingestion implementation to `src/db/ingest.py`.
  - Remove its `__main__` block and direct CSV command-line interface.
  - Preserve it as the single source for raw-row validation, ATP transforms,
    bronze match/profile/ranking writes, approved-map validation, and bronze
    Wikipedia helpers.
  - Update every import, comment, docstring, test module path, dbt source
    description, and operational reference.
- **Acceptance criteria:**
  - There are no production or test imports of `src.flows.ingest`.
  - Running `src/db/ingest.py` does not offer or execute a CLI operation.
  - Existing input validation and idempotent write contracts remain unchanged.
- **Guardrails:** Do not move Prefect flows or network/browser orchestration
  into `src/db`.

### [ ] Task 2: Rename the post-seed rankings flow to scrape

- **Files:** `src/flows/rankings.py` -> `src/flows/scrape.py`,
  `infra/prefect/worker.py`, `tests/test_rankings_flow.py` ->
  `tests/test_scrape_flow.py`, `justfile`, `README.md`, `pyproject.toml`
- **Description:**
  - Rename the Prefect module while retaining the existing weekly ranking
    watermark, browser lifecycle, parser, mapping, and independent-week commit
    behavior.
  - Change its database imports to `src.db.ingest`.
  - Update the Prefect deployment entrypoint and worker registration import.
  - Make `scrape.py` the documented future home for post-seed match scraping;
    do not add an unrequested match scraper.
- **Acceptance criteria:**
  - The Monday rankings deployment registers and runs from
    `src/flows/scrape.py`.
  - No active code, command, or documentation points at `rankings.py`.
  - Current scrape behavior and parser tests remain unchanged after rename.
- **Guardrails:** Preserve the no-challenge-bypass policy and the empty-table
  behavior that avoids a historical web scrape.

### [ ] Task 3: Make seed the complete initial-bootstrap orchestrator

- **Files:** `src/db/seed.py`, `src/db/ingest.py`, `tests/test_seed.py`,
  `tests/test_ingest.py`, `README.md`
- **Description:**
  - Add `--enrich` independently of `--all`.
  - Keep seed responsible for local archive discovery, deterministic miniset
    selection, and sequencing shared ingestion-library operations.
  - Have both miniset and all-data paths derive and retain their canonical
    player-ID set.
  - Import every matching local official top-200 ranking-history row for that
    set, after profile writes.
  - Add an optional canonical-ID filter to the ranking ingestion operation;
    report seeded players absent from the reviewed map and continue.
  - Run bronze Wikipedia enrichment only when `--enrich` is provided.
- **Acceptance criteria:**
  - Default seed is offline and imports no ranking data for unseeded players.
  - `--all`, `--enrich`, and their combination are deterministic and idempotent.
  - Mapping gaps are visible in output and never resolved by name matching.
- **Guardrails:** Seed orchestrates; it must not duplicate validation or SQL
  write logic held by `src.db.ingest`.

### [ ] Task 4: Enforce bronze-only metadata and aggregate-only gold profiles

- **Files:** `infra/postgres/init.sql`, `dbt/models/gold/player_profiles.sql`,
  `dbt/models/gold/player_profiles.yml`, `dbt/models/sources.yml`,
  `src/constants.py`, `src/flows/etl.py`, relevant dbt tests
- **Description:**
  - Preserve the user-updated init behavior: do not create
    `gold.player_profiles` in init SQL.
  - Make dbt's `gold.player_profiles` output `player_id` plus only derived
    counts, rankings, service/return metrics, surface values, and rolling
    values.
  - Remove all copied bronze fields from gold: name variants, biographical and
    physical attributes, coach/style fields, IOC, summary, and enrichment time.
  - Correct ETL documentation to state that dbt creates aggregate gold tables
    and does not enrich/copy profile metadata.
  - Rename/reword constants whose comments imply gold owns consumer metadata.
  - Let the first dbt build replace an existing legacy wide
    `gold.player_profiles` relation; no manual data migration is required
    because bronze remains the metadata system of record.
- **Acceptance criteria:**
  - Immediately after `db-init` and `db-seed`, `gold.player_profiles` is absent.
  - `db-etl` creates it with only `player_id` and derived columns.
  - No dbt model writes or duplicates bronze profile metadata.
- **Guardrails:** Do not create an empty gold-profile stub merely to make the
  API boot; the serving fallback handles pre-ETL absence.

### [ ] Task 5: Join ownership tables at every metadata consumer

- **Files:** `src/serving/service.py`, `src/features/inference.py`,
  `src/models/similarity.py`, `src/db/ingest.py`, `tests/test_service_*.py`,
  `tests/test_inference_features.py`, `tests/test_similarity.py`
- **Description:**
  - Update `/players` and `/player_profile` to join bronze metadata to the
    gold aggregate row by `player_id`.
  - Read match-history opponent names directly from bronze metadata.
  - Update inference and similarity profile queries to select bronze metadata
    and gold-derived values explicitly rather than treating gold as a wide
    profile table.
  - Make all enrichment reads/writes target bronze only.
  - Retain a narrowly-scoped **missing-relation-only** fallback for pre-ETL
    empty-state data endpoints.
- **Acceptance criteria:**
  - API response contracts remain unchanged after ETL.
  - Enrichment changes persist in bronze and appear in serving results after
    the next ETL without duplicate storage in gold.
  - A fresh initialized database returns an empty `/players` list rather than
    a 500 response.
- **Guardrails:** Do not mask unexpected post-ETL query failures as empty data.

### [ ] Task 6: Replace direct Python recipes with project commands

- **Files:** `justfile`, `pyproject.toml`, entry-point modules for init, ETL,
  seed, snapshot, deploy, scrape, training, drift check, and worker; tests/docs
  that cite commands
- **Description:**
  - Add a prominent justfile header stating that recipes must not invoke Python
    modules/files directly or contain ad-hoc Python snippets, and that recipe
    arguments are passed directly (`just recipe --arg`), never after `--`.
  - Define installed `[project.scripts]` entry points with small callable `main`
    functions for each supported Python operation currently launched by just.
  - Change recipes such as `db-init`, `db-reset`, `db-seed`, `db-etl`,
    `db-snapshot`, `deploy-bento`, `rankings-fetch`, `train`, `worker`, and
    `check-drift` to call those installed commands.
  - Keep direct external-tool recipes (`docker`, `kubectl`, `k3d`, `uv`,
    `pytest`, `pre-commit`) unchanged where appropriate.
  - Do not restore the removed `db-enrich`, `db-rankings`, or `db-dbt` recipes.
- **Acceptance criteria:**
  - No just recipe contains `python`, `python -m`, or `python -c`.
  - Every supported Python operation is callable directly through its installed
    console-script command.
  - `setup` and dependent recipes still run in dependency order.
- **Guardrails:** Use one small entry point per existing application action;
  do not add a generic command framework or duplicate orchestration.

### [ ] Task 7: Finalize docs and regression coverage

- **Files:** `README.md`, `AGENTS.md`, `tests/test_seed.py`,
  `tests/test_ingest.py`, `tests/test_scrape_flow.py`,
  `tests/test_e2e_ingest_to_inference.py`, `tests/test_dbt_helper.py`,
  `tests/test_service_*.py`, `tests/test_similarity.py`,
  `tests/test_inference_features.py`
- **Description:**
  - Replace outdated command/path references and the old match-only seed
    contract.
  - Cover the module moves, installed commands, all seed flag combinations,
    filtered ranking history, unmapped reporting, bronze-only enrichment, and
    aggregate-only gold schema.
  - Cover the full lifecycle: fresh database -> empty API data -> seed -> ETL
    -> populated serving/inference behavior.
- **Acceptance criteria:**
  - No user-facing documentation describes a removed recipe/path.
  - Focused unit, dbt, and end-to-end tests prove all stated contracts.

## Dependencies

1. Task 1 before Tasks 2, 3, and 5.
2. Task 2 before the scrape console-script/justfile change in Task 6.
3. Task 4 before Task 5.
4. Task 6 after all callable entry points are identified by Tasks 1-3.
5. Task 7 validates the completed sequence.

## Final audit checklist

- [ ] No `src/flows/ingest.py` or `src/flows/rankings.py` references remain.
- [ ] No just recipe invokes a Python module/file directly.
- [ ] No init/seed path creates or writes `gold.player_profiles`.
- [ ] No gold profile column duplicates bronze metadata.
- [ ] No automatic Wikipedia calls occur outside `db-seed --enrich`.
- [ ] Justfile usage and README examples pass arguments as `just recipe --arg`,
      never `just recipe -- --arg`.
- [ ] Empty-state serving fallback handles only an absent dbt relation, never
      an unexpected missing column.
- [ ] Post-seed ranking scraping writes through `src.db.ingest` and remains
      independent from base-history seeding.
