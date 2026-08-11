# Plan: bootstrap seed, rankings, and profile-schema ownership

## Goal

Make `db-seed` a complete bootstrap for the selected local match corpus:
bronze matches, player metadata, matching official top-200 ranking history,
and optional Wikipedia enrichment via `--enrich`. Complete the bronze/gold
profile ownership split and apply only query-backed schema improvements
required by the resulting seed and read paths.

## Reconciliation audit

- The current code still uses `src/flows/ingest.py` and `src/flows/rankings.py`;
  the planned `src/db/ingest.py` and `src/flows/scrape.py` moves have not happened.
- `src/db/seed.py` currently seeds matches and profiles, but rank-history data is
  loaded separately by `ingest_rankings()` into `bronze.rankings`. This explains
  the empty profile rank-history chart after a normal seed.
- The serving endpoint reads `bronze.rankings` directly for `/rank_history`, so
  no dbt model is required to make seeded rank history visible.
- The existing ranking import already validates the reviewed identity map,
  filters to official top-200 rows, deduplicates `(ranking_date, player_id)`, and
  upserts idempotently. Seed should reuse that path rather than reimplement it.
- Schema optimization is not a general rewrite: keep the normalized bronze/
  silver/gold design and add or retain only constraints/indexes validated against
  seed upserts, latest-rank lookup, chronological rank-history lookup, and the
  bronze-to-gold player joins.

## Scope

### In scope

- Move ingestion ownership out of `src/flows` without changing ingestion rules.
- Make the seed path import locally available, filtered rank history for seeded
  canonical players.
- Preserve separate weekly post-bootstrap rankings catch-up.
- Add optional seed-time Wikipedia enrichment through an independent
  `--enrich` flag.
- Establish bronze metadata versus gold derived-profile ownership.
- Audit and tune the affected PostgreSQL relation definitions for the known
  write/read paths.
- Update focused, self-contained tests and user-facing commands/docs.

### Out of scope

- New match scraping, live ranking backfill from seed, or automatic Wikipedia
  enrichment outside explicit `db-seed --enrich`.
- Changing the reviewed ranking identity-map policy.
- Broad schema redesign, speculative indexes, partitioning, or model-feature
  changes.
- Running the seed, dbt, or destructive database commands during this work.

## Tasks

### [x] Task 1: Establish the seed bootstrap flags and rank-history contract

- **Files:** `src/db/seed.py`, `src/flows/ingest.py`, `tests/test_seed.py`,
  `tests/test_ingest.py`
- **Description:**
  - Keep `seed.py` as the bootstrap orchestrator and add `--enrich` as an
    independent, non-mutually-exclusive optional flag alongside `--all`.
    It solely enables Wikipedia enrichment for the player set selected by the
    chosen seed mode.
  - Derive the canonical player-id set from the exact seeded miniset or `--all`
    match corpus.
  - Extend the existing ranking import operation with an optional canonical-id
    filter and call it from seed after match/profile writes.
  - Import only locally available official top-200 rows for the seeded players;
    retain map validation, unmapped reporting, deduplication, and idempotent
    upsert behavior.
  - Run the existing bronze Wikipedia enrichment only for the seeded player set
    when `--enrich` is supplied; `--all --enrich` is valid.
  - Report when no ranking files exist or a seeded player has no approved-map
    coverage; never guess by name.
- **Acceptance criteria:**
  - A default offline seed populates `bronze.rankings` only for selected seeded
    players when local ranking CSVs are available.
  - Re-running the same seed does not duplicate rankings.
  - Default and `--all` seeds make no Wikipedia request; `--enrich` enriches
    only the selected player set.
  - An empty local ranking archive remains a successful seed with explicit
    zero-import output.
- **Guardrails:** The only seed-time network operation is explicit
  `--enrich`; do not add ranking browser work or duplicate ranking
  parsing/write logic.

### [x] Task 2: Move database ingestion ownership without behavior changes

- **Files:** `src/flows/ingest.py` -> `src/db/ingest.py`, `src/db/seed.py`,
  `src/flows/rankings.py`, `src/countries.py`, `tests/test_ingest.py`,
  `tests/test_seed.py`, `tests/test_e2e_ingest_to_inference.py`,
  `tests/test_dbt_helper.py`, `dbt/models/sources.yml`, `AGENTS.md`
- **Description:**
  - Relocate raw CSV validation, transforms, bronze writes, ranking import, and
    profile-enrichment helpers to a non-runnable database library.
  - Repair all import paths and references while preserving public behavior.
- **Acceptance criteria:**
  - No production or test import remains on `src.flows.ingest`.
  - The moved module has no CLI entry point.
  - Existing validation and idempotency tests still cover the same boundaries.
- **Guardrails:** Do not move Prefect orchestration or browser code into `src/db`.

### [x] Task 3: Keep weekly catch-up separate from initial seed

- **Files:** `src/flows/rankings.py` -> `src/flows/scrape.py`,
  `infra/prefect/worker.py`, `tests/test_rankings_flow.py` ->
  `tests/test_scrape_flow.py`, `justfile`, `README.md`, `pyproject.toml`
- **Description:**
  - Rename the weekly rankings Prefect flow to its post-seed scraping role.
  - Keep its watermark behavior: it must not attempt an unbounded historical
    browser scrape when `bronze.rankings` is empty.
  - Point it at the moved ingestion library for its validated database writes.
- **Acceptance criteria:**
  - The Monday deployment retains its parser, map validation, and independent
    per-week commit behavior.
  - The initial seed and weekly catch-up are independently runnable.
- **Guardrails:** Do not implement future match scraping or challenge bypasses.

### [x] Task 4: Complete bronze/gold profile ownership

- **Files:** `infra/postgres/init.sql`, `dbt/models/gold/player_profiles.sql`,
  `dbt/models/gold/player_profiles.yml`, `dbt/models/sources.yml`,
  `src/constants.py`, `src/flows/etl.py`, `src/serving/service.py`,
  `src/features/inference.py`, `src/models/similarity.py`, relevant dbt tests
- **Description:**
  - Leave player metadata and enrichment exclusively in `bronze.player_profiles`.
  - Have dbt create `gold.player_profiles` only after ETL, containing `player_id`
    and derived aggregates.
  - Update consumers to join bronze metadata and gold aggregates explicitly.
  - Retain only the narrow pre-ETL missing-relation empty-state fallback.
- **Acceptance criteria:**
  - `db-init` and `db-seed` do not create or write a gold profile relation.
  - After ETL, API payloads remain contract-compatible and metadata is not
    duplicated in gold.
- **Guardrails:** Do not add a gold stub table or conceal unexpected post-ETL
  query failures as empty data.

### [x] Task 5: Audit and apply minimal schema optimizations

- **Files:** `infra/postgres/init.sql`, `dbt/models/sources.yml`,
  `dbt/models/gold/player_profiles.sql`, relevant dbt schema tests,
  `tests/test_init_db.py`, `tests/test_service_profile.py`
- **Description:**
  - Audit actual relation constraints and indexes before changing them.
  - Ensure `bronze.rankings` supports its idempotent conflict key and the two
    serving patterns: latest rank per player and chronological rank history per
    player.
  - Ensure the bronze profile key and the gold-profile player key support the
    explicit ownership joins.
  - Prefer existing primary/unique keys where they already provide the required
    access path; add no index without a named query it supports.
- **Acceptance criteria:**
  - Schema definitions enforce the ranking identity required by upserts.
  - Rank-history and current-rank queries have an intentional access path.
  - No redundant index duplicates a primary-key or unique-key index.
- **Guardrails:** No partitioning, materialized rank-history copy, broad
  denormalization, or performance claim without an inspected query plan.

### [x] Task 6: Normalize commands, documentation, and regression coverage

- **Files:** `justfile`, `pyproject.toml`, `README.md`, `AGENTS.md`,
  `tests/test_seed.py`, `tests/test_ingest.py`, `tests/test_scrape_flow.py`,
  `tests/test_init_db.py`, `tests/test_service_profile.py`,
  `tests/test_e2e_ingest_to_inference.py`
- **Description:**
  - Document the bootstrap sequence: init -> seed (including local rank history
    and optional `--enrich`) -> ETL -> weekly catch-up.
  - Update command/module references after the ownership moves.
  - Add self-contained tests using mocked database boundaries or fixtures only.
- **Acceptance criteria:**
  - Documentation no longer says rank history requires a separate manual step
    after normal seed.
  - Tests cover miniset and `--all` rank filtering, each `--enrich` combination,
    no-ranking-file behavior, idempotency, and pre/post-ETL serving behavior.
- **Guardrails:** Tests must not require a live database or prebuilt gold state.

### [x] Task 7: Reduce dbt validation to critical contracts

- **Files:** `dbt/models/sources.yml`, `dbt/models/silver/*.yml`,
  `dbt/models/gold/*.yml`, `dbt/tests/**/*.sql`, `dbt/dbt_project.yml`,
  `README.md` only if its validation commands change
- **Description:**
  - Inventory existing dbt schema and singular tests by the contract they prove.
  - Retain only critical tests: source/model identity and referential integrity,
    non-null model-training columns, finite/bounded values where the bound is a
    real business invariant, prevention of current-match leakage, and the
    bronze/gold ownership plus ranking-grain contracts established by this plan.
  - Delete or consolidate duplicated, derivable, observability-only, and
    overlapping tests that materially lengthen every `dbt build` without adding
    a distinct safety property.
  - Keep a compact written test inventory explaining each retained contract.
- **Acceptance criteria:**
  - Every retained dbt test maps to a distinct critical data/feature contract.
  - Redundant tests are removed rather than merely disabled.
  - `dbt build` still validates the canonical training dataset, ownership split,
    ranking identity, and leakage boundary.
- **Guardrails:** Do not weaken unique/not-null/relationship guarantees, model
  feature validity, or leakage checks. Do not delete tests solely to make a
  failure disappear; fix an invalid critical contract instead.

## Dependencies

1. Task 1 defines the behavior to preserve during Task 2.
2. Task 2 precedes Tasks 3 and 4 import changes.
3. Task 4 precedes consumer and schema verification in Task 5.
4. Task 6 validates the completed lifecycle after Tasks 1-5.
5. Task 7 runs after Task 4's ownership and Task 5's schema contracts are final.

## QA scenarios

1. Fresh database, ranking CSVs present: seed imports only selected players'
   official rank rows; `/rank_history` returns chronologically ordered data.
2. Fresh database, no ranking CSVs: seed succeeds, reports no import, and the
   rank chart has no data without an API failure.
3. Repeated identical seed: match, profile, and ranking row counts do not grow.
4. `--enrich` enriches only the selected players; default and `--all` runs make
   no enrichment request.
5. Seeded player missing from the reviewed map: output reports the gap and no
   name-based mapping occurs.
6. Pre-ETL: player data endpoints degrade only for the absent derived relation.
7. Post-ETL: bronze metadata plus gold aggregates produce the existing profile,
   inference, and similarity contracts.
8. Explain/schema inspection confirms every retained or added index serves a
   named seed or read query.
9. dbt test inventory maps every retained validation to a distinct critical
   contract; the canonical training and leakage checks still pass.
