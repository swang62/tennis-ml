# Plan: Local development and matchup predictions refinement

## Goal

Make pnpm the only JavaScript package manager, provide reliable root-level local and Docker startup commands, simplify the top navigation to two centered tabs, and consolidate the matchup page around prediction, head-to-head comparison, and direct meetings.

## Scope

### In scope

- Root-level pnpm workspace and commands.
- Verified and seeded Homebrew PostgreSQL as the active host-development database.
- Standard PostgreSQL for every operational environment; pg_duckdb removed without affecting the Python DuckDB training snapshot.
- One-command local development against an already-running Homebrew PostgreSQL configured through `.env`.
- BentoML reload plus Vite HMR lifecycle management.
- Compose startup that builds the web image before starting containers.
- Anonymous API abuse hardening without accounts, OAuth, CAPTCHA, or client-visible secrets.
- Centered two-tab navigation.
- Compact searchable player dropdowns and consolidated matchup content.
- Relevant documentation and focused tests/checks.

### Out of scope

- Installing, starting, stopping, or reconfiguring Homebrew PostgreSQL automatically.
- Installing pg_duckdb into any PostgreSQL instance.
- Guaranteeing that a public anonymous browser API cannot be called outside the SPA. Browser requests are reproducible; controls reduce attack surface and abuse rather than establishing caller identity.
- Changing API contracts, model artifacts, model promotion, deployment registry behavior, route paths, or database schemas.
- Adding SSR or SSG. The application remains a client-rendered Vite SPA whose production bundle is static files served by nginx.
- Continuous Docker file watching. HMR remains the development workflow; `pnpm docker` rebuilds when invoked.

## Tasks

### [x] Task 0: Verify and configure Homebrew PostgreSQL

- **Description**:
  - Inspect `brew services` to identify the installed/running PostgreSQL formula and version without starting, stopping, upgrading, or reinstalling it implicitly.
  - Use `pg_isready`/`psql` to determine the actual host, port, role, and database. Verify that the target is the intended local Homebrew instance, not Compose on port 6543 or a remote database.
  - In the git-ignored `.env`, comment the current Compose `DATABASE_URL` that targets `127.0.0.1:6543` and add the verified Homebrew `DATABASE_URL` as the active value. Preserve all unrelated environment values byte-for-byte and never print either URL.
  - Validate basic connectivity through psycopg before making repository changes.
- **Files**:
  - `.env` (git-ignored local configuration only)
  - `README.md` only if the local database setup instructions need correction.
- **Acceptance Criteria**:
  - The exact Homebrew PostgreSQL service/version and listening port are recorded in the verification report.
  - The active `.env` `DATABASE_URL` connects to the verified Homebrew instance; the previous Compose URL remains directly above it as a comment.
  - `.env` remains untracked and no credential or complete URL appears in terminal logs, diffs, tests, or documentation.
  - A basic parameterized psycopg query succeeds against the verified target.
- **Guardrails**:
  - Do not inspect, install, upgrade, compile, or enable pg_duckdb.
  - Do not initialize, seed, reset, or alter schemas until Task 1 removes the extension requirement.
  - Do not modify unrelated `.env` lines.

### [ ] Task 1: Remove pg_duckdb and establish standard PostgreSQL parity

- **Description**:
  - Remove `CREATE EXTENSION pg_duckdb` and extension-specific bootstrap comments from the initialization SQL so it runs on ordinary PostgreSQL 17.
  - Replace `pgduckdb/pgduckdb:17-v1.1.1` with the standard official PostgreSQL `postgres:18.4` image and simplify its health check to database readiness. Do not use an Alpine variant or derive a custom PostgreSQL image.
  - Update the Compose volume mount for PostgreSQL 18's version-specific data layout (`/var/lib/postgresql`, with the image-managed PGDATA beneath it). Never mount the existing PostgreSQL 17 data directory directly into PostgreSQL 18.
  - Preserve existing Compose data through a logical PostgreSQL migration when present: create a separate PostgreSQL 18 named volume, dump from the old PostgreSQL 17 container/image, restore into 18, validate counts/constraints, and leave the old volume and dump intact until explicit cleanup approval. If no prior Compose data exists, initialize the new 18 volume normally.
  - Remove `analytical_df` and its `duckdb.force_execution` / `duckdb.convert_unsupported_numeric_to_double` settings. Route the six inference aggregate reads through the existing parameterized `execute_df` path.
  - Audit all executable SQL in `infra/postgres`, `dbt`, Python query constants, health checks, and tests against PostgreSQL 18. Remove pg_duckdb-only casts/workarounds and extension-era comments. Retain or simplify casts required by PostgreSQL itself, especially casts preventing integer division and preserving floating-point feature semantics.
  - Compile and execute dbt models/tests on PostgreSQL 18 rather than relying on textual compatibility inspection alone.
  - Rewrite extension-specific unit/integration tests as standard PostgreSQL result, parameter-binding, cold-start-imputation, and inference parity tests. Remove diagnostics that inspect pg_duckdb fallback notices.
  - Retain the Python `duckdb` package, `src/db/snapshot.py`, PostgreSQL scanner attachment, local `.duckdb` training snapshot, and all training behavior.
  - Update architecture and operational documentation to say PostgreSQL is the operational backend and DuckDB is used only for training snapshots.
  - After the code no longer requires the extension, initialize the verified Homebrew database, load only the deterministic default seed, run ETL/dbt, and execute a representative prediction feature build through plain PostgreSQL.
- **Files**:
  - `infra/postgres/init.sql`
  - `compose.yaml`
  - `src/db/client.py`
  - `src/features/inference.py`
  - `src/flows/init_db.py`
  - `dbt/models/**/*.sql`
  - `dbt/tests/**/*.sql` where present
  - `tests/test_db_client.py`
  - `tests/test_inference_features.py`
  - `tests/test_inference_units.py`
  - `tests/conftest.py`
  - `tests/test_deploy.py`
  - `README.md`
  - `AGENTS.md` only where its architecture statement would otherwise be false
- **Acceptance Criteria**:
  - A repository search finds no operational `pg_duckdb`, `duckdb.force_execution`, or `duckdb.convert_unsupported_numeric_to_double` references; historical plan/draft text is excluded.
  - Compose uses exactly `postgres:18.4`, mounts its named volume at the PostgreSQL 18-supported parent path, and reports healthy without checking any extension.
  - Existing PostgreSQL 17 Compose data, if present, is logically migrated and validated before the PostgreSQL 18 stack becomes authoritative; the old volume remains untouched pending separate deletion approval.
  - Every SQL model/query compiles and executes on PostgreSQL 18; dbt tests pass.
  - Initialization succeeds against the verified Homebrew PostgreSQL without pg_duckdb and is idempotent.
  - Default seed produces exactly 28 distinct bronze matches and 35 players.
  - ETL produces 56 `silver.player_matches`, 56 `silver.rolling_features`, and 28 `gold.match_features` rows with exactly the existing 36 feature columns.
  - Re-running the default seed does not duplicate rows.
  - Cold-start and known-player inference feature outputs remain behaviorally identical to the existing expected fixtures, including float conversion and no-NaN guarantees.
  - A representative `/predict_from_ids` request succeeds using plain PostgreSQL.
  - Training snapshot creation still produces exactly the two expected DuckDB tables and passes existing validation.
- **Guardrails**:
  - Do not remove Python DuckDB, the PostgreSQL scanner used to build training snapshots, or `.duckdb` training artifacts.
  - Do not detect pg_duckdb, introduce conditional extension handling, or retain dual query paths.
  - Do not rewrite aggregate SQL unless standard PostgreSQL exposes a real incompatibility.
  - Do not remove PostgreSQL numeric casts merely because they were introduced during the old DuckDB migration; prove they are extension-only or preserve them for result parity.
  - Do not reuse a PostgreSQL 17 physical data directory with PostgreSQL 18.
  - Do not delete the old Compose volume or logical dump without explicit approval.
  - Never run `just db-seed --all` or any equivalent full-corpus load.
  - Never run `db-reset`, drop schemas, delete data, or overwrite unexpected existing Homebrew data without separate explicit approval.
  - If the target database already contains unexpected data, stop before seeding and request approval.

### [ ] Task 2: Establish pnpm as the repository JavaScript package manager

- **Description**:
  - Add a root `package.json` marked private with a pinned `packageManager` value and root scripts for `dev`, `docker`, `build`, and any narrowly required service-specific checks.
  - Add `pnpm-workspace.yaml` with `web` as the only package.
  - Import `web/package-lock.json` into `web/pnpm-lock.yaml`, then remove the npm lockfile.
  - Ensure all JavaScript commands in repository-owned scripts and documentation use pnpm; do not add `only-allow` or another package solely to police package-manager choice.
- **Files**:
  - `package.json` (new)
  - `pnpm-workspace.yaml` (new)
  - `web/package.json`
  - `web/pnpm-lock.yaml` (new)
  - `web/package-lock.json` (remove after successful import)
  - `justfile`
  - `README.md`
- **Acceptance Criteria**:
  - A clean `pnpm install --frozen-lockfile` succeeds from the repository root.
  - `pnpm build` runs the web typecheck and Vite production build.
  - No repository-owned runtime/build command references npm or `package-lock.json`.
  - The pnpm version used to create the lockfile is pinned in the root manifest.
- **Guardrails**:
  - Keep Python dependency management on uv.
  - Do not introduce a monorepo tool or process-manager dependency.
  - Do not change frontend dependency versions except lockfile normalization required by pnpm import.

### [ ] Task 3: Add one-command local development against Homebrew PostgreSQL

- **Description**:
  - Add a small POSIX-compatible development launcher invoked by root `pnpm dev`.
  - Load the existing repository `.env` without printing credentials.
  - Preflight the configured `DATABASE_URL` or `POSTGRES_*` components before starting servers: connection succeeds, the expected database is selected, and required application schemas/tables exist.
  - Reject an accidental Compose target such as `127.0.0.1:6543` when the requested local workflow is meant to use Homebrew PostgreSQL; report the expected `.env` keys without displaying values.
  - Start `uv run bentoml serve src/serving/service.py:TennisPredictor --host 127.0.0.1 --port 3000 --reload` and the web workspace's Vite dev server concurrently.
  - Keep Vite's existing `/api` proxy to Bento and terminate both child processes when the launcher exits or receives an interrupt.
- **Files**:
  - `package.json`
  - `scripts/dev.sh` (new)
  - `web/vite.config.ts`
  - `justfile`
  - `README.md`
- **Acceptance Criteria**:
  - With a valid Homebrew PostgreSQL target in `.env`, `pnpm dev` starts Bento on `127.0.0.1:3000` and Vite HMR on its documented port.
  - Editing a React source file updates through HMR without rebuilding Docker.
  - Browser `/api/healthz` and one read endpoint succeed through the Vite proxy.
  - Ctrl-C exits both Bento and Vite without leaving an orphan process.
  - Invalid/unreachable database configuration or missing schemas/data fails before server startup with a concise, non-secret error.
- **Guardrails**:
  - Do not start or stop Homebrew services from `pnpm dev`.
  - Do not hard-code a username, password, database name, or unknown Homebrew formula/version.
  - Do not run migrations, reset schemas, or seed data automatically.
  - Do not stop the Compose stack automatically; fail on occupied ports and let the user choose what to stop.

### [ ] Task 4: Add deterministic Docker Compose startup with web rebuild

- **Description**:
  - Define root `pnpm docker` as the explicit Compose path using `docker compose up -d --build` so the web build is evaluated before startup on every invocation.
  - Convert the web Docker build stage from npm to pnpm using Corepack and the pnpm lockfile with a frozen install.
  - Preserve nginx SPA serving, `/api` proxying, health checks, dependency ordering, PostgreSQL volume behavior, and the promoted Bento image contract.
  - Remove Bento's host-published port from the production-style Compose stack so only nginx can reach it over the internal Compose network. Keep port 3000 only for the separate host-local `pnpm dev` workflow.
  - Replace the broad nginx `/api/` proxy with exact allowlisted SPA routes and methods: required read endpoints for players/profiles/ranks/history/head-to-head plus `POST /api/predict_from_ids`. Reject model-only `/predict`, unknown API paths, wrong methods, non-JSON prediction bodies, and oversized bodies before they reach Bento.
  - Add nginx per-IP request-rate and connection/concurrency limits, bounded proxy timeouts, and generic gateway errors. Use a stricter limit for compute-heavy prediction requests than for read endpoints.
  - Preserve strict validation in Bento and confirm prediction inputs have bounded values and no caller-controlled way to trigger unbounded database/model work.
  - Update deploy-flow command construction/tests only where the pnpm Dockerfile or consolidated Compose invocation changes observable expectations.
- **Files**:
  - `package.json`
  - `compose.yaml`
  - `web/Dockerfile`
  - `web/nginx.conf`
  - `src/serving/service.py` only if an input/work bound is missing
  - `web/pnpm-lock.yaml`
  - `src/flows/deploy.py` only if command behavior must stay aligned
  - `tests/test_deploy.py` only for affected deployment assertions
  - `README.md`
- **Acceptance Criteria**:
  - `pnpm docker` builds the current web source and starts all three services.
  - A subsequent source change followed by `pnpm docker` produces a web image containing that change without a separate manual build command.
  - Docker dependency layers remain cacheable; no unconditional `--no-cache` is added.
  - PostgreSQL, Bento, and web health checks pass, and the SPA can call Bento through nginx `/api`.
  - Bento is not reachable through a host-published Compose port; nginx is the only public API entrypoint.
  - Required SPA reads and predictions succeed, while `/api/predict`, unknown paths, wrong methods, wrong content types, and oversized prediction bodies are rejected.
  - A bounded request burst demonstrates the configured prediction rate/concurrency limit without high-volume load testing.
- **Guardrails**:
  - Do not claim the SPA is SSG; document it as a Vite SPA static bundle.
  - Do not add source bind mounts or HMR to the production-style Compose stack.
  - Do not force-rebuild or republish the promoted Bento image as part of `pnpm docker`.
  - Do not remove the explicit force-rebuild behavior from the existing deployment flow.
  - Do not embed a shared token, API key, or signing secret in the SPA.
  - Do not describe CORS, Referer, or Origin checks as authentication. They may be supplemental filters only.
  - Do not add Turnstile/CAPTCHA or user authentication.

### [ ] Task 5: Center and simplify the primary navigation

- **Description**:
  - Recompose the desktop top bar so its navigation is visually centered independently of the left brand and right theme control, using a stable three-column layout rather than positional offsets.
  - Expose exactly two primary tabs: `Player Profiles` linking to `/` and `Matchup Predictions` linking to `/h2h`.
  - Keep player detail pages under the Player Profiles tab's active navigation context.
  - Simplify the mobile layout for two tabs while preserving the theme toggle, keyboard navigation, active indication, and a single-line desktop bar.
- **Files**:
  - `web/src/router.tsx`
  - `web/src/index.css`
- **Acceptance Criteria**:
  - Desktop displays exactly two centered tabs on one line.
  - `/`, `/players/$playerId`, and `/h2h` remain unchanged and navigable.
  - Player profile routes identify `Player Profiles` as the active section; `/h2h` identifies `Matchup Predictions`.
  - The header remains usable at mobile widths with no horizontal overflow and 44px minimum touch targets.
  - Theme toggle and focus states remain accessible in light and dark modes.
- **Guardrails**:
  - Do not rename route paths or change the brand mark.
  - Do not add a third navigation item or duplicate navigation intent.

### [ ] Task 6: Convert player selection into compact searchable dropdowns

- **Description**:
  - Refine the shared player picker into an accessible combobox-style dropdown: closed by default, search field visible when interacting, filtered names in a bounded popover, and selected player name in the trigger.
  - Search and display by player name only, but bind every option's underlying value to `player_id`. Selection state, TanStack query keys, route parameters, and every Bento request must continue using IDs; names are presentation labels only and are never used as database/API lookup keys.
  - Remove ID-based search and every visible secondary ID label from picker options. Ensure screen-reader option names also contain only the display name and selection state.
  - Close on selection and Escape, support keyboard traversal/selection, maintain focus correctly, and exclude or disable the player selected on the opposite side.
  - Keep the two selectors at the top of Matchup Predictions with the predictor immediately below.
- **Files**:
  - `web/src/components.tsx`
  - `web/src/pages/H2H.tsx`
  - `web/src/index.css`
- **Acceptance Criteria**:
  - The full player list is never permanently expanded.
  - No player ID is visible in selector labels, results, or selected state.
  - Users can search, choose, replace, and clear players with mouse, touch, and keyboard.
  - Selecting the same player on both sides is prevented rather than accepted and rejected later.
  - Empty-search and no-result states are clear and local to the control.
- **Guardrails**:
  - Use existing React and CSS; add no combobox dependency unless native implementation proves inaccessible during verification.
  - Preserve player IDs in internal state and API calls.
  - Do not introduce name-to-ID backend lookups, name-based queries, or display-name uniqueness assumptions.

### [ ] Task 7: Consolidate Matchup Predictions around direct head-to-head data

- **Description**:
  - Rename page copy from Head-to-Head to Matchup Predictions and remove unrelated marketing-style labels.
  - Preserve the order: player selectors, match predictor controls and Predict button/result, combined Matchup Comparison, then Meetings.
  - Remove recent-form API queries, helpers, imports, and UI from this page.
  - Remove the rank graph and standalone rank cards; place each current rank within the combined comparison.
  - Remove standalone all-time-series and surface-mix panels.
  - Build one mirrored comparison panel with player names at the top, all-time direct-meeting win counts as the lead row, current ranks, and one row per surface represented in their direct meetings. Surface rows show each player's direct H2H wins/rate for that surface, not general career form. Do not add an all-time win-rate row.
  - Keep the center label and outward bars: left player grows left, right player grows right. Retain a text equivalent for screen readers and handle zero-meeting/one-sided values without misleading bar proportions.
  - Keep model probabilities in the prediction result rather than mixing them into historical H2H comparison.
  - Replace ID-returning display fallbacks such as `nameById.get(id) ?? id` with a neutral non-ID fallback. Prediction winner/result labels, chart series/tooltips, comparison text summaries, errors, and assistive descriptions must resolve to player names only.
- **Files**:
  - `web/src/pages/H2H.tsx`
  - `web/src/index.css`
  - `web/src/components.tsx` only for shared picker behavior
- **Acceptance Criteria**:
  - No recent-form section, rank graph, standalone all-time panel, or surface pie remains.
  - Every historical comparison metric is derived from direct meetings except explicitly labeled current rank.
  - All-time wins lead the panel, followed by current rank and per-surface direct H2H rows.
  - Prediction probability remains correctly oriented to the selected left/right players regardless of canonical backend ordering.
  - No player ID appears in visible text, tooltip content, fallback content, error content, document metadata, or ARIA output.
  - Loading, empty, prediction-error, and H2H-error states render in place without hiding usable controls.
  - Layout is readable in both themes and collapses cleanly below 768px.
- **Guardrails**:
  - Do not invent metrics or infer absent data.
  - Do not change Bento endpoint schemas.
  - Do not reintroduce general career or recent-form statistics.

### [ ] Task 8: Reduce match history to one newest-first Meetings section

- **Description**:
  - Render one combined list containing only matches played between the selected pair.
  - Sort by `match_date` descending in the UI even if the API is already ordered.
  - For each meeting, show date, tournament, surface, round, and winner by display name, with a clear left/right winner treatment.
  - Retain a concise zero-meetings state and avoid duplicate summary cards.
- **Files**:
  - `web/src/pages/H2H.tsx`
  - `web/src/index.css`
- **Acceptance Criteria**:
  - Meetings appear once and newest first.
  - Winner identity is unambiguous for every row without exposing player IDs.
  - Rows remain readable on mobile without horizontal scrolling.
  - Missing optional labels degrade to plain functional text rather than placeholders that resemble data.
- **Guardrails**:
  - Do not fetch each player's general match history.
  - Do not add pagination unless the existing head-to-head endpoint truncates real meeting data.

### [ ] Task 9: Verify workflows and UI behavior

- **Description**:
  - Run package-manager, local-service, Docker, type/build, and focused backend checks in dependency order.
  - Browser-test Player Profiles and Matchup Predictions at desktop and mobile widths in both themes.
  - Mechanically audit rendered text, accessible names/descriptions, chart tooltips, errors, and empty/loading states for known player ID values. Inspect Home, Profile, and Matchup Predictions, not only the picker.
  - Verify selector keyboard behavior, prediction orientation, comparison calculations, newest-first meetings, route active states, and clean process shutdown.
  - Update operational documentation only where commands or architecture descriptions changed.
- **Files**:
  - `README.md`
  - Relevant existing frontend/backend tests; add only focused tests justified by extracted pure comparison logic or deployment command behavior.
- **Acceptance Criteria**:
  - `pnpm install --frozen-lockfile`, `pnpm build`, and `git diff --check` pass.
  - Focused Python tests covering Compose/deploy behavior and serving contracts pass.
  - `pnpm dev` passes the valid Homebrew database smoke path and fails safely for invalid configuration.
  - `pnpm docker` starts a healthy stack and serves the latest built web assets.
  - Browser inspection confirms no overflow, no visible IDs, centered two-tab navigation, and correct light/dark rendering.
- **Guardrails**:
  - Never print `.env` contents or credentials in test output.
  - Never run destructive database reset/seed commands as verification.
  - Never use the full historical seed load.

## Dependencies

1. Task 0 establishes the verified local connection target.
2. Task 1 removes the extension requirement, then initializes and seeds that target; it must complete before serving or application verification.
3. Task 2 is foundational for every pnpm command.
4. Task 3 depends on Tasks 0-2 and the verified standard PostgreSQL database configured in `.env` with required schemas and deterministic data.
5. Task 4 depends on Tasks 1-2 and the generated pnpm lockfile.
6. Task 5 can begin after infrastructure tasks are stable.
7. Tasks 6-8 share `H2H.tsx` and CSS, so implement them sequentially to avoid conflicting edits.
8. Task 9 follows all implementation tasks.

## QA/Testing Scenarios

- **Homebrew discovery**: identify the active service/version and distinguish it from Compose without exposing credentials.
- **Existing database safety**: if the Homebrew target contains unexpected data, stop before destructive work and report the mismatch.
- **No-extension PostgreSQL**: initialize and exercise inference on Homebrew PostgreSQL without querying, installing, or configuring pg_duckdb; no extension fallback path exists.
- **PostgreSQL 18 SQL compatibility**: compile and run every dbt model/test plus representative service and inference queries against PostgreSQL 18, verifying numeric outputs match expected fixtures.
- **Compose major-version migration**: when an old PostgreSQL 17 volume contains data, logically dump/restore into a new PostgreSQL 18 volume, compare schema/table counts and constraints, and retain the old volume.
- **Deterministic seed parity**: initialize, seed without `--all`, run ETL, verify 28/35/56/56/28 counts and exactly 36 gold features, then confirm idempotent re-run behavior.
- **Local happy path**: Homebrew PostgreSQL is reachable through `.env`; `pnpm dev` starts Bento and Vite; profile and matchup APIs load through `/api`.
- **Wrong database target**: `.env` still points to Compose port 6543; launcher stops with instructions to update the target, without printing credentials.
- **Missing data**: connection succeeds but expected schemas/tables are absent; launcher fails before starting servers and names the missing prerequisite.
- **Port collision**: Bento port 3000 or Vite port is occupied; startup fails clearly and cleans up any process already started.
- **Shutdown**: interrupting `pnpm dev` stops both child processes.
- **Docker rebuild**: change visible web copy, run `pnpm docker`, and confirm port 8187 serves the changed bundle.
- **API exposure**: confirm Compose publishes web and PostgreSQL as intended but not Bento; direct host access to Bento's container port fails while same-origin nginx API calls succeed.
- **Route allowlist**: verify every SPA-required GET route and the prediction POST route, then verify blocked model-only/unknown routes and unsupported methods.
- **Abuse controls**: send only a small bounded burst to confirm prediction requests receive rate-limit responses and recover after the configured window.
- **Request constraints**: reject non-JSON, oversized, malformed, and out-of-range prediction requests without leaking stack traces.
- **Navigation**: exactly two centered desktop tabs; active state is correct on home, player detail, and matchup routes; mobile remains operable.
- **Selectors**: search by partial name, keyboard select, clear/change, no visible IDs, no duplicate-player matchup.
- **Global ID-visibility audit**: exercise normal, loading, empty, missing-name, and API-error states across all routes; known player IDs must not appear in rendered or assistive UI. IDs remain permitted in `/players/$playerId`, API requests/responses, and internal state.
- **No meetings**: predictor remains usable; comparison and meetings show concise empty states.
- **Meeting history**: multiple direct meetings display once in descending date order with correct winners.
- **Comparison**: all-time wins, current ranks, and surface-specific direct H2H rows map to the correct selected side and have accessible text equivalents.
- **Prediction orientation**: swap left/right player selection and confirm probabilities/winner labels follow selected positions while canonical request semantics remain intact.
- **Responsive/theme**: test representative desktop and mobile widths in light and dark mode with keyboard focus visible and no horizontal overflow.
