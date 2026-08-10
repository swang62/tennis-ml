# Plan: player-flags-rankings

## Goal

Show an accessible country flag next to player names on the player profile and the two matchup-comparison headings, and replace match-derived ranking history with official weekly top-200 ATP rankings from `data/raw/rankings/`.

## Scope

### In

- A non-null IOC value for each `bronze.player_profiles` row, using `UNK` only when no verified country is available.
- CDN-hosted country SVG flags, with country-name hover text and the native white-flag glyph (`🏳️`) as the generic fallback for `UNK` and failed image loads.
- Official weekly rankings (rank 1-200 only) in a new idempotent `bronze.rankings` table.
- An explicit, reviewed ranking-source-ID to canonical-player-ID mapping CSV; ambiguous/unmapped rows are skipped and reported.
- The profile chart and matchup “Current rank” read official rankings.

### Out

- Downloading or committing flag assets.
- Ingesting ranks outside the top 200.
- Silent name-based identity resolution.
- Changing model features, training data, or prediction behavior.
- Flag display in matchup pickers, prediction output, odds, or other non-comparison name labels.

## Technical Decisions

- Keep IOC as the stored canonical source country code; resolve it through a versioned local `IOC -> ISO alpha-2 + country name` reference CSV. This supplies stable human-readable tooltip text and the two-letter code required by FlagCDN.
- Load country flags from `https://flagcdn.com/<iso2>.svg`; render `UNK`, an unresolvable IOC, or a failed flag-image request as the native white-flag glyph (`🏳️`), with accessible name and title “Country unknown.” This needs no additional asset or dependency.
- The mapping file is authoritative by ranking-source player ID, with source name retained for audit. Name normalization is only a mapping-review aid, never an automatic write path.
- The rankings primary key is `(ranking_date, player_id)` and accepts only ranks 1 through 200. Re-imports upsert the same record deterministically.
- Existing endpoint shape stays stable: `rank_history` still returns `{ rank_date, rank }`, but is sourced from `bronze.rankings`.
- The profile rank chart retains all official weekly points for hover/detail but renders one category per calendar month, using that month's latest weekly rank. The numeric rank axis remains inverted; the date axis has one monthly tick, and each December gets a prominent year-boundary marker plus a `Dec\nYYYY` label rather than overlapping weekly date text.
- Initial historical rankings remain a reviewed, file-based backfill. After that backfill is complete, a host-run Prefect flow may fetch missing weekly rankings directly with the CloakBrowser Python library. The MCP server is interactive tooling and is not a dependency of the flow. The database is the self-healing watermark: each run queries `MAX(bronze.rankings.ranking_date)` and processes every later ATP ranking Monday through the latest completed Monday. The flow runs every Monday and remains manually triggerable through Prefect. Do not automate CAPTCHA solving, persist challenge credentials, or bypass access controls; a normal browser-navigation block is an observable failed run.

## Tasks

### [ ] Task 1: Add country-reference data and enforce the IOC invariant

- **Description**:
  - Add a versioned IOC reference CSV containing `ioc`, `iso2`, and `country_name`, including the explicit `UNK` sentinel.
  - Add one shared loader/lookup module so profile import and serving use the same IOC validation and country resolution rules.
  - Normalize valid imported IOC values (trim/uppercase), preserve verified values, fill missing or invalid values with `UNK`, and report the number of unresolved profiles.
  - Extend the idempotent database bootstrap/upgrade SQL so existing databases are backfilled to `UNK`, new rows default to it, and `ioc` becomes non-null after the backfill.
- **Files**:
  - New `data/ioc_countries.csv`
  - New `src/countries.py`
  - `src/flows/ingest.py`
  - `infra/postgres/init.sql`
  - `tests/test_ingest.py`
- **Acceptance Criteria**:
  - Every `bronze.player_profiles.ioc` value is non-null after initialization/import.
  - Known IOC values resolve to the expected ISO alpha-2 code and country name; missing/invalid values resolve to `UNK` only.
  - Re-running initialization/backfill does not overwrite a valid IOC or fail on an already-upgraded database.
- **Guardrails**:
  - Do not infer nationality from a player name, birthplace, or Wikipedia.
  - Do not call a third-party country API at runtime.

### [ ] Task 2: Add the reviewed ranking identity-map contract

- **Description**:
  - Add the authoritative mapping CSV with columns such as `ranking_player_id`, `ranking_name`, and `player_id`; document that source ID is the match key and the name is an audit/review field.
  - Implement deterministic normalized-name candidate generation for maintainers to review, including collision/ambiguity reporting, without changing the mapping file automatically.
  - Validate mapping-file structure, duplicate source IDs, duplicate/conflicting targets, and unknown canonical player IDs before inserting any rankings.
- **Files**:
  - New `data/ranking_player_map.csv`
  - `src/flows/ingest.py`
  - `tests/test_ingest.py`
  - `README.md`
- **Acceptance Criteria**:
  - A source ID is mapped only through one approved mapping-file row.
  - Exact and normalized-name candidate reports are deterministic and clearly label ambiguous candidates.
  - Invalid mapping files fail before database writes; unmapped ranking rows do not fail a valid import and appear in the import report with source ID, name, and skipped-row count.
- **Guardrails**:
  - Do not use a name match as an implicit production mapping.
  - Do not add a mapping-management UI or external identity service.

### [ ] Task 3: Create and ingest `bronze.rankings`

- **Description**:
  - Define `bronze.rankings` in bootstrap SQL with `ranking_date`, canonical `player_id`, `rank`, and `points`; add a primary key on date/player, rank range check, and the player/date lookup index required by the API.
  - Add ingest functions that discover only `data/raw/rankings/atp_rankings_*.csv`, validate the documented four-column shape, combine files, filter rank `<= 200`, join the approved identity map, and upsert valid rows.
  - During the same import, use `atp_players.csv` as a higher-confidence IOC source for mapped players, subject to Task 1 validation/fallback rules.
  - Produce a concise import summary: files read, source rows, retained top-200 rows, inserted/updated rows, and unmapped/invalid rows skipped.
  - Add a dedicated `just db-rankings` recipe that invokes this idempotent import; leave `db-seed` and `db-seed -- --all` match-only.
- **Files**:
  - `infra/postgres/init.sql`
  - `src/flows/ingest.py`
  - `justfile`
  - `tests/test_ingest.py`
  - `tests/test_seed.py`
  - `tests/test_e2e_ingest_to_inference.py`
  - `README.md`
- **Acceptance Criteria**:
  - Every stored rank is in `[1, 200]`; ranks above 200 are absent.
  - Re-running `just db-rankings` leaves the same logical rows and does not duplicate records.
  - The importer handles all seven supplied ranking-period files and rejects malformed input before writing partial data.
  - The import summary makes unmapped top-200 source records actionable.
- **Guardrails**:
  - Do not modify the existing deterministic match seed data set or introduce ranking ingestion into its default path.
  - Do not derive official history from match rows once this table is available.

### [ ] Task 4: Serve official rankings and country metadata

- **Description**:
  - Extend `/players` and `/player_profile` responses with IOC plus resolved ISO alpha-2 code and country name.
  - Replace `_RANK_HISTORY_SQL`’s two-sided match-event expansion with a parameterized query over `bronze.rankings`, ordered by weekly ranking date, while preserving the existing response envelope and point names.
  - Retain existing empty-history behavior for players with no approved official ranking rows.
- **Files**:
  - `src/serving/service.py`
  - `web/src/api.ts`
  - `tests/test_service_profile.py`
  - New focused service-ranking test file if existing service tests do not cover `/rank_history`
- **Acceptance Criteria**:
  - Profile and player-list API responses expose consistent country metadata for known, `UNK`, and missing-reference cases.
  - `/rank_history` returns weekly official entries only, in chronological order, and no longer reads `bronze.match_events`.
  - Existing consumers keep receiving `rank_date` and `rank` without a frontend-breaking endpoint change.
- **Guardrails**:
  - Do not expose raw mapping-file internals in public API responses.
  - Do not alter prediction endpoint contracts.

### [ ] Task 5: Render flags only in the requested UI locations

- **Description**:
  - Add a small shared flag/name presentation component that accepts resolved country metadata and handles FlagCDN image failure by replacing the `<img>` with the native white-flag glyph (`🏳️`).
  - Render it to the left of the player name in the profile `<h1>` and in the two matchup comparison headings only.
  - Give the icon meaningful `alt`/accessible name and a native hover title containing the country name; the white-flag fallback uses `role="img"`, `aria-label="Country unknown"`, and `title="Country unknown"`.
  - Add the minimal component/CSS sizing and alignment needed to keep headings readable on narrow screens.
- **Files**:
  - `web/src/components.tsx`
  - `web/src/pages/Profile.tsx`
  - `web/src/pages/H2H.tsx`
  - Existing web stylesheet(s) that define `.page-title` / `.mirror-name`
  - `web/src/api.ts`
- **Acceptance Criteria**:
  - A known player has a flag immediately left of their name on the profile and matchup comparison headings.
  - Hovering a known flag exposes its country name; screen readers receive equivalent text.
  - `UNK`/bad image URLs show the native white-flag fallback instead of a broken image icon.
  - Flags do not appear in matchup selectors, prediction labels, odds, or unrelated player-name locations.
- **Guardrails**:
  - Do not add a flag-icon dependency or local asset bundle.
  - Do not change the selected-player picker UI.

### [ ] Task 6: Simplify the official-ranking chart time axis

- **Description**:
  - In the profile rank-chart view model, group the API's weekly history by UTC `YYYY-MM` and retain each month's latest dated rank as its plotted point; leave the underlying API history untouched for tooltips and other callers.
  - Change the x-axis to those monthly categories so it produces exactly one tick per month, never a weekly timestamp tick.
  - Format normal labels as abbreviated month names and December labels as `Dec` plus the year. Add a high-contrast, full-height year-boundary mark at each December point so year transitions are visually larger than normal monthly ticks without custom chart-canvas drawing.
  - Keep rank direction, tooltip exact weekly-date/rank information for the selected monthly point, theme behavior, and responsive chart sizing unchanged.
- **Files**:
  - `web/src/pages/Profile.tsx`
  - `web/src/lib/charts.ts` only if the existing shared chart helpers can express the year-boundary styling without affecting unrelated charts
  - Existing frontend test location, if the repository has a configured frontend test runner
- **Acceptance Criteria**:
  - The visible x-axis has one label/tick per month and no overlapping weekly timestamps at desktop or mobile widths.
  - December labels identify the year and have a visually prominent year-boundary marker.
  - The latest available rank within a month is the plotted rank; rank order stays inverted and hover remains intelligible.
- **Guardrails**:
  - Do not reduce the stored/imported cadence below weekly.
  - Do not introduce a custom SVG/canvas axis renderer or new chart dependency just for uneven tick lengths.

### [ ] Task 7: Verify the full data-to-UI contract

- **Description**:
  - Add focused unit coverage for country lookup/fallback, mapping validation, top-200 filtering, unmapped-row reporting, idempotent ranking upsert, and service query behavior.
  - Extend the PostgreSQL end-to-end fixture with official ranking rows and verify the API/chart contract observes those rows rather than embedded match ranks.
  - Run the repository’s focused tests, then the configured complete test suite and frontend checks available in the project.
  - Perform a manual browser pass for known-country and `UNK` players on profile and matchup views, including image-load failure fallback and tooltip/accessibility behavior.
- **Files**:
  - `tests/test_ingest.py`
  - `tests/test_seed.py`
  - `tests/test_e2e_ingest_to_inference.py`
  - `tests/test_service_profile.py`
  - Any new focused test files introduced by Tasks 1 and 4
- **Acceptance Criteria**:
  - Tests cover the requested positive paths and the explicitly chosen fallback/skip paths.
  - `just db-rankings`, `just test`, and the project’s applicable web validation complete successfully in the implementation environment.
  - Manual UI verification confirms only the agreed locations display flags.
- **Guardrails**:
  - Do not weaken existing match-ingestion or inference tests to accommodate the new table.

### [ ] Task 8: Automate rankings catch-up after the initial backfill

- **Description**:
  - Add a host-executed Prefect flow that launches CloakBrowser through its Python API and visits each missing weekly ATP singles-ranking URL.
  - Wait for the rankings table, then extract rank, points, display name, and player ATP identifier/slug for ranks 1-200.
  - Reuse the approved identity-map validation and existing idempotent `bronze.rankings` upsert.
  - Query `MAX(ranking_date)` before every run; when no weekly dates are missing, log and exit successfully without browser work.
  - Process dates chronologically and commit each completed week independently. If a date cannot load or validate, raise a date-specific failure and leave later dates for the next run.
  - Create a Prefect deployment scheduled every Monday, with manual triggering through the existing host worker.
  - Add a `just` command for local/manual invocation and document required host-browser dependencies.
- **Files**:
  - `pyproject.toml`
  - `uv.lock`
  - New `src/flows/rankings.py`
  - `src/flows/ingest.py`
  - `infra/prefect/worker.py`
  - `justfile`
  - `README.md`
  - New focused rankings-flow test file
  - `tests/test_ingest.py`
- **Acceptance Criteria**:
  - With rankings through date `D`, the flow attempts each valid weekly ranking date after `D`, oldest first.
  - A successful run stores only ranks 1-200 and does not create duplicate `(ranking_date, player_id)` records.
  - Re-running after successful catch-up does not launch a browser or change rows.
  - A failed date fails clearly, preserves all previously ingested dates, and is retried on the next run.
  - The deployment runs on Mondays and can be triggered manually using the host worker.
  - Parser tests use saved fixture HTML; no test requires ATP network access.
- **Guardrails**:
  - Do not make the Prefect worker call CloakBrowser MCP.
  - Do not automate CAPTCHA solving or preserve/replay challenge credentials.
  - Do not skip identity-map validation or silently ingest unmapped players.
  - Do not schedule this before the initial historical backfill is complete.

## Dependencies

1. Task 1 precedes Tasks 3-5 because IOC normalization and resolution are shared contracts.
2. Task 2 precedes Task 3; no ranking ingestion occurs without the reviewed map.
3. Task 3 precedes Task 4; service history must have the canonical bronze source.
4. Task 4 precedes Task 5; frontend rendering depends on country metadata in the API.
5. Task 4 precedes Task 6; the chart consumes the official-history response.
6. Task 7 follows all implementation tasks.
7. Task 8 is the final task and depends on Tasks 2, 3, and 7; it is enabled only after initial historical rankings are fully ingested and validated.

## QA / Testing Scenarios

| Scenario | Expected result |
| --- | --- |
| Valid profile IOC | API returns IOC, ISO code, country name; CDN flag has that country tooltip. |
| Missing/invalid profile IOC | Database stores `UNK`; UI displays generic flag and “Country unknown.” |
| Rank 1 and rank 200 source rows | Both import and appear in weekly history. |
| Rank 201 source row | Never reaches `bronze.rankings`. |
| Unmapped top-200 source player | Row is skipped and included in the import report; valid mapped rows still import. |
| Duplicate import | No duplicate primary-key rows; re-import stays deterministic. |
| Official rank differs from match-row rank | Profile chart and matchup current rank show the official weekly value. |
| Multiple weekly rankings in one month | Chart plots the month's latest rank at one monthly tick; tooltip identifies the retained date. |
| Year transition | December has one month label plus a prominent year-boundary marker; no weekly labels overlap. |
| Flag CDN image fails | UI replaces it with the generic accessible flag, not a broken image. |
| Database has no rankings | The automated catch-up flow does not run before the initial backfill is complete. |
| Database is current | The flow exits successfully without opening a browser. |
| Two missed Mondays | Both weeks ingest in chronological order. |
| First missed week succeeds, second fails | The first remains committed; the second is retried later. |
| ATP table is unavailable or challenged | The flow fails with URL/date evidence and preserves the existing watermark. |
| Manual Prefect run | The host worker uses the same catch-up logic as the Monday schedule. |

## ATP Tour Automation Investigation

The supplied `dateWeek=2025-07-28` URL was tested with four access paths. Results:

| Access path | Result |
| --- | --- |
| CloakBrowser MCP (headless stealth Chromium) | HTTP 403; Cloudflare “Performing security verification” challenge; no rankings payload reached, even after 20s/30s/60s waits and reloads |
| agent-browser CLI (headed) | HTTP 403; same Cloudflare challenge page |
| chrome-devtools MCP (real interactive Chrome) | Success. Full rankings page loads; table is fully server-rendered HTML |
| `fetch()` from inside the working browser session | HTTP 403 for the same URL |

The successful path showed the rankings are server-rendered directly in the page HTML, including rank, `J. Sinner`-style names, player slug URLs (`/en/players/jannik-sinner/s0ag/overview`), flag sprite references, and points. However, the page cannot be fetched by an HTTP client even with the browser's session cookies — every non-interactive request gets 403. The only working path is an interactive real Chrome session that completes Cloudflare's challenge on each navigation.

**Updated conclusion:** the Prefect flow must not call CloakBrowser MCP. After the initial historical backfill, Task 8 may use the installed CloakBrowser Python library directly from the host worker to perform normal interactive browser navigation and parse the rendered table. The flow must fail visibly when navigation is blocked and must not bypass challenges or persist challenge credentials. The existing `just db-rankings` import remains the supported historical/backfill mechanism.

**Explicit exclusion:** do not add 2Captcha (or another CAPTCHA-solving service), CAPTCHA tokens, persistent challenge/session storage, or a browser-bypass workflow. Those would automate circumvention of the ATP Tour site's access controls.

**Prerequisite for scheduled automation:** complete and validate the initial historical ranking backfill first. Then Task 8 adds Monday scheduling, manual triggering, chronological catch-up from the database watermark, per-week commits, retry behavior, source attribution, and failure reporting that leaves prior rankings intact.
