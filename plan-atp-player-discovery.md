# Plan: ATP Player Discovery

## Goal

When either ATP scrape encounters a player ID missing from `bronze.player_profiles`, fetch that player's ATP overview page through the existing persistent browser session, validate the identity, append only new canonical/reference rows, insert the profile, and continue processing the dependent ranking or match.

## Scope

### In

- Shared, ATP-only player discovery used by both `rankings_flow` and `matches_flow`.
- ATP ID, non-empty display name, and IOC (valid or `UNK`) are required; optional profile fields may be absent.
- Deduplicated, validated, append-only updates to `data/ATP_player_database.csv` and `data/ranking_player_map.csv`.
- Idempotent insertion into `bronze.player_profiles`, without updating existing player data.
- Explicit per-player reporting and dependent-row skips on ATP navigation, parsing, validation, or identity-conflict failure.
- Hermetic tests for discovery, persistence boundaries, and both caller integrations.

### Out

- General web search, Wikipedia enrichment, and any non-ATP identity source.
- Modifying or overwriting existing CSV or database player records.
- Database schema migrations, changes to the browser-profile topology, ETL, model, or deployment behavior.

## Tasks

### [ ] Task 1: Add validated append-only player persistence

- **Description**: In `src/db/ingest.py`, add the small persistence seam for newly discovered ATP identities. It must validate the candidate against the existing ATP profile-column contract and IOC normalizer; reject an ID already associated with conflicting canonical/profile or map data; append a deduplicated full canonical row to `data/ATP_player_database.csv`; append the self-mapping `ranking_player_id`, ATP display name, and canonical `player_id` to `data/ranking_player_map.csv`; then reuse `load_atp_profiles(..., player_ids={...}, force=False)` to insert only the new DB profile. Validate every affected file and candidate before any append.
- **Files**: `src/db/ingest.py`, `data/ATP_player_database.csv`, `data/ranking_player_map.csv`, `tests/test_ingest.py`
- **Acceptance Criteria**:
  - A valid unseen ATP identity produces exactly one canonical CSV row, one map row, and one `bronze.player_profiles` insert.
  - Repeating the same discovery writes nothing and does not alter the existing player data.
  - Invalid required data, malformed optional typed data, duplicate/conflicting IDs, or map conflicts produce a reason and no dependent-data approval.
  - A prior partial append can be retried safely and reconciles the missing idempotent step without duplicate rows.
- **Guardrails**: Do not use name matching as the write key; do not call the enrichment path; do not use `force=True`; do not rewrite the CSVs or edit old rows.

### [ ] Task 2: Implement the shared ATP-profile discovery helper

- **Description**: Extend the shared `src/flows/rankings.py` identity layer with one helper used by both flows. Given a run's existing browser page and unique candidate `{ATP ID, slug, displayed name}` values, query `bronze.player_profiles` once for known IDs, process only missing players, navigate the same persistent page to `https://www.atptour.com/en/players/{slug}/{id}/overview`, parse its identity fields, require ID/name/IOC, and call Task 1's persistence seam. Update the in-memory canonical player map, ranking map, and profile metadata map for each success. Batch/deduplicate candidates within the run and return successes plus structured failure reasons; a failed profile must not abort the scrape.
- **Files**: `src/flows/rankings.py`, `src/db/ingest.py`, `tests/test_scrape_flow.py`
- **Acceptance Criteria**:
  - Both flows can call the same helper without creating a second browser context or a non-ATP network path.
  - One ATP profile navigation occurs per unique, DB-missing player ID per run.
  - The parsed ID must equal the ID from the ATP link; a mismatch, absent name, failed navigation, unparseable page, or invalid identity is reported as a non-fatal failure.
  - Successful discoveries immediately resolve using the refreshed in-memory maps; existing database players never cause a profile-page fetch or write.
- **Guardrails**: Preserve the existing headed persistent page and jitter discipline; do not add caching infrastructure or a separate web-search client; do not weaken the reviewed-map validation for pre-existing identities.

### [ ] Task 3: Wire rankings discovery before ranking translation

- **Description**: Preserve the ATP profile slug alongside each parsed rankings row in `src/flows/rankings.py`. Before `translate_rank_rows` filters against the identity map, invoke the shared helper for the parsed page's candidates. Feed the updated map into the existing translation/upsert path, and emit concise success/failure counts plus player-specific skip reasons.
- **Files**: `src/flows/rankings.py`, `tests/test_scrape_flow.py`
- **Acceptance Criteria**:
  - A new valid player shown in a rankings page is added to both identity stores and included in that week's `ingest_rankings` payload.
  - Existing rankings behavior and the 1–200 rank filter remain unchanged.
  - A failed discovery skips only that player’s ranking row while the rest of the week is stored.
- **Guardrails**: Do not change watermark behavior, weekly scheduling, or the current retry semantics for whole unavailable ranking pages.

### [ ] Task 4: Wire matches discovery before match identity resolution

- **Description**: Preserve each player’s ATP profile slug when `extract_matches_from_results` builds discovery rows in `src/flows/matches.py`. Before `resolve_discovered_matches` performs canonical resolution, invoke the same shared helper for all players from the tournament page, then resolve against refreshed maps. Propagate discovery failures into the existing per-match unresolved reason/reporting path so a match with either unresolved player is skipped before Hawkeye fetch and bronze/CSV writes.
- **Files**: `src/flows/matches.py`, `src/flows/rankings.py`, `tests/test_scrape_flow.py`
- **Acceptance Criteria**:
  - A valid new ATP player in a result card is persisted and its match can proceed through the existing resolution path in the same run.
  - Repeated player appearances across a tournament produce one lookup/write only.
  - If either player fails discovery, the match is reported and skipped; no Hawkeye request, bronze upsert, or raw-match CSV append occurs for it.
- **Guardrails**: Preserve winner orientation (`winner_id` remains aligned with the original player order), tournament-tier handling, physical-match dedupe, and existing raw-match append behavior.

### [ ] Task 5: Verify behavior at public seams

- **Description**: Add hermetic fixtures/fake browser pages for ATP overview markup and mock the DB/file persistence boundary. Cover valid discovery, duplicate retries, existing-player no-op, profile-ID mismatch, missing required identity, persistence conflict, rankings inclusion, and match-level skip propagation. Run the focused suites, lint, then the repository suite.
- **Files**: `tests/test_ingest.py`, `tests/test_scrape_flow.py`
- **Acceptance Criteria**:
  - Tests make no live PostgreSQL, ATP, CloakBrowser, or general-web requests.
  - Tests assert externally observable rows/actions/reasons rather than parser-private implementation details.
  - `just lint`, focused ingestion/scrape tests, and the full test suite pass.
- **Guardrails**: Do not add tests that merely freeze selectors, constants, SQL spelling, or mock call ordering.

## Dependencies

1. Task 1 defines the validated persistence contract.
2. Task 2 consumes that contract and provides the shared discovery behavior.
3. Tasks 3 and 4 can then be completed independently.
4. Task 5 validates all completed seams and integrations.

## QA Scenarios

1. Rankings page includes a new valid ATP ID: one profile lookup, validated append to both CSVs, DB insert, and ranking insert in the same week.
2. Tournament results include the same new player in multiple cards: one profile lookup and insert; all resolvable matches continue.
3. ATP page has a different ID, no name, or invalid/missing identity: no new profile/map rows; affected ranking/match rows are skipped and reported; unrelated rows continue.
4. Player exists in the DB: no ATP navigation and no CSV or DB mutation.
5. A prior run appended only part of the identity state before interruption: the next run completes the missing idempotent step without duplicates or overwrites.
6. Existing player IDs, rankings, match orientation, raw-match CSV dedupe, and browser lifecycle retain their current behavior.
