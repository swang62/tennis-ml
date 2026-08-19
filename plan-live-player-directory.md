# Plan: live-player-directory

## Goal

Replace the deploy/build-time player-directory and serialized MiniSearch artifacts with a live `GET /api/directory` Bento response. Build MiniSearch from that response in browser memory once per page load.

## Scope

### In

- One envelope-backed response with `{ players, latest_match_date, total_matches }` from Bento.
- Browser-side MiniSearch construction from the fetched players.
- Removal of all directory/MiniSearch raw artifacts, generated files, builder scripts, manifests, hashes, and reuse tracking.
- Consolidation of the standalone `/api/directory_info` consumer into the shared directory query.

### Out

- The deploy-time FAISS similarity index and its `data/deploy/player_metadata.json` remain unchanged.
- No persistent browser storage, automatic refetching, explicit retry control, endpoint authentication change, or HTTP-cache policy change.

## Response contract

`GET /api/directory` returns the existing standard Bento envelope:

```json
{
  "ok": true,
  "data": {
    "players": [
      {
        "player_id": "...",
        "display_name": "...",
        "matches_played": 0,
        "current_rank": null,
        "ioc": "...",
        "iso2": "..."
      }
    ],
    "latest_match_date": null,
    "total_matches": 0
  }
}
```

`players` retains the current ranking/name/id ordering. Database failures use the service's existing 500 error envelope.

## Tasks

### [ ] Task 1: Serve the directory on demand

- **Description**: Reuse `PLAYERS_SQL` and `directory_players()` to query/normalize player rows per `GET /directory` request. Combine them with the existing directory summary calculation in one handler, register the route on `DATA_APP`, and remove `/directory_info` plus its dedicated handler/SQL after moving the summary contract.
- **Files**: `src/serving/directory.py`, `src/serving/service.py`.
- **Acceptance Criteria**:
  - `GET /api/directory` returns the standard successful envelope with ordered player records, `latest_match_date`, and physical-match `total_matches`.
  - Empty data returns `players: []`, `latest_match_date: null`, and `total_matches: 0`.
  - Database errors return the existing 500 envelope without exposing a stack trace.
  - `/api/directory_info` is no longer registered.
- **Guardrails**: Do not add a server cache, alter player field values/order, or touch the FAISS similarity endpoints/assets.

### [ ] Task 2: Fetch once and construct MiniSearch in browser memory

- **Description**: Replace generated-asset imports and serialized-index deserialization with the API client fetch. When the React Query directory request resolves, construct MiniSearch with the existing `display_name`/`player_id`, fuzzy, prefix, and boost options; retain its search function in the shared query result for both Home and H2H. Fold directory summary fields into this same result and remove the separate idle-gated directory-info hook.
- **Files**: `web/src/api.ts`, `web/src/lib/playerIndex.ts`, `web/src/lib/directoryInfo.ts`, `web/src/pages/Home.tsx`, `web/src/routes.tsx`.
- **Acceptance Criteria**:
  - The first app load makes one `/api/directory` request per in-memory React Query lifecycle.
  - The API response is the only player-list source; no generated JSON or serialized MiniSearch asset is imported/fetched.
  - MiniSearch is constructed using the fetched players and preserves current fuzzy/prefix name-search results.
  - Home player/match counts and footer last-updated text use the shared directory response.
  - Existing page-level loading and error behavior remains; a reload retries a failed request.
- **Guardrails**: No localStorage, polling, background refresh, new UI retry control, or duplicate directory request for summary data.

### [ ] Task 3: Remove directory/MiniSearch build and deploy pipeline

- **Description**: Delete the Node index-builder script and generated web assets; remove raw-directory staging, manifest creation, navigation source hashing, and reuse-state logic that exists only for the directory/MiniSearch pipeline. Simplify deploy/local-dev orchestration so it only stages the required FAISS similarity assets; remove build preconditions/copy steps related to generated player assets.
- **Files**: `web/scripts/build-player-index.mjs`, `web/src/assets/generated/player-directory.json`, `web/src/assets/generated/player-search.json`, `data/deploy/player-directory.json`, `data/deploy/player-index.manifest.json`, `src/flows/deploy.py`, `justfile`, `scripts/dev.sh`, `web/Dockerfile`, `web/.gitignore`.
- **Acceptance Criteria**:
  - Neither `just deploy` nor local web development invokes the MiniSearch builder or requires a player-directory input/artifact.
  - No directory-specific hash/state/manifest code remains in deployment code.
  - The web build does not reference or expect `web/src/assets/generated/`.
  - FAISS `player_similarity.index` and `player_metadata.json` generation and packaging still work as before.
- **Guardrails**: Do not remove the FAISS metadata/index, model-serving artifacts, or unrelated deployment fingerprinting.

### [ ] Task 4: Align tests and documentation

- **Description**: Replace static-artifact and deserialization tests with live directory endpoint and in-memory indexing tests. Update deploy/dev-script tests to assert no directory build/staging remains, while preserving similarity staging coverage. Correct artifact-boundary documentation so it only describes rebuilds that still occur at deploy time.
- **Files**: `tests/test_service_data_endpoints.py`, `tests/test_directory.py`, `tests/test_deploy.py`, `tests/test_dev_script.py`, `web/tests/playerIndex.test.mjs`, `web/tests/buildPlayerIndex.test.mjs`, `AGENTS.md`.
- **Acceptance Criteria**:
  - Hermetic endpoint tests cover success, empty data, query failure, response shape, and removal of the legacy route.
  - Browser-unit tests prove the fetched players create a working in-memory MiniSearch index with current search options.
  - Directory-builder test file is deleted; no test fixture relies on generated directory/index files.
  - Deployment/dev-script tests confirm directory artifacts are not built, but similarity artifacts remain supported.
  - `AGENTS.md` no longer claims deploy builds directory/MiniSearch navigation assets.
- **Guardrails**: Tests must remain self-contained: no live PostgreSQL, network, Bento, or external calls.

## Dependencies

1. Task 1 establishes the API contract.
2. Task 2 consumes that contract and replaces both static directory and summary consumers.
3. Task 3 can remove artifacts once no consumer imports or invokes them.
4. Task 4 verifies the final contract and cleanup.

## QA/Testing Scenarios

- API returns all directory fields in current sort order plus match summary, including players without a gold profile (`matches_played: 0`, nullable rank).
- API returns the documented empty directory response and an existing-style error envelope on a mocked database failure.
- Cold page load waits for the directory response; Home, H2H, player pickers, counts, and footer converge on the same cached result.
- The first non-empty picker search uses the in-memory MiniSearch index and returns fuzzy/prefix matches; no extra search-asset network request occurs.
- Refreshing the browser obtains a fresh directory; navigation within the SPA does not refetch it.
- Web build/dev and deploy paths succeed without any player-directory/MiniSearch artifact while similarity assets remain available.
