# Plan: Deployment-built player search index

## Goal

Generate the full MiniSearch directory index during `just deploy`, package it in the web image as a static Nginx asset, and download/deserialise it in Home and H2H. Browsers must not call `/api/players` or construct the MiniSearch index from raw directory rows. The content-hashed asset is cached by the browser's normal HTTP cache.

## Tasks

### [x] 1. Generate the deploy-time player-index artifact

- **Files:** `src/flows/deploy.py`, `src/serving/service.py` or a minimal shared directory-query module, `tests/`
- **Implement:** Reuse one canonical directory-data query/normalization path to produce the fields currently required by both pickers. During deployment, query PostgreSQL once, write a generated raw directory input under `web/public/`, and include the actual database `MAX(match_date)` in that artifact for the footer. Do not call `/api/directory_info` at runtime. Fail the deploy before image publication if either query/artifact write fails. Keep it ignored by Git.
- **Acceptance:** The deploy path creates a deterministic directory artifact from the same data contract as the existing `/players` endpoint plus `latest_match_date`; no duplicate SQL/data mapping remains. The value is the database's latest match date, never deployment time.

### [x] 2. Build and serve a versioned serialized MiniSearch asset

- **Files:** `web/scripts/`, `web/Dockerfile`, `web/nginx.conf.template`, `web/.gitignore` as needed
- **Implement:** During the web-image build, use the existing MiniSearch dependency to serialize the full index and its player records into one content-hashed static asset. Emit a tiny build-time manifest/URL reference for the frontend. Nginx serves the hashed artifact with immutable cache headers and gzip; generated inputs/outputs are not committed.
- **Acceptance:** The web image contains the index artifact; no runtime API or database request is required to retrieve directory data. A new deploy changes the asset URL when player data changes.

### [x] 3. Consume one static directory source in Home and H2H

- **Files:** `web/src/api.ts`, `web/src/pages/Home.tsx`, `web/src/pages/H2H.tsx`, focused web tests
- **Implement:** Replace `getPlayers()`/`/api/players` and `getDirectoryInfo()`/`/api/directory_info` queries with one shared static-index loader that downloads and deserializes the deployment-built MiniSearch payload in the browser. Home and H2H consume the same query key/source. Remove the duplicate Home `getPlayers()` hook and all browser-side `MiniSearch.addAll()`/localStorage construction; browser caching is HTTP caching of the immutable hashed asset.
- **Acceptance:** Initial directory load fetches only the static index asset; the application makes no `/api/players` or `/api/directory_info` request. Search, picker/player-count, and footer latest-match-date fields remain intact.

### [x] 4. Verify asset generation and frontend behavior

- **Files:** focused tests only
- **Checks:** Add hermetic tests for the shared directory artifact and static loader as appropriate; run focused Python/web tests, web production build, Ruff, and basedpyright. Inspect built assets or Docker build output to confirm the index is present and `/api/players` is not required.

### [x] 5. Rebuild the static index for every local dev start

- **Files:** `scripts/dev.sh`, focused script tests if a suitable seam exists
- **Implement:** After local database preflight and before starting Vite, generate `web/public/player-directory.json` from the configured local PostgreSQL database, then run the existing Node index-builder. This overwrites any fixture/stale index, so Vite always serves the current database directory and latest match date. Abort `just dev` when either generation step fails.
- **Acceptance:** `just dev` never serves a test fixture such as Player A/B/C, does not call `/api/players` or `/api/directory_info`, and starts Vite only after the static index is rebuilt from the local database.

## Exclusions

- No browser polling, localStorage directory cache, server-side request cache, new database index, service-worker cache, or change to `/api/players` for backward compatibility.
