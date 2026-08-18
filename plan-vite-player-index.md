# Plan: Vite-backed player directory and deferred directory info

## Goal

Remove the browser-side manifest-to-directory waterfall while preserving deterministic, snapshot-derived MiniSearch generation and hash-verified reuse. Bundle the small player directory in the Vite entry, emit the serialized MiniSearch index as a separate Vite-hashed asset that loads only on first search input, and move `directory_info` out of the initial network work. Add a direct, accessible ATP Tour overview link from each profile's name in the main bio section.

## Scope

### In scope

- Keep a generated player-index manifest under `data/deploy/` as build cache/integrity state only.
- Build the directory and MiniSearch index from the deploy-produced raw directory JSON, not from DuckDB in the web build and not in the browser.
- Make `just deploy` and `just dev` run the same host-side generation step before a web build/server starts, then exercise the same Vite import-based runtime loader.
- Eliminate runtime requests for `player-index.manifest.json` and `player-directory.*.json`.
- Lazy-load and memoize the MiniSearch index after the first non-empty picker search.
- Defer and deduplicate `/api/directory_info`; render no placeholder value while it is pending.
- Link the profile name to its ATP Tour overview page with a small external-link icon.

### Out of scope

- Changing DuckDB snapshot contents, directory SQL, player fields, MiniSearch search behavior, or player-picker UX.
- Loading MiniSearch from DuckDB at runtime.
- Preconnect hints or analytics changes.
- Champion/MLflow lineage changes; navigation assets remain outside champion pins.

## Architecture decisions

1. **Snapshot remains the upstream source.** `generate_navigation_artifacts()` reads the local DuckDB snapshot and writes raw `{"players": [...]}` to `data/deploy/player-directory.json`. The Node builder reads that file, calls `MiniSearch.addAll(players)`, and serializes the index. The Node builder never opens DuckDB.
2. **The manifest is build-only.** Store `data/deploy/player-index.manifest.json`; never copy or serve it from `web/public` or `dist`.
3. **Manifest records content, not paths.** Use a versioned schema containing `sourceHash`, `optionsHash`, `directoryHash`, and `searchHash`. It contains no generated filenames or browser URLs.
4. **Generated Vite inputs live under `web/src/assets/generated/`.** Write `player-directory.json` and `player-search.json` there before Vite/Docker runs. They are gitignored but included by `COPY . .` in the web Docker build context.
5. **Directory is inlined; search index is emitted.** A standard JSON import bundles the directory in the entry chunk. An explicit `?no-inline` URL import makes Vite emit the search payload under `/assets/` with a Vite content hash.
6. **`directory_info` is non-critical.** Both current consumers share one React Query key and idle-gated fetch. Until data arrives, the existing conditional footer date and the home total-match value are omitted rather than rendered as `0` or a loading placeholder.
7. **ATP URL identity is code-led.** Build the readable slug from `display_name`, but use the existing canonical `player_id` as the URL identifier: `https://www.atptour.com/en/players/{slug}/{player_id-lowercase}/overview`. The project data contains Felix as `AG37`, matching ATP's `ag37` URL code; ATP also resolves that code when the slug is intentionally wrong. Do not obtain URLs through name search or network calls at runtime.

## Tasks

### [ ] Task 1: Relocate the raw directory artifact and define generated-artifact ownership

- **Description**: Change deploy staging so the snapshot-generated raw directory writes to `data/deploy/player-directory.json`, alongside the navigation artifacts. Define constants for the raw directory, build manifest, and generated Vite directory/search inputs. Update comments and docstrings to describe the host-side builder contract and to state that the raw input is retained rather than deleted.
- **Files**: `src/flows/deploy.py`, `src/serving/directory.py`, `src/constants.py` if shared path constants are appropriate, `.gitignore`, `web/.gitignore`.
- **Acceptance Criteria**:
  - `generate_navigation_artifacts()` produces `data/deploy/player-directory.json` from the DuckDB snapshot in both the reuse and rebuild branches.
  - The raw directory is not written to `web/public/` and is not a web-served file.
  - All generated deploy and Vite-input artifacts are ignored by Git; no fixture/generated JSON is committed.
- **Guardrails**:
  - Do not change the player directory schema or query.
  - Do not make the raw directory a model/champion artifact.

### [ ] Task 2: Rework the Node generator around hash-only build state

- **Description**: Refactor `buildPlayerIndex` to read the raw directory from `data/deploy/`, retain its manifest in `data/deploy/player-index.manifest.json`, and write two deterministic inputs to `web/src/assets/generated/`:
  - `player-directory.json`: normalized `{ players }` payload for a normal Vite JSON import.
  - `player-search.json`: `{ index: "..." }` payload produced by `MiniSearch.addAll(players)`.

  Compute hashes of the raw input, the serialized MiniSearch options/schema, and both generated byte payloads. Reuse is valid only when the input/options hashes match the manifest and both generated files exist with bytes matching the manifest's directory/search hashes. Otherwise regenerate both outputs atomically, then write the manifest. Do not use or record filenames in the manifest.
- **Files**: `web/scripts/build-player-index.mjs`, `data/deploy/player-index.manifest.json` (generated), `web/src/assets/generated/player-directory.json` (generated), `web/src/assets/generated/player-search.json` (generated), `web/tests/buildPlayerIndex.test.mjs`.
- **Acceptance Criteria**:
  - The generator still indexes `players` parsed from the raw directory JSON via `MiniSearch.addAll(players)`.
  - The manifest has a schema/version plus `sourceHash`, `optionsHash`, `directoryHash`, and `searchHash`; it contains no path or filename fields.
  - Identical raw bytes plus identical options reuse verified generated files without re-indexing.
  - Changed raw bytes, changed options, missing output, or hash-corrupt output rebuilds both generated files and updates the manifest.
  - No `player-index.manifest.json` or player payload is written under `web/public/`.
- **Guardrails**:
  - Keep the test suite hermetic using temporary directories only.
  - Do not substitute browser-side MiniSearch construction for the serialized index.

### [ ] Task 3: Make development and production consume one generated Vite-input contract

- **Description**: Move the Node-generation invocation out of the web Dockerfile because the authoritative manifest/input are under repository-level `data/deploy/`, outside the existing `web/` Docker context. Have `just deploy` invoke the generator after `deploy.py` stages navigation artifacts and before `docker buildx build ... web/`. Update `scripts/dev.sh` to invoke the same generator using the same input, manifest, and output locations after it stages snapshot navigation artifacts and before starting Vite. Remove the old in-container builder invocation.

  Development must use the same application module path as production: direct JSON import for the directory and `?no-inline` URL import for the search index. Vite development may serve source URLs rather than production-hashed `/assets/` URLs, but application code must not branch by environment, fetch from `public/`, or retain the old runtime manifest protocol. A restarted `just dev` after a changed snapshot therefore exercises the identical generated-input -> React-loader pipeline as a deployment; only Vite's transport/cache implementation differs by mode.
- **Files**: `justfile`, `scripts/dev.sh`, `web/Dockerfile`, `tests/test_dev_script.py`, any existing deployment-command tests in `tests/test_deploy.py` that assert command order.
- **Acceptance Criteria**:
  - `just deploy` order is: deploy stages raw navigation data -> Node generator verifies/reuses or rebuilds Vite inputs -> Docker builds the web image.
  - `just dev` stages and generates the same `web/src/assets/generated/*` files before `pnpm dev` starts; Vite dev and production use the same imports and runtime loader.
  - Neither development nor production serves `web/public/player-directory.json`, reads a browser manifest, or selects a loader based on mode.
  - A web Docker build consumes pre-generated `web/src/assets/generated/*` files and does not attempt to read `data/deploy/` from inside its `web/` build context.
  - Missing generated Vite inputs fail the Docker/Vite build rather than silently serving old public artifacts.
- **Guardrails**:
  - Preserve Vite's normal distinction: dev source URLs are fresh/HMR-friendly; production files are Vite-hashed immutable `/assets/` files. Do not try to fake production cache headers in the dev server.
  - Preserve the current `web/` Docker build context and published-image flow.
  - Do not add a second, divergent generation command for development.

### [ ] Task 4: Replace the browser manifest loader with direct Vite assets

- **Description**: Update the player-index module to statically import the generated directory JSON so player data is immediately available with the entry bundle. Import the generated search JSON using an explicit non-inline URL. Preserve the `loadSearch` API, but on its first call concurrently fetch the search asset and dynamically import MiniSearch, deserialize with the existing options, and cache the promise in memory. Remove all manifest fetching and path parsing.

  Preserve current page contracts without a network-loading state for the directory: the directory query/hook must synchronously expose the bundled players and retain a stable shared result for Home and H2H.
- **Files**: `web/src/lib/playerIndex.ts`, `web/src/assets/generated/player-directory.json` (generated), `web/src/assets/generated/player-search.json` (generated), `web/src/pages/Home.tsx`, `web/src/pages/H2H.tsx`, `web/tests/playerIndex.test.mjs`.
- **Acceptance Criteria**:
  - In both Vite dev and the production build, an initial homepage load makes no request for `/player-index.manifest.json` or `/player-directory.*.json`.
  - Initial player picker data comes from the Vite entry chunk with no fetch-induced loading/error state.
  - The first non-empty picker search triggers one request for the Vite-emitted search asset and one dynamic MiniSearch chunk load; subsequent searches reuse the resolved in-memory search function without further index fetches.
  - Search result behavior (prefix/fuzzy matching and mapping IDs back to complete player records) is unchanged.
- **Guardrails**:
  - Do not bundle the serialized search index into the initial entry chunk.
  - Do not introduce localStorage, polling, or an API fallback for the static directory.

### [ ] Task 5: Remove obsolete public caching rules and defer one shared directory-info query

- **Description**: Delete nginx rules and ignores that support the browser discovery manifest and custom root-level hashed player payloads; Vite's `/assets/` immutable-cache rule becomes the only search-index cache policy.

  Replace the two current `getDirectoryInfo()` queries (`["directory_info"]` in the root layout and `["directory-info"]` in Home) with one shared idle-gated hook and one query key. Schedule it with `requestIdleCallback` and a delayed fallback where idle callbacks are unavailable. Do not display total matches as `0` while it is pending; retain the existing no-date footer behavior until a response arrives.
- **Files**: `web/nginx.conf.template`, `web/.gitignore`, `web/src/lib/directoryInfo.ts` (new), `web/src/routes.tsx`, `web/src/pages/Home.tsx`, related web tests.
- **Acceptance Criteria**:
  - `/api/directory_info` is not started during the first render task and is requested once after browser idle.
  - Layout and Home share the same cached query/result; no duplicate endpoint request occurs due to mismatched query keys.
  - Footer date and home match count remain absent until data resolves, then render their existing values.
  - Production caching remains: Vite `/assets/` files are immutable for one year; `index.html` remains non-cacheable.
- **Guardrails**:
  - Do not defer or alter selected-player/profile data requests.
  - Do not show incorrect zero values as a loading fallback.

### [ ] Task 6: Add ATP overview links to profile headings

- **Description**: Add a small pure URL helper that derives an ATP overview URL from the profile's canonical `player_id` and a display-name slug. Render the main profile bio name as an external anchor to that URL, with a compact external-link SVG. It opens in a new tab and exposes an accessible label that states it opens the official ATP Tour profile.
- **Files**: `web/src/lib/atpProfile.ts` (new), `web/src/pages/Profile.tsx`, `web/src/index.css`, focused web test file(s).
- **Acceptance Criteria**:
  - Felix Auger-Aliassime (`AG37`) resolves to `https://www.atptour.com/en/players/felix-auger-aliassime/ag37/overview`.
  - The player name in the main profile bio heading is the link; the external icon is visually secondary and `aria-hidden`.
  - The anchor uses `target="_blank"` and `rel="noopener noreferrer"`, has an accessible indication that it opens the official ATP profile in a new tab, and preserves visible keyboard focus and mobile layout.
  - URL generation is deterministic and requires no ATP request, scrape, or name matching.
- **Guardrails**:
  - Do not make external ATP availability part of page rendering or tests.
  - Do not add the URL to the directory payload, API response, database schema, or generated MiniSearch index; it is derived client-side from data already present in `PlayerProfile`.

### [ ] Task 7: Verify artifact lifecycle, ATP links, and network behavior

- **Description**: Expand hermetic tests for build-state reuse, generated artifact integrity, and loader laziness. Run the project web checks plus focused Node/Python tests. Use a browser/network trace in dev or a built image to validate the intended request graph.
- **Files**: `web/tests/buildPlayerIndex.test.mjs`, `web/tests/playerIndex.test.mjs`, focused ATP-profile-link test(s), `tests/test_directory.py`, `tests/test_deploy.py`, `tests/test_dev_script.py`, `scripts/web_checks.sh` if its covered commands need adjustment.
- **Acceptance Criteria**:
  - All existing and updated hermetic tests run with no database or network access.
  - `just dev` rebuilds or verifies the same generated assets used by `just deploy` before Vite serves the app; the browser follows the same direct-directory/lazy-search loader path in both modes.
  - Initial network traces in Vite dev and a production build contain no manifest/directory payload request and no eager search-index request. Production additionally serves the lazy search payload as Vite's immutable hashed `/assets/` file.
  - Typing the first search character loads the index once and returns the expected player matches.
  - `directory_info` appears only after the idle gate and is issued once.
  - The URL helper covers punctuation, accents, repeated whitespace, and the `AG37` fixture; rendered profile markup has secure external-link attributes.

## Dependencies

1. Task 1 establishes the new raw-input location.
2. Task 2 defines and produces Vite source inputs from that location.
3. Task 3 wires those inputs into both production and development before Docker/Vite consumes them.
4. Task 4 changes browser consumption after the generated inputs exist.
5. Task 5 removes obsolete runtime paths and fixes the separate non-critical API request.
6. Task 6 adds the independent profile-link UI using the existing player identity.
7. Task 7 validates the complete lifecycle.

## QA/testing scenarios

1. **Unchanged snapshot**: stage identical raw directory twice; generator verifies stored hashes and reuses generated Vite inputs without calling `MiniSearch.addAll` again.
2. **Changed directory**: alter one player field; generator regenerates both Vite inputs, updates all output hashes, and Vite emits a new entry/search asset reference.
3. **Changed MiniSearch configuration**: alter serialization options; `optionsHash` invalidates reuse and regenerates the search payload.
4. **Corrupt/missing generated input**: manifest/source hashes match but either generated file fails verification; generator rebuilds both rather than using stale data.
5. **Fresh web image**: run the generation step, build from `web/`, and verify Docker contains no raw directory or browser manifest endpoint.
6. **Initial page load**: player picker renders from bundled data; neither a player manifest nor a directory JSON request occurs.
7. **First search**: entering a non-empty string loads MiniSearch and the separate index exactly once; repeated searches use the cached in-memory function.
8. **Directory metadata**: first paint succeeds without `directory_info`; after idle, Home and Layout share a single response and display values only when available.
9. **ATP profile link**: a player name opens the expected ATP overview in a new tab; a malformed external URL never blocks profile rendering because the helper has no network dependency.
