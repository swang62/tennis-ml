# Plan: transparent Bento API proxy

## Goal

Make production nginx behave like the local Vite proxy: a request under
`/api/` is forwarded to Bento with only the `/api/` prefix removed. Remove
bespoke route renames and `/api/internal` paths, retain API-key protection on
explicitly selected operational endpoints, and expose BentoML's built-in
interactive API UI and OpenAPI schema under `/api/`.

## Current Findings

- Local Vite already rewrites `/api/*` to Bento `/*` with
  `path.replace(/^\/api/, '')`.
- BentoML 1.4.39 serves its Swagger UI at `/`, its generated OpenAPI JSON at
  `/docs.json`, and its UI assets under `/static_content/`.
- `location /api/ { proxy_pass ${BENTO_API_URL}/; }` reproduces the Vite
  behavior:
  - `/api/` -> Bento `/` (interactive Swagger UI)
  - `/api/docs.json` -> Bento `/docs.json` (OpenAPI JSON)
  - `/api/static_content/*` -> Bento `/static_content/*` (Swagger assets)
  - `/api/players` -> Bento `/players`
  - `/api/predict_from_ids` -> Bento `/predict_from_ids`
- Current nginx individually maps public routes and renames protected routes:
  `/api/internal/model-info` -> `/model_info`, and
  `/api/internal/predict-batch` -> `/predict_from_ids_bulk`.
- BentoML bulk input expects `{ "rows": [...] }`; drift now sends that shape.

## Scope

### In

- Replace per-route public nginx mappings with one transparent `/api/` proxy.
- Remove `/api/internal` from the public contract.
- Protect chosen endpoints with `X-API-Key` without changing their Bento path.
- Expose Bento's Swagger UI at `/api/` and OpenAPI spec at `/api/docs.json`.
- Align drift constants/tests with the new direct public route names.
- Include `/api/` in the generated sitemap if it is intended as a crawlable
  interactive API documentation page.

### Out

- Changing Bento method implementations or request/response schemas.
- Exposing model-only `/predict`.
- Changing the Vite proxy behavior (it is already the desired contract).
- Publishing/deploying rebuilt images unless explicitly requested.

## Contract After Change

| Public URL | Bento URL | Access |
| --- | --- | --- |
| `/api/` | `/` | public Swagger UI |
| `/api/docs.json` | `/docs.json` | public OpenAPI schema |
| `/api/static_content/*` | `/static_content/*` | public Swagger assets |
| `/api/players`, `/api/health`, etc. | same path without `/api` | public |
| `/api/predict_from_ids` | `/predict_from_ids` | public POST |
| `/api/model_info` | `/model_info` | API key required |
| `/api/predict_from_ids_bulk` | `/predict_from_ids_bulk` | API key required |
| `/api/predict` | n/a | explicit 403 |

## Tasks

### [x] Task 1: Replace nginx route mapping with transparent proxying

- **File**: `web/nginx.conf.template`
- **Description**:
  - Remove the individual public GET and prediction proxy locations.
  - Remove the old `/api/internal/model-info` and `/api/internal/predict-batch`
    route names.
  - Add one public `location /api/` with
    `proxy_pass ${BENTO_API_URL}/;` so nginx strips `/api/` exactly as Vite
    does.
  - Keep an explicit `location /api/predict` 403 response.
  - Remove the nginx `/api/` catch-all 404; unknown paths should naturally
    receive Bento's response.
- **Acceptance Criteria**:
  - `/api/` serves Bento's Swagger UI.
  - `/api/docs.json` returns BentoML OpenAPI JSON.
  - Swagger UI assets at `/api/static_content/*` load through the same proxy.
  - Existing public frontend calls still route to their root Bento endpoints.

### [x] Task 2: Gate selected operational routes without path renames

- **File**: `web/nginx.conf.template`
- **Description**:
  - Add more-specific locations for `/api/model_info` and
    `/api/predict_from_ids_bulk`.
  - Enforce `X-API-Key` in those locations; retain the existing request-method,
    content-type, rate/body-size, and timeout controls where applicable.
  - Proxy each to the matching root Bento route, preserving the uniform
    `/api`-prefix-strip convention rather than using bespoke aliases.
- **Acceptance Criteria**:
  - Missing or invalid keys return 401 for the selected routes.
  - Valid keyed calls reach `/model_info` and `/predict_from_ids_bulk`.
  - Public routes remain accessible without the key.
- **Decision Needed**:
  - Default protected set is `model_info` and `predict_from_ids_bulk`, matching
    the existing operational routes. Confirm or amend this set before merge.

### [x] Task 3: Align shared client route constants and drift diagnostics

- **Files**: `src/constants.py`, `src/flows/drift.py`,
  `tests/test_drift_monitor.py`
- **Description**:
  - Set `MODEL_INFO_ROUTE` to `/api/model_info`.
  - Set `PREDICT_BATCH_ROUTE` to `/api/predict_from_ids_bulk`.
  - Remove stale `/api/internal/...` text from drift errors and test stubs.
  - Preserve the corrected BentoML bulk payload envelope:
    `{"rows": contexts}`.
- **Acceptance Criteria**:
  - Drift calls direct API-contract route names with a valid key.
  - `/api/predict_from_ids_bulk` accepts the `rows` envelope and returns a list
    of prediction records.
  - No code references `/api/internal`.

### [x] Task 4: Make sitemap include the API UI page

- **File**: `web/vite.config.ts`
- **Description**:
  - Add `${SITE_URL}/api/` as a sitemap entry alongside `/` and `/h2h`.
  - Keep `/api/docs.json` and all other API endpoints out of the sitemap; they
    are machine endpoints, while `/api/` is the interactive documentation UI.
- **Acceptance Criteria**:
  - Production `sitemap.xml` includes the canonical `/api/` URL when `SITE_URL`
    is configured.
  - `robots.txt` continues to point at the generated sitemap.

### [x] Task 5: Update nginx contract tests and verify rendered proxy config

- **Files**: `tests/test_service_health.py` and any affected nginx tests.
- **Description**:
  - Replace assumptions about a narrow nginx allowlist with checks for the
    transparent proxy, the two auth-gated routes, Swagger/OpenAPI exposure,
    and the blocked `/api/predict` route.
  - Render the nginx template with test values and run `nginx -t` in an
    `nginx:alpine` container.
- **Acceptance Criteria**:
  - Focused Python tests pass.
  - Rendered nginx configuration is syntactically valid.
  - OpenAPI schema documents the current Bento endpoints, including the bulk
    request object with its required `rows` field.

## QA Scenarios

1. `GET /api/` loads Swagger UI; its relative `docs.json` and
   `static_content/*` requests succeed.
2. `GET /api/docs.json` returns valid JSON containing the current Bento paths
   and bulk request schema.
3. `POST /api/predict_from_ids` remains public and works with the documented
   request shape.
4. `GET /api/model_info` without a key returns 401; with the `.env`
   `BENTO_API_KEY`, it succeeds.
5. `POST /api/predict_from_ids_bulk` without a key returns 401; with a key and
   `{ "rows": [...] }`, it succeeds.
6. Drift flow uses those exact keyed routes and completes scoring.
7. Generated sitemap includes `/api/` but not raw API data/spec endpoints.

## Deployment Note

The nginx template is packaged in the `web` image. After implementation, the
web image must be rebuilt and deployed before production serves the new proxy
or `/api/` Swagger UI. A worker restart alone does not apply nginx changes.
