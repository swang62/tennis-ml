# Plan: Add Prefect Cloud as Production Orchestrator

## ⚠️ CRITICAL: Infrastructure Preservation

**The local k3d cluster and all its resources are DEV infrastructure. They are NOT being removed, modified, or replaced.**

### DO NOT TOUCH (under any circumstances):

- `infra/manifests/default/prefect-server.yaml` — local Prefect server Deployment/PVC/Service
- `infra/manifests/default/ingress.yaml` — local ingress rules including `prefect.macsteve.lan`
- `infra/manifests/default/config-map.yaml` — local Prefect config entries
- `infra/manifests/default/mlflow.yaml` — local MLflow server
- `infra/manifests/default/traefik-helm-chart-config.yaml` — ingress controller
- Any k8s resources in the local cluster
- `.env` existing lines — only ADDING new lines, never deleting or modifying
- `.env.example` existing lines — only ADDING new lines, never deleting or modifying
- `AGENTS.md` — local Prefect references stay (document dev environment)
- `README.md` — local Prefect references stay (document dev environment)
- `justfile` — no changes needed
- Flow code (`src/flows/etl.py`, `src/flows/scrape.py`) except adding `retries=2` to scrape
- Worker code (`infra/prefect/worker.py`) except updating docstring
- BentoML serving — unchanged
- MLflow — unchanged
- PostgreSQL — unchanged

### What we ARE doing:

- **Adding** Prefect Cloud as a parallel production environment
- **Adding** `PREFECT_WORKER_QUERY_SECONDS=3600` to `.env` (hourly polling)
- **Adding** `PREFECT_WORKER_QUERY_SECONDS` to `.env.example`
- **Updating** `infra/prefect/worker.py` docstring to document Cloud connection
- **Adding** `retries=2` to `scrape_flow` in `src/flows/scrape.py`
- **Creating** `tennis-pool` work pool in Cloud workspace (Cloud-side only)
- **Registering** scrape + etl deployments against Cloud (Cloud-side only)
- **Verifying** everything works via E2E test

## Goal

Add Prefect Cloud as the production orchestration layer alongside the existing
local k3d Prefect server (which stays as dev). The worker, flows, and
deployments are unchanged — only the API endpoint differs. Local cluster
remains completely untouched and functional.

## Architecture

| Environment | Prefect server | Worker | Use case |
|-------------|---------------|--------|----------|
| **Dev** (local) | k8s `prefect-server` at `prefect.macsteve.lan` | `infra/prefect/worker.py` with `PREFECT_API_URL=https://prefect.macsteve.lan/api` | Local development, testing |
| **Prod** (Cloud) | Prefect Cloud managed workspace | `infra/prefect/worker.py` with `PREFECT_API_URL=https://api.prefect.cloud/...` | Production scheduling |

Both environments use the same worker script, same flow code, same `tennis-pool`
work pool name. Switching is just changing `PREFECT_API_URL` + `PREFECT_API_KEY`
in `.env`. The local dev environment continues to work exactly as before.

## Scope

### In scope (ADDITIVE ONLY):

- **ADD** `PREFECT_WORKER_QUERY_SECONDS=3600` to `.env` (hourly polling)
- **ADD** `PREFECT_WORKER_QUERY_SECONDS` to `.env.example`
- **UPDATE** `infra/prefect/worker.py` docstring to document Cloud connection
- **ADD** `retries=2` to `scrape_flow` in `src/flows/scrape.py`
- **CREATE** `tennis-pool` work pool in Cloud workspace (Cloud-side only)
- **REGISTER** scrape + etl deployments against Cloud (Cloud-side only)
- **VERIFY** everything works via E2E test (delete rankings after Aug 3, run scrape, confirm backfill)

### Out of scope (NOT CHANGING):

- **Local k8s Prefect server** — stays as dev, completely untouched
- **`infra/manifests/default/prefect-server.yaml`** — NOT deleting, NOT modifying
- **`infra/manifests/default/ingress.yaml`** — NOT deleting, NOT modifying
- **`infra/manifests/default/config-map.yaml`** — NOT deleting, NOT modifying
- **`AGENTS.md`** — local Prefect references stay (document dev environment)
- **`README.md`** — local Prefect references stay (document dev environment)
- **`.env` existing lines** — only ADDING, never deleting or modifying
- **`.env.example` existing lines** — only ADDING, never deleting or modifying
- **Flow code (`etl.py`)** — completely unchanged
- **BentoML serving** — completely unchanged
- **MLflow** — completely unchanged
- **PostgreSQL** — completely unchanged

## Tasks

### [x] Task 1: Create Prefect Cloud workspace

- **Status**: DONE — credentials already in `.env`
- `PREFECT_API_URL` set to Cloud workspace URL
- `PREFECT_API_KEY` set
- **Local infrastructure**: NOT affected

### [ ] Task 2: Add `PREFECT_WORKER_QUERY_SECONDS` to `.env` and `.env.example`

- **Action**: ADD new lines to `.env` and `.env.example`
- **Files**:
  - `.env` — **ADD** `PREFECT_WORKER_QUERY_SECONDS=3600` (new line). **DO NOT delete or modify any existing lines.**
  - `.env.example` — **ADD** `PREFECT_WORKER_QUERY_SECONDS=3600` with docs explaining it controls worker polling interval (default 15s, 3600 = hourly). **DO NOT delete or modify any existing lines.**
- **Acceptance Criteria**:
  - `.env` has `PREFECT_WORKER_QUERY_SECONDS=3600` (new line added)
  - `.env.example` documents the new var (new line added)
  - **No existing lines modified or deleted**
- **Local infrastructure**: NOT affected (env vars only)

### [ ] Task 3: Update `infra/prefect/worker.py` docstring

- **Action**: UPDATE docstring only, NO code changes
- **Files**:
  - `infra/prefect/worker.py` — **UPDATE** docstring to mention both dev (local) and prod (Cloud) modes. **DO NOT change any code.**
- **Acceptance Criteria**:
  - Docstring mentions both dev (local) and prod (Cloud) modes
  - **No code changes**
- **Local infrastructure**: NOT affected (docstring only)

### [ ] Task 4: Add `retries=2` to `scrape_flow`

- **Action**: ADD `retries=2` parameter to `@flow` decorator
- **Files**:
  - `src/flows/scrape.py` — **UPDATE** `@flow` decorator to add `retries=2`. If the scrape fails (network, Cloudflare block, etc.), Prefect retries up to 2 times before marking as failed. The ETL flow is triggered by the scrape, so it doesn't need its own retries (it only runs if scrape succeeds).
- **Acceptance Criteria**:
  - `scrape_flow` has `retries=2` in its decorator
  - If scrape fails, Cloud UI shows retry attempts (up to 2), then final state
- **Local infrastructure**: NOT affected (flow code change applies to both dev and prod)

### [ ] Task 5: Create Cloud work pool

- **Action**: CREATE work pool in Cloud workspace (Cloud-side only)
- **Description**: Create the `tennis-pool` work pool in the Cloud workspace. Type: `process` (same as local dev — the host worker picks up work items).
- **CLI**: `prefect work-pool create tennis-pool --type process`
- **Acceptance Criteria**:
  - Command succeeds against Cloud
  - Work pool visible in Cloud UI
- **Local infrastructure**: NOT affected (Cloud-side only)

### [ ] Task 6: Register deployments against Cloud

- **Action**: REGISTER deployments in Cloud workspace (Cloud-side only)
- **Description**: Run the worker registration to push the `scrape` and `etl` deployments to the Cloud workspace. Verify they appear in the Cloud UI with correct schedules.
- **CLI**: `uv run python -c "from src.flows.scrape import register_deployment; register_deployment(); from src.flows.etl import register_deployment as r2; r2()"` (or just run the worker, which calls `_register_deployments()` on startup)
- **Acceptance Criteria**:
  - `scrape` deployment visible in Cloud UI with cron `0 6 * * 1`
  - `etl` deployment visible in Cloud UI (no cron, scrape-triggered)
  - Both show work pool `tennis-pool`
- **Local infrastructure**: NOT affected (Cloud-side only)

### [ ] Task 7: End-to-end verification

- **Action**: VERIFY everything works by running a real scrape
- **Description**: Force a real scrape by deleting rankings after Aug 3 2026 (so the watermark is behind and the scrape has work to do), then trigger a manual scrape run and verify it executes successfully against Cloud.

  **Steps:**
  1. Delete rankings after Aug 3 2026:
     ```sql
     DELETE FROM bronze.rankings WHERE ranking_date > '2026-08-03';
     ```
  2. Trigger a manual scrape: `just scrape`
  3. Watch the worker logs — should see the scrape flow start, process weeks from Aug 3 to today, and store new rankings.
  4. If the scrape stored new rows, it triggers ETL automatically via `run_deployment("etl-flow/etl")`. Verify ETL runs and completes.
  5. Check Cloud UI for both run logs (scrape + ETL).
  6. Verify the rankings table now has data up to the current week.

- **Acceptance Criteria**:
  - `just scrape` triggers a run visible in Cloud UI
  - Scrape flow completes successfully, processes multiple weeks (Aug 3 → today)
  - ETL flow triggers and completes (if scrape stored new rows)
  - Worker logs show clean Cloud connection (no errors)
  - Rankings table has data up to the current week after the run
- **Local infrastructure**: NOT affected (verification only)

## Execution Order

```
Task 1 (Cloud workspace) ✅ DONE
  ├─→ Task 2 (.env polling) ──┐
  ├─→ Task 3 (worker.py docs) ┤
  └─→ Task 4 (scrape retries) ┤
                               ├─→ Task 5 (create work pool)
                               └─→ Task 6 (register deployments)
                                    └─→ Task 7 (e2e verify)
```

Tasks 2, 3, 4 can run in parallel.
Task 5 and 6 can run in parallel after 2-4.
Task 7 runs last.

## QA / Testing Scenarios

1. **Worker startup**: `uv run python infra/prefect/worker.py` connects to Cloud, registers both deployments, starts polling `tennis-pool`.
2. **Polling interval**: Worker logs show polling every ~3600 seconds (1 hour), not every 15 seconds.
3. **Manual scrape**: `just scrape` triggers a run visible in Cloud UI; scrape completes and triggers ETL if new data exists.
4. **Missed schedule recovery**: Stop the worker, wait past a scheduled cron time, restart the worker. Verify it picks up the queued run and executes it.
5. **Watermark backfill**: Run scrape with no date params; verify it finds the newest stored ranking and backfills to today (check logs for weeks processed).
6. **Retry on failure**: If scrape fails (e.g., Cloudflare block), Cloud UI shows up to 2 retry attempts before final failure state.
7. **Local dev still works**: Switching `PREFECT_API_URL` back to the local server and restarting the worker connects to local Prefect (no code changes).

## Safety Constraints

### What this plan does NOT do:

- **Does NOT delete any k8s resources** — local Prefect server stays
- **Does NOT modify any k8s manifests** — ingress, config-map, prefect-server.yaml all stay
- **Does NOT delete any `.env` lines** — only adding new lines
- **Does NOT delete any `.env.example` lines** — only adding new lines
- **Does NOT modify AGENTS.md or README.md** — local Prefect docs stay
- **Does NOT modify the local Prefect server** — it continues running as dev
- **Does NOT modify the local ingress** — `prefect.macsteve.lan` stays
- **Does NOT modify the local config-map** — Prefect config stays
- **Does NOT modify flow code** except adding `retries=2` to scrape
- **Does NOT modify worker code** except updating docstring
- **Does NOT affect BentoML, MLflow, or PostgreSQL**

### What this plan DOES:

- **Adds** Cloud as a parallel production environment
- **Adds** polling interval config to `.env`
- **Adds** retries to scrape flow
- **Creates** Cloud-side work pool and deployments
- **Verifies** everything works

## Notes

- **Dev vs prod switching**: Change `PREFECT_API_URL` in `.env` to point at either the local server (`https://prefect.macsteve.lan/api`) or Cloud (`https://api.prefect.cloud/api/accounts/.../workspaces/...`). The worker script is the same; only the endpoint differs.
- **Why process pool**: Both flows need local resources (PostgreSQL, CloakBrowser, dbt, file system). A push pool or remote execution would require containerizing all of that — not worth it for a single-host setup.
- **Hourly polling for weekly cron**: The scrape runs once per week (Monday 6am). Hourly polling is fine — the worker catches the scheduled time on the next poll. If the computer is off during the cron, Prefect queues the run; when the worker restarts, it picks up the queued run. The scrape flow uses watermark-based backfill, so a single run handles all missed weeks.
- **Retries**: 2 retries handle transient failures (network, Cloudflare blocks). If all retries fail, the run is marked failed in Cloud UI. No deduplication — the watermark logic ensures a single run backfills all missing data.
- **Local dev fallback**: If Cloud has issues, you can switch back to local Prefect by changing `PREFECT_API_URL` in `.env` to `https://prefect.macsteve.lan/api` and restarting the worker. No code changes needed.
