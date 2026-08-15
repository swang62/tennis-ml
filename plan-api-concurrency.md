# Plan: Remove obsolete directory APIs and pool database reads

## Findings

- `/players` and `/directory_info` have no frontend callers after the deployment-built directory index; remove them.
- Starlette routes are synchronous `def` handlers. Starlette runs them in a thread pool, so they do not block the ASGI event loop; changing them to `async def` while retaining synchronous psycopg would be worse.
- The current process-global psycopg connection serializes database work within each Bento worker. With four Bento processes that permits at most four active database queries, not the intended concurrent reads.
- Frontend query functions return fetch promises through TanStack Query and are non-blocking. Independent Home/H2H queries start concurrently. `MiniSearch.loadJSON` is the only synchronous client work; it is a one-time deserialization after the cached static asset downloads.

## Tasks

### [x] 1. Replace the process-global connection with a bounded Psycopg pool

- **Files:** `pyproject.toml`, `uv.lock`, `src/db/client.py`, `tests/test_db_client.py`, relevant dependent tests
- **Implement:** Add the official Psycopg pool package and lazily create a process-local `ConnectionPool` using `DATABASE_URL`, autocommit, health checking, and an explicit bounded size. Use a checked-out connection for every `execute_df()` call and a single checked-out connection throughout `transaction()`. Preserve the hermetic boundary and add shutdown/reset support for tests/workers.
- **Sizing:** `min_size=1`, `max_size=2` per Bento worker, so four Bento workers allow up to 8 concurrent PostgreSQL queries; document the PostgreSQL connection-capacity requirement.
- **Acceptance:** Concurrent callers can receive distinct connections; a failed/broken connection is discarded/replaced; no statement is replayed automatically; all tests remain live-DB-free.

### [x] 2. Delete obsolete directory endpoints

- **Files:** `src/serving/service.py`, API documentation/comments, endpoint tests
- **Implement:** Remove `/players` and `/directory_info` handlers, their SQL/imports, their mounted `DATA_APP` routes, and tests dedicated solely to them. Preserve the shared deploy-time directory query module and static asset pipeline.
- **Acceptance:** The backend no longer advertises or serves these routes; Home/H2H/footer remain entirely static-directory based.

### [x] 3. Verify concurrency contracts and frontend behavior

- **Files:** focused tests and documentation only where needed
- **Checks:** Add a hermetic pool-concurrency test, run relevant serving/database/web tests, type/lint checks, and confirm no frontend `/api/players` or `/api/directory_info` references. Document that sync Starlette handlers are threadpooled and intentionally remain sync while using pooled sync psycopg connections.

## Explicitly not changing

- Keep `/player_profile`, `/rank_history`, `/match_history`, and `/similar_players` as progressive independent fetches.
- Do not make sync database handlers `async def`, add an async driver, pool on the client/browser, or combine profile chunks into one response.
