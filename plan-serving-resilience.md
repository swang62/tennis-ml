# Plan: Serving resilience and concurrency

## Goal

Recover automatically from interrupted PostgreSQL connections and run four Nginx workers plus four BentoML service worker processes.

## Tasks

### [x] 1. Reset stale PostgreSQL connections

- **Files:** `src/db/client.py`, `tests/test_db_client.py`
- **Implement:** Reopen the cached connection when it is closed. When a connection-level psycopg error occurs during `execute_df`, close/reset the shared connection before re-raising so TanStack Query's retry or the next request reconnects. Do not replay potentially ambiguous writes automatically.
- **Acceptance:** A mocked broken connection cannot poison later requests; the next `execute_df` obtains a fresh connection. Tests remain hermetic with no live database.

### [x] 2. Configure four serving workers

- **Files:** `src/serving/service.py`, `web/Dockerfile`
- **Implement:** Set BentoML's documented `workers=4` service setting and change Nginx `worker_processes` from 1 to 4. Keep all existing proxy limits, request validation, and routes unchanged.
- **Acceptance:** Generated Bento configuration declares four service processes; the web image validates its Nginx config contains four workers.
- **Operational constraint:** Four Bento workers each load model state and maintain an independent PostgreSQL connection; the host must have sufficient CPU/RAM and PostgreSQL connection capacity for four service processes.

### [x] 3. Verify focused behavior

- **Files:** relevant changed tests only
- **Checks:** Run `uv run pytest tests/test_db_client.py`, relevant serving tests, `uv run ruff check src/db/client.py src/serving/service.py`, and `uv run basedpyright src/db/client.py src/serving/service.py`.

## Exclusions

- No async rewrite, connection pool, query/index redesign, automatic retry of writes, Nginx rate-limit changes, or Compose deployment changes.
