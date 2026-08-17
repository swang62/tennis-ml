"""PostgreSQL client for the tennis-ml pipeline.

PostgreSQL (via psycopg) is the only operational backend. Every query uses
psycopg's `%s` placeholders — request data is never concatenated into SQL —
and results come back as pandas DataFrames.

Each process shares a lazily-created `psycopg_pool.ConnectionPool`
(min_size=0, max_size=1, autocommit, ~30s connection/checkout/readiness
timeouts, 30s idle close) that starts with no connections and, across the two
Bento workers, caps the app at two checked-out connections. A checked-out
connection returns to the pool the moment its caller exits; surplus physical
connections are closed after `MAX_IDLE_S` of disuse, so an idle app does not
retain PostgreSQL connections (psycopg's default idle timeout is 10 minutes,
far too long to hold server capacity). `execute_df()` checks out one
connection per call; multi-step writes run inside an explicit
`transaction()` context manager that holds one checked-out connection,
commits on success and rolls back on error. A broken connection is discarded
by the pool — statements are never replayed automatically.

DuckDB remains installed solely for the training database snapshots; it is
not part of the operational query path.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, LiteralString, cast

import pandas as pd
import psycopg
from psycopg.rows import tuple_row
from psycopg_pool import ConnectionPool

# Pool bounds: two Bento workers x max 1 connection per process cap the app
# at 2 concurrent PostgreSQL queries; min 0 starts with no connections.
# Subject to the database's capacity.
MIN_POOL_SIZE = 0
MAX_POOL_SIZE = 1

# Idle lifecycle: a returned connection that goes unused is closed after
# MAX_IDLE_S (psycopg defaults to 10 minutes, which would let an idle app
# retain server capacity). Checkouts return immediately; only the physical
# connection is closed after the idle period.
MAX_IDLE_S = 30.0

# Connection bounds: every wait is capped at 30 seconds so a wedged or
# high-latency server fails a query instead of hanging a worker forever.
# connect_timeout caps each TCP/SSL handshake (libpq defaults wait
# indefinitely; integer seconds), pool timeout caps checkout waits, and
# reconnect_timeout caps how long the pool keeps retrying an unreachable
# server before giving up.
CONNECT_TIMEOUT_S = 30
POOL_TIMEOUT_S = 30.0
RECONNECT_TIMEOUT_S = 30.0

_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


def get_pool() -> ConnectionPool:
    """Return the process-local connection pool, creating it on first use.

    The pool is bounded (min_size=0, max_size=1) and autocommit, and closes
    connections left idle for `MAX_IDLE_S`. `wait()` surfaces an unreachable
    DATABASE_URL at first use instead of failing on the first query, with a
    bounded readiness timeout. `check` runs a health probe on every checkout
    so connections broken by the server (e.g. an SSL reset) are discarded
    before reuse, and `reconnect_timeout` bounds how long the pool keeps
    retrying after a failure.
    """
    global _pool
    pool = _pool
    if pool is None:
        with _pool_lock:
            if _pool is None:
                url = os.environ.get("DATABASE_URL")
                if not url:
                    raise RuntimeError(
                        "DATABASE_URL not set — set it to postgresql://user@host:port/db in your shell or .env"
                    )
                _pool = ConnectionPool(
                    url,
                    min_size=MIN_POOL_SIZE,
                    max_size=MAX_POOL_SIZE,
                    max_idle=MAX_IDLE_S,
                    kwargs={"autocommit": True, "connect_timeout": CONNECT_TIMEOUT_S},
                    check=ConnectionPool.check_connection,
                    timeout=POOL_TIMEOUT_S,
                    reconnect_timeout=RECONNECT_TIMEOUT_S,
                    name="tennis-pool",
                )
                _pool.wait(timeout=POOL_TIMEOUT_S)
            pool = _pool
    return pool


def close() -> None:
    """Close the pool and reset it, releasing every connection.

    Call when a worker or test finishes so the pool never holds stale
    connections across runs.
    """
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


@contextmanager
def connection() -> Iterator[psycopg.Connection[Any]]:
    """Check out one pooled connection for the duration of the block.

    The connection is returned to the pool on exit, so callers never hold a
    pooled connection longer than their work needs.
    """
    with get_pool().connection() as conn:
        yield conn


@contextmanager
def transaction() -> Iterator[psycopg.Cursor[Any]]:
    """Run a multi-step write atomically on one pooled connection.

    Commits on success, rolls back on error; the checked-out connection is
    returned to the pool when the context exits.
    """
    with connection() as conn, conn.transaction(), conn.cursor(row_factory=tuple_row) as cur:
        yield cur


def _cursor_to_df(cur: psycopg.Cursor[Any]) -> pd.DataFrame:
    columns = [d.name for d in cur.description] if cur.description is not None else []
    return pd.DataFrame(cur.fetchall(), columns=columns)


def execute_df(sql: str, params: list[object] | tuple[object, ...] | None = None) -> pd.DataFrame:
    """Run a parameterized query and return the results as a DataFrame.

    Positional `%s` placeholders in `sql` are bound to `params` by psycopg, so
    bound values containing quotes or other SQL metacharacters stay safe.

    Each call checks out one pooled connection and returns it on exit. A
    connection-level failure (OperationalError/InterfaceError) is discarded by
    the pool and replaced asynchronously. Statements are never replayed
    automatically here: callers may be writing.
    """
    with connection() as conn, conn.cursor() as cur:
        cur.execute(cast(LiteralString, sql), params)
        return _cursor_to_df(cur)


def to_dataframe(sql: str) -> pd.DataFrame:
    """Run a query with no bound parameters and return a DataFrame."""
    return execute_df(sql)


def first_row_dict(df: pd.DataFrame) -> dict[str, Any]:
    """First row of a result frame as a dict with string keys.

    pandas-stubs types ``DataFrame.to_dict`` as ``dict[Hashable, Any]`` even
    though the keys are the column names; normalize to str so the result fits
    the typed ``dict[str, ...]`` parameters downstream.
    """
    return {str(k): v for k, v in df.iloc[0].to_dict().items()}
