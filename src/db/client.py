"""PostgreSQL client for the tennis-ml pipeline.

PostgreSQL (via psycopg) is the only operational backend. Every query uses
psycopg's `%s` placeholders — request data is never concatenated into SQL —
and results come back as pandas DataFrames.

Each process shares a lazily-created `psycopg_pool.ConnectionPool`
(min_size=1, max_size=2, autocommit, health-checked at every checkout)
instead of one global connection, so concurrent Bento worker requests each
run on their own connection. `execute_df()` checks out one connection per
call; multi-step writes run inside an explicit `transaction()` context
manager that holds one checked-out connection, commits on success and rolls
back on error. A broken connection is discarded by the pool — statements are
never replayed automatically.

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

# Pool bounds: four Bento workers x max 2 connections per process permit up
# to 8 concurrent PostgreSQL queries; min 1 per process keeps 4 warm
# connections. Subject to the database's capacity.
MIN_POOL_SIZE = 1
MAX_POOL_SIZE = 2

_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


def get_pool() -> ConnectionPool:
    """Return the process-local connection pool, creating it on first use.

    The pool is bounded (min_size=1, max_size=2), autocommit, and
    health-checked: every checkout verifies the connection with a trivial
    query before handing it out. `wait()` surfaces an unreachable
    DATABASE_URL at first use instead of failing on the first query.
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
                    kwargs={"autocommit": True},
                    check=ConnectionPool.check_connection,
                    name="tennis-pool",
                )
                _pool.wait()
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
    the pool — the next checkout health-checks the replacement — and the
    statement is never replayed automatically here: callers may be writing.
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
