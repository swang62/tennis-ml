"""PostgreSQL client with bounded pooling and retry-safe checkout."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, LiteralString, cast

import pandas as pd
import psycopg
from psycopg.rows import tuple_row
from psycopg_pool import ConnectionPool, PoolTimeout

MIN_POOL_SIZE = 0
MAX_POOL_SIZE = 4

MAX_IDLE_S = 10.0

CONNECT_TIMEOUT_S = 30
POOL_TIMEOUT_S = 30.0
RECONNECT_TIMEOUT_S = 30.0

DB_RETRY_ATTEMPTS = 4
DB_RETRY_BASE_S = 1.0
DB_RETRY_MAX_S = 5.0

TRANSIENT_ERRORS: tuple[type[psycopg.Error], ...] = (
    psycopg.OperationalError,
    psycopg.InterfaceError,
    PoolTimeout,
)

_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


def get_pool() -> ConnectionPool:
    """Return the lazy, bounded process-local connection pool."""
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
    """Close and reset the process-local connection pool."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


@contextmanager
def connection() -> Iterator[psycopg.Connection[Any]]:
    """Check out a connection, retrying transient failures before the body runs."""
    delay = DB_RETRY_BASE_S
    for attempt in range(DB_RETRY_ATTEMPTS):
        checkout = get_pool().connection()
        try:
            conn = checkout.__enter__()
        except TRANSIENT_ERRORS:
            if attempt == DB_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, DB_RETRY_MAX_S)
            continue
        try:
            yield conn
        except BaseException as exc:
            checkout.__exit__(type(exc), exc, exc.__traceback__)
            raise
        else:
            checkout.__exit__(None, None, None)
        return


def clear_active_sessions() -> tuple[list[int], list[int]]:
    """Cancel active queries and terminate idle transactions in this database."""
    with (
        psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=CONNECT_TIMEOUT_S) as conn,
        conn.cursor() as cur,
    ):
        cur.execute(
            """
            SELECT pid, state
            FROM pg_stat_activity
            WHERE datname = current_database()
              AND pid <> pg_backend_pid()
              AND backend_type = 'client backend'
              AND state IN ('active', 'idle in transaction', 'idle in transaction (aborted)')
            """
        )
        sessions = cur.fetchall()
        cancelled = [pid for pid, state in sessions if state == "active"]
        terminated = [pid for pid, state in sessions if state != "active"]
        for pid in cancelled:
            cur.execute("SELECT pg_cancel_backend(%s)", (pid,))
        for pid in terminated:
            cur.execute("SELECT pg_terminate_backend(%s)", (pid,))
    return cancelled, terminated


@contextmanager
def transaction() -> Iterator[psycopg.Cursor[Any]]:
    """Run a multi-step write atomically on one pooled connection."""
    with connection() as conn, conn.transaction(), conn.cursor(row_factory=tuple_row) as cur:
        yield cur


def _cursor_to_df(cur: psycopg.Cursor[Any]) -> pd.DataFrame:
    columns = [d.name for d in cur.description] if cur.description is not None else []
    return pd.DataFrame(cur.fetchall(), columns=columns)


def execute_df(sql: str, params: list[object] | tuple[object, ...] | None = None) -> pd.DataFrame:
    """Run a parameterized ``%s`` query and return its results as a DataFrame."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(cast(LiteralString, sql), params)
        return _cursor_to_df(cur)


def to_dataframe(sql: str) -> pd.DataFrame:
    """Run a query with no bound parameters and return a DataFrame."""
    return execute_df(sql)


def first_row_dict(df: pd.DataFrame) -> dict[str, Any]:
    """First row of a result frame as a dict with string keys."""
    return {str(k): v for k, v in df.iloc[0].to_dict().items()}
