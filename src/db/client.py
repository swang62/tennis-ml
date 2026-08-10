"""PostgreSQL client for the tennis-ml pipeline.

PostgreSQL (via psycopg) is the only operational backend. Every query uses
psycopg's `%s` placeholders — request data is never concatenated into SQL —
and results come back as pandas DataFrames.

Multi-step writes run inside an explicit `transaction()` context manager that
commits on success and rolls back on error, so Prefect tasks and Bento workers
never leave an idle transaction behind.

DuckDB remains installed solely for the training database snapshots; it is
not part of the operational query path.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, LiteralString, cast

import pandas as pd
import psycopg
from psycopg.rows import tuple_row

_conn: psycopg.Connection[Any] | None = None


def _connect() -> psycopg.Connection[Any]:
    """Open a PostgreSQL connection from the DATABASE_URL env var.

    Reads directly from os.environ so each process picks up its own
    environment: compose sets postgres:5432 for the Bento container, the
    host shell sets localhost:6543 for dev scripts.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL not set — set it to postgresql://user@host:port/db in your shell or .env"
        )
    return psycopg.connect(url, autocommit=True)


def get_conn() -> psycopg.Connection[Any]:
    """Return the process-wide lazy PostgreSQL connection (autocommit)."""
    global _conn
    if _conn is None:
        _conn = _connect()
    return _conn


def close() -> None:
    """Close and reset the process-wide connection.

    Call when a task or worker finishes so the pool never holds stale
    connections across runs.
    """
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None


@contextmanager
def transaction() -> Iterator[psycopg.Cursor[Any]]:
    """Run a multi-step write atomically; commits on success, rolls back on error."""
    conn = get_conn()
    with conn.transaction(), conn.cursor(row_factory=tuple_row) as cur:
        yield cur


def _cursor_to_df(cur: psycopg.Cursor[Any]) -> pd.DataFrame:
    columns = [d.name for d in cur.description] if cur.description is not None else []
    return pd.DataFrame(cur.fetchall(), columns=columns)


def execute_df(sql: str, params: list[object] | tuple[object, ...] | None = None) -> pd.DataFrame:
    """Run a parameterized query and return the results as a DataFrame.

    Positional `%s` placeholders in `sql` are bound to `params` by psycopg, so
    bound values containing quotes or other SQL metacharacters stay safe.
    """
    with get_conn().cursor() as cur:
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
