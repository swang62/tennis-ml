"""Read-only DuckDB access to the training snapshot."""

from __future__ import annotations

import duckdb
import pandas as pd

from src.db.snapshot import SNAPSHOT_PATH

_conn: duckdb.DuckDBPyConnection | None = None


def connect() -> duckdb.DuckDBPyConnection:
    """Return the process-wide read-only snapshot connection (lazy)."""
    global _conn
    if _conn is None:
        if not SNAPSHOT_PATH.exists():
            raise FileNotFoundError(
                f"training snapshot not found at {SNAPSHOT_PATH}; run `just snapshot` first"
            )
        _conn = duckdb.connect(str(SNAPSHOT_PATH), read_only=True)
    return _conn


def close() -> None:
    """Close and reset the process-wide snapshot connection."""
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None


def to_dataframe(sql: str) -> pd.DataFrame:
    """Run a read-only query against the training snapshot as a DataFrame."""
    return connect().execute(sql).df()
