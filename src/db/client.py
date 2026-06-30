"""DuckDB client for the tennis-ml pipeline."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DB_PATH = DATA_DIR / "tennis.duckdb"

_conn: duckdb.DuckDBPyConnection | None = None


def get_conn() -> duckdb.DuckDBPyConnection:
    global _conn
    if _conn is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _conn = duckdb.connect(str(DB_PATH))
    return _conn


def get_client():
    return get_conn()


def query(sql: str) -> list[dict]:
    conn = get_conn()
    rows = conn.sql(sql).fetchall()
    columns = [desc[0] for desc in conn.sql(sql).description]
    return [dict(zip(columns, row, strict=False)) for row in rows]


def to_dataframe(sql: str) -> pd.DataFrame:
    return get_conn().sql(sql).fetchdf()
