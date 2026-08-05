"""DuckDB client for the tennis-ml pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from src.constants import ROOT, tennis_db_path

DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "tennis.duckdb"

_conn: duckdb.DuckDBPyConnection | None = None


def _resolve_db_path() -> Path:
    """DB file to open: $TENNIS_DB_PATH when set (tests), else the repo default.

    Tests set TENNIS_DB_PATH to a throwaway file so no test ever touches the
    dev/staging/prod database at DB_PATH.
    """
    env_path = tennis_db_path()
    if env_path:
        return Path(env_path)
    return DB_PATH


def get_conn() -> duckdb.DuckDBPyConnection:
    global _conn
    if _conn is None:
        db_path = _resolve_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _conn = duckdb.connect(str(db_path))
    return _conn


def to_dataframe(sql: str) -> pd.DataFrame:
    return get_conn().sql(sql).fetchdf()


def execute_df(sql: str, params: list[object] | None = None) -> pd.DataFrame:
    """Run a query and return results as a DataFrame.

    When `params` is provided, the SQL uses positional `?` placeholders and
    the query is executed as a prepared statement (no string interpolation).
    """
    conn = get_conn()
    if params is None:
        return conn.sql(sql).fetchdf()
    return conn.execute(sql, params).fetchdf()


def first_row_dict(df: pd.DataFrame) -> dict[str, Any]:
    """First row of a result frame as a dict with string keys.

    pandas-stubs types ``DataFrame.to_dict`` as ``dict[Hashable, Any]`` even
    though the keys are the column names; normalize to str so the result fits
    the typed ``dict[str, ...]`` parameters downstream.
    """
    return {str(k): v for k, v in df.iloc[0].to_dict().items()}
