"""DuckDB client for the tennis-ml pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from src import constants

_conn: duckdb.DuckDBPyConnection | None = None


def get_conn() -> duckdb.DuckDBPyConnection:
    global _conn
    if _conn is None:
        db_path = (
            Path(constants.TENNIS_DB_PATH)
            if constants.TENNIS_DB_PATH
            else constants.ROOT / "data" / "tennis.duckdb"
        )
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
    if params is None:
        return to_dataframe(sql)
    return get_conn().execute(sql, params).fetchdf()


def first_row_dict(df: pd.DataFrame) -> dict[str, Any]:
    """First row of a result frame as a dict with string keys.

    pandas-stubs types ``DataFrame.to_dict`` as ``dict[Hashable, Any]`` even
    though the keys are the column names; normalize to str so the result fits
    the typed ``dict[str, ...]`` parameters downstream.
    """
    return {str(k): v for k, v in df.iloc[0].to_dict().items()}
