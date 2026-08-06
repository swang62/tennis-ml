"""DuckDB client for the tennis-ml pipeline.

Two serving modes, selected by the single explicit `ENVIRONMENT` switch:

- `dev` (default): a local embedded DuckDB at `TENNIS_DB_PATH` (or
  `data/tennis.duckdb`). Used by ETL, training, and local `bentoml serve`.
- `production`: a Quack remote served by the companion container
  (infra/duckdb). The production client opens a local session, loads the
  `quack` extension, ATTACHes the remote URI with the runtime token, and makes
  it the default catalog (`USE`) so the existing schema-qualified SQL
  (`bronze.*`, `silver.*`, `gold.*`) resolves verbatim against the remote.

A missing or invalid `ENVIRONMENT` (or missing Quack config in production mode)
fails fast with a clear error — it never silently falls back to the dev DB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from src import constants

_conn: duckdb.DuckDBPyConnection | None = None


def _connect_dev() -> duckdb.DuckDBPyConnection:
    db_path = (
        Path(constants.TENNIS_DB_PATH)
        if constants.TENNIS_DB_PATH
        else constants.ROOT / "data" / "tennis.duckdb"
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def _connect_production() -> duckdb.DuckDBPyConnection:
    uri = constants.QUACK_URI
    token = constants.QUACK_TOKEN
    if not uri or not token:
        raise RuntimeError(
            "ENVIRONMENT=production requires QUACK_URI and QUACK_TOKEN to be set; "
            "refusing to fall back to the dev database"
        )
    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL quack FROM core")
    conn.execute("LOAD quack")
    # The URI is trusted deployment config (not user SQL), so it is formatted
    # into the statement; the token is a runtime secret and always bound as a
    # prepared `?` parameter.
    conn.execute(
        f"ATTACH '{uri}' AS {constants.QUACK_CATALOG} (TYPE quack, TOKEN ?, DISABLE_SSL true)",
        [token],
    )
    # Make the remote the default catalog so unqualified `silver.x` /
    # `gold.x` SQL resolves against it, exactly as it does against the
    # embedded dev DB.
    conn.execute(f"USE {constants.QUACK_CATALOG}")
    return conn


def get_conn() -> duckdb.DuckDBPyConnection:
    global _conn
    if _conn is None:
        if constants.ENVIRONMENT == "production":
            _conn = _connect_production()
        elif constants.ENVIRONMENT == "dev":
            _conn = _connect_dev()
        else:
            raise RuntimeError(
                f"invalid ENVIRONMENT={constants.ENVIRONMENT!r}; expected 'dev' or 'production'"
            )
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
