"""ClickHouse client for the tennis-ml pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import clickhouse_connect
import pandas as pd
from dotenv import load_dotenv

if TYPE_CHECKING:
    from clickhouse_connect.driver import Client

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "clickhouse")
CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT", "8123"))
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "")
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "")


def get_client(host: str | None = None, port: int | None = None):
    return clickhouse_connect.get_client(
        host=host or CLICKHOUSE_HOST,
        port=port or CLICKHOUSE_PORT,
        password=CLICKHOUSE_PASSWORD,
        username=CLICKHOUSE_USER,
    )


def query(sql: str, client: Client | None = None) -> list[dict[str, Any]]:
    c = client or get_client()
    result = c.query(sql)
    return [dict(row) for row in result.result_rows]


def to_dataframe(sql: str, client: Client | None = None) -> pd.DataFrame:
    c = client or get_client()
    return cast(pd.DataFrame, c.query_df(sql))
