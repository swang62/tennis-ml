"""Bootstrap PostgreSQL structure and safely reset the local development database."""

from __future__ import annotations

import os
import sys
from typing import LiteralString, cast

import psycopg
from psycopg import sql as pg_sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.errors import DuplicateDatabase

from src import constants
from src.constants import ROOT
from src.db.client import get_conn

INIT_SQL = ROOT / "infra" / "postgres" / "init.sql"

# reset permits only local targets; it validates the live connection.
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def init() -> None:
    """Create the configured database if needed, then apply init.sql."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL not set")
    conninfo = conninfo_to_dict(database_url)
    database = str(conninfo.get("dbname") or "")
    if not database:
        raise RuntimeError("DATABASE_URL must include a database name")
    # template1 always exists, unlike postgres when POSTGRES_DB names a
    # different initial database in the official container image.
    conninfo["dbname"] = "template1"
    maintenance_url = make_conninfo(
        **{key: str(value) for key, value in conninfo.items() if value is not None}
    )
    with psycopg.connect(maintenance_url, autocommit=True).cursor() as cur:
        try:
            cur.execute(pg_sql.SQL("CREATE DATABASE {}").format(pg_sql.Identifier(database)))
            print(f"PostgreSQL database created: {database}")
        except DuplicateDatabase:
            pass
    init_sql = INIT_SQL.read_text()
    with get_conn().cursor() as cur:
        cur.execute(cast(LiteralString, init_sql))
    print("PostgreSQL init: done")


def actual_target() -> tuple[str | None, int, str]:
    """Return the configured client-side connection endpoint, including behind NAT."""
    info = get_conn().info
    return info.host, int(info.port), info.dbname


def reset() -> None:
    """Drop and recreate schemas only when the live target is the expected local DB."""
    expected = conninfo_to_dict(constants.DATABASE_URL or "")
    expected_host = str(expected["host"] or "127.0.0.1")
    expected_port = int(expected["port"] or "5432")
    expected_db = str(expected["dbname"] or "tennis")
    host, port, database = actual_target()
    if host not in LOCAL_HOSTS or port != expected_port or database != expected_db:
        raise RuntimeError(
            f"refusing to reset non-local target {host}:{port}/{database}; "
            f"expected local {expected_host}:{expected_port}/{expected_db}"
        )
    conn = get_conn()
    with conn.transaction(), conn.cursor() as cur:
        for schema in ("bronze", "silver", "gold"):
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    init()
    print("PostgreSQL reset: schemas recreated, data restored via `just db-seed`")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "init"
    if command == "init":
        init()
    elif command == "reset":
        reset()
    else:
        print("Usage: uv run python -m src.flows.init_db [init|reset]")
        sys.exit(1)
