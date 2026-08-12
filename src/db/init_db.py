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
from src.constants import INIT_SQL, ROOT, get_database_url
from src.db.client import get_conn


def init() -> None:
    """Create the configured database if needed, then apply init.sql."""
    conninfo = conninfo_to_dict(get_database_url())
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
    """Return the configured client-side connection endpoint."""
    info = get_conn().info
    return info.host, int(info.port), info.dbname


def reset() -> None:
    """Drop and recreate schemas."""
    host, port, database = actual_target()

    conn = get_conn()
    with conn.transaction(), conn.cursor() as cur:
        for schema in ("bronze", "silver", "gold"):
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    init()
    print(f"PostgreSQL {host}:{port}/{database} reset and schemas recreated")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "init"
    if command == "init":
        init()
    elif command == "reset":
        reset()
    else:
        print("Usage: uv run python src/db/init_db.py [init|reset]")
        sys.exit(1)
