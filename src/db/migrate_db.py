"""Apply idempotent PostgreSQL schema migrations and safely reset local development data."""

from __future__ import annotations

import os
import sys
from typing import LiteralString, cast

import psycopg
from psycopg import sql as pg_sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.errors import DuplicateDatabase

from src import constants
from src.constants import ROOT, SCHEMA_SQL, get_database_url
from src.db.client import CONNECT_TIMEOUT_S, connection


def migrate() -> None:
    """Create the configured database if needed, then apply schema.sql."""
    conninfo = conninfo_to_dict(get_database_url())
    database = str(conninfo.get("dbname") or "")
    if not database:
        raise RuntimeError("DATABASE_URL must include a database name")

    # One bounded, sanitized progress line before the first network call (the
    # maintenance connect) so a wedged server shows up instead of hanging
    # silently. host/port/db only — never the URL, user, or password.
    host = str(conninfo.get("host") or "localhost")
    port = str(conninfo.get("port") or "5432")
    print(f"Connecting to {host}:{port}/{database}...")

    # template1 always exists, unlike postgres when POSTGRES_DB names a
    # different initial database in the official container image.
    conninfo["dbname"] = "template1"
    maintenance_url = make_conninfo(
        **{key: str(value) for key, value in conninfo.items() if value is not None}
    )
    with psycopg.connect(
        maintenance_url, autocommit=True, connect_timeout=CONNECT_TIMEOUT_S
    ).cursor() as cur:
        try:
            cur.execute(pg_sql.SQL("CREATE DATABASE {}").format(pg_sql.Identifier(database)))
            print(f"PostgreSQL database created: {database}")
        except DuplicateDatabase:
            pass
    schema_sql = SCHEMA_SQL.read_text()
    with connection() as conn, conn.transaction(), conn.cursor() as cur:
        cur.execute(cast(LiteralString, schema_sql))
    print("[db] PostgreSQL migration: done")


def actual_target() -> tuple[str | None, int, str, str]:
    """Return the configured client-side connection endpoint."""
    with connection() as conn, conn.cursor() as cur:
        info = conn.info
        cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
        row = cur.fetchone()
        size = str(row[0]) if row else "unknown"
    return info.host, int(info.port), info.dbname, size


def reset() -> None:
    """Drop and recreate schemas."""
    host, port, database, size = actual_target()
    print(f"Database target: {host}:{port}/{database} (current size: {size})")
    try:
        answer = input("Reset this database? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    if answer not in {"y", "yes"}:
        print("Database reset cancelled.")
        return

    with connection() as conn, conn.transaction(), conn.cursor() as cur:
        for schema in ("bronze", "silver", "gold"):
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    migrate()
    print(f"PostgreSQL {host}:{port}/{database} reset and schemas recreated")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "migrate"
    if command == "migrate":
        migrate()
    elif command == "reset":
        reset()
    else:
        print("Usage: uv run python src/db/migrate_db.py [migrate|reset]")
        sys.exit(1)
