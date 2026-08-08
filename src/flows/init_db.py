"""PostgreSQL bootstrap: structure only, plus a guarded destructive reset.

`init` runs infra/postgres/init.sql against the configured PostgreSQL (the
three schemas and the two non-dbt-owned base tables). It never loads data —
seeding is the explicit `just db-seed` / `just db-seed --all` step.

`reset` drops and recreates the bronze/silver/gold schemas, but only after
checking the ACTUAL connection target (server address, port, and database
name, read from the live connection). It refuses to run against anything other
than the expected local development database (the host/port/db of the single
DATABASE_URL contract), so a stray environment name can never reset a
non-local database.
"""

from __future__ import annotations

import sys
from typing import LiteralString, cast

from psycopg.conninfo import conninfo_to_dict

from src import constants
from src.constants import ROOT
from src.db.client import get_conn

INIT_SQL = ROOT / "infra" / "postgres" / "init.sql"

# The only targets `reset` is allowed to drop. inet_server_addr() reports the
# address the client actually connected to, so an ENVIRONMENT value alone can
# never authorize a reset of a remote database.
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def init() -> None:
    """Apply infra/postgres/init.sql (idempotent, structure only)."""
    sql = INIT_SQL.read_text()
    with get_conn().cursor() as cur:
        cur.execute(cast(LiteralString, sql))
    print("PostgreSQL init: done")


def actual_target() -> tuple[str | None, int, str]:
    """(host, port, database) the current connection actually uses.

    Read from the live psycopg connection's client-side info, not the
    server-reported address: behind Docker NAT (the Compose postgres service)
    ``inet_server_addr()`` reports the container-internal bridge address and
    port, which would always look non-local. The client-side endpoint is
    exactly what the operator configured (DATABASE_URL), so a stray
    environment name still can never authorize resetting a remote database —
    a remote URL reports the remote host here.
    """
    info = get_conn().info
    return info.host, int(info.port), info.dbname


def reset() -> None:
    """Drop and recreate the schemas, refusing non-local targets.

    The check is against the live connection target, not an environment name:
    if the database is not reachable at the expected local host/port/database
    (parsed from the DATABASE_URL contract), nothing is dropped. On success
    the schemas are recreated from init.sql (structure only — data is restored
    by `just db-seed`).
    """
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
