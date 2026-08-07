"""PostgreSQL bootstrap: structure only, plus a guarded destructive reset.

`init` runs infra/postgres/init.sql against the configured PostgreSQL (the
three schemas and the two non-dbt-owned base tables). It never loads data —
seeding is the explicit `just db-seed` / `just db-seed --all` step.

`reset` drops and recreates the bronze/silver/gold schemas, but only after
checking the ACTUAL connection target (server address, port, and database
name, read from the live connection). It refuses to run against anything other
than the expected local development database (the configured POSTGRES_HOST /
POSTGRES_PORT / POSTGRES_DB), so a stray environment name can never reset a
non-local database.
"""

from __future__ import annotations

import sys
from typing import LiteralString, cast

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
    if the database is not reachable at the expected local host/port/database,
    nothing is dropped. On success the schemas are recreated from init.sql
    (structure only — data is restored by `just db-seed`).
    """
    host, port, database = actual_target()
    expected_port = int(constants.POSTGRES_PORT or "6543")
    if host not in LOCAL_HOSTS or port != expected_port or database != constants.POSTGRES_DB:
        raise RuntimeError(
            f"refusing to reset non-local target {host}:{port}/{database}; "
            f"expected local {constants.POSTGRES_HOST}:{expected_port}/{constants.POSTGRES_DB}"
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
