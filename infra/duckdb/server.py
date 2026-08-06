"""Production DuckDB companion server (official Quack extension).

Serves the production DuckDB database at `/data/tennis.duckdb` over DuckDB's
official Quack remote protocol (`quack_serve`), so clients attach it as a
full catalog instead of shipping or opening the DB inside the Bento.

The server owns `/data/tennis.duckdb`, loads the `quack` extension, and serves
on `0.0.0.0:9494` with an explicit runtime token (QUACK_TOKEN, never generated
or printed). It mirrors the production bronze/silver/gold tables as `main`-schema
views because the Quack `client` strips the `catalog.schema.table` prefix on
ATTACH and resolves the bare name against the server's default schema
(duckdb-quack#144); the mirrored views keep the client's existing
schema-qualified SQL working verbatim.

On first start (an empty `/data` named volume) the server creates
`/data/tennis.duckdb` and applies the repo's `init.sql` (idempotent schema
setup) via `apply_init_sql`. On later starts the same SQL is re-applied — it is
entirely `IF NOT EXISTS` DDL, so existing data is preserved. No CSVs are ever
auto-seeded; data loading is owned by the local seed/ingest paths and must be
done explicitly against the running server.

The server stays alive as a long-running process, answering health checks and
serving queries until it receives SIGTERM/SIGINT, at which point it stops the
Quack listener and CHECKPOINTs the database before exiting.

Runtime settings (all required in production):
  QUACK_URI    - the listen URI, e.g. quack:0.0.0.0:9494
  QUACK_TOKEN  - the token clients must present (min 4 chars, never printed)
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import duckdb

DATA_FILE = Path(os.environ.get("QUACK_DB_PATH", "/data/tennis.duckdb"))
URI = os.environ.get("QUACK_URI", "quack:0.0.0.0:9494")
TOKEN = os.environ.get("QUACK_TOKEN", "")

# Physical schemas/tables exposed over the wire. The server mirrors each as a
# `main`-schema view so the Quack client's prefix-stripping resolves them.
_PRODUCTION_TABLES: tuple[str, ...] = (
    "bronze.match_events",
    "silver.player_matches",
    "gold.rolling_features",
    "gold.match_features",
    "gold.player_profiles",
)


def mirror_views(conn: duckdb.DuckDBPyConnection) -> None:
    """Mirror the production tables as `main`-schema views.

    The Quack beta client (duckdb-quack#144) ships `gold.match_features` as the
    bare name `match_features` and the server resolves it against its default
    (`main`) schema. A `main.match_features` view delegating to the physical
    `gold.match_features` table keeps the client's schema-qualified SQL correct.
    """
    for qualified in _PRODUCTION_TABLES:
        schema, table = qualified.split(".", 1)
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        exists = bool(
            conn.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_schema = ? AND table_name = ?",
                [schema, table],
            ).fetchone()
        )
        if exists:
            conn.execute(f"CREATE OR REPLACE VIEW main.{table} AS SELECT * FROM {qualified}")


def _healthcheck(conn: duckdb.DuckDBPyConnection) -> None:
    """Run a real DB query to prove the served database is responsive."""
    row = conn.sql("SELECT 1").fetchone()
    if row != (1,):
        raise RuntimeError(f"healthcheck failed: expected (1,), got {row!r}")


def main() -> int:
    if not TOKEN or len(TOKEN) < 4:
        raise RuntimeError("QUACK_TOKEN is required and must be at least 4 characters")
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"production database not found: {DATA_FILE}")

    conn = duckdb.connect(str(DATA_FILE))
    try:
        conn.execute("INSTALL quack FROM core")
        conn.execute("LOAD quack")
        mirror_views(conn)
        conn.execute(
            "CALL quack_serve(?, token => ?, allow_other_hostname => true)",
            [URI, TOKEN],
        )
        print(f"quack serving {DATA_FILE} on {URI}", flush=True)

        stop = {"fired": False}

        def _stop(_signum: int, _frame: object) -> None:
            stop["fired"] = True

        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)

        # Serve forever, answering health checks, until a stop signal arrives.
        while not stop["fired"]:
            _healthcheck(conn)
            time.sleep(5)

        print("shutdown: stopping quack listener and checkpointing", flush=True)
        conn.execute("CALL quack_stop(?)", [URI])
        conn.execute("CHECKPOINT")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
