"""Container entrypoint: apply init.sql idempotently, then exec the Quack server.

The DB lives on the /data named volume (never baked into the image). On every
start we open /data/tennis.duckdb, run the schema SQL (all IF NOT EXISTS DDL,
so re-runs are safe and never drop data), close it, then replace this process
with the long-running Quack server.
"""

import os
import sys

import duckdb

INIT_SQL_PATH = "/usr/local/share/quack/init.sql"
DB_PATH = "/data/tennis.duckdb"
SERVER_PATH = "/usr/local/bin/quack_server.py"


def init_statements(sql: str):
    """Yield nonempty statements from semicolon-delimited SQL, ignoring comments."""
    stripped_lines = [line for line in sql.splitlines() if not line.lstrip().startswith("--")]
    for stmt in "\n".join(stripped_lines).split(";"):
        if stmt.strip():
            yield stmt


def main() -> None:
    with open(INIT_SQL_PATH) as f:
        sql = f.read()

    os.makedirs("/data", exist_ok=True)

    con = duckdb.connect(DB_PATH)
    try:
        for stmt in init_statements(sql):
            con.execute(stmt)
    finally:
        con.close()

    os.execv(sys.executable, [sys.executable, SERVER_PATH])


if __name__ == "__main__":
    main()
