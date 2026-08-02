"""Run DuckDB init SQL (schema setup only). Data seeding lives in seed.py."""

import sys
from pathlib import Path

import duckdb

DB_PATH = Path("data/tennis.duckdb")
INIT_SQL = "infra/duckdb/init.sql"


def main():
    if len(sys.argv) < 2 or sys.argv[1] != "init":
        print(f"Usage: uv run python {__file__} init")
        sys.exit(1)

    sql = Path(INIT_SQL).read_text()
    conn = duckdb.connect(str(DB_PATH))
    for stmt in sql.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt + ";")
    conn.close()

    print("DuckDB init: done")


if __name__ == "__main__":
    main()
