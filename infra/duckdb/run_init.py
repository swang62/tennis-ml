"""Run DuckDB init or seed SQL."""

import sys
from pathlib import Path

import duckdb

DB_PATH = Path("data/tennis.duckdb")
SQL_FILES = {
    "init": "infra/duckdb/init.sql",
    "seed": "infra/duckdb/seed.sql",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SQL_FILES:
        print(f"Usage: uv run python {__file__} [{'|'.join(SQL_FILES)}]")
        sys.exit(1)

    key = sys.argv[1]
    sql = Path(SQL_FILES[key]).read_text()

    conn = duckdb.connect(str(DB_PATH))
    for stmt in sql.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt + ";")
    conn.close()

    print(f"DuckDB {key}: done")


if __name__ == "__main__":
    main()
