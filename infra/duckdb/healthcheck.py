"""Quack server healthcheck for the companion image.

Attaches to the local Quack server (`127.0.0.1:9494`) with the QUACK_TOKEN
from the environment and runs a real `SELECT 1` to prove the served database
is responsive. Exits non-zero on any failure so the container is marked
unhealthy.
"""

from __future__ import annotations

import os

import duckdb

URI = os.environ.get("QUACK_URI_CANONICAL", "quack:127.0.0.1:9494")
TOKEN = os.environ.get("QUACK_TOKEN", "")


def main() -> int:
    if not TOKEN:
        print("QUACK_TOKEN is not set")
        return 1
    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL quack FROM core")
    conn.execute("LOAD quack")
    conn.execute(
        f"ATTACH '{URI}' AS hc (TYPE quack, TOKEN ?, DISABLE_SSL true)",
        [TOKEN],
    )
    row = conn.execute("SELECT 1").fetchone()
    if row != (1,):
        print(f"healthcheck failed: expected (1,), got {row!r}")
        return 1
    print("quack healthcheck ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
