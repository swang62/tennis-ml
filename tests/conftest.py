"""Shared pytest bootstrap for the tennis-ml test suite.

Pytest environment variables are configured in pyproject.toml. The session
fixture rebuilds the fixed test database under tests/___db, so the development
database at data/tennis.duckdb is never touched.
"""

import os
import subprocess
from pathlib import Path

import duckdb
import pytest

from src.constants import ROOT
from src.flows.etl import run_dbt_build

# dbt needs a profiles.yml pointing at the test DB (same shape as
# tests/test_e2e_ingest_to_inference.py) so the gold build never touches
# data/tennis.duckdb.
PROFILES_YML = """tennis_ml:
  target: local
  outputs:
    local:
      type: duckdb
      path: {db}
      schema: gold
      threads: 1
"""


@pytest.fixture(scope="session", autouse=True)
def seeded_test_db():
    """Rebuild the fixed, ignored test database once per test session."""
    db_path = ROOT / os.environ["TENNIS_DB_PATH"]
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = duckdb.connect(str(db_path))
    conn.execute((ROOT / "infra" / "duckdb" / "init.sql").read_text())
    conn.close()

    subprocess.run(
        ["uv", "run", "python", "infra/duckdb/seed.py"],
        cwd=ROOT,
        check=True,
    )

    profiles_dir = db_path.parent / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    (profiles_dir / "profiles.yml").write_text(PROFILES_YML.format(db=db_path))
    run_dbt_build(profiles_dir=profiles_dir)

    yield

    import src.db.client as db_client

    if db_client._conn is not None:
        db_client._conn.close()
        db_client._conn = None
