"""Shared pytest bootstrap for the tennis-ml test suite.

Sets KMP_DUPLICATE_LIB_OK so torch and faiss can coexist in one test process.

On macOS both torch and faiss-cpu ship their own libomp (LLVM OpenMP runtime).
The first to initialize wins; the second aborts the interpreter with
"OMP: Error #15". The fast suite imports torch (test_nn.py) and faiss
(test_similarity.py) in the same process, so faiss's first native search
crashes the whole run unless this env var is set. This is the documented
PyTorch workaround for exactly this duplicate-libomp situation.

Pretest DB setup: the session-scoped autouse fixture builds a TEMPORARY seeded
miniset database in a fresh temp dir and redirects all DB access to it via the
TENNIS_DB_PATH env var, so the dev/staging/prod database at
`data/tennis.duckdb` is never touched during tests. src.db.client resolves
TENNIS_DB_PATH on every get_conn() call, so every DB-backed test (inference,
ingest, seed) runs against the test DB. The temp DB is built per session and
cleaned up in teardown; it is never written into the repo tree.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import duckdb
import pytest

from src.constants import ROOT
from src.flows.etl import run_dbt_build

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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
    """Build a TEMPORARY seeded miniset test DB; point all DB access at it.

    Sets TENNIS_DB_PATH to a fresh temp-dir DuckDB so every get_conn()
    (inference, ingest, seed) targets the test DB, never the dev/staging/prod
    data/tennis.duckdb. Built once per session from infra/duckdb/init.sql +
    `seed.py --offline` + a dbt gold build, always (a fresh temp dir means the
    build can never be cached). The prior TENNIS_DB_PATH value is restored and
    the temp dir deleted in teardown, so no test DB ever persists.
    """
    previous = os.environ.get("TENNIS_DB_PATH")
    tmp = tempfile.TemporaryDirectory(prefix="tennis-test-db-")
    db = Path(tmp.name) / "tennis.duckdb"
    try:
        os.environ["TENNIS_DB_PATH"] = str(db)

        conn = duckdb.connect(str(db))
        conn.execute((ROOT / "infra" / "duckdb" / "init.sql").read_text())
        conn.close()

        command = ["uv", "run", "python", "infra/duckdb/seed.py", "--offline"]
        proc = subprocess.run(
            command,
            cwd=ROOT,
            env={**os.environ, "TENNIS_DB_PATH": str(db)},
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"test DB seed command failed: {' '.join(command)}\n"
                f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
            )

        profiles_dir = Path(tmp.name) / "profiles"
        profiles_dir.mkdir(parents=True, exist_ok=True)
        (profiles_dir / "profiles.yml").write_text(PROFILES_YML.format(db=db))
        run_dbt_build(profiles_dir=profiles_dir)

        # The singleton connection may have opened a DB before the env var was
        # set; drop it so the next get_conn() reconnects to the test DB.
        import src.db.client as db_client

        db_client._conn = None

        yield
    finally:
        if previous is None:
            os.environ.pop("TENNIS_DB_PATH", None)
        else:
            os.environ["TENNIS_DB_PATH"] = previous
        tmp.cleanup()
