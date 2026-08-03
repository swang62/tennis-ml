"""Shared small helpers for the tennis-ml repo.

Environment loading is explicit only: call load_env() at process entry points
(pipeline runner, notebooks). Nothing is loaded on import. Also hosts the
repo-local kernelspec registration used by the notebook pipeline runner.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.constants import KERNEL_DIR, KERNEL_NAME, ROOT


def load_env() -> None:
    """Load the env file (idempotent)."""
    load_dotenv(ROOT / ".env", override=True)


DBT_BUILD_CMD = ["uv", "run", "dbt", "build", "--project-dir", "dbt", "--profiles-dir", "dbt"]


def run_dbt_build(profiles_dir: str | Path = "dbt") -> subprocess.CompletedProcess:
    """Run `dbt build` (gold layer) from the repo root; raise on failure.

    `profiles_dir` overrides the profiles directory (the repo default `dbt/`).
    Tests pass a temp dir containing a profiles.yml that points dbt at a
    throwaway DuckDB.
    """
    cmd = DBT_BUILD_CMD
    if str(profiles_dir) != "dbt":
        cmd = [*DBT_BUILD_CMD[:-2], "--profiles-dir", str(profiles_dir)]
    return subprocess.run(cmd, cwd=ROOT, check=True)


def ensure_kernel() -> str:
    """Register a repo-local kernelspec for the running interpreter; return its name.

    Notebook metadata kernelspecs are machine-specific: 'python3' resolves to a
    pyenv interpreter without project deps, and a stale user spec points at a
    removed venv. A repo-local spec for sys.executable (the interpreter actually
    running this pipeline) executes deterministically on any machine.
    """
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    (KERNEL_DIR / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": KERNEL_NAME,
                "language": "python",
            },
            indent=2,
        )
    )
    # JUPYTER_PATH entries are searched before user kernel dirs, so the
    # repo-local spec wins over any stale machine-specific spec of the same name.
    repo_path = str(KERNEL_DIR.parents[1])
    existing = os.environ.get("JUPYTER_PATH", "")
    os.environ["JUPYTER_PATH"] = repo_path if not existing else f"{repo_path}{os.pathsep}{existing}"
    return KERNEL_NAME
