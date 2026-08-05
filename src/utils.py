"""Shared small helpers for the tennis-ml repo.

Environment loading is explicit only: call load_env() at process entry points
(pipeline runner, notebooks). Nothing is loaded on import. Also hosts the
repo-local kernelspec registration used by the notebook pipeline runner.
"""

import json
import os
import sys
from pathlib import Path

from src.constants import IMAGE_NAME, ROOT, load_env

# Sentinel kept for monkeypatch in tests; actual path computed in ensure_kernel().
KERNEL_DIR: Path | None = None


def ensure_kernel() -> str:
    """Register a repo-local kernelspec for the running interpreter; return its name.

    Notebook metadata kernelspecs are machine-specific: 'python3' resolves to a
    pyenv interpreter without project deps, and a stale user spec points at a
    removed venv. A repo-local spec for sys.executable (the interpreter actually
    running this pipeline) executes deterministically on any machine.
    """
    global KERNEL_DIR
    name = IMAGE_NAME
    if not name:
        raise RuntimeError("IMAGE_NAME not set in env; call load_env() first")
    if KERNEL_DIR is None:
        KERNEL_DIR = ROOT / ".jupyter" / "kernels" / name
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    (KERNEL_DIR / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": name,
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
    return name
