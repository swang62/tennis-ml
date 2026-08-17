"""Environment loading and repository-local Jupyter kernel helpers."""

import json
import os
import sys
import warnings
from pathlib import Path

from src.constants import IMAGE_NAME, ROOT, load_env

# Sentinel kept for monkeypatch in tests; actual path computed in ensure_kernel().
KERNEL_DIR: Path | None = None


def ensure_kernel() -> str:
    """Register a kernel for this interpreter, avoiding stale machine-specific specs."""
    global KERNEL_DIR
    name = IMAGE_NAME
    if KERNEL_DIR is None:
        KERNEL_DIR = ROOT / ".jupyter" / "kernels" / name
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    (KERNEL_DIR / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": name,
                "language": "python",
                "env": {
                    "MLFLOW_TRACKING_INSECURE_TLS": "true",
                    "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", ""),
                },
            },
            indent=2,
        )
    )
    # Prefer the repository kernel over a stale user kernel of the same name.
    repo_path = str(KERNEL_DIR.parents[1])
    existing = os.environ.get("JUPYTER_PATH", "")
    os.environ["JUPYTER_PATH"] = repo_path if not existing else f"{repo_path}{os.pathsep}{existing}"
    return name


def suppress_insecure_tls_warning() -> None:
    """Silence urllib3's InsecureRequestWarning for self-signed cluster HTTPS.

    Only when an insecure-TLS env setting is enabled (MLFLOW_TRACKING_INSECURE_TLS
    or PREFECT_API_TLS_INSECURE_SKIP_VERIFY) — i.e. the operator already opted
    into skipping TLS verification — so the expected warning is not surfaced.
    No other warnings are touched.
    """
    opted_in = any(
        os.environ.get(key, "").strip().lower() in {"true", "1"}
        for key in ("MLFLOW_TRACKING_INSECURE_TLS", "PREFECT_API_TLS_INSECURE_SKIP_VERIFY")
    )
    if not opted_in:
        return
    import urllib3

    warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
