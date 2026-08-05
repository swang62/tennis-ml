#!/usr/bin/env python
"""Start the Prefect work-pool worker on the host.

The Prefect server runs in the cluster; the worker runs on the host so it can
access local artifacts (DuckDB, trained models). It loads the repo's .env
(which defines PREFECT_API_URL pointing at the ingress) and starts a worker
attached to the process-based `tennis-pool`.

Run from the repo root:
    uv run python infra/prefect/worker.py

PREFECT_API_URL must be reachable from the host (defaults to the .env value).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env", override=False)

api_url = os.getenv("PREFECT_API_URL", "")
if not api_url:
    sys.exit("PREFECT_API_URL is not set in .env — cannot reach the Prefect server.")

print(f"Connecting worker to Prefect API at {api_url}")
cmd = ["prefect", "worker", "start", "--pool", "tennis-pool"]
raise SystemExit(subprocess.call(cmd))
