#!/usr/bin/env python
"""Start the Prefect work-pool worker on the host.

The Prefect server runs in the cluster; the worker runs on the host so it can
access local resources (PostgreSQL, trained models). It loads the repo's .env
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


def _register_deployments() -> None:
    """Register host-run scheduled deployments (idempotent upserts by name).

    The Monday scrape and ETL deployments live on the host work pool, so they
    are created/updated whenever this worker starts and stay manually
    triggerable from the Prefect UI or `prefect deployment run`.
    """
    from src.flows.etl import register_deployment as register_etl
    from src.flows.scrape import register_deployment as register_scrape

    register_scrape()
    register_etl()


_register_deployments()
cmd = ["prefect", "worker", "start", "--pool", "tennis-pool"]
raise SystemExit(subprocess.call(cmd))
