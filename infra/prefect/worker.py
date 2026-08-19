#!/usr/bin/env python
"""Start the Prefect work-pool worker on the host.

The Prefect server runs in the cluster; the worker runs on the host so it can
access local resources (PostgreSQL, trained models). It loads the repo's .env
(which defines PREFECT_API_URL pointing at the k3d node port) and starts a
worker attached to the process-based `tennis-pool`.

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

    Independent Monday deployments for rankings (``rankings-flow/rankings``,
    06:00 UTC) and matches (``matches-flow/matches``, 06:30 UTC), plus the
    automation-triggered ETL and drift deployments, all live on the host work
    pool, so they are created/updated whenever this worker starts and stay
    manually triggerable from the Prefect UI or `prefect deployment run`.
    Each module registers exactly its own deployment — no duplicates, and no
    combined scrape flow.
    """
    from src.flows.drift import register_deployment as register_drift
    from src.flows.etl import register_deployment as register_etl
    from src.flows.matches import register_deployment as register_matches
    from src.flows.rankings import register_deployment as register_rankings

    register_rankings()
    register_matches()
    register_etl()
    register_drift()


def _register_automations() -> None:
    """Register event-driven automations (idempotent upserts by name).

    The rankings/matches -> ETL trigger is a visible Prefect automation, not an
    in-flow command, so it lives here alongside the deployments it wires
    together.
    """
    from src.flows.etl import register_automation

    register_automation()


_register_deployments()
_register_automations()
# Imported after the PREFECT_API_URL check: src.constants loads .env with
# override=True, which must not shadow the value read above.
from src.constants import WORK_POOL_NAME  # noqa: E402

cmd = ["prefect", "worker", "start", "--pool", WORK_POOL_NAME]
raise SystemExit(subprocess.call(cmd))
