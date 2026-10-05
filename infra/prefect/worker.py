#!/usr/bin/env python
"""Start the host Prefect worker attached to the process-based `tennis-pool`."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env", override=False)

api_url = os.getenv("PREFECT_API_URL", "")
if not api_url:
    sys.exit("PREFECT_API_URL is not set in .env — cannot reach the Prefect server.")

print(f"Connecting worker to Prefect API at {api_url}")


def _register_deployments() -> None:
    """Register the host scheduled deployments as idempotent name-based upserts."""
    from src.flows.drift import register_deployment as register_drift
    from src.flows.etl import register_deployment as register_etl
    from src.flows.matches import register_deployment as register_matches
    from src.flows.rankings import register_deployment as register_rankings

    register_rankings()
    register_matches()
    register_etl()
    register_drift()


def _register_automations() -> None:
    """Register the visible rankings/matches-to-ETL automation."""
    from src.flows.etl import register_automation

    register_automation()


RETRY_SECONDS = 300


def _retry_forever(step: Callable[[], None]) -> None:
    """Run a step until it succeeds, waiting RETRY_SECONDS between attempts."""
    while True:
        try:
            step()
        except Exception as error:
            print(f"{step.__name__} failed: {error!r}; retrying in {RETRY_SECONDS}s", flush=True)
            time.sleep(RETRY_SECONDS)
        else:
            return


_retry_forever(_register_deployments)
_retry_forever(_register_automations)
# Imported after the PREFECT_API_URL check: src.constants loads .env with
# override=True, which must not shadow the value read above.
from src.constants import WORK_POOL_NAME  # noqa: E402

cmd = ["prefect", "worker", "start", "--pool", WORK_POOL_NAME]
while True:
    exit_code = subprocess.call(cmd)
    print(f"prefect worker exited with {exit_code}; restarting in {RETRY_SECONDS}s", flush=True)
    time.sleep(RETRY_SECONDS)
