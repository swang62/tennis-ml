"""Prefect flow: Bronze → Gold ETL.

Runs `dbt build` which builds the medallion layers in dependency order:
silver.player_matches (player-perspective rows) -> silver.rolling_features
(post-match snapshots) -> gold.match_features (canonical one-row-per-match
training table) -> gold.player_profiles (derived player-grain aggregates).

Wikipedia bio enrichment happens at seed time via `just seed --enrich`
(never after ETL); re-run `just etl` to pick up new summaries.

ETL defaults to a full refresh (`dbt build --full-refresh`); pass
`--incremental` to append only new rows. The scrape-triggered Prefect
deployment runs incremental.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import TextIO, cast
from uuid import UUID

from prefect import flow, task
from prefect.automations import Automation
from prefect.client.orchestration import get_client
from prefect.events.actions import RunDeployment
from prefect.events.schemas.automations import EventTrigger

from src import constants
from src.constants import (
    BRONZE_MATCHES_TABLE,
    GOLD_MATCHES_TABLE,
    GOLD_PROFILES_TABLE,
    LOGS,
    SILVER_PLAYER_MATCHES,
    SILVER_ROLLING_FEATURES,
    WORK_POOL_NAME,
)
from src.db.client import connection
from src.db.conninfo import dbt_env
from src.utils import load_env

DBT_BUILD_CMD = ["uv", "run", "dbt", "build", "--project-dir", "dbt", "--profiles-dir", "dbt"]
DBT_RUN_RESULTS = constants.ROOT / "dbt" / "target" / "run_results.json"

ETL_DEPLOYMENT_NAME = "etl"
# No cron: ETL is triggered by the "scrape-triggers-etl" automation (see
# register_automation) whenever a scrape flow run completes successfully. The
# trigger is a visible Prefect automation, not an in-flow command.

SCRAPE_FLOW_NAME = "scrape-flow"  # Prefect flow name of src/flows/scrape.py:scrape_flow
SCRAPE_ETL_AUTOMATION_NAME = "scrape-triggers-etl"


def run_dbt_build(
    profiles_dir: str | Path = "dbt",
    log_file: Path | None = None,
    incremental: bool = False,
) -> subprocess.CompletedProcess:
    """Build dbt models, optionally streaming output to ``log_file``."""
    cmd = DBT_BUILD_CMD if incremental else [*DBT_BUILD_CMD, "--full-refresh"]
    if str(profiles_dir) != "dbt":
        cmd = [*DBT_BUILD_CMD[:-2], "--profiles-dir", str(profiles_dir)]
    env = {**os.environ, **dbt_env(constants.get_database_url())}
    if log_file is None:
        return subprocess.run(cmd, cwd=constants.ROOT, check=True, env=env)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w") as log:
        return _run_streamed(cmd, log, env)


def _run_streamed(cmd: list[str], log: TextIO, env: dict[str, str]) -> subprocess.CompletedProcess:
    proc = subprocess.Popen(
        cmd, cwd=constants.ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        text = line.decode(errors="replace")
        sys.stdout.write(text)
        sys.stdout.flush()
        log.write(text)
        log.flush()
    returncode = proc.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)
    return subprocess.CompletedProcess(cmd, returncode)


def _etl_log_file() -> Path:
    """Timestamped dbt build log under artifacts/logs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOGS / f"etl_dbt_{timestamp}.log"


@task(retries=0)
def bronze_to_gold(incremental: bool = False) -> int:
    log_file = _etl_log_file()
    mode = "incremental" if incremental else "full_refresh"
    print(f"dbt mode: {mode}")
    run_dbt_build(log_file=log_file, incremental=incremental)
    with connection() as conn, conn.cursor() as cur:
        counts = {
            BRONZE_MATCHES_TABLE: _table_count(cur, BRONZE_MATCHES_TABLE),
            SILVER_PLAYER_MATCHES: _table_count(cur, SILVER_PLAYER_MATCHES),
            SILVER_ROLLING_FEATURES: _table_count(cur, SILVER_ROLLING_FEATURES),
            GOLD_MATCHES_TABLE: _table_count(cur, GOLD_MATCHES_TABLE),
            GOLD_PROFILES_TABLE: _table_count(cur, GOLD_PROFILES_TABLE),
        }
    for table, count in counts.items():
        print(f"{table}: {count} current rows")
    for model, rows in _dbt_model_rows().items():
        action = (
            "rebuilt"
            if not incremental or model in {"player_profiles", "tour_averages"}
            else "inserted/replaced"
        )
        print(f"dbt {model}: {rows} rows {action}")
    print(f"dbt ({mode}): {_dbt_summary(log_file)}")
    return counts[GOLD_MATCHES_TABLE]


def _table_count(cur, table: str) -> int:
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    row = cur.fetchone()
    return int(row[0]) if row is not None else 0


def _dbt_summary(log_file: Path) -> str:
    """The final dbt result line, e.g. 'Done. PASS=41 WARN=0 ERROR=0 SKIP=0 TOTAL=41'."""
    for line in reversed(log_file.read_text().splitlines()):
        if re.search(r"Done\.\s+PASS=", line):
            return re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
    return "(no summary line in dbt log)"


def _dbt_model_rows() -> dict[str, int]:
    """Return this invocation's dbt model write counts, not table totals."""
    results = json.loads(DBT_RUN_RESULTS.read_text()).get("results", [])
    return {
        result["unique_id"].rsplit(".", maxsplit=1)[-1]: int(
            result.get("adapter_response", {}).get("rows_affected") or 0
        )
        for result in results
        if result.get("unique_id", "").startswith("model.")
    }


@flow(log_prints=True, retries=2)
def etl_flow(incremental: bool = False):
    """Bronze → gold ETL: dbt build only. Enrichment is a seed-time step —
    run `just seed --enrich`, then re-run `just etl`.

    Full refresh by default; `incremental=True` runs dbt without
    `--full-refresh`, so only new rows are appended.
    """
    load_env()
    rows = bronze_to_gold(incremental=incremental)
    print(f"ETL complete: {rows} gold rows")


def register_deployment() -> None:
    """Create/update the ETL deployment (idempotent by name).

    Registered on the host ``tennis-pool`` work pool so the automation's
    ``RunDeployment`` action can resolve it by ``etl-flow/etl``. No cron — ETL
    runs only when the scrape flow completes successfully. The deployment pins
    ``incremental=True`` so scrape-triggered ETL appends new rows instead of
    rebuilding from bronze.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    from typing import Any, cast

    deployment = cast(
        Any,
        etl_flow.from_source(
            source=str(repo_root),
            entrypoint="src/flows/etl.py:etl_flow",
        ),
    )
    deployment.deploy(
        name=ETL_DEPLOYMENT_NAME,
        work_pool_name=WORK_POOL_NAME,
        parameters={"incremental": True},
        build=False,
        ignore_warnings=True,
        print_next_steps=False,
    )
    print(
        f"Registered deployment {ETL_DEPLOYMENT_NAME!r} "
        "(no cron — automation-triggered, incremental)"
    )


def build_scrape_etl_automation(etl_deployment_id: UUID) -> Automation:
    """Automation spec: run ETL when the scrape flow completes successfully.

    Pure builder (no API calls) so the trigger/action wiring is unit-testable
    without a Prefect server. The trigger matches ``prefect.flow-run.Completed``
    events whose related ``flow`` resource is the scrape flow, so a scrape run
    that fails (or any other flow completing) never fires ETL.
    """
    return Automation(
        name=SCRAPE_ETL_AUTOMATION_NAME,
        description="Run ETL after a successful scrape flow run.",
        trigger=EventTrigger(
            expect={"prefect.flow-run.Completed"},
            match_related={
                "prefect.resource.role": "flow",
                "prefect.resource.name": SCRAPE_FLOW_NAME,
            },
        ),
        actions=[RunDeployment.model_validate({"deployment_id": etl_deployment_id})],
    )


def register_automation() -> None:
    """Idempotent upsert of the scrape -> ETL automation (by name).

    Replaces any existing automation with the same name, so worker restarts
    converge on the current spec. Runs on the host worker (alongside the
    deployments) so the trigger is a first-class, visible Prefect automation.
    """
    with get_client(sync_client=True) as client:
        deployment = client.read_deployment_by_name(f"{etl_flow.name}/{ETL_DEPLOYMENT_NAME}")
    automation = build_scrape_etl_automation(deployment.id)
    with suppress(ValueError):
        cast(Automation, Automation.read(name=SCRAPE_ETL_AUTOMATION_NAME)).delete()
    automation.create()
    print(
        f"Registered automation {SCRAPE_ETL_AUTOMATION_NAME!r}: "
        f"{SCRAPE_FLOW_NAME} success -> {etl_flow.name}/{ETL_DEPLOYMENT_NAME}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="append only new rows (dbt build without --full-refresh)",
    )
    args, _ignored = parser.parse_known_args(argv)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    etl_flow(incremental=args.incremental)


if __name__ == "__main__":
    main()
