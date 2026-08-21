"""Prefect flow: Bronze → Gold ETL.

Runs `dbt build` which builds the medallion layers in dependency order:
silver.player_matches (player-perspective rows) -> silver.rolling_features
(post-match snapshots) -> gold.match_features (canonical one-row-per-match
training table) -> gold.player_profiles (derived player-grain aggregates).

Enrichment is a seed-time step (``just seed --enrich``), never post-ETL;
re-run ``just etl`` to pick up new summaries. ETL defaults to a full refresh
(``dbt build --full-refresh``); pass ``--incremental`` to append only new rows.
The scrape-triggered Prefect deployment runs incremental.
"""

import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import sys
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import TextIO, cast
from uuid import UUID

import psycopg
from prefect import flow, get_run_logger, task
from prefect.automations import Automation
from prefect.client.orchestration import get_client
from prefect.events.actions import RunDeployment
from prefect.events.schemas.automations import EventTrigger

from src.constants import (
    BRONZE_MATCHES_TABLE,
    GOLD_MATCHES_TABLE,
    GOLD_PROFILES_TABLE,
    LOGS,
    ROOT,
    SILVER_PLAYER_MATCHES,
    SILVER_ROLLING_FEATURES,
    WORK_POOL_NAME,
    get_database_url,
    load_env,
)
from src.db.client import CONNECT_TIMEOUT_S
from src.db.conninfo import dbt_env

DBT_BUILD_CMD = [
    "uv",
    "run",
    "dbt",
    "build",
    "--project-dir",
    "dbt",
    "--profiles-dir",
    "dbt",
]
DBT_RUN_RESULTS = ROOT / "dbt" / "target" / "run_results.json"

ETL_DEPLOYMENT_NAME = "etl"
# No cron: ETL is triggered by the "scrape-triggers-etl" automation (see
# register_automation) whenever a rankings or matches flow run completes
# successfully. The trigger is a visible Prefect automation, not an in-flow
# command.

RANKINGS_FLOW_NAME = "rankings-flow"  # Prefect flow name of src/flows/rankings.py:rankings_flow
MATCHES_FLOW_NAME = "matches-flow"  # Prefect flow name of src/flows/matches.py:matches_flow
SCRAPE_ETL_AUTOMATION_NAME = "scrape-triggers-etl"


def run_dbt_build(
    profiles_dir: str | Path = "dbt",
    log_file: Path | None = None,
    incremental: bool = False,
    select: list[str] | None = None,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
) -> subprocess.CompletedProcess:
    """Build dbt models, streaming output to ``log_file`` and ``logger``."""
    cmd = [*DBT_BUILD_CMD]
    if str(profiles_dir) != "dbt":
        cmd[-2:] = ["--profiles-dir", str(profiles_dir)]
    if not incremental:
        cmd.append("--full-refresh")
    if select:
        cmd.extend(["--select", *select])
    if logger is not None:
        logger.info(f"dbt command: {' '.join(shlex.quote(part) for part in cmd)}")
    env = {**os.environ, **dbt_env(get_database_url())}
    if log_file is None:
        return subprocess.run(cmd, cwd=ROOT, check=True, env=env)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w") as log:
        log.write(f"$ {' '.join(shlex.quote(part) for part in cmd)}\n")
        return _run_streamed(cmd, log, env, logger)


def _run_streamed(
    cmd: list[str],
    log: TextIO,
    env: dict[str, str],
    logger: logging.Logger | logging.LoggerAdapter | None = None,
) -> subprocess.CompletedProcess:
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            text = line.decode(errors="replace")
            sys.stdout.write(text)
            sys.stdout.flush()
            log.write(text)
            log.flush()
            if logger is not None:
                logger.info(text.rstrip())
    except BaseException:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise
    returncode = proc.wait()
    if logger is not None:
        logger.info(f"dbt build exited with code {returncode}")
    if returncode != 0:
        message = (
            f"dbt build failed with exit code {returncode}; see the artifact log for dbt's error"
        )
        log.write(f"{message}\n")
        log.flush()
        if logger is not None:
            logger.error(message)
        raise subprocess.CalledProcessError(returncode, cmd)
    return subprocess.CompletedProcess(cmd, returncode)


def _etl_log_file() -> Path:
    """Timestamped dbt build log under artifacts/logs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOGS / f"etl_dbt_{timestamp}.log"


def _incremental_watermarks() -> tuple[datetime | None, datetime | None]:
    """Return the bronze source and last successfully-built watermarks."""
    with (
        psycopg.connect(get_database_url(), connect_timeout=CONNECT_TIMEOUT_S) as conn,
        conn.cursor() as cur,
    ):
        cur.execute(f"SELECT MAX(ingested_at) FROM {BRONZE_MATCHES_TABLE}")
        source = cur.fetchone()
        cur.execute(
            "SELECT source_watermark FROM bronze.etl_state WHERE pipeline = %s",
            ("dbt",),
        )
        built = cur.fetchone()
    return (
        source[0] if source is not None else None,
        built[0] if built is not None else None,
    )


def _record_incremental_watermark(watermark: datetime | None) -> None:
    if watermark is None:
        return
    with (
        psycopg.connect(get_database_url(), connect_timeout=CONNECT_TIMEOUT_S) as conn,
        conn.cursor() as cur,
    ):
        cur.execute(
            """
            INSERT INTO bronze.etl_state (pipeline, source_watermark)
            VALUES (%s, %s)
            ON CONFLICT (pipeline) DO UPDATE
            SET source_watermark = EXCLUDED.source_watermark
            """,
            ("dbt", watermark),
        )


@task(retries=0)
def bronze_to_gold(incremental: bool = False) -> int:
    log_file = _etl_log_file()
    mode = "incremental" if incremental else "full_refresh"
    print(f"dbt mode: {mode}")
    source_watermark, built_watermark = _incremental_watermarks()
    select = None
    if (
        incremental
        and source_watermark is not None
        and built_watermark is not None
        and source_watermark <= built_watermark
    ):
        select = ["player_profiles"]
        print("no changed bronze matches: refreshing player_profiles only")
    try:
        logger = get_run_logger()
    except RuntimeError:
        # No active run context (e.g. hermetic .fn() tests); skip Prefect logs.
        logger = None
    run_dbt_build(log_file=log_file, incremental=incremental, select=select, logger=logger)
    if select is None:
        _record_incremental_watermark(source_watermark)
    # dbt owns the ETL write connection; use a separate bounded connection for
    # post-build counts rather than contending with serving's process-local pool.
    with (
        psycopg.connect(
            get_database_url(),
            connect_timeout=CONNECT_TIMEOUT_S,
            options="-c statement_timeout=30000",
        ) as conn,
        conn.cursor() as cur,
    ):
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


@flow(log_prints=True, retries=1)
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
    runs only when a rankings or matches flow run completes successfully. The
    deployment pins ``incremental=True`` so automation-triggered ETL appends
    new rows instead of rebuilding from bronze.
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
    """Automation spec: run ETL when a rankings or matches flow completes successfully.

    Pure builder (no API calls) so the trigger/action wiring is unit-testable
    without a Prefect server. The trigger requires the primary event resource to
    be a flow run (``match={"prefect.resource.id": "prefect.flow-run.*"}``) and
    its related ``flow`` resource to be the rankings or matches flow (one
    ``match_related`` spec with both flow names, i.e. OR across names). The
    primary-resource constraint is what keeps unrelated completions out: without
    it, a single ``match_related`` on the flow name alone can still match events
    whose primary resource is not a flow run. A failed run — or any other flow
    completing — never fires ETL.
    """
    return Automation(
        name=SCRAPE_ETL_AUTOMATION_NAME,
        description="Run ETL after a successful rankings or matches flow run.",
        trigger=EventTrigger(
            expect={"prefect.flow-run.Completed"},
            match={"prefect.resource.id": "prefect.flow-run.*"},
            match_related={
                "prefect.resource.role": "flow",
                "prefect.resource.name": [RANKINGS_FLOW_NAME, MATCHES_FLOW_NAME],
            },
        ),
        actions=[RunDeployment.model_validate({"deployment_id": etl_deployment_id})],
    )


def register_automation() -> None:
    """Idempotent upsert of the rankings/matches -> ETL automation (by name).

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
        f"{RANKINGS_FLOW_NAME} or {MATCHES_FLOW_NAME} success -> "
        f"{etl_flow.name}/{ETL_DEPLOYMENT_NAME}"
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
