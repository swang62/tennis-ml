"""Prefect flow: Bronze → Gold ETL.

Runs `dbt build` which builds the medallion layers in dependency order:
silver.player_matches (player-perspective rows) -> silver.rolling_features
(post-match snapshots) -> gold.match_features (canonical one-row-per-match
training table) -> gold.player_profiles (derived player-grain aggregates).

Wikipedia bio enrichment happens at seed time via `just db-seed --enrich`
(never after ETL); re-run `just db-etl` to pick up new summaries.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO

from prefect import flow, task

from src import constants
from src.constants import (
    BRONZE_TABLE,
    GOLD_TABLE,
    LOGS,
    PROFILES_TABLE,
    SILVER_PLAYER_MATCHES,
    SILVER_ROLLING_FEATURES,
    WORK_POOL_NAME,
)
from src.db.client import get_conn
from src.db.conninfo import dbt_env
from src.utils import load_env

DBT_BUILD_CMD = ["uv", "run", "dbt", "build", "--project-dir", "dbt", "--profiles-dir", "dbt"]
DBT_RUN_RESULTS = constants.ROOT / "dbt" / "target" / "run_results.json"

ETL_DEPLOYMENT_NAME = "etl"
# No cron: ETL is triggered by the scrape flow via run_deployment only when new
# rows were stored, so an empty or Cloudflare-blocked scrape never runs it.


def run_dbt_build(
    profiles_dir: str | Path = "dbt",
    log_file: Path | None = None,
    full_refresh: bool = False,
) -> subprocess.CompletedProcess:
    """Build dbt models, optionally streaming output to ``log_file``."""
    cmd = [*DBT_BUILD_CMD, "--full-refresh"] if full_refresh else DBT_BUILD_CMD
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
def bronze_to_gold(full_refresh: bool = False) -> int:
    log_file = _etl_log_file()
    mode = "full refresh" if full_refresh else "incremental"
    print(f"dbt mode: {mode}")
    run_dbt_build(log_file=log_file, full_refresh=full_refresh)
    with get_conn().cursor() as cur:
        counts = {
            BRONZE_TABLE: _table_count(cur, BRONZE_TABLE),
            SILVER_PLAYER_MATCHES: _table_count(cur, SILVER_PLAYER_MATCHES),
            SILVER_ROLLING_FEATURES: _table_count(cur, SILVER_ROLLING_FEATURES),
            GOLD_TABLE: _table_count(cur, GOLD_TABLE),
            PROFILES_TABLE: _table_count(cur, PROFILES_TABLE),
        }
    for table, count in counts.items():
        print(f"{table}: {count} current rows")
    for model, rows in _dbt_model_rows().items():
        action = (
            "rebuilt"
            if full_refresh or model in {"player_profiles", "tour_averages"}
            else "inserted/replaced"
        )
        print(f"dbt {model}: {rows} rows {action}")
    print(f"dbt ({mode}): {_dbt_summary(log_file)}")
    return counts[GOLD_TABLE]


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


@flow(log_prints=True)
def etl_flow(full_refresh: bool = False):
    """Bronze → gold ETL: dbt build only. Enrichment is a seed-time step —
    run `just db-seed --enrich`, then re-run `just db-etl`.
    """
    load_env()
    rows = bronze_to_gold(full_refresh=full_refresh)
    print(f"ETL complete: {rows} gold rows")


def register_deployment() -> None:
    """Create/update the ETL deployment (idempotent by name).

    Registered on the host ``tennis-pool`` work pool so the scrape flow's
    ``run_deployment("etl-flow/etl")`` trigger resolves to it. No cron — ETL
    runs only when new data was actually scraped.
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
        build=False,
        ignore_warnings=True,
        print_next_steps=False,
    )
    print(f"Registered deployment {ETL_DEPLOYMENT_NAME!r} (no cron — scrape-triggered)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="rebuild all dbt-managed silver/gold models from bronze",
    )
    etl_flow(full_refresh=parser.parse_args().full_refresh)
