"""Prefect flow: Bronze → Gold ETL.

Runs `dbt build` which builds the medallion layers in dependency order:
silver.player_matches (player-perspective rows) -> silver.rolling_features
(post-match snapshots) -> gold.match_features (canonical one-row-per-match
training table). Also enriches player bios (first `Playing style` paragraph,
lead fallback) once the gold layer exists.
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO

from prefect import flow, task

from src import constants
from src.constants import GOLD_TABLE, LOGS
from src.db.client import get_conn
from src.db.conninfo import dbt_env
from src.flows.ingest import enrich_missing as _enrich_missing
from src.utils import load_env

DBT_BUILD_CMD = ["uv", "run", "dbt", "build", "--project-dir", "dbt", "--profiles-dir", "dbt"]


def run_dbt_build(
    profiles_dir: str | Path = "dbt", log_file: Path | None = None
) -> subprocess.CompletedProcess:
    """Build dbt models, optionally streaming output to ``log_file``."""
    cmd = DBT_BUILD_CMD
    if str(profiles_dir) != "dbt":
        cmd = [*DBT_BUILD_CMD[:-2], "--profiles-dir", str(profiles_dir)]
    env = {**os.environ, **dbt_env(constants.build_database_url())}
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


@task(retries=2, retry_delay_seconds=30)
def bronze_to_gold() -> int:
    run_dbt_build(log_file=_etl_log_file())
    with get_conn().cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {GOLD_TABLE}")
        count_row = cur.fetchone()
        row_count = int(count_row[0]) if count_row is not None else 0
    print(f"Gold: {row_count} rows")
    return row_count


@task(retries=1, retry_delay_seconds=10)
def enrich_bios():
    inserted = _enrich_missing()
    print(f"Bios enriched: {inserted} new")
    return inserted


@flow(log_prints=True)
def etl_flow(enrich: bool = False):
    """Bronze → gold ETL. Offline by default: never triggers Wikipedia
    enrichment. Pass `enrich=True` (or `just db-etl -- --enrich`) as the
    explicit operator opt-in for bio enrichment.
    """
    load_env()
    rows = bronze_to_gold()
    print(f"ETL complete: {rows} gold rows")
    if enrich:
        enrich_bios()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enrich",
        action="store_true",
        help="explicit opt-in to Wikipedia bio enrichment after the gold build (default: offline)",
    )
    args = parser.parse_args()
    etl_flow(enrich=args.enrich)
