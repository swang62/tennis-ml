"""Prefect flow: Bronze → Gold ETL.

Runs `dbt build` which builds the medallion layers in dependency order:
silver.player_matches (player-perspective rows) and silver.player_rankings
(ranking series) -> gold.rolling_features (post-match snapshots) ->
gold.match_features (canonical one-row-per-match training table). Also
enriches player bios once the gold layer exists.
"""

import subprocess
from pathlib import Path

from prefect import flow, task

from src.constants import GOLD_TABLE, ROOT
from src.db.client import get_conn
from src.flows.ingest import enrich_missing as _enrich_missing
from src.utils import load_env

# --- ETL-specific dbt gold build (only used by this flow) ---
DBT_BUILD_CMD = ["uv", "run", "dbt", "build", "--project-dir", "dbt", "--profiles-dir", "dbt"]


def run_dbt_build(profiles_dir: str | Path = "dbt") -> subprocess.CompletedProcess:
    """Run `dbt build` (gold layer) from the repo root; raise on failure.

    `profiles_dir` overrides the profiles directory (the repo default `dbt/`).
    Tests pass a temp dir containing a profiles.yml that points dbt at a
    throwaway DuckDB.
    """
    cmd = DBT_BUILD_CMD
    if str(profiles_dir) != "dbt":
        cmd = [*DBT_BUILD_CMD[:-2], "--profiles-dir", str(profiles_dir)]
    return subprocess.run(cmd, cwd=ROOT, check=True)


@task(retries=2, retry_delay_seconds=30)
def bronze_to_gold() -> int:
    run_dbt_build()
    conn = get_conn()
    row = conn.sql(f"SELECT COUNT(*) FROM {GOLD_TABLE}").fetchone()
    row_count = int(row[0]) if row is not None else 0
    print(f"Gold: {row_count} rows")
    return row_count


@task(retries=1, retry_delay_seconds=10)
def enrich_bios():
    inserted = _enrich_missing()
    print(f"Bios enriched: {inserted} new")
    return inserted


@flow(log_prints=True)
def etl_flow():
    load_env()
    rows = bronze_to_gold()
    if rows > 0:
        enrich_bios()
        print(f"ETL complete: {rows} gold rows")
    else:
        print("No rows in bronze, skipping validation")


if __name__ == "__main__":
    etl_flow()
