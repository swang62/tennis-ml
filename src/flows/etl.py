"""Prefect flow: Bronze → Gold ETL.

The gold table is built by the dbt model `gold.match_features`
(dbt/models/gold/match_features.sql); this flow runs `dbt build` (which also
runs the gold data tests) and then enriches player bios once the gold table
exists.
"""

import subprocess

from prefect import flow, task

from src.constants import GOLD_TABLE, ROOT
from src.db.client import get_conn
from src.flows.ingest import enrich_missing as _enrich_missing


@task(retries=2, retry_delay_seconds=30)
def bronze_to_gold() -> int:
    subprocess.run(
        ["uv", "run", "dbt", "build", "--project-dir", "dbt", "--profiles-dir", "dbt"],
        cwd=ROOT,
        check=True,
    )
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
    rows = bronze_to_gold()
    if rows > 0:
        enrich_bios()
        print(f"ETL complete: {rows} gold rows")
    else:
        print("No rows in bronze, skipping validation")


if __name__ == "__main__":
    etl_flow()
