"""Prefect flow: Bronze → Gold ETL.

Runs `dbt build` which builds the medallion layers in dependency order:
silver.player_matches (player-perspective rows) -> silver.rolling_features
(post-match snapshots) -> gold.match_features (canonical one-row-per-match
training table). Also enriches player bios (first `Playing style` paragraph,
lead fallback) once the gold layer exists.
"""

from datetime import datetime
from pathlib import Path

from prefect import flow, task

from src.constants import GOLD_TABLE, LOGS
from src.db.client import get_conn
from src.db.dbt import run_dbt_build
from src.flows.ingest import enrich_missing as _enrich_missing
from src.utils import load_env


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
