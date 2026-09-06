"""Prefect flow for Bronze-to-Gold dbt ETL."""

import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import urllib.parse
from contextlib import suppress
from datetime import datetime, timedelta
from pathlib import Path
from typing import TextIO, cast
from uuid import UUID

import psycopg
from prefect import flow, get_run_logger, task
from prefect.automations import Automation
from prefect.blocks.notifications import CustomWebhookNotificationBlock
from prefect.events import DeploymentEventTrigger, emit_event
from prefect.events.actions import SendNotification
from prefect.events.schemas.automations import EventTrigger, Posture
from prefect.settings import PREFECT_UI_URL

from src.constants import (
    BRONZE_MATCHES_TABLE,
    GOLD_MATCHES_TABLE,
    GOLD_PROFILES_TABLE,
    LOGS,
    PREFECT_FLOW_TIMEOUT_SECONDS,
    ROOT,
    SILVER_ELO_SNAPSHOTS,
    SILVER_PLAYER_MATCHES,
    SILVER_ROLLING_FEATURES,
    WORK_POOL_NAME,
    get_database_url,
    load_env,
)
from src.db.client import CONNECT_TIMEOUT_S
from src.db.conninfo import dbt_env
from src.db.ingest import clear_etl_state
from src.features.elo import materialize_elo

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
NTFY_BLOCK_NAME = "tennis-etl-ntfy"
EMPTY_SCRAPES_AUTOMATION_NAME = "empty-scrapes"


def run_dbt_build(
    profiles_dir: str | Path = "dbt",
    log_file: Path | None = None,
    incremental: bool = False,
    select: list[str] | None = None,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
    subcommand: str = "build",
) -> subprocess.CompletedProcess:
    """Run a dbt subcommand (``build`` includes tests, ``run`` models only).

    ``select``/``incremental`` and command logging are preserved for either.
    """
    cmd = [*DBT_BUILD_CMD]
    cmd[3] = subcommand  # "build" (with tests) vs "run" (models only)
    if str(profiles_dir) != "dbt":
        cmd[-2:] = ["--profiles-dir", str(profiles_dir)]
    if not incremental and subcommand != "test":
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
        logger.info(f"dbt command exited with code {returncode}")
    if returncode != 0:
        message = (
            f"dbt command failed with exit code {returncode}; see the artifact log for dbt's error"
        )
        log.write(f"{message}\n")
        log.flush()
        if logger is not None:
            logger.error(message)
        raise subprocess.CalledProcessError(returncode, cmd)
    return subprocess.CompletedProcess(cmd, returncode)


def _etl_log_file(run_id: str, phase: str) -> Path:
    """Phase-specific, timestamped dbt build log under artifacts/logs."""
    return LOGS / f"etl_dbt_{run_id}_{phase}.log"


# Base dbt phase: silver player/rolling features and gold tour averages/profiles.
BASE_PHASE_MODELS = [
    "player_matches",
    "rolling_features",
    "tour_averages",
    "player_profiles",
]
# Final dbt phase: gold.match_features plus its dbt tests ("+" selects downstream).
FINAL_PHASE_MODELS = ["match_features+"]


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


@task()
def bronze_to_gold(incremental: bool = False, profile_only: bool = False) -> tuple[int, bool]:
    """Build bronze-to-gold models with dbt, split around Elo materialization."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "incremental" if incremental else "full_refresh"
    print(f"dbt mode: {mode}")
    if not incremental:
        clear_etl_state()
    source_watermark, built_watermark = _incremental_watermarks()
    auto_derived = (
        source_watermark is not None
        and built_watermark is not None
        and source_watermark <= built_watermark
    )
    profile_only = (profile_only or auto_derived) and incremental
    try:
        logger = get_run_logger()
    except RuntimeError:
        logger = None

    if profile_only:
        # No new bronze matches: refresh profiles only. Elo and match_features
        # are skipped so unchanged rows aren't needlessly rebuilt.
        log_file = _etl_log_file(run_id, "profiles")
        print(
            "no changed bronze matches: refreshing gold.player_profiles only "
            "(skipping Elo and gold.match_features)"
        )
        run_dbt_build(
            log_file=log_file,
            incremental=incremental,
            select=["player_profiles"],
            logger=logger,
            subcommand="run",
        )
        _report_phase(log_file, incremental, mode, "profiles")
        return _current_gold_count(), True

    # Phase 1 — base dbt models: player_matches, rolling_features, tour_averages, player_profiles.
    # Run without tests so they don't execute against stale Elo/match_features state.
    base_log = _etl_log_file(run_id, "base")
    run_dbt_build(
        log_file=base_log,
        incremental=incremental,
        select=BASE_PHASE_MODELS,
        logger=logger,
        subcommand="run",
    )
    _report_phase(base_log, incremental, mode, "base")

    # Phase 2 — Elo materialization. Runs between base and match_features so the
    # newly calculated Elo is visible in the same run; its failure aborts before
    # the watermark advances.
    print("\n==================== ELO ====================")
    print("ELO phase: rate newly ingested matches and materialize player snapshots")
    elo_before = _elo_counts()
    print(
        "ELO before: "
        f"{elo_before['matches']} source matches, "
        f"{elo_before['snapshots']} materialized matches"
    )
    if logger is not None:
        logger.info("ELO phase: rating new matches and materializing snapshots")
    elo_result = materialize_elo()
    elo_after = _elo_counts()
    skipped = max(0, elo_before["matches"] - elo_result.processed)
    print(
        "ELO detected: "
        f"{elo_result.processed} matches needed rating, "
        f"{skipped} skipped (already materialized)"
    )
    print(
        "ELO changes: "
        f"{elo_result.snapshots} snapshot rows added, "
        f"{elo_after['snapshots'] - elo_before['snapshots']} net snapshot rows, "
        f"{elo_after['snapshots']} total materialized matches"
    )
    print("ELO phase complete")
    print("================================================\n")

    # Phase 3 — final dbt models: gold.match_features. Tests run separately after
    # all five models are materialized, so base-model tests see current state.
    final_log = _etl_log_file(run_id, "final")
    run_dbt_build(
        log_file=final_log,
        incremental=incremental,
        select=FINAL_PHASE_MODELS,
        logger=logger,
        subcommand="run",
    )
    _report_phase(final_log, incremental, mode, "final")

    # Phase 4 — all project data tests. This restores the full 9-test check
    # while avoiding tests against stale Elo during the base phase.
    print("\n==================== TESTS ====================")
    print("TESTS phase: validate all dbt models after Elo and feature materialization")
    tests_log = _etl_log_file(run_id, "tests")
    run_dbt_build(
        log_file=tests_log,
        incremental=True,
        select=["test_type:data"],
        logger=logger,
        subcommand="test",
    )
    _report_phase(tests_log, True, mode, "tests")
    print("TESTS phase complete")
    print("================================================\n")

    # Only advance the watermark after every phase above succeeded.
    _record_incremental_watermark(source_watermark)
    return _current_gold_count(), False


def _report_phase(log_file: Path, incremental: bool, mode: str, phase: str) -> None:
    """Print this dbt phase's per-model write counts and summary line."""
    for model, rows in _dbt_model_rows().items():
        action = (
            "rebuilt"
            if not incremental or model in {"player_profiles", "tour_averages"}
            else "inserted/replaced"
        )
        print(f"dbt {model}: {rows} rows {action} [{phase}]")
    print(f"dbt ({mode}) [{phase}]: {_dbt_summary(log_file)}")


def _current_gold_count() -> int:
    """Return the current gold match row count and print all table sizes."""
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
    return counts[GOLD_MATCHES_TABLE]


def _elo_counts() -> dict[str, int]:
    """Return distinct source-match and Elo-snapshot counts for phase logging."""
    with (
        psycopg.connect(
            get_database_url(),
            connect_timeout=CONNECT_TIMEOUT_S,
            options="-c statement_timeout=30000",
        ) as conn,
        conn.cursor() as cur,
    ):
        cur.execute(f"SELECT COUNT(DISTINCT match_id) FROM {BRONZE_MATCHES_TABLE}")
        source = cur.fetchone()
        cur.execute(f"SELECT COUNT(DISTINCT match_id) FROM {SILVER_ELO_SNAPSHOTS}")
        snapshots = cur.fetchone()
    return {
        "matches": int(source[0]) if source is not None else 0,
        "snapshots": int(snapshots[0]) if snapshots is not None else 0,
    }


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


VALID_ETL_SOURCES = frozenset({"rankings", "matches"})


def etl_run_name(source: str | None) -> str:
    """Return ``etl-{source}`` for scrape triggers, otherwise ``etl-manual``."""
    if source in VALID_ETL_SOURCES:
        return f"etl-{source}"
    return "etl-manual"


def _etl_flow_run_name() -> str:
    """Prefect ``flow_run_name`` callable: resolve the name from the run's params."""
    from prefect.runtime import flow_run as flow_run_runtime

    try:
        params = flow_run_runtime.get_parameters()
    except Exception:
        params = {}
    return etl_run_name(params.get("source"))


@flow(
    log_prints=True,
    flow_run_name=_etl_flow_run_name,
    timeout_seconds=PREFECT_FLOW_TIMEOUT_SECONDS,
)
def etl_flow(
    incremental: bool = False,
    source: str | None = None,
    profile_only: bool = False,
):
    """Build bronze-to-gold models with dbt, split around Elo materialization.

    Phase order: base dbt models -> Elo snapshots -> gold.match_features (+ tests).
    The bronze.etl_state watermark advances only after every phase succeeds.
    """
    if source is not None and source not in VALID_ETL_SOURCES:
        raise ValueError(
            f"etl_flow source must be one of {sorted(VALID_ETL_SOURCES)} or None, got {source!r}"
        )
    load_env()
    gold_rows, resolved_profile_only = bronze_to_gold(
        incremental=incremental, profile_only=profile_only
    )
    print(f"ETL complete: {gold_rows} gold rows")
    try:
        emit_event(
            event="etl.completed",
            resource={"prefect.resource.id": "tennis.gold.match_features"},
            payload={
                "source": source,
                "profile_only": resolved_profile_only,
                "gold_rows": gold_rows,
            },
        )
    except Exception as exc:
        print(f"etl.completed emit failed (audit-only): {exc}")


def register_deployment() -> None:
    """Create or update the automation-triggered incremental ETL deployment."""
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
        concurrency_limit=1,
        triggers=[
            DeploymentEventTrigger(
                name="etl_rankings",
                match={"prefect.resource.id": "tennis.bronze.rankings"},
                expect={"rankings.scraped"},
                posture=Posture.Reactive,
                threshold=1,
                within=timedelta(0),
                parameters={
                    "source": "rankings",
                    "incremental": True,
                    "profile_only": True,
                },
            ),
            DeploymentEventTrigger(
                name="etl_matches",
                match={"prefect.resource.id": "tennis.bronze.matches"},
                expect={"matches.scraped"},
                posture=Posture.Reactive,
                threshold=1,
                within=timedelta(0),
                parameters={"source": "matches", "incremental": True},
            ),
        ],
        build=False,
        ignore_warnings=True,
        print_next_steps=False,
    )
    print(
        f"Registered deployment {ETL_DEPLOYMENT_NAME!r} "
        "(event-triggered, incremental, concurrency_limit=1)"
    )


def _prefect_runs_url() -> str | None:
    """Prefect UI URL, or None when the server exposes no UI URL."""
    ui_url = PREFECT_UI_URL.value()
    if not ui_url:
        return None
    return f"{ui_url.rstrip('/')}/runs"


def register_automation() -> None:
    """Register the empty-scrapes automation idempotently."""
    for name in (
        "scrape-triggers-etl",
        "scrape-triggers-etl-rankings",
        "scrape-triggers-etl-matches",
        "scrape-dead-man-switch",
    ):
        with suppress(ValueError):
            cast(Automation, Automation.read(name=name)).delete()

    ntfy_url = os.environ.get("NTFY_URL")
    parsed = urllib.parse.urlparse(ntfy_url) if ntfy_url else None
    if parsed is None or parsed.scheme not in ("http", "https"):
        print("NTFY_URL not set — no empty-scrapes notifications registered")
        return
    block_url = f"{parsed.scheme}://{parsed.netloc}"
    topic = parsed.path.strip("/")
    print(f"ntfy topic: {topic} (anyone who knows it can read alerts — set NTFY_URL to control it)")
    json_data = {
        "topic": topic,
        "title": "{{ subject }}",
        "message": "{{ body }}",
        "priority": 5,
        "tags": ["skull"],
    }
    click_url = _prefect_runs_url()
    if click_url is not None:
        json_data["click"] = click_url
    block = CustomWebhookNotificationBlock(
        name=NTFY_BLOCK_NAME,
        url=block_url,
        # basedpyright misreads Field(None) defaults on this block as required.
        headers=None,
        cookies=None,
        json_data=json_data,
    )
    block_id = cast(UUID, block.save(name=NTFY_BLOCK_NAME, overwrite=True))

    with suppress(ValueError):
        cast(Automation, Automation.read(name=EMPTY_SCRAPES_AUTOMATION_NAME)).delete()
    automation = Automation(
        name=EMPTY_SCRAPES_AUTOMATION_NAME,
        description=(
            "Alert via ntfy when neither rankings.scraped nor matches.scraped fires within 8 days."
        ),
        trigger=EventTrigger(
            posture=Posture.Proactive,
            expect={"rankings.scraped", "matches.scraped"},
            match={
                "prefect.resource.id": [
                    "tennis.bronze.rankings",
                    "tennis.bronze.matches",
                ]
            },
            threshold=1,
            within=timedelta(days=8),
        ),
        actions=[
            SendNotification(
                block_document_id=block_id,
                subject="tennis-ml: scrapes stale",
                body=(
                    "No rankings.scraped or matches.scraped event in 8 days — "
                    "check Prefect worker / scraper logs."
                ),
            )
        ],
    )
    automation.create()
    print(f"Registered automation {automation.name!r} (empty-scrapes, 8-day window, ntfy alerts)")


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
