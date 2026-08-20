import logging
import subprocess
from typing import cast
from uuid import uuid4

import pytest
from prefect.events.actions import RunDeployment
from prefect.events.schemas.automations import EventTrigger
from prefect.events.schemas.events import ResourceSpecification

import src.flows.etl as etl
from src.constants import ROOT
from src.flows.etl import DBT_BUILD_CMD, etl_flow, run_dbt_build

# Used by etl_flow to build gold; patched so the flow test never runs dbt.
FAKE_GOLD_COUNT = 1


def test_run_dbt_build_invokes_dbt_build_with_exact_args(monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return "fake-result"

    monkeypatch.setattr("src.flows.etl.subprocess.run", fake_run)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db:5432/tennis")

    result = run_dbt_build()

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ([*DBT_BUILD_CMD, "--full-refresh"],)
    assert kwargs["cwd"] == ROOT
    assert kwargs["check"] is True
    # dbt receives fields derived from the single DATABASE_URL.
    env = kwargs["env"]
    assert env["POSTGRES_HOST"] == "db"
    assert env["POSTGRES_PORT"] == "5432"
    assert env["POSTGRES_USER"] == "u"
    assert env["POSTGRES_PASSWORD"] == "p"
    assert env["POSTGRES_DB"] == "tennis"
    assert result == "fake-result"


def test_run_dbt_build_propagates_called_process_error(monkeypatch):
    def fail_run(*_args, **_kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=DBT_BUILD_CMD)

    monkeypatch.setattr("src.flows.etl.subprocess.run", fail_run)

    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_dbt_build()

    assert excinfo.value.returncode == 1
    assert excinfo.value.cmd == DBT_BUILD_CMD


def test_run_dbt_build_incremental_controls_full_refresh(monkeypatch):
    """incremental=False (default) appends --full-refresh; True omits it."""
    calls = []

    def fake_run(*args, **_kwargs):
        calls.append(args[0])

    monkeypatch.setattr("src.flows.etl.subprocess.run", fake_run)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db:5432/tennis")

    run_dbt_build(incremental=True)
    run_dbt_build(incremental=False)

    assert calls == [DBT_BUILD_CMD, [*DBT_BUILD_CMD, "--full-refresh"]]


def test_run_dbt_build_custom_profiles_dir_keeps_full_refresh(monkeypatch):
    """A non-default profiles_dir replaces the --profiles-dir pair without
    dropping --full-refresh from the command."""
    calls = []

    def fake_run(*args, **_kwargs):
        calls.append(args[0])

    monkeypatch.setattr("src.flows.etl.subprocess.run", fake_run)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db:5432/tennis")

    run_dbt_build(profiles_dir="/tmp/profiles")
    run_dbt_build(profiles_dir="/tmp/profiles", incremental=True)

    assert calls[0] == [
        "uv",
        "run",
        "dbt",
        "build",
        "--project-dir",
        "dbt",
        "--profiles-dir",
        "/tmp/profiles",
        "--full-refresh",
    ]
    assert calls[1] == [
        "uv",
        "run",
        "dbt",
        "build",
        "--project-dir",
        "dbt",
        "--profiles-dir",
        "/tmp/profiles",
    ]


def test_streamed_build_streams_lines_to_log_and_logger(monkeypatch, tmp_path, caplog):
    class FakePopen:
        def __init__(self, *_args, **_kwargs):
            self.stdout = [b"line one\n", b"line two\n"]

        def wait(self):
            return 0

    monkeypatch.setattr("src.flows.etl.subprocess.Popen", FakePopen)
    log_file = tmp_path / "etl.log"

    with (
        caplog.at_level(logging.INFO, logger="dbt-test"),
        log_file.open("w") as log,
    ):
        result = etl._run_streamed(DBT_BUILD_CMD, log, {"K": "V"}, logging.getLogger("dbt-test"))

    assert result.returncode == 0
    assert log_file.read_text() == "line one\nline two\n"
    assert ("dbt-test", logging.INFO, "line one") in caplog.record_tuples
    assert ("dbt-test", logging.INFO, "line two") in caplog.record_tuples
    assert ("dbt-test", logging.INFO, "dbt build exited with code 0") in caplog.record_tuples


def test_streamed_build_failure_logs_error_and_raises(monkeypatch, tmp_path, caplog):
    class FailingPopen:
        def __init__(self, *_args, **_kwargs):
            self.stdout = [b"boom\n"]

        def wait(self):
            return 3

    monkeypatch.setattr("src.flows.etl.subprocess.Popen", FailingPopen)
    log_file = tmp_path / "fail.log"

    with (
        caplog.at_level(logging.INFO, logger="dbt-test"),
        pytest.raises(subprocess.CalledProcessError) as excinfo,
        log_file.open("w") as log,
    ):
        etl._run_streamed(DBT_BUILD_CMD, log, {"K": "V"}, logging.getLogger("dbt-test"))

    assert excinfo.value.returncode == 3
    assert excinfo.value.cmd == DBT_BUILD_CMD
    assert "failed with exit code 3" in log_file.read_text()
    assert any(
        level == logging.ERROR and "failed with exit code 3" in msg
        for _name, level, msg in caplog.record_tuples
    )


def test_etl_flow_builds_without_enrichment(monkeypatch):
    """ETL builds bronze-to-gold without enrichment — it's a separate step."""
    monkeypatch.setattr("src.flows.etl.bronze_to_gold", lambda **_: FAKE_GOLD_COUNT)
    monkeypatch.setattr("src.flows.etl.load_env", lambda: None)

    # .fn() bypasses the Prefect engine — a bare etl_flow() call would register
    # a real flow run on the Prefect server (PREFECT_API_URL is set in .env),
    # polluting the deployment history from a hermetic test.
    etl_flow.fn()


def test_etl_flow_maps_incremental_to_bronze_to_gold(monkeypatch):
    """etl_flow's user-facing incremental option flows through untouched."""
    received = []

    def fake_bronze_to_gold(**kwargs):
        received.append(kwargs)
        return FAKE_GOLD_COUNT

    monkeypatch.setattr("src.flows.etl.bronze_to_gold", fake_bronze_to_gold)
    monkeypatch.setattr("src.flows.etl.load_env", lambda: None)

    etl_flow.fn(incremental=True)
    etl_flow.fn()

    assert received == [{"incremental": True}, {"incremental": False}]


def test_bronze_to_gold_maps_incremental_to_dbt_full_refresh(monkeypatch):
    """bronze_to_gold translates incremental=False (default) into dbt
    --full-refresh; incremental=True skips it. run_dbt_build keeps its
    full_refresh parameter for other callers (e.g. drift)."""
    from unittest.mock import MagicMock

    dbt_calls = []
    monkeypatch.setattr("src.flows.etl.run_dbt_build", lambda **kwargs: dbt_calls.append(kwargs))
    monkeypatch.setattr("src.flows.etl._dbt_model_rows", lambda: {})
    monkeypatch.setattr("src.flows.etl._dbt_summary", lambda _log: "Done. PASS=1")

    conn = MagicMock()
    direct_conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    conn.__enter__.return_value = direct_conn
    direct_conn.cursor.return_value.__enter__.return_value = cursor
    monkeypatch.setattr("src.flows.etl.psycopg.connect", lambda *_args, **_kwargs: conn)

    etl.bronze_to_gold.fn(incremental=False)
    etl.bronze_to_gold.fn(incremental=True)

    assert [call["incremental"] for call in dbt_calls] == [False, True]
    assert conn.__enter__.call_count == 2


def test_cli_default_runs_full_refresh(monkeypatch):
    """`just etl` (no args) invokes the flow with incremental=False."""
    calls = []
    monkeypatch.setattr("src.flows.etl.etl_flow", lambda **kwargs: calls.append(kwargs))

    etl.main(["--incremental"])
    etl.main([])

    assert calls == [{"incremental": True}, {"incremental": False}]


def test_register_deployment_pins_incremental_parameters(monkeypatch):
    """The ETL deployment carries explicit incremental=True parameters, so the
    scrape-triggered ETL appends new rows regardless of flow defaults."""
    deploy_kwargs = {}

    class FakeDeployment:
        def deploy(self, **kwargs):
            deploy_kwargs.update(kwargs)

    class FakeFlow:
        name = "etl-flow"

        @staticmethod
        def from_source(**_kwargs):
            return FakeDeployment()

    monkeypatch.setattr("src.flows.etl.etl_flow", FakeFlow)

    etl.register_deployment()

    assert deploy_kwargs["parameters"] == {"incremental": True}


def test_enrich_missing_is_callable():
    """Enrichment is a separate, idempotent module-level function."""
    from src.db.ingest import enrich_missing

    assert callable(enrich_missing)


def test_scrape_etl_automation_triggers_etl_on_scrape_completion():
    """The rankings/matches -> ETL trigger is a visible Prefect automation (not an
    in-flow command): it fires on a rankings or matches flow's Completed event
    and runs the ETL deployment. Pure builder, so no Prefect server or database
    is touched.
    """
    deployment_id = uuid4()
    automation = etl.build_scrape_etl_automation(deployment_id)

    assert automation.name == etl.SCRAPE_ETL_AUTOMATION_NAME
    trigger = cast(EventTrigger, automation.trigger)
    assert trigger.expect == {"prefect.flow-run.Completed"}
    match_related = cast(ResourceSpecification, trigger.match_related)
    assert match_related.root == {
        "prefect.resource.role": "flow",
        "prefect.resource.name": [etl.RANKINGS_FLOW_NAME, etl.MATCHES_FLOW_NAME],
    }
    assert len(automation.actions) == 1
    action = cast(RunDeployment, automation.actions[0])
    assert action.deployment_id == deployment_id
    assert action.source == "selected"
