import logging
import subprocess
from datetime import datetime
from typing import cast
from uuid import uuid4

import pytest
from prefect.events.actions import RunDeployment
from prefect.events.schemas.automations import EventTrigger
from prefect.events.schemas.events import Resource, ResourceSpecification

import src.flows.etl as etl
from src.flows.etl import DBT_BUILD_CMD, etl_flow, run_dbt_build

# Used by etl_flow to build gold; patched so the flow test never runs dbt.
FAKE_GOLD_COUNT = 1


def test_run_dbt_build_wires_flags_and_env_from_single_url(monkeypatch):
    """run_dbt_build derives the per-var dbt env from one DATABASE_URL and maps
    its options onto flags: incremental=False -> --full-refresh, select adds
    --select <model>, and a custom profiles_dir replaces the default pair."""
    calls = []
    passed_env = {}

    monkeypatch.setattr(
        "src.flows.etl.subprocess.run",
        lambda *args, **kwargs: calls.append(args[0]) or passed_env.update(kwargs.get("env", {})),
    )
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db:5432/tennis")

    run_dbt_build()
    run_dbt_build(incremental=True, select=["player_profiles"], profiles_dir="/tmp/profiles")
    run_dbt_build(incremental=True)

    assert len(calls) == 3
    assert calls[0][-1] == "--full-refresh"  # default incremental=False
    assert "--full-refresh" not in calls[1]
    assert calls[1][calls[1].index("--select") + 1] == "player_profiles"
    assert calls[1][calls[1].index("--profiles-dir") + 1] == "/tmp/profiles"
    assert calls[2] == calls[0][:-1]  # incremental=True only drops --full-refresh
    # dbt receives all connection fields derived from the single DATABASE_URL.
    assert passed_env["POSTGRES_HOST"] == "db"
    assert passed_env["POSTGRES_PORT"] == "5432"
    assert passed_env["POSTGRES_USER"] == "u"
    assert passed_env["POSTGRES_PASSWORD"] == "p"
    assert passed_env["POSTGRES_DB"] == "tennis"


def test_run_dbt_build_propagates_called_process_error(monkeypatch):
    def fail_run(*_args, **_kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=DBT_BUILD_CMD)

    monkeypatch.setattr("src.flows.etl.subprocess.run", fail_run)

    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_dbt_build()

    assert excinfo.value.returncode == 1
    assert excinfo.value.cmd == DBT_BUILD_CMD


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


def test_streamed_build_terminates_dbt_when_interrupted(monkeypatch, tmp_path):
    class InterruptingPopen:
        def __init__(self, *_args, **_kwargs):
            self.stdout = self
            self.terminated = False

        def __iter__(self):
            raise KeyboardInterrupt
            yield b""  # pragma: no cover

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            assert self.terminated
            assert timeout == 5
            return -15

    proc = InterruptingPopen()
    monkeypatch.setattr("src.flows.etl.subprocess.Popen", lambda *_args, **_kwargs: proc)

    with pytest.raises(KeyboardInterrupt), (tmp_path / "etl.log").open("w") as log:
        etl._run_streamed(DBT_BUILD_CMD, log, {"K": "V"})

    assert proc.terminated


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
    monkeypatch.setattr(
        "src.flows.etl._incremental_watermarks",
        lambda: (datetime(2026, 1, 2), datetime(2026, 1, 1)),
    )
    monkeypatch.setattr("src.flows.etl._record_incremental_watermark", lambda _watermark: None)

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


def test_bronze_to_gold_skips_expensive_models_without_new_matches(monkeypatch):
    from unittest.mock import MagicMock

    dbt_calls = []
    monkeypatch.setattr("src.flows.etl.run_dbt_build", lambda **kwargs: dbt_calls.append(kwargs))
    monkeypatch.setattr("src.flows.etl._dbt_model_rows", lambda: {})
    monkeypatch.setattr("src.flows.etl._dbt_summary", lambda _log: "Done. PASS=1")
    monkeypatch.setattr(
        "src.flows.etl._incremental_watermarks",
        lambda: (datetime(2026, 1, 1), datetime(2026, 1, 1)),
    )

    conn = MagicMock()
    direct_conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    conn.__enter__.return_value = direct_conn
    direct_conn.cursor.return_value.__enter__.return_value = cursor
    monkeypatch.setattr("src.flows.etl.psycopg.connect", lambda *_args, **_kwargs: conn)

    etl.bronze_to_gold.fn(incremental=True)

    assert dbt_calls[0]["select"] == ["player_profiles"]


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
    in-flow command): one automation per scrape flow fires on that flow's
    Completed event and runs the ETL deployment, passing its source so the ETL
    run can be named. Pure builder, so no Prefect server or database is touched.
    """
    deployment_id = uuid4()
    for source, flow_name in (
        ("rankings", etl.RANKINGS_FLOW_NAME),
        ("matches", etl.MATCHES_FLOW_NAME),
    ):
        automation = etl.build_scrape_etl_automation(source, deployment_id)

        assert automation.name == f"{etl.SCRAPE_ETL_AUTOMATION_NAME}-{source}"
        trigger = cast(EventTrigger, automation.trigger)
        assert trigger.expect == {"prefect.flow-run.Completed"}
        match = cast(ResourceSpecification, trigger.match)
        assert match.root == {"prefect.resource.id": "prefect.flow-run.*"}
        match_related = cast(ResourceSpecification, trigger.match_related)
        assert match_related.root == {
            "prefect.resource.role": "flow",
            "prefect.resource.name": [flow_name],
        }
        assert len(automation.actions) == 1
        action = cast(RunDeployment, automation.actions[0])
        assert action.deployment_id == deployment_id
        assert action.source == "selected"


def test_scrape_etl_automation_excludes_drift_and_unrelated_flows():
    """Behavior check using Prefect's own ResourceSpecification matching: each
    per-source automation matches only its own scrape flow's completion, while a
    drift-flow (or any other flow) completion does not. The primary-resource
    constraint (``prefect.flow-run.*``) plus the related-flow-name filter are
    what keep unrelated completions from firing ETL.
    """
    deployment_id = uuid4()
    for source, flow_name in (
        ("rankings", etl.RANKINGS_FLOW_NAME),
        ("matches", etl.MATCHES_FLOW_NAME),
    ):
        automation = etl.build_scrape_etl_automation(source, deployment_id)
        trigger = cast(EventTrigger, automation.trigger)
        primary_match = cast(ResourceSpecification, trigger.match)
        match_related = cast(ResourceSpecification, trigger.match_related)

        # primary event resource must be a flow run
        assert primary_match.matches(
            cast(Resource, {"prefect.resource.id": "prefect.flow-run.<uuid>"})
        )
        assert not primary_match.matches(
            cast(Resource, {"prefect.resource.id": "prefect.task-run.<uuid>"})
        )

        # related flow resource: only this source's flow name is allowed
        assert match_related.matches(
            cast(
                Resource,
                {"prefect.resource.role": "flow", "prefect.resource.name": flow_name},
            )
        )
        # the other scrape flow is excluded (no cross-firing)
        other = etl.MATCHES_FLOW_NAME if source == "rankings" else etl.RANKINGS_FLOW_NAME
        assert not match_related.matches(
            cast(Resource, {"prefect.resource.role": "flow", "prefect.resource.name": other})
        )
        # drift flow is excluded
        assert not match_related.matches(
            cast(
                Resource,
                {"prefect.resource.role": "flow", "prefect.resource.name": "drift-flow"},
            )
        )
        # unrelated flow is excluded even with the flow role
        assert not match_related.matches(
            cast(
                Resource,
                {"prefect.resource.role": "flow", "prefect.resource.name": "some-other-flow"},
            )
        )
        # wrong role (e.g. a task) never matches, even with a rankings name
        assert not match_related.matches(
            cast(
                Resource,
                {
                    "prefect.resource.role": "task-run",
                    "prefect.resource.name": flow_name,
                },
            )
        )
