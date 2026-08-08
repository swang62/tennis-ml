import subprocess

import pytest

import src.db.dbt as dbt
import src.flows.etl as etl
from src.constants import ROOT
from src.db.dbt import DBT_BUILD_CMD, run_dbt_build
from src.flows.etl import etl_flow

# Used by etl_flow to build gold; patched so the flow test never runs dbt.
FAKE_GOLD_COUNT = 1


def test_run_dbt_build_invokes_dbt_build_with_exact_args(monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return "fake-result"

    monkeypatch.setattr("src.db.dbt.subprocess.run", fake_run)
    monkeypatch.setattr(dbt.constants, "DATABASE_URL", "postgresql://u:p@db:5432/tennis")

    result = run_dbt_build()

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (DBT_BUILD_CMD,)
    assert kwargs["cwd"] == ROOT
    assert kwargs["check"] is True
    # dbt gets the discrete connection fields derived from the single
    # DATABASE_URL (password-bearing here, matching the Compose stack).
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

    monkeypatch.setattr("src.db.dbt.subprocess.run", fail_run)

    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_dbt_build()

    assert excinfo.value.returncode == 1
    assert excinfo.value.cmd == DBT_BUILD_CMD


def test_dbt_build_cmd_is_exact():
    assert DBT_BUILD_CMD == [
        "uv",
        "run",
        "dbt",
        "build",
        "--project-dir",
        "dbt",
        "--profiles-dir",
        "dbt",
    ]


def test_etl_flow_does_not_trigger_wikipedia_enrichment(monkeypatch):
    """Task 2 guard: `just db-etl` must build silver/gold from bronze with no
    online enrichment. enrich_bios (the Wikipedia path) must never be called."""
    monkeypatch.setattr("src.flows.etl.bronze_to_gold", lambda: FAKE_GOLD_COUNT)
    monkeypatch.setattr(
        "src.flows.etl.enrich_bios",
        lambda: (_ for _ in ()).throw(AssertionError("etl_flow must not call enrich_bios")),
    )
    monkeypatch.setattr("src.flows.etl.load_env", lambda: None)

    etl_flow()


def test_etl_flow_enrich_true_calls_enrich_bios(monkeypatch):
    """Only the explicit `enrich=True` opt-in triggers bio enrichment."""
    monkeypatch.setattr("src.flows.etl.bronze_to_gold", lambda: FAKE_GOLD_COUNT)
    calls = []
    monkeypatch.setattr("src.flows.etl.enrich_bios", lambda: calls.append("enrich"))
    monkeypatch.setattr("src.flows.etl.load_env", lambda: None)

    etl_flow(enrich=True)

    assert calls == ["enrich"]
