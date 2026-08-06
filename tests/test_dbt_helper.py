import subprocess

import pytest

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

    result = run_dbt_build()

    assert calls == [((DBT_BUILD_CMD,), {"cwd": ROOT, "check": True})]
    assert result == "fake-result"


def test_run_dbt_build_propagates_called_process_error(monkeypatch):
    def fail_run(*_args, **_kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=DBT_BUILD_CMD)

    monkeypatch.setattr("src.flows.etl.subprocess.run", fail_run)

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
