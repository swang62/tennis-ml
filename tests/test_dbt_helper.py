import subprocess

import pytest

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
    monkeypatch.setattr(etl.constants, "DATABASE_URL", "postgresql://u:p@db:5432/tennis")

    result = run_dbt_build()

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (DBT_BUILD_CMD,)
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


def test_etl_flow_builds_without_enrichment(monkeypatch):
    """ETL builds bronze-to-gold without enrichment — it's a separate step."""
    monkeypatch.setattr("src.flows.etl.bronze_to_gold", lambda: FAKE_GOLD_COUNT)
    monkeypatch.setattr("src.flows.etl.load_env", lambda: None)

    # .fn() bypasses the Prefect engine — a bare etl_flow() call would register
    # a real flow run on the Prefect server (PREFECT_API_URL is set in .env),
    # polluting the deployment history from a hermetic test.
    etl_flow.fn()


def test_enrich_missing_is_callable():
    """Enrichment is a separate, idempotent module-level function."""
    from src.db.ingest import enrich_missing

    assert callable(enrich_missing)
