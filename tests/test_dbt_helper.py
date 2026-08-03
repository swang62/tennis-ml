import subprocess

import pytest

from src.constants import ROOT
from src.flows.etl import DBT_BUILD_CMD, run_dbt_build


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
