"""Hermetic guard for scripts/probe.py — never touches live services.

scripts/ is not an installed package, so the module is loaded from its file
path. Monkeypatches the I/O boundary (shutil.which, subprocess, urllib,
psycopg) and asserts fail-fast ordering, the required checks, and the rule
that a filesystem MLflow store is never contacted over HTTP.
"""

import importlib.util
from io import StringIO
from pathlib import Path

import pytest

PROBE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "probe.py"


class _FakeTTY:
    """Minimal stdout stand-in whose isatty() reports a terminal."""

    def __init__(self) -> None:
        self.text = ""

    def isatty(self) -> bool:
        return True

    def write(self, text: str) -> int:
        self.text += text
        return len(text)

    def flush(self) -> None:
        pass


@pytest.fixture
def probe():
    spec = importlib.util.spec_from_file_location("probe", PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    """Base environment: required vars set, MLflow on the filesystem store."""
    for name in ("DATABASE_URL", "PREFECT_API_URL", "POSTGRES_PASSWORD", "BENTO_API_KEY"):
        monkeypatch.setenv(name, f"test-{name}")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *_args, **_kwargs: None)
    return env_path


def test_all_checks_pass_with_filesystem_mlflow(probe, monkeypatch, capsys):
    monkeypatch.setattr(probe.shutil, "which", lambda name: f"/usr/bin/{name}")
    run_calls = []
    monkeypatch.setattr(probe, "run", lambda cmd: run_calls.append(cmd) or 0)
    db_calls = []
    monkeypatch.setattr(probe, "db_ok", lambda url: db_calls.append(url) or None)
    http_calls = []
    monkeypatch.setattr(probe, "http_is_ok", lambda url: http_calls.append(url) or True)

    probe._probe()

    # Order-sensitive: Docker daemon before compose, both via `run`.
    docker_info = run_calls.index(["docker", "info"])
    compose_config = run_calls.index(["docker", "compose", "config", "-q"])
    assert docker_info < compose_config
    # Host PostgreSQL probed with the configured DATABASE_URL.
    assert db_calls == ["test-DATABASE_URL"]
    # Filesystem MLflow store: exactly one HTTP contact (Prefect), never MLflow.
    assert http_calls == ["test-PREFECT_API_URL/health"]
    assert "local filesystem store" in capsys.readouterr().out


def test_fail_fast_on_missing_command(probe, monkeypatch, capsys):
    monkeypatch.setattr(
        probe.shutil,
        "which",
        lambda name: None if name == "serviceman" else f"/usr/bin/{name}",
    )
    run_calls = []
    monkeypatch.setattr(probe, "run", lambda cmd: run_calls.append(cmd) or 0)

    with pytest.raises(SystemExit):
        probe._probe()

    assert "required command not found: serviceman" in capsys.readouterr().err
    # No later step ran: docker info was never attempted.
    assert run_calls == []


def test_fail_fast_on_docker_daemon_down(probe, monkeypatch, capsys):
    monkeypatch.setattr(probe.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(probe, "run", lambda *_args: 1)
    monkeypatch.setattr(probe, "db_ok", lambda *_args: pytest.fail("db probe must not run"))

    with pytest.raises(SystemExit):
        probe._probe()

    assert "Docker daemon is not running" in capsys.readouterr().err


def test_fail_fast_on_missing_env_var(probe, monkeypatch, capsys):
    monkeypatch.setattr(probe.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(probe, "run", lambda *_args: 0)
    monkeypatch.delenv("POSTGRES_PASSWORD")
    monkeypatch.setattr(probe, "db_ok", lambda *_args: pytest.fail("db probe must not run"))

    with pytest.raises(SystemExit):
        probe._probe()

    assert "missing POSTGRES_PASSWORD" in capsys.readouterr().err


def test_http_mlflow_is_checked_and_failure_stops(probe, monkeypatch, capsys):
    monkeypatch.setattr(probe.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(probe, "run", lambda *_args: 0)
    monkeypatch.setattr(probe, "db_ok", lambda *_args: None)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    monkeypatch.setattr(
        probe, "http_is_ok", lambda url: not url.startswith("http://127.0.0.1:5000")
    )

    with pytest.raises(SystemExit):
        probe._probe()

    assert "MLflow not reachable" in capsys.readouterr().err


def test_compose_config_failure_fails_last(probe, monkeypatch, capsys):
    monkeypatch.setattr(probe.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(probe, "db_ok", lambda *_args: None)
    monkeypatch.setattr(probe, "http_is_ok", lambda *_args: True)
    monkeypatch.setattr(
        probe,
        "run",
        lambda cmd: 1 if cmd == ["docker", "compose", "config", "-q"] else 0,
    )

    with pytest.raises(SystemExit):
        probe._probe()

    assert "docker compose config failed" in capsys.readouterr().err


def test_no_credentials_in_output(probe, monkeypatch, capsys):
    monkeypatch.setattr(probe.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(probe, "run", lambda *_args: 0)
    monkeypatch.setattr(probe, "db_ok", lambda *_args: None)
    monkeypatch.setattr(probe, "http_is_ok", lambda *_args: True)
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:secret@127.0.0.1:5432/tennis")

    probe._probe()

    captured = capsys.readouterr()
    assert "secret" not in captured.out + captured.err
    assert "postgresql://" not in captured.out + captured.err
