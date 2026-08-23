"""Hermetic guard tests for scripts/probe.py."""

import importlib.util
from pathlib import Path

import pytest

PROBE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "probe.py"


@pytest.fixture
def probe():
    spec = importlib.util.spec_from_file_location("probe", PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    """Baseline env (DATABASE_URL, PREFECT_API_URL, POSTGRES_PASSWORD,
    BENTO_API_KEY) is provided by [tool.pytest_env]; MLflow stays on the
    filesystem store for the default code path."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *_args, **_kwargs: None)
    return env_path


def test_all_checks_pass_with_filesystem_mlflow(probe, monkeypatch, capsys):
    monkeypatch.setattr(probe.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(probe, "run", lambda _cmd: 0)
    monkeypatch.setattr(probe, "db_ok", lambda _url: None)
    monkeypatch.setattr(probe, "http_is_ok", lambda _url: True)

    probe._probe()

    assert "local filesystem store" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("api_url", "health_url"),
    (
        ("https://prefect.example.com", "https://prefect.example.com/api/health"),
        ("https://prefect.example.com/api", "https://prefect.example.com/api/health"),
        ("https://prefect.example.com/api/", "https://prefect.example.com/api/health"),
    ),
)
def test_prefect_health_url_supports_api_prefix(probe, api_url, health_url):
    assert probe._prefect_health_url(api_url) == health_url


def test_fail_fast_on_missing_command(probe, monkeypatch, capsys):
    monkeypatch.setattr(
        probe.shutil,
        "which",
        lambda name: None if name == "serviceman" else f"/usr/bin/{name}",
    )
    monkeypatch.setattr(probe, "run", lambda _cmd: 0)

    with pytest.raises(SystemExit):
        probe._probe()

    assert "required command not found: serviceman" in capsys.readouterr().err


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
