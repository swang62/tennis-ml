import json
import os
import sys
import warnings

import urllib3

from src.config import ensure_kernel, suppress_insecure_tls_warning
from src.constants import load_env


def test_ensure_kernel_registers_repo_local_kernelspec(monkeypatch, tmp_path):
    kernel_dir = tmp_path / "kernels"
    monkeypatch.setattr("src.config.KERNEL_DIR", kernel_dir)
    monkeypatch.setenv("JUPYTER_PATH", "/existing/path")
    monkeypatch.setattr("src.config.IMAGE_NAME", "test-image")

    name = ensure_kernel()

    assert name == "test-image"
    kernel_json = json.loads((kernel_dir / "kernel.json").read_text())
    assert kernel_json["argv"][0] == sys.executable
    assert "ipykernel_launcher" in kernel_json["argv"]
    repo_path = str(kernel_dir.parents[1])
    assert os.environ["JUPYTER_PATH"] == f"{repo_path}{os.pathsep}/existing/path"


def test_ensure_kernel_sets_jupyter_path_when_unset(monkeypatch, tmp_path):
    kernel_dir = tmp_path / "kernels"
    monkeypatch.setattr("src.config.KERNEL_DIR", kernel_dir)
    monkeypatch.delenv("JUPYTER_PATH", raising=False)
    monkeypatch.setattr("src.config.IMAGE_NAME", "test-image")

    ensure_kernel()

    assert os.environ["JUPYTER_PATH"] == str(kernel_dir.parents[1])


def test_load_env_loads_dotenv_file(monkeypatch, tmp_path):
    monkeypatch.setattr("src.constants.ROOT", tmp_path)
    monkeypatch.delenv("FOO", raising=False)
    (tmp_path / ".env").write_text("FOO=bar\n")

    load_env()

    assert os.environ.get("FOO") == "bar"


def test_load_env_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.setattr("src.constants.ROOT", tmp_path)
    monkeypatch.delenv("FOO", raising=False)
    (tmp_path / ".env").write_text("FOO=bar\n")

    load_env()
    load_env()

    assert os.environ.get("FOO") == "bar"


def _has_insecure_ignore() -> bool:
    return any(
        action == "ignore" and category is urllib3.exceptions.InsecureRequestWarning
        for action, _message, category, _module, _lineno in warnings.filters
    )


def test_suppress_insecure_tls_warning_requires_opted_in_env(monkeypatch):
    """No insecure-TLS env setting -> no urllib3 warning is suppressed."""
    monkeypatch.delenv("MLFLOW_TRACKING_INSECURE_TLS", raising=False)
    monkeypatch.delenv("PREFECT_API_TLS_INSECURE_SKIP_VERIFY", raising=False)

    with warnings.catch_warnings():
        suppress_insecure_tls_warning()
        assert not _has_insecure_ignore()


def test_suppress_insecure_tls_warning_opt_in_via_mlflow_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_INSECURE_TLS", "true")

    with warnings.catch_warnings():
        suppress_insecure_tls_warning()
        assert _has_insecure_ignore()


def test_suppress_insecure_tls_warning_opt_in_via_prefect_env(monkeypatch):
    monkeypatch.setenv("PREFECT_API_TLS_INSECURE_SKIP_VERIFY", "True")

    with warnings.catch_warnings():
        suppress_insecure_tls_warning()
        assert _has_insecure_ignore()
