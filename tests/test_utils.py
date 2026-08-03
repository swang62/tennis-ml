import json
import os
import sys

from src.constants import REPO_NAME
from src.utils import ensure_kernel, load_env


def test_ensure_kernel_registers_repo_local_kernelspec(monkeypatch, tmp_path):
    kernel_dir = tmp_path / "kernels"
    monkeypatch.setattr("src.utils.KERNEL_DIR", kernel_dir)
    monkeypatch.setenv("JUPYTER_PATH", "/existing/path")

    name = ensure_kernel()

    assert name == REPO_NAME
    kernel_json = json.loads((kernel_dir / "kernel.json").read_text())
    assert kernel_json["argv"][0] == sys.executable
    assert "ipykernel_launcher" in kernel_json["argv"]
    repo_path = str(kernel_dir.parents[1])
    assert os.environ["JUPYTER_PATH"] == f"{repo_path}{os.pathsep}/existing/path"


def test_ensure_kernel_sets_jupyter_path_when_unset(monkeypatch, tmp_path):
    kernel_dir = tmp_path / "kernels"
    monkeypatch.setattr("src.utils.KERNEL_DIR", kernel_dir)
    monkeypatch.delenv("JUPYTER_PATH", raising=False)

    ensure_kernel()

    assert os.environ["JUPYTER_PATH"] == str(kernel_dir.parents[1])


def test_load_env_loads_dotenv_file(monkeypatch, tmp_path):
    monkeypatch.setattr("src.utils.ROOT", tmp_path)
    monkeypatch.delenv("FOO", raising=False)
    (tmp_path / ".env").write_text("FOO=bar\n")

    load_env()

    assert os.environ.get("FOO") == "bar"


def test_load_env_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.setattr("src.utils.ROOT", tmp_path)
    monkeypatch.delenv("FOO", raising=False)
    (tmp_path / ".env").write_text("FOO=bar\n")

    load_env()
    load_env()

    assert os.environ.get("FOO") == "bar"
