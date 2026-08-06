"""Offline tests for the host deployment path (src/flows/deploy.py).

These test command construction, secure credential handling, and force
behavior WITHOUT a Docker daemon, Docker Hub login/push, Docker Compose, or a
model deployment. build_bento_image / deploy_bento are exercised with their
heavy dependencies (MLflow, BentoML, Docker) mocked away.
"""

import importlib
import sys
from types import SimpleNamespace


def _deploy():
    return importlib.import_module("src.flows.deploy")


# --- Secure credential handling ---


def test_docker_login_passes_token_via_stdin_only(monkeypatch):
    d = _deploy()
    monkeypatch.setenv("DOCKER_TOKEN", "super-secret-token")
    monkeypatch.setenv("DOCKER_USERNAME", "hubuser")
    monkeypatch.setattr(d, "DOCKER_REPO", "acme")

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["input"] = kwargs.get("input")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)

    d._docker_login()

    assert "--password-stdin" in captured["cmd"]
    assert "hubuser" in captured["cmd"]
    # Token must travel only via stdin, never in argv.
    assert "super-secret-token" not in captured["cmd"]
    assert captured["input"] == "super-secret-token\n"


def test_docker_login_username_wins_over_repo_owner(monkeypatch):
    d = _deploy()
    monkeypatch.setenv("DOCKER_TOKEN", "tok")
    monkeypatch.setenv("DOCKER_USERNAME", "explicit-user")
    monkeypatch.setattr(d, "DOCKER_REPO", "acme")

    captured = {}

    def fake_run(cmd, **_kwargs):
        captured["cmd"] = list(cmd)
        return SimpleNamespace(returncode=0)


def test_docker_login_username_falls_back_to_repo_owner(monkeypatch):
    d = _deploy()
    monkeypatch.setenv("DOCKER_TOKEN", "tok")
    monkeypatch.delenv("DOCKER_USERNAME", raising=False)
    monkeypatch.setattr(d, "DOCKER_REPO", "acme")

    captured = {}

    def fake_run(cmd, **_kwargs):
        captured["cmd"] = list(cmd)
        return SimpleNamespace(returncode=0)


def test_docker_login_skips_when_token_unset(monkeypatch):
    d = _deploy()
    monkeypatch.delenv("DOCKER_TOKEN", raising=False)

    called = []

    def fake_run(*_args, **_kwargs):
        called.append(_args)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)

    d._docker_login()

    assert called == []


def test_docker_login_raises_on_failure(monkeypatch):
    d = _deploy()
    monkeypatch.setenv("DOCKER_TOKEN", "tok")
    monkeypatch.setattr("subprocess.run", lambda *_a, **_k: SimpleNamespace(returncode=1))

    import pytest

    with pytest.raises(RuntimeError):
        d._docker_login()


# --- Command construction (push latest + compose up) ---


def _stub_subprocess(monkeypatch):
    """Replace subprocess.run with a no-op returning success."""
    monkeypatch.setattr("subprocess.run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))


def test_deploy_bento_pushes_only_latest_and_composes_up(monkeypatch, tmp_path):
    d = _deploy()
    monkeypatch.setattr(d, "DOCKER_REPO", "acme")
    monkeypatch.setattr(d, "IMAGE_NAME", "tennis-ml")
    monkeypatch.setattr(d, "LOGS", tmp_path)
    monkeypatch.setattr(d, "build_bento_image", lambda **_kwargs: ("acme/tennis-ml:abc", 5))
    monkeypatch.setattr(d, "_docker_login", lambda: None)
    monkeypatch.setattr(d, "_read_state", lambda: {})
    monkeypatch.setattr(d, "_write_state", lambda _s: None)
    monkeypatch.setenv("QUACK_TOKEN", "super-secret-deploy-token")
    _stub_subprocess(monkeypatch)

    calls = []

    def fake_run_teed(cmd, _log, env=None):
        calls.append((list(cmd), dict(env or {})))
        return None

    monkeypatch.setattr(d, "_run_teed", fake_run_teed)

    d.deploy_bento(force=False)

    pushes = [c for c in calls if c[0][:2] == ["docker", "push"]]
    assert len(pushes) == 1
    assert pushes[0][0] == ["docker", "push", "acme/tennis-ml:latest"]

    composes = [c for c in calls if c[0][:2] == ["docker", "compose"]]
    # Two compose calls, ordered: build web, then up.
    assert len(composes) == 2
    build_cmd, _ = composes[0]
    assert build_cmd[-1] == "web"
    assert "build" in build_cmd
    assert "--no-cache" not in build_cmd  # only forced rebuilds skip the cache
    assert "-f" in build_cmd
    up_cmd, up_env = composes[1]
    assert up_cmd[-4:] == ["up", "-d", "--pull", "always"]
    assert "-f" in up_cmd
    assert up_env["DOCKER_IMAGE"] == "acme/tennis-ml"


def test_deploy_bento_force_builds_web_no_cache(monkeypatch, tmp_path):
    d = _deploy()
    monkeypatch.setattr(d, "DOCKER_REPO", "acme")
    monkeypatch.setattr(d, "IMAGE_NAME", "tennis-ml")
    monkeypatch.setattr(d, "LOGS", tmp_path)
    monkeypatch.setattr(d, "build_bento_image", lambda **_kwargs: ("acme/tennis-ml:abc", 5))
    monkeypatch.setattr(d, "_docker_login", lambda: None)
    monkeypatch.setattr(d, "_read_state", lambda: {})
    monkeypatch.setattr(d, "_write_state", lambda _s: None)
    monkeypatch.setenv("QUACK_TOKEN", "super-secret-deploy-token")
    _stub_subprocess(monkeypatch)

    calls = []

    def fake_run_teed(cmd, _log, env=None):
        calls.append((list(cmd), dict(env or {})))
        return None

    monkeypatch.setattr(d, "_run_teed", fake_run_teed)

    d.deploy_bento(force=True)

    composes = [c for c in calls if c[0][:2] == ["docker", "compose"]]
    assert len(composes) == 2
    build_cmd, _ = composes[0]
    assert build_cmd[-1] == "web"
    assert build_cmd[build_cmd.index("build") + 1] == "--no-cache"


def test_deploy_bento_requires_quack_token(monkeypatch, tmp_path):
    """Production deploy must fail fast when QUACK_TOKEN is unset — never
    boot Compose without the Quack secret."""
    import pytest

    d = _deploy()
    monkeypatch.setattr(d, "DOCKER_REPO", "acme")
    monkeypatch.setattr(d, "IMAGE_NAME", "tennis-ml")
    monkeypatch.setattr(d, "LOGS", tmp_path)
    monkeypatch.setattr(d, "build_bento_image", lambda **_kwargs: ("acme/tennis-ml:abc", 5))
    monkeypatch.setattr(d, "_docker_login", lambda: None)
    monkeypatch.setattr(d, "_read_state", lambda: {})
    monkeypatch.setattr(d, "_write_state", lambda _s: None)
    monkeypatch.delenv("QUACK_TOKEN", raising=False)
    _stub_subprocess(monkeypatch)

    with pytest.raises(RuntimeError, match="QUACK_TOKEN is required"):
        d.deploy_bento(force=False)


# --- Force behavior ---


def _stub_bento_build(monkeypatch):
    """Patch the MLflow/BentoML/Docker machinery behind build_bento_image."""
    d = _deploy()
    from pathlib import Path as _Path

    monkeypatch.setattr(d, "IMAGE_NAME", "tennis-ml")
    monkeypatch.setattr(d, "BENTO_TAG_FILE", _Path("/tmp/bento_tag.txt"))
    monkeypatch.setattr(d, "STATE_FILE", _Path("/tmp/state.json"))
    monkeypatch.setattr(
        d,
        "_latest_production_version",
        lambda _client: SimpleNamespace(version="7", run_id="r"),
    )
    monkeypatch.setattr(
        d,
        "_resolve_pins",
        lambda _client, _prod: {k: {} for k in ("nn", "linear", "gbdt", "production")},
    )
    monkeypatch.setattr(d, "_materialize_nn_onnx", lambda _nn: None)
    monkeypatch.setattr(d, "_check_aux_files", lambda: None)
    monkeypatch.setattr(d, "_state_fingerprint", lambda _v: "fp")
    monkeypatch.setattr(d, "_read_state", lambda: {"fingerprint": "fp"})
    # A cache HIT: _cached_tag would return a tag when not forced.
    monkeypatch.setattr(d, "_cached_tag", lambda _state, _fp: "bento:abc")
    monkeypatch.setattr(
        d, "_import_models", lambda _pins: {"linear": "l", "gbdt": "g", "production": "p"}
    )
    monkeypatch.setattr(d, "_write_pinned_bentofile", lambda _tags: "pinned")
    monkeypatch.setattr(d, "_write_state", lambda _s: None)
    monkeypatch.setattr(d, "_image_exists", lambda _image: True)
    _stub_subprocess(monkeypatch)

    built = {"bento": 0, "container": 0}
    bentoml = {
        "bentoml": SimpleNamespace(
            bentos=SimpleNamespace(get=lambda _tag: (_ for _ in ()).throw(Exception())),
            container=SimpleNamespace(),
        )
    }
    monkeypatch.setitem(sys.modules, "bentoml", bentoml["bentoml"])
    monkeypatch.setitem(
        sys.modules,
        "mlflow.tracking.client",
        SimpleNamespace(MlflowClient=lambda: None),
    )

    def build_bentofile(**_kwargs):
        built["bento"] += 1
        return SimpleNamespace(tag="bento:abc")

    def container_build(*_args, **_kwargs):
        built["container"] += 1

    bentoml["bentoml"].bentos.build_bentofile = build_bentofile
    bentoml["bentoml"].container.build = container_build
    return d, built


def test_build_force_rebuilds_over_cache_hit(monkeypatch):
    d, built = _stub_bento_build(monkeypatch)

    # force=True: rebuild the Bento and containerize even though the cache hits.
    d.build_bento_image(force=True)
    assert built["bento"] == 1
    assert built["container"] == 1


def test_build_no_force_reuses_cache_hit(monkeypatch):
    d, built = _stub_bento_build(monkeypatch)

    # force=False: cache hit -> no rebuild, and the local image exists -> no
    # containerization.
    d.build_bento_image(force=False)
    assert built["bento"] == 0
    assert built["container"] == 0


def test_deploy_flow_forwards_force(monkeypatch):
    d = _deploy()
    seen = {}

    def fake_deploy_bento(force=False):
        seen["force"] = force

    monkeypatch.setattr(d, "deploy_bento", fake_deploy_bento)

    d.deploy_flow(force=True)
    assert seen["force"] is True
    d.deploy_flow(force=False)
    assert seen["force"] is False
