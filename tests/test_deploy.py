"""Offline deployment tests with mocked MLflow, BentoML, Docker, and Compose."""

import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


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

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["input"] = kwargs.get("input")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)

    d._docker_login()

    assert "--password-stdin" in captured["cmd"]
    assert "explicit-user" in captured["cmd"]
    assert "acme" not in captured["cmd"]  # repo owner is NOT the username here
    assert captured["input"] == "tok\n"


def test_docker_login_username_falls_back_to_repo_owner(monkeypatch):
    d = _deploy()
    monkeypatch.setenv("DOCKER_TOKEN", "tok")
    monkeypatch.delenv("DOCKER_USERNAME", raising=False)
    monkeypatch.setattr(d, "DOCKER_REPO", "acme")

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["input"] = kwargs.get("input")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)

    d._docker_login()

    assert "--password-stdin" in captured["cmd"]
    assert "acme" in captured["cmd"]  # DOCKER_REPO owner is the fallback username
    assert captured["input"] == "tok\n"


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


# --- Command construction (Buildx multi-arch push; no Compose, no web build) ---


def _stub_subprocess(monkeypatch):
    """Replace subprocess.run with a no-op returning success."""
    monkeypatch.setattr("subprocess.run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))


class _FakeMlflowRun:
    """Context manager standing in for mlflow.start_run's returned run."""

    info = SimpleNamespace(run_id="deploy-run-id")

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _stub_deploy_mlflow_tracking(monkeypatch):
    """Stub deploy_bento's post-build MLflow tracking block.

    After the Buildx push, deploy_bento records the deployment in MLflow
    (experiment + run + the two pipeline_last_deploy_* version tags). Without
    this stub the tracking calls reach the DagsHub tracking server from tests.
    """
    fake_mlflow = SimpleNamespace(
        set_experiment=lambda _name: None,
        set_experiment_tag=lambda _key, _value: None,
        start_run=lambda **_kwargs: _FakeMlflowRun(),
        log_param=lambda *_args: None,
        log_artifact=lambda *_args: None,
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setitem(
        sys.modules,
        "mlflow.tracking.client",
        SimpleNamespace(
            MlflowClient=lambda: SimpleNamespace(set_model_version_tag=lambda *_a: None)
        ),
    )


def test_deploy_bento_logs_in_before_build(monkeypatch, tmp_path):
    d = _deploy()
    monkeypatch.setattr(d, "DOCKER_REPO", "acme")
    monkeypatch.setattr(d, "IMAGE_NAME", "tennis-bento")
    monkeypatch.setattr(d, "LOGS", tmp_path)
    monkeypatch.setattr(d, "_read_state", lambda: {})
    monkeypatch.setattr(d, "_write_state", lambda _s: None)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    order = []
    monkeypatch.setattr(d, "_docker_login", lambda: order.append("login"))
    monkeypatch.setattr(
        d,
        "build_bento_image",
        lambda: order.append("build") or ("acme/tennis-bento:dev", 5),
    )
    _stub_deploy_mlflow_tracking(monkeypatch)

    d.deploy_bento()

    assert order == ["login", "build"]


def test_deploy_bento_logs_in_before_build_then_writes_state(monkeypatch, tmp_path):
    """Login happens before the build (whose Buildx `--push` is the push); deploy
    itself runs no docker tag/push and only ever targets the Docker Hub DOCKER_TAG image."""
    d = _deploy()
    monkeypatch.setattr(d, "DOCKER_REPO", "acme")
    monkeypatch.setattr(d, "DOCKER_TAG", "dev")
    monkeypatch.setattr(d, "IMAGE_NAME", "tennis-bento")
    monkeypatch.setattr(d, "LOGS", tmp_path)
    monkeypatch.setattr(d, "_read_state", lambda: {})
    written = {}
    monkeypatch.setattr(d, "_write_state", lambda s: written.update(s))
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    order = []
    docker_calls = []

    def fake_run(cmd, **_kwargs):
        docker_calls.append(list(cmd))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(d, "generate_navigation_artifacts", lambda: None)
    monkeypatch.setattr(d, "_docker_login", lambda: order.append("login"))
    monkeypatch.setattr(
        d,
        "build_bento_image",
        lambda: order.append("build") or ("acme/tennis-bento:dev", 5),
    )
    _stub_deploy_mlflow_tracking(monkeypatch)

    d.deploy_bento()

    # Login precedes the build because `--push` is part of the Buildx build.
    assert order == ["login", "build"]
    # Deploy itself performs no separate docker tag/push — Buildx pushed already.
    assert docker_calls == []
    assert written["deployed_version"] == 5
    assert written["deployed_image"] == "acme/tennis-bento:dev"


def test_deploy_bento_does_not_require_postgres_password(monkeypatch, tmp_path):
    """Deploy does not require a PostgreSQL credential."""
    d = _deploy()
    monkeypatch.setattr(d, "DOCKER_REPO", "acme")
    monkeypatch.setattr(d, "IMAGE_NAME", "tennis-bento")
    monkeypatch.setattr(d, "LOGS", tmp_path)
    monkeypatch.setattr(d, "build_bento_image", lambda **_kwargs: ("acme/tennis-bento:dev", 5))
    monkeypatch.setattr(d, "generate_navigation_artifacts", lambda: None)
    monkeypatch.setattr(d, "_docker_login", lambda: None)
    monkeypatch.setattr(d, "_read_state", lambda: {})
    monkeypatch.setattr(d, "_write_state", lambda _s: None)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    teed = []
    monkeypatch.setattr(d, "_run_teed", lambda cmd, _log=None: teed.append(list(cmd)))
    _stub_subprocess(monkeypatch)
    _stub_deploy_mlflow_tracking(monkeypatch)

    d.deploy_bento()  # must not raise

    # Nothing is pushed by deploy directly; Buildx pushed inside the build.
    assert teed == []


def test_deploy_bento_fails_when_buildx_push_fails(monkeypatch, tmp_path):
    d = _deploy()
    monkeypatch.setattr(d, "LOGS", tmp_path)
    monkeypatch.setattr(d, "generate_navigation_artifacts", lambda: None)
    monkeypatch.setattr(d, "_docker_login", lambda: None)

    def fail_build():
        raise subprocess.CalledProcessError(1, ["docker", "buildx", "build"])

    monkeypatch.setattr(d, "build_bento_image", fail_build)

    import pytest

    with pytest.raises(subprocess.CalledProcessError):
        d.deploy_bento()


# --- Whole deploy tees build + Buildx output into one deploy_<timestamp>.log ---


def test_deploy_bento_logs_build_and_buildx_to_single_file(monkeypatch, tmp_path):
    """Build prints and Buildx output go into one deploy_*.log that exists
    before the build starts, while the console still sees them."""
    d = _deploy()
    monkeypatch.setattr(d, "LOGS", tmp_path)
    monkeypatch.setattr(d, "DOCKER_REPO", "acme")
    monkeypatch.setattr(d, "IMAGE_NAME", "tennis-bento")
    monkeypatch.setattr(d, "generate_navigation_artifacts", lambda: None)
    monkeypatch.setattr(d, "_docker_login", lambda: None)
    monkeypatch.setattr(d, "_read_state", lambda: {})
    monkeypatch.setattr(d, "_write_state", lambda _s: None)
    _stub_subprocess(monkeypatch)

    class _FakeProc:
        stdout = iter([b"#90 exporting manifest list 0.3s\n"])

        def wait(self):
            return 0

    monkeypatch.setattr("subprocess.Popen", lambda *_a, **_k: _FakeProc())

    build_output = {}

    def fake_build():
        build_output["log_at_build_start"] = sorted(p.name for p in tmp_path.glob("deploy_*.log"))
        print("building bento image")
        # The Buildx build streams its output through _run_teed into the open log.
        d._run_teed(["docker", "buildx", "build", "--push"])
        return "acme/tennis-bento:dev", 5

    monkeypatch.setattr(d, "build_bento_image", fake_build)
    _stub_deploy_mlflow_tracking(monkeypatch)

    d.deploy_bento()

    logs = sorted(tmp_path.glob("deploy_*.log"))
    assert len(logs) == 1
    # The log file already existed when the build started.
    assert build_output["log_at_build_start"] == [logs[0].name]
    content = logs[0].read_text()
    assert "building bento image" in content
    assert "exporting manifest list" in content


def test_deploy_bento_leaves_log_when_build_fails(monkeypatch, tmp_path):
    """A raising build still leaves a non-empty deploy_*.log with its output."""
    d = _deploy()
    monkeypatch.setattr(d, "LOGS", tmp_path)
    monkeypatch.setattr(d, "generate_navigation_artifacts", lambda: None)

    def fail_build():
        print("champion missing lineage tags")
        raise RuntimeError("no champion to build")

    monkeypatch.setattr(d, "build_bento_image", fail_build)

    import pytest

    with pytest.raises(RuntimeError, match="no champion"):
        d.deploy_bento()

    (log,) = tmp_path.glob("deploy_*.log")
    assert "champion missing lineage tags" in log.read_text()


# --- Force behavior (removed — deployments always rebuild) ---


def test_reuse_or_materialize_nn_onnx_uses_exact_model_uri(monkeypatch, tmp_path):
    d = _deploy()
    onnx_file = tmp_path / "nn_best.onnx"
    onnx_file.write_bytes(b"onnx")
    monkeypatch.setattr(d, "NN_ONNX_FILE", onnx_file)
    materialized = []
    monkeypatch.setattr(d, "_materialize_nn_onnx", lambda pin: materialized.append(pin))
    pin = {"model_uri": "models:/nn_best/7"}

    assert d._reuse_or_materialize_nn_onnx({"nn_onnx_model_uri": pin["model_uri"]}, pin)
    assert materialized == []

    assert not d._reuse_or_materialize_nn_onnx({"nn_onnx_model_uri": "models:/nn_best/6"}, pin)
    assert materialized == [pin]


class _FakeContext:
    """Context manager standing in for _buildx_context's temp directory."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return self.path

    def __exit__(self, *_exc):
        return False


def _stub_bento_build(monkeypatch):
    """Patch the MLflow/BentoML/Docker machinery behind build_bento_image."""
    d = _deploy()
    from pathlib import Path as _Path

    monkeypatch.setattr(d, "IMAGE_NAME", "tennis-bento")
    monkeypatch.setattr(d, "DOCKER_REPO", "acme")
    monkeypatch.setattr(d, "DOCKER_TAG", "dev")
    monkeypatch.setattr(d, "BENTO_TAG_FILE", _Path("/tmp/bento_tag.txt"))
    monkeypatch.setattr(d, "STATE_FILE", _Path("/tmp/state.json"))
    monkeypatch.setattr(d, "MODEL_INFO_FILE", _Path("/tmp/model_info.json"))
    monkeypatch.setattr(d, "BENTO_CONTAINERFILE", _Path("/tmp/Containerfile.bento"))
    monkeypatch.setattr(
        d,
        "_latest_production_version",
        lambda _client: SimpleNamespace(version="7", run_id="r"),
    )
    monkeypatch.setattr(
        d,
        "_lineage_pins",
        lambda _client, _prod: {
            k: {"model_uri": f"models:/{k}/7"} for k in ("nn", "linear", "gbdt", "production")
        },
    )
    monkeypatch.setattr(d, "_reuse_or_materialize_nn_onnx", lambda _state, _nn: False)
    monkeypatch.setattr(d, "_download_aux_artifacts", lambda _client, _tags: None)
    monkeypatch.setattr(d, "generate_navigation_artifacts", lambda: None)
    monkeypatch.setattr(d, "build_input_fingerprint", lambda _client, _prod: "fp")
    monkeypatch.setattr(d, "_read_state", lambda: {"fingerprint": "fp"})
    monkeypatch.setattr(
        d, "_import_models", lambda _pins: {"linear": "l", "gbdt": "g", "production": "p"}
    )
    monkeypatch.setattr(d, "_write_pinned_bentofile", lambda _tags: "pinned")
    monkeypatch.setattr(d, "_write_state", lambda _s: None)
    monkeypatch.setattr(d, "_ensure_buildx_builder", lambda: "tennis-multiarch")
    monkeypatch.setattr(
        d, "_write_bento_containerfile", lambda _bento: _Path("/tmp/Containerfile.bento")
    )
    monkeypatch.setattr(d, "_buildx_context", lambda _bento: _FakeContext("/ctx"))
    docker_calls = []
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **_k: (
            (docker_calls.append(list(a[0])) if a else None) or SimpleNamespace(returncode=0)
        ),
    )
    teed_calls = []
    monkeypatch.setattr(d, "_run_teed", lambda cmd, _log=None: teed_calls.append(list(cmd)))

    built: dict[str, Any] = {
        "bento": 0,
        "teed_calls": teed_calls,
        "docker_calls": docker_calls,
    }
    bentoml = {
        "bentoml": SimpleNamespace(
            bentos=SimpleNamespace(get=lambda _tag: (_ for _ in ()).throw(Exception()))
        )
    }
    monkeypatch.setitem(sys.modules, "bentoml", bentoml["bentoml"])
    # The canonical-manifest writer (_write_model_info) is the only real client
    # use left unstubbed; the fake serves the champion version's lineage tags
    # and creation timestamp so the cache/force paths run without MLflow.
    monkeypatch.setitem(
        sys.modules,
        "mlflow.tracking.client",
        SimpleNamespace(
            MlflowClient=lambda: SimpleNamespace(
                get_model_version=lambda _name, _version: SimpleNamespace(
                    tags=dict(_lineage_tags()), creation_timestamp=123
                )
            )
        ),
    )

    def build_bentofile(**_kwargs):
        built["bento"] += 1
        return SimpleNamespace(tag="bento:abc")

    bentoml["bentoml"].bentos.build_bentofile = build_bentofile
    return d, built


def test_build_bento_image_publishes_multiarch_via_buildx(monkeypatch):
    """build_bento_image runs one `docker buildx build` that targets the remote
    DOCKER_TAG image, both platforms, the generated Containerfile, and the Bento
    build context — with `--push` as part of the build."""
    d, built = _stub_bento_build(monkeypatch)

    image, version = d.build_bento_image()
    assert built["bento"] == 1
    assert image == "acme/tennis-bento:dev"
    assert version == 7
    (cmd,) = built["teed_calls"]
    assert cmd[:3] == ["docker", "buildx", "build"]
    assert cmd[cmd.index("--builder") + 1] == "tennis-multiarch"
    assert cmd[cmd.index("--file") + 1] == "/tmp/Containerfile.bento"
    assert cmd[cmd.index("--platform") + 1] == "linux/amd64,linux/arm64"
    assert cmd[cmd.index("--tag") + 1] == "acme/tennis-bento:dev"
    assert "--push" in cmd
    # The build context that actually contains the Bento is the last argument.
    assert cmd[-1] == "/ctx"
    # No separate docker tag/push anywhere: the only docker subprocess is Buildx.
    assert built["docker_calls"] == []


def test_build_bento_image_no_separate_docker_push(monkeypatch):
    """Every deploy rebuilds the Bento and publishes it with the single Buildx
    command — the old docker tag + docker push route is gone."""
    d, built = _stub_bento_build(monkeypatch)

    d.build_bento_image()
    assert built["bento"] == 1
    assert len(built["teed_calls"]) == 1
    cmd = built["teed_calls"][0]
    assert cmd[:3] == ["docker", "buildx", "build"]
    assert "--push" in cmd


# --- Buildx command and builder setup ---


def test_buildx_build_cmd_exact():
    """The generated command is exactly the Buildx multi-platform push."""
    d = _deploy()
    cmd = d._buildx_build_cmd(
        builder="b1",
        containerfile=Path("/cf"),
        context=Path("/ctx"),
        image="acme/tennis-bento:latest",
    )
    assert cmd == [
        "docker",
        "buildx",
        "build",
        "--builder",
        "b1",
        "--file",
        "/cf",
        "--platform",
        "linux/amd64,linux/arm64",
        "--tag",
        "acme/tennis-bento:latest",
        "--push",
        "/ctx",
    ]


def test_ensure_buildx_builder_reuses_existing(monkeypatch):
    d = _deploy()
    calls = []

    def fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)

    assert d._ensure_buildx_builder() == "tennis-multiarch"
    # Inspect only — the named builder already exists, so it is reused.
    assert calls == [["docker", "buildx", "inspect", "tennis-multiarch"]]


def test_ensure_buildx_builder_creates_docker_container_when_missing(monkeypatch):
    d = _deploy()
    calls = []

    def fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        # First call (inspect) fails; the create that follows succeeds.
        return SimpleNamespace(returncode=1 if len(calls) == 1 else 0)

    monkeypatch.setattr("subprocess.run", fake_run)

    assert d._ensure_buildx_builder() == "tennis-multiarch"
    assert calls == [
        ["docker", "buildx", "inspect", "tennis-multiarch"],
        [
            "docker",
            "buildx",
            "create",
            "--name",
            "tennis-multiarch",
            "--driver",
            "docker-container",
        ],
    ]


def test_write_bento_containerfile_uses_bentoml_generator(monkeypatch, tmp_path):
    """The Containerfile comes from BentoML's get_containerfile, written to the
    deploy artifacts path for the current Bento tag."""
    d = _deploy()
    containerfile = tmp_path / "Containerfile.bento"
    monkeypatch.setattr(d, "BENTO_CONTAINERFILE", containerfile)
    calls = {}
    fake_bentoml = SimpleNamespace(
        container=SimpleNamespace(
            get_containerfile=lambda tag, **kwargs: calls.update(tag=str(tag), **kwargs) or None
        )
    )
    monkeypatch.setitem(sys.modules, "bentoml", fake_bentoml)

    assert d._write_bento_containerfile(SimpleNamespace(tag="bento:abc")) == containerfile
    assert calls["tag"] == "bento:abc"
    assert calls["output_path"] == str(containerfile)


def test_buildx_context_copies_bento_and_materializes_models(monkeypatch, tmp_path):
    """The build context contains the whole Bento: its files plus the model
    files resolved from the model store (bento.path has no models/ itself)."""
    d = _deploy()
    bento_root = tmp_path / "bento"
    (bento_root / "env" / "python").mkdir(parents=True)
    (bento_root / "env" / "python" / "requirements.txt").write_text("x")
    store_root = tmp_path / "store"
    model_root = store_root / "m1" / "v1"
    model_root.mkdir(parents=True)
    (model_root / "model.pkl").write_bytes(b"data")
    stored = SimpleNamespace(tag=SimpleNamespace(name="m1", version="v1"), path=model_root)
    bento = SimpleNamespace(
        path=bento_root,
        info=SimpleNamespace(all_models=[SimpleNamespace(tag="m1:v1")]),
    )
    fake_bentoml = SimpleNamespace(models=SimpleNamespace(get=lambda _tag: stored))
    monkeypatch.setitem(sys.modules, "bentoml", fake_bentoml)

    with d._buildx_context(bento) as context:
        assert (context / "env" / "python" / "requirements.txt").read_text() == "x"
        assert (context / "models" / "m1" / "v1" / "model.pkl").read_bytes() == b"data"
        context_path = context

    # The temporary context is cleaned up after the build.
    assert not context_path.exists()


# --- Every deploy packages the on-disk artifacts into the build context ---


def test_pinned_bentofile_preserves_packaged_artifact_includes(monkeypatch, tmp_path):
    """The pinned bentofile keeps every packaged artifact in `include`, so each
    deploy rebuilds the Bento from the current data/processed files on disk."""
    d = _deploy()
    pinned = tmp_path / "bentofile.pinned.yaml"
    monkeypatch.setattr(d, "PINNED_BENTOFILE", pinned)

    d._write_pinned_bentofile(
        {"linear": "linear_best:v3", "gbdt": "gbdt_best:v2", "production": "ensemble_lr_model:v7"}
    )

    import yaml

    config = yaml.safe_load(pinned.read_text())
    # nn_best is served via the ONNX artifact, not as a BentoModel dep.
    assert config["models"] == ["linear_best:v3", "gbdt_best:v2", "ensemble_lr_model:v7"]
    for artifact in [*d.AUX_FILES, d.MODEL_INFO_FILE]:
        assert artifact.relative_to(d.ROOT).as_posix() in config["include"]


def _aux_tags_and_files(monkeypatch, tmp_path, d, pre_populate=True):
    """Full URI/hash lineage tag set whose hashes match real source files, plus
    a stubbed mlflow.artifacts.download_artifacts that copies the matching
    source into the staging dir. Returns (specs, tags, downloaded, artifacts_dir)."""
    src_dir = tmp_path / "remote"
    src_dir.mkdir()
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    specs = [
        ("base_linear_scaler_uri", "base_linear_scaler_hash", "linear_scaler.pkl"),
        ("aux_embeddings_uri", "aux_embeddings_hash", "bio_embeddings.npz"),
        ("aux_bio_feature_cols_uri", "aux_bio_feature_cols_hash", "bio_feature_cols.json"),
    ]
    tags = {}
    for uri_tag, hash_tag, name in specs:
        src = src_dir / name
        src.write_bytes(f"content-{name}".encode())
        tags[uri_tag] = f"runs:/run-aux/{name}"
        tags[hash_tag] = d._file_hash(src)
    if pre_populate:
        for _, _, name in specs:
            (artifacts_dir / name).write_bytes((src_dir / name).read_bytes())

    downloaded = []

    def fake_download(uri, dst_path=""):
        downloaded.append(uri)
        dest = Path(dst_path) / uri.rsplit("/", 1)[-1]
        dest.write_bytes((src_dir / uri.rsplit("/", 1)[-1]).read_bytes())
        return str(dest)

    monkeypatch.setitem(
        sys.modules,
        "mlflow",
        SimpleNamespace(artifacts=SimpleNamespace(download_artifacts=fake_download)),
    )
    return specs, tags, downloaded, artifacts_dir


def test_download_aux_artifacts_reuses_matching_local_files(monkeypatch, tmp_path):
    """An existing local artifact whose hash matches the pin is reused: no download."""
    d = _deploy()
    _specs, tags, downloaded, artifacts_dir = _aux_tags_and_files(
        monkeypatch, tmp_path, d, pre_populate=True
    )
    monkeypatch.setattr(d, "DEPLOY_ARTIFACTS", artifacts_dir)

    temp = d._download_aux_artifacts(None, tags)

    assert downloaded == []  # all model artifacts reused
    assert temp == 1.0  # no calibration tags -> no-op temperature
    assert sorted(p.name for p in artifacts_dir.iterdir()) == sorted(name for _, _, name in _specs)


def test_download_aux_artifacts_requires_no_navigation_tags(monkeypatch, tmp_path):
    """Navigation artifacts are rebuilt from the snapshot, never downloaded."""
    d = _deploy()
    _specs, tags, _downloaded, artifacts_dir = _aux_tags_and_files(
        monkeypatch, tmp_path, d, pre_populate=True
    )
    monkeypatch.setattr(d, "DEPLOY_ARTIFACTS", artifacts_dir)

    assert not any("similarity" in tag or "directory" in tag for tag in tags)
    d._download_aux_artifacts(None, tags)
    assert not (artifacts_dir / "player_directory.json").exists()


def test_download_aux_artifacts_downloads_missing_files(monkeypatch, tmp_path):
    """Missing artifacts are downloaded once each and verified against the pin."""
    d = _deploy()
    specs, tags, downloaded, artifacts_dir = _aux_tags_and_files(
        monkeypatch, tmp_path, d, pre_populate=False
    )
    monkeypatch.setattr(d, "DEPLOY_ARTIFACTS", artifacts_dir)

    d._download_aux_artifacts(None, tags)

    assert sorted(downloaded) == sorted(tags[uri_tag] for uri_tag, _, _ in specs)
    for _, _, name in specs:
        assert (artifacts_dir / name).exists()


def test_download_aux_artifacts_missing_tag_fails(monkeypatch, tmp_path):
    """A lineage tag missing from the champion is a hard failure: the artifact
    is never silently skipped."""
    d = _deploy()
    _specs, tags, _downloaded, artifacts_dir = _aux_tags_and_files(monkeypatch, tmp_path, d)
    monkeypatch.setattr(d, "DEPLOY_ARTIFACTS", artifacts_dir)
    del tags["aux_embeddings_uri"]

    import pytest

    with pytest.raises(RuntimeError, match="aux_embeddings_uri"):
        d._download_aux_artifacts(None, tags)


def test_download_aux_artifacts_hash_mismatch_redownloads_then_fails(monkeypatch, tmp_path):
    """A local artifact with a stale hash is re-downloaded; a re-download whose
    content still mismatches the pin is rejected before the build proceeds."""
    d = _deploy()
    _specs, tags, downloaded, artifacts_dir = _aux_tags_and_files(
        monkeypatch, tmp_path, d, pre_populate=True
    )
    monkeypatch.setattr(d, "DEPLOY_ARTIFACTS", artifacts_dir)
    tags["base_linear_scaler_hash"] = "0" * 64  # wrong pin

    import pytest

    with pytest.raises(RuntimeError, match=r"linear_scaler.pkl"):
        d._download_aux_artifacts(None, tags)

    # The stale local copy was NOT reused; a fresh download was attempted.
    assert downloaded == [tags["base_linear_scaler_uri"]]


def test_download_aux_artifacts_retries_once_on_transient_failure(monkeypatch, tmp_path):
    """A failing download is retried once; the file is present and verified after."""
    d = _deploy()
    _specs, tags, _downloaded, artifacts_dir = _aux_tags_and_files(
        monkeypatch, tmp_path, d, pre_populate=True
    )
    monkeypatch.setattr(d, "DEPLOY_ARTIFACTS", artifacts_dir)
    name = "bio_embeddings.npz"
    (artifacts_dir / name).unlink()  # missing -> needs a download
    attempts = []

    def flaky_download(uri, dst_path=""):
        attempts.append(uri)
        if len(attempts) == 1:
            raise ConnectionError("transient fetch failure")
        dest = Path(dst_path) / name
        dest.write_bytes((tmp_path / "remote" / name).read_bytes())
        return str(dest)

    monkeypatch.setitem(
        sys.modules,
        "mlflow",
        SimpleNamespace(artifacts=SimpleNamespace(download_artifacts=flaky_download)),
    )

    d._download_aux_artifacts(None, tags)

    assert attempts == [tags["aux_embeddings_uri"]] * 2
    assert (artifacts_dir / name).exists()


def test_download_aux_artifacts_raises_after_retry_fails(monkeypatch, tmp_path):
    """Two failed downloads re-raise with a clear error naming the file and URI."""
    d = _deploy()
    _specs, tags, _downloaded, artifacts_dir = _aux_tags_and_files(
        monkeypatch, tmp_path, d, pre_populate=True
    )
    monkeypatch.setattr(d, "DEPLOY_ARTIFACTS", artifacts_dir)
    name = "bio_embeddings.npz"
    (artifacts_dir / name).unlink()  # missing -> needs a download
    calls = []

    def always_fail(uri, dst_path=""):  # noqa: ARG001 — signature matches download_artifacts
        calls.append(uri)
        raise ConnectionError("network down")

    monkeypatch.setitem(
        sys.modules,
        "mlflow",
        SimpleNamespace(artifacts=SimpleNamespace(download_artifacts=always_fail)),
    )

    import pytest

    with pytest.raises(RuntimeError, match=r"bio_embeddings\.npz"):
        d._download_aux_artifacts(None, tags)
    assert len(calls) == 2  # initial attempt + one retry, then no more


# --- Calibration artifact materialization (CALIBRATION_URI_TAG/HASH_TAG) ---


def _calibration_source(tmp_path, temperature=1.7):
    """Write a valid calibration file into the fake remote dir and return it."""
    import json

    import src.constants as c

    src = tmp_path / "remote" / c.CALIBRATION_ARTIFACT
    src.write_text(json.dumps({"temperature": temperature}) + "\n")
    return src


def _pin_calibration(tmp_path, tags, temperature=1.7):
    """Add both calibration lineage tags pinning a real source file."""
    import src.constants as c

    src = _calibration_source(tmp_path, temperature)
    tags[c.CALIBRATION_URI_TAG] = f"runs:/run-aux/{c.CALIBRATION_ARTIFACT}"
    tags[c.CALIBRATION_HASH_TAG] = _deploy()._file_hash(src)
    return src


def test_download_aux_artifacts_legacy_champion_writes_noop_calibration(monkeypatch, tmp_path):
    """A legacy champion with no calibration tags resolves to the explicit no-op
    temperature (1.0) — never a failure, never a stale temperature, and no
    separate calibration file is written."""
    import json

    import src.constants as c

    d = _deploy()
    _specs, tags, _downloaded, artifacts_dir = _aux_tags_and_files(
        monkeypatch, tmp_path, d, pre_populate=True
    )
    monkeypatch.setattr(d, "DEPLOY_ARTIFACTS", artifacts_dir)
    assert c.CALIBRATION_URI_TAG not in tags
    assert c.CALIBRATION_HASH_TAG not in tags

    temperature = d._download_aux_artifacts(None, tags)

    assert temperature == 1.0
    # the temp value is embedded in model_info at build time, not shipped as a file
    assert not (artifacts_dir / c.CALIBRATION_ARTIFACT).exists()


def test_download_aux_artifacts_untagged_champion_overwrites_stale_calibration(
    monkeypatch, tmp_path
):
    """A stale local calibration file (a previous champion's temperature) is not
    reused by a legacy champion with no tags: it resolves to 1.0 and the stale
    file is removed since it is no longer shipped."""
    import json

    import src.constants as c

    d = _deploy()
    _specs, tags, _downloaded, artifacts_dir = _aux_tags_and_files(
        monkeypatch, tmp_path, d, pre_populate=True
    )
    monkeypatch.setattr(d, "DEPLOY_ARTIFACTS", artifacts_dir)
    stale = artifacts_dir / c.CALIBRATION_ARTIFACT
    stale.write_text(json.dumps({"temperature": 2.0}) + "\n")

    temperature = d._download_aux_artifacts(None, tags)

    assert temperature == 1.0
    assert not stale.exists()


def test_calibration_reuses_matching_local_file(monkeypatch, tmp_path):
    """A tagged champion with a matching local calibration file reuses it: no download."""
    import json

    import src.constants as c

    d = _deploy()
    _specs, tags, downloaded, artifacts_dir = _aux_tags_and_files(
        monkeypatch, tmp_path, d, pre_populate=True
    )
    _pin_calibration(tmp_path, tags, temperature=1.7)
    (artifacts_dir / c.CALIBRATION_ARTIFACT).write_text(
        (tmp_path / "remote" / c.CALIBRATION_ARTIFACT).read_text()
    )
    monkeypatch.setattr(d, "DEPLOY_ARTIFACTS", artifacts_dir)

    temperature = d._download_aux_artifacts(None, tags)

    assert downloaded == []  # model files AND calibration file reused
    assert temperature == 1.7
    payload = json.loads((artifacts_dir / c.CALIBRATION_ARTIFACT).read_text())
    assert payload == {"temperature": 1.7}


def test_calibration_downloads_and_places_pinned_file(monkeypatch, tmp_path):
    """A tagged champion's calibration file is downloaded, hash-verified, and
    placed at DEPLOY_ARTIFACTS/calibration_t.json."""
    import json

    import src.constants as c

    d = _deploy()
    _specs, tags, downloaded, artifacts_dir = _aux_tags_and_files(
        monkeypatch, tmp_path, d, pre_populate=False
    )
    _pin_calibration(tmp_path, tags, temperature=0.8)
    monkeypatch.setattr(d, "DEPLOY_ARTIFACTS", artifacts_dir)

    temperature = d._download_aux_artifacts(None, tags)

    assert tags[c.CALIBRATION_URI_TAG] in downloaded
    assert temperature == 0.8
    local = artifacts_dir / c.CALIBRATION_ARTIFACT
    assert local.exists()
    assert d._file_hash(local) == tags[c.CALIBRATION_HASH_TAG]
    assert json.loads(local.read_text()) == {"temperature": 0.8}


def test_calibration_rejects_malformed_verified_artifact(monkeypatch, tmp_path):
    """A hash-verified calibration file with malformed JSON is rejected."""
    import src.constants as c

    d = _deploy()
    _specs, tags, _downloaded, artifacts_dir = _aux_tags_and_files(
        monkeypatch, tmp_path, d, pre_populate=True
    )
    _pin_calibration(tmp_path, tags)
    src = tmp_path / "remote" / c.CALIBRATION_ARTIFACT
    src.write_text("{not json")
    tags[c.CALIBRATION_HASH_TAG] = d._file_hash(src)
    (artifacts_dir / c.CALIBRATION_ARTIFACT).write_text("{not json")
    monkeypatch.setattr(d, "DEPLOY_ARTIFACTS", artifacts_dir)

    import pytest

    with pytest.raises(RuntimeError, match="invalid calibration file"):
        d._download_aux_artifacts(None, tags)


def test_calibration_rejects_non_positive_verified_temperature(monkeypatch, tmp_path):
    """A hash-verified calibration file with a non-positive temperature is rejected."""
    import json

    import src.constants as c

    d = _deploy()
    _specs, tags, _downloaded, artifacts_dir = _aux_tags_and_files(
        monkeypatch, tmp_path, d, pre_populate=True
    )
    _pin_calibration(tmp_path, tags)
    src = tmp_path / "remote" / c.CALIBRATION_ARTIFACT
    src.write_text(json.dumps({"temperature": 0.0}))
    tags[c.CALIBRATION_HASH_TAG] = d._file_hash(src)
    (artifacts_dir / c.CALIBRATION_ARTIFACT).write_text(json.dumps({"temperature": 0.0}))
    monkeypatch.setattr(d, "DEPLOY_ARTIFACTS", artifacts_dir)

    import pytest

    with pytest.raises(RuntimeError, match="invalid calibration file"):
        d._download_aux_artifacts(None, tags)


def test_write_model_info_records_calibration_only_with_both_tags(monkeypatch, tmp_path):
    """The canonical manifest carries the calibration pin only when the champion
    has both calibration lineage tags; a legacy champion gets no key at all."""
    import json

    import src.constants as c

    d = _deploy()
    manifest_file = tmp_path / "model_info.json"
    monkeypatch.setattr(d, "MODEL_INFO_FILE", manifest_file)
    monkeypatch.setattr(d, "_cached_gbdt_framework", lambda _version: None)
    monkeypatch.setattr(d, "_gbdt_framework", lambda _model_uri: "xgboost")
    production = SimpleNamespace(version="7", run_id="run-prod")

    tags = _lineage_tags()
    tags[c.CALIBRATION_URI_TAG] = "runs:/run-aux/calibration_t.json"
    tags[c.CALIBRATION_HASH_TAG] = "abc"
    tagged = _FakeMlflowClient(_FakeModelVersion(tags))
    pins = d._lineage_pins(tagged, production)
    manifest = json.loads(
        d._write_model_info(tagged, production, pins, "fp", calibration_temperature=1.7).read_text()
    )
    assert manifest["calibration"] == {
        "uri": "runs:/run-aux/calibration_t.json",
        "sha256": "abc",
        "temperature": 1.7,
    }

    legacy = _FakeMlflowClient(_FakeModelVersion(_lineage_tags()))
    legacy_manifest = json.loads(d._write_model_info(legacy, production, pins, "fp").read_text())
    assert "calibration" not in legacy_manifest


# --- Task 2: exact champion lineage; base models carry no aliases ---


def test_build_database_url_returns_passwordless_local_url(monkeypatch):
    """Homebrew trust path: the passwordless DATABASE_URL is returned verbatim."""
    import src.constants as c

    monkeypatch.setenv("DATABASE_URL", "postgresql://steve@127.0.0.1:5432/postgres")
    assert c.get_database_url() == "postgresql://steve@127.0.0.1:5432/postgres"


def test_build_database_url_returns_password_bearing_compose_url(monkeypatch):
    """Compose path: the password-bearing DATABASE_URL is returned verbatim."""
    import src.constants as c

    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:password@postgres:5432/tennis")
    assert c.get_database_url() == "postgresql://postgres:password@postgres:5432/tennis"


def test_build_database_url_missing_contract_fails_fast(monkeypatch):
    """No DATABASE_URL means no application connection: fail fast rather than
    silently target some default database."""
    import pytest

    import src.constants as c

    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="missing PostgreSQL configuration"):
        c.get_database_url()


# --- Task 2: exact champion lineage; base models carry no aliases ---


def _lineage_tags():
    """The exact tag set 05_evaluate writes onto the promoted ensemble version."""
    import json

    import src.constants as c
    from src.features.columns import FEATURE_COLS

    return {
        "base_linear_registered_model_name": "linear_best",
        "base_linear_version": "3",
        "base_linear_run_id": "run-linear",
        "base_linear_model_uri": "runs:/run-linear/linear_model",
        "base_linear_scaler_uri": "runs:/run-linear/linear_scaler.pkl",
        "base_linear_scaler_hash": "aaa",
        "base_gbdt_registered_model_name": "gbdt_best",
        "base_gbdt_version": "2",
        "base_gbdt_run_id": "run-gbdt",
        "base_gbdt_model_uri": "runs:/run-gbdt/gbdt_model",
        "base_nn_registered_model_name": "nn_best",
        "base_nn_version": "1",
        "base_nn_run_id": "run-nn",
        "base_nn_model_uri": "runs:/run-nn/nn_model",
        "aux_embeddings_uri": "runs:/run-aux/bio_embeddings.npz",
        "aux_embeddings_hash": "bbb",
        "aux_bio_feature_cols_uri": "runs:/run-aux/bio_feature_cols.json",
        "aux_bio_feature_cols_hash": "ccc",
        c.FEATURE_COLS_TAG: json.dumps(FEATURE_COLS, separators=(",", ":")),
        c.FEATURE_COLS_HASH_TAG: _deploy()._feature_cols_hash(FEATURE_COLS),
    }


class _FakeModelVersion:
    def __init__(self, tags, version="7", run_id="run-prod"):
        self.tags = dict(tags)
        self.version = version
        self.run_id = run_id
        self.creation_timestamp = 123


class _FakeMlflowClient:
    """Records every alias lookup; only ensemble @champion resolution is legal."""

    def __init__(self, model_version):
        self._mv = model_version
        self.alias_queries = []

    def get_model_version(self, name, version):
        assert name == "ensemble_lr_model"
        assert str(version) == str(self._mv.version)
        return self._mv

    def get_model_version_by_alias(self, name, alias):
        self.alias_queries.append((name, alias))
        assert name == "ensemble_lr_model" and alias == "champion"
        return self._mv


def test_lineage_pins_resolve_exact_versions_from_champion_tags(monkeypatch):
    """Deploy resolves bases from champion tags: exact versions, never aliases."""
    d = _deploy()
    # Framework detection loads the pinned GBDT model via MLflow; stub it out.
    monkeypatch.setattr(d, "_gbdt_framework", lambda _uri: "xgboost")
    client = _FakeMlflowClient(_FakeModelVersion(_lineage_tags()))
    production = SimpleNamespace(version="7", run_id="run-prod")

    pins = d._lineage_pins(client, production)

    assert pins["production"]["registered_model_name"] == "ensemble_lr_model"
    assert pins["production"]["version"] == "7"
    for cls, name in (("linear", "linear_best"), ("gbdt", "gbdt_best"), ("nn", "nn_best")):
        assert pins[cls]["registered_model_name"] == name
        assert "alias" not in pins[cls]
    assert pins["linear"]["version"] == "3"
    assert pins["linear"]["scaler_uri"] == "runs:/run-linear/linear_scaler.pkl"
    assert pins["linear"]["scaler_hash"] == "aaa"
    assert pins["gbdt"]["run_id"] == "run-gbdt"
    assert pins["gbdt"]["framework"] == "xgboost"
    assert pins["nn"]["model_uri"] == "runs:/run-nn/nn_model"
    # Only the ensemble @champion alias is ever resolved — no base alias lookups.
    assert client.alias_queries == []


def test_lineage_pins_missing_tags_fail_fast():
    """A champion without its exact lineage tags is not deployable."""
    import pytest

    d = _deploy()
    tags = _lineage_tags()
    del tags["base_nn_run_id"]
    client = _FakeMlflowClient(_FakeModelVersion(tags))
    with pytest.raises(RuntimeError, match="base_nn_run_id"):
        d._lineage_pins(client, SimpleNamespace(version="7", run_id="run-prod"))


def _resolve_with_warning(monkeypatch, tags):
    """Run _lineage_pins capturing _log output; return (pins, warnings)."""
    d = _deploy()
    monkeypatch.setattr(d, "_gbdt_framework", lambda _uri: "xgboost")
    warnings: list[str] = []
    monkeypatch.setattr(d, "_log", lambda _category, message: warnings.append(message))
    pins = d._lineage_pins(
        _FakeMlflowClient(_FakeModelVersion(tags)),
        SimpleNamespace(version="7", run_id="run-prod"),
    )
    return pins, warnings


def test_lineage_pins_stale_feature_hash_warns_and_resolves_pins(monkeypatch):
    """A stale feature-contract hash warns but the lineage pins still resolve."""
    from src.features.columns import FEATURE_COLS

    tags = _lineage_tags()
    tags["feature_cols_hash"] = "deadbeef"
    pins, warnings = _resolve_with_warning(monkeypatch, tags)

    assert pins["nn"]["version"] == "1"
    assert len(warnings) == 1
    assert "ensemble_lr_model v7" in warnings[0]
    assert "stale" in warnings[0]
    expected = _deploy()._feature_cols_hash(list(FEATURE_COLS))
    assert f"expected contract hash {expected}" in warnings[0]
    assert "tagged deadbeef" in warnings[0]


def test_lineage_pins_old_feature_columns_warn_and_resolve_pins(monkeypatch):
    """A champion trained on different FEATURE_COLS warns and still deploys."""
    import json as _json

    old_columns = ["rank_diff", "age_diff"]
    tags = _lineage_tags()
    tags["feature_cols"] = _json.dumps(old_columns, separators=(",", ":"))
    tags["feature_cols_hash"] = _deploy()._feature_cols_hash(old_columns)
    pins, warnings = _resolve_with_warning(monkeypatch, tags)

    assert pins["gbdt"]["run_id"] == "run-gbdt"
    assert len(warnings) == 1
    assert "packaging its pinned feature columns" in warnings[0]


def test_lineage_pins_malformed_feature_contract_warns_and_resolves_pins(monkeypatch):
    """Malformed feature-contract JSON warns and the lineage pins still resolve."""
    tags = _lineage_tags()
    tags["feature_cols"] = "{not json"
    pins, warnings = _resolve_with_warning(monkeypatch, tags)

    assert pins["linear"]["version"] == "3"
    assert len(warnings) == 1
    assert "malformed" in warnings[0]


def test_lineage_pins_missing_feature_contract_warns_and_resolves_pins(monkeypatch):
    """A champion tagged before feature contracts existed warns and still deploys."""
    tags = _lineage_tags()
    del tags["feature_cols"]
    del tags["feature_cols_hash"]
    pins, warnings = _resolve_with_warning(monkeypatch, tags)

    assert pins["production"]["version"] == "7"
    assert len(warnings) == 1
    assert "missing" in warnings[0]


def test_lineage_pins_stale_contract_still_fails_on_missing_lineage(monkeypatch):
    """The stale-contract warning never bypasses exact lineage-tag safety."""
    import pytest

    tags = _lineage_tags()
    tags["feature_cols_hash"] = "deadbeef"
    del tags["base_nn_run_id"]
    with pytest.raises(RuntimeError, match="base_nn_run_id"):
        _resolve_with_warning(monkeypatch, tags)


def test_build_input_fingerprint_includes_lineage_and_sources(monkeypatch, tmp_path):
    """The fingerprint covers canonical lineage and source/artifact hashes but
    excludes the generated manifest and all post-build Bento/Docker identities."""
    d = _deploy()
    monkeypatch.setattr(d, "ROOT", tmp_path)
    src_files = []
    for name in ("bentofile.yaml", "service.py", "sim.index", "sim.meta.json"):
        p = tmp_path / name
        p.write_text("content-" + name)
        src_files.append(p)
    monkeypatch.setattr(d, "SOURCE_FINGERPRINT_FILES", src_files)

    client = _FakeMlflowClient(_FakeModelVersion(_lineage_tags()))
    production = SimpleNamespace(version="7", run_id="run-prod")
    fp = d.build_input_fingerprint(client, production)

    assert "ensemble_lr_model@champion=v7" in fp
    for key, value in _lineage_tags().items():
        assert f"{key}={value}" in fp
    for path in src_files:
        assert path.relative_to(tmp_path).as_posix() in fp
    # Non-circular: generated manifest, deploy state, Bento tag, and Docker
    # identity must never appear in the fingerprint.
    assert "bentofile.pinned" not in fp
    assert "bento_build_state" not in fp
    assert "bento_tag" not in fp
    assert "tennis-bento:latest" not in fp


def test_build_input_fingerprint_ignores_generated_outputs(monkeypatch, tmp_path):
    """Touching generated files must not change the fingerprint; changing the
    champion lineage must."""
    d = _deploy()
    monkeypatch.setattr(d, "ROOT", tmp_path)
    src = tmp_path / "service.py"
    src.write_text("v1")
    monkeypatch.setattr(d, "SOURCE_FINGERPRINT_FILES", [src])

    production = SimpleNamespace(version="7", run_id="run-prod")
    before = d.build_input_fingerprint(
        _FakeMlflowClient(_FakeModelVersion(_lineage_tags())), production
    )

    # Post-build outputs written after the fingerprint is computed.
    (tmp_path / "bentofile.pinned.yaml").write_text("generated manifest")
    (tmp_path / "bento_tag.txt").write_text("bento:abc")
    (tmp_path / "bento_build_state.json").write_text('{"tag": "bento:abc"}')
    after = d.build_input_fingerprint(
        _FakeMlflowClient(_FakeModelVersion(_lineage_tags())), production
    )
    assert after == before  # deterministic and non-self-referential

    changed_tags = dict(_lineage_tags())
    changed_tags["base_linear_version"] = "4"
    changed = d.build_input_fingerprint(
        _FakeMlflowClient(_FakeModelVersion(changed_tags)), production
    )
    assert changed != before  # lineage is canonical: a repoint changes the fingerprint


def test_build_lineage_tags_flattens_exact_pins():
    """build_lineage_tags turns base/aux pins into the champion tag set."""
    import src.constants as c

    base_pins = {
        "linear": {
            "registered_model_name": "linear_best",
            "version": "3",
            "run_id": "run-linear",
            "model_uri": "runs:/run-linear/linear_model",
            "scaler_uri": "runs:/run-linear/linear_scaler.pkl",
            "scaler_hash": "aaa",
        },
        "gbdt": {
            "registered_model_name": "gbdt_best",
            "version": "2",
            "run_id": "run-gbdt",
            "model_uri": "runs:/run-gbdt/gbdt_model",
        },
        "nn": {
            "registered_model_name": "nn_best",
            "version": "1",
            "run_id": "run-nn",
            "model_uri": "runs:/run-nn/nn_model",
        },
    }
    aux_pins = {
        "embeddings_uri": "runs:/run-aux/bio_embeddings.npz",
        "embeddings_hash": "bbb",
        "bio_feature_cols_uri": "runs:/run-aux/bio_feature_cols.json",
        "bio_feature_cols_hash": "ccc",
    }
    tags = c.build_lineage_tags(base_pins, aux_pins)

    assert tags["base_linear_version"] == "3"
    assert tags["base_linear_scaler_uri"] == "runs:/run-linear/linear_scaler.pkl"
    assert "base_gbdt_scaler_uri" not in tags  # only linear has a scaler
    assert tags["base_gbdt_run_id"] == "run-gbdt"
    assert tags["base_nn_model_uri"] == "runs:/run-nn/nn_model"
    assert tags["aux_embeddings_hash"] == "bbb"
    assert tags["aux_bio_feature_cols_uri"] == "runs:/run-aux/bio_feature_cols.json"
    # Navigation artifacts (similarity index/metadata, player directory) are not
    # model lineage: build_lineage_tags never emits their tags.
    for nav_key in (
        "aux_similarity_index_uri",
        "aux_similarity_metadata_uri",
        "aux_player_directory_uri",
    ):
        assert nav_key not in tags


# --- Navigation boundary contracts ---


def test_similarity_artifacts_are_snapshot_built_not_fingerprint_inputs():
    """Navigation artifacts are rebuilt at deploy, not champion lineage inputs."""
    d = _deploy()
    assert d.SIMILARITY_INDEX not in d.SOURCE_FINGERPRINT_FILES
    assert d.SIMILARITY_METADATA not in d.SOURCE_FINGERPRINT_FILES


def test_deploy_module_is_not_a_prefect_flow():
    """Deploy is deliberately manual: the module never imports Prefect or
    decorates anything with @flow, so running it can never register a run."""
    src = (_deploy().ROOT / "src" / "flows" / "deploy.py").read_text()
    assert "import prefect" not in src
    assert "from prefect" not in src
    assert "@flow" not in src


def test_deploy_source_reads_champion_and_writes_only_deploy_bookkeeping_tags():
    """Deploy reads MLflow — the sole alias lookup is ensemble @champion — and
    its only registry mutation is the two pipeline_last_deploy_* bookkeeping
    tags on the champion version. It never re-aliases, re-stages, updates, or
    deletes model versions."""
    src = (_deploy().ROOT / "src" / "flows" / "deploy.py").read_text()
    assert src.count("get_model_version_by_alias") == 1  # champion resolution only
    import re

    assert re.search(r"models:/[^\"']*@", src) is None  # no alias URIs
    # Post-build bookkeeping: exactly the two deploy tags, nothing else.
    assert src.count("set_model_version_tag") == 2
    assert src.count("pipeline_last_deploy_run_id") == 1
    assert src.count("pipeline_last_deploy_image") == 1
    for api in (
        "set_registered_model_alias",
        "set_model_tag",
        "update_model_version",
        "delete_model_version",
        "transition_model_version_stage",
    ):
        assert api not in src


def test_base_notebooks_never_create_best_alias():
    """02 notebooks register numbered versions but never set a @best alias."""
    root = _deploy().ROOT
    for name in ("02_tune_linear", "02_tune_gbdt", "02_tune_nn"):
        src = (root / "notebooks" / "parameters" / f"{name}.ipynb").read_text()
        assert "set_registered_model_alias" not in src
        assert '"best"' not in src


def test_promotion_tags_lineage_before_champion_alias():
    """05 tags the promoted version with exact lineage BEFORE assigning @champion,
    and the only alias in the training path is the ensemble @champion."""
    root = _deploy().ROOT
    src = (root / "notebooks" / "parameters" / "04_evaluate.ipynb").read_text()
    assert src.index("set_model_version_tag") < src.index("set_registered_model_alias")
    assert "build_lineage_tags" in src
    assert src.count("set_registered_model_alias") == 1
    for name in ("02_tune_linear", "02_tune_gbdt", "02_tune_nn"):
        nb = (root / "notebooks" / "parameters" / f"{name}.ipynb").read_text()
        assert "set_registered_model_alias" not in nb


# --- Web image build consumes host-generated Vite inputs ---


def test_just_deploy_runs_staging_then_node_builder_then_web_build():
    """`just deploy` order is: deploy.py stages snapshot navigation artifacts ->
    the node generator verifies/reuses or rebuilds web/src/assets/generated/* ->
    the Docker build, so the web image never reads data/deploy/ itself."""
    justfile = (_deploy().ROOT / "justfile").read_text()
    staging = justfile.index("uv run python src/flows/deploy.py")
    generator = justfile.index("node web/scripts/build-player-index.mjs")
    web_build = justfile.index("docker buildx build")
    assert staging < generator < web_build
    assert "--push web/" in justfile[web_build:]


def test_web_dockerfile_has_no_in_container_index_builder():
    """The node builder runs host-side only; the Docker image build just
    consumes the pre-generated inputs via `COPY . .` in its web/ context."""
    dockerfile = (_deploy().ROOT / "web" / "Dockerfile").read_text()
    assert "build-player-index" not in dockerfile
    assert "RUN node" not in dockerfile
    assert "COPY . ." in dockerfile
