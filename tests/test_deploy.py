"""Offline deployment tests with mocked MLflow, BentoML, Docker, and Compose."""

import importlib
import subprocess
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


# --- Command construction (push latest only; no Compose, no web build) ---


def _stub_subprocess(monkeypatch):
    """Replace subprocess.run with a no-op returning success."""
    monkeypatch.setattr("subprocess.run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))


def test_deploy_bento_pushes_only_latest_no_compose_no_web(monkeypatch, tmp_path):
    d = _deploy()
    monkeypatch.setattr(d, "DOCKER_REPO", "acme")
    monkeypatch.setattr(d, "IMAGE_NAME", "tennis-bento")
    monkeypatch.setattr(d, "LOGS", tmp_path)
    monkeypatch.setattr(d, "build_bento_image", lambda **_kwargs: ("tennis-bento:latest", 5))
    monkeypatch.setattr(d, "_docker_login", lambda: None)
    monkeypatch.setattr(d, "_read_state", lambda: {})
    written = {}
    monkeypatch.setattr(d, "_write_state", lambda s: written.update(s))
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    _stub_subprocess(monkeypatch)

    calls = []

    def fake_run_teed(cmd, _log):
        calls.append(list(cmd))
        return None

    monkeypatch.setattr(d, "_run_teed", fake_run_teed)

    d.deploy_bento()

    # Push only the Docker Hub latest image.
    assert calls == [["docker", "push", "acme/tennis-bento:latest"]]
    assert written["deployed_version"] == 5
    assert written["deployed_image"] == "acme/tennis-bento:latest"


def test_deploy_bento_does_not_require_postgres_password(monkeypatch, tmp_path):
    """Deploy does not require a PostgreSQL credential."""
    d = _deploy()
    monkeypatch.setattr(d, "DOCKER_REPO", "acme")
    monkeypatch.setattr(d, "IMAGE_NAME", "tennis-bento")
    monkeypatch.setattr(d, "LOGS", tmp_path)
    monkeypatch.setattr(d, "build_bento_image", lambda **_kwargs: ("tennis-bento:latest", 5))
    monkeypatch.setattr(d, "_docker_login", lambda: None)
    monkeypatch.setattr(d, "_read_state", lambda: {})
    monkeypatch.setattr(d, "_write_state", lambda _s: None)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    pushed = []
    monkeypatch.setattr(d, "_run_teed", lambda cmd, _log: pushed.append(list(cmd)))
    _stub_subprocess(monkeypatch)

    d.deploy_bento()  # must not raise

    assert pushed == [["docker", "push", "acme/tennis-bento:latest"]]


def test_deploy_bento_fails_when_push_fails(monkeypatch, tmp_path):
    d = _deploy()
    monkeypatch.setattr(d, "LOGS", tmp_path)
    monkeypatch.setattr(d, "build_bento_image", lambda **_kwargs: ("tennis-bento:latest", 5))
    monkeypatch.setattr(d, "_docker_login", lambda: None)
    _stub_subprocess(monkeypatch)

    def fail_push(_cmd, _log):
        raise subprocess.CalledProcessError(1, ["docker", "push"])

    monkeypatch.setattr(d, "_run_teed", fail_push)

    import pytest

    with pytest.raises(subprocess.CalledProcessError):
        d.deploy_bento()


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


def _stub_bento_build(monkeypatch):
    """Patch the MLflow/BentoML/Docker machinery behind build_bento_image."""
    d = _deploy()
    from pathlib import Path as _Path

    monkeypatch.setattr(d, "IMAGE_NAME", "tennis-bento")
    monkeypatch.setattr(d, "BENTO_TAG_FILE", _Path("/tmp/bento_tag.txt"))
    monkeypatch.setattr(d, "STATE_FILE", _Path("/tmp/state.json"))
    monkeypatch.setattr(d, "MODEL_INFO_FILE", _Path("/tmp/model_info.json"))
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
    monkeypatch.setattr(d, "_check_aux_files", lambda: None)
    monkeypatch.setattr(d, "build_input_fingerprint", lambda _client, _prod: "fp")
    monkeypatch.setattr(d, "_read_state", lambda: {"fingerprint": "fp"})
    monkeypatch.setattr(
        d, "_import_models", lambda _pins: {"linear": "l", "gbdt": "g", "production": "p"}
    )
    monkeypatch.setattr(d, "_write_pinned_bentofile", lambda _tags: "pinned")
    monkeypatch.setattr(d, "_write_state", lambda _s: None)
    docker_calls = []
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **_k: (
            (docker_calls.append(list(a[0])) if a else None) or SimpleNamespace(returncode=0)
        ),
    )

    built = {"bento": 0, "container": 0, "image_tags": [], "docker_calls": docker_calls}
    bentoml = {
        "bentoml": SimpleNamespace(
            bentos=SimpleNamespace(get=lambda _tag: (_ for _ in ()).throw(Exception())),
            container=SimpleNamespace(),
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

    def container_build(*_args, **_kwargs):
        built["container"] += 1
        built["image_tags"].append(_kwargs.get("image_tag"))

    bentoml["bentoml"].bentos.build_bentofile = build_bentofile
    bentoml["bentoml"].container.build = container_build
    return d, built


def test_build_force_rebuilds(monkeypatch):
    d, built = _stub_bento_build(monkeypatch)

    image, version = d.build_bento_image()
    assert built["bento"] == 1
    assert built["container"] == 1
    # Build only the moving local latest tag.
    assert built["image_tags"] == [("tennis-bento:latest",)]
    assert image == "tennis-bento:latest"
    assert version == 7
    assert built["docker_calls"] == []


def test_build_without_force_rebuilds(monkeypatch):
    d, built = _stub_bento_build(monkeypatch)

    d.build_bento_image()
    assert built["bento"] == 1
    assert built["container"] == 1
    assert built["image_tags"] == [("tennis-bento:latest",)]


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


def test_check_aux_files_requires_every_packaged_artifact(monkeypatch, tmp_path):
    """Deploy fails fast when any packaged artifact is missing from disk."""
    d = _deploy()
    monkeypatch.setattr(d, "ROOT", tmp_path)
    files = [tmp_path / "a.pkl", tmp_path / "b.npz", tmp_path / "c.json"]
    for f in files:
        f.write_bytes(b"x")
    monkeypatch.setattr(d, "AUX_FILES", files)

    d._check_aux_files()  # all present: no error

    files[1].unlink()
    import pytest

    with pytest.raises(RuntimeError, match=r"b\.npz"):
        d._check_aux_files()


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
    import src.constants as c

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
    }


class _FakeModelVersion:
    def __init__(self, tags, version="7", run_id="run-prod"):
        self.tags = dict(tags)
        self.version = version
        self.run_id = run_id


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


# --- Task 2: static notebook/deploy contracts ---


def test_similarity_artifacts_are_fingerprinted_build_inputs():
    """The on-disk similarity index and metadata are build inputs: changing
    them changes the fingerprint even when the MLflow lineage is unchanged."""
    d = _deploy()
    assert d.SIMILARITY_INDEX in d.SOURCE_FINGERPRINT_FILES
    assert d.SIMILARITY_METADATA in d.SOURCE_FINGERPRINT_FILES


def test_deploy_module_is_not_a_prefect_flow():
    """Deploy is deliberately manual: the module never imports Prefect or
    decorates anything with @flow, so running it can never register a run."""
    src = (_deploy().ROOT / "src" / "flows" / "deploy.py").read_text()
    assert "import prefect" not in src
    assert "from prefect" not in src
    assert "@flow" not in src


def test_deploy_source_never_mutates_mlflow_or_resolves_base_aliases():
    """Deploy only reads MLflow: the sole alias lookup is ensemble @champion, and
    no tag/alias/version mutation API appears anywhere in deploy."""
    src = (_deploy().ROOT / "src" / "flows" / "deploy.py").read_text()
    assert src.count("get_model_version_by_alias") == 1  # champion resolution only
    import re

    assert re.search(r"models:/[^\"']*@", src) is None  # no alias URIs
    for api in (
        "set_model_version_tag",
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
