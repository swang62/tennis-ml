"""Local deployment artifact and manifest tests."""

import contextlib
import hashlib
import importlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest


def _deploy():
    return importlib.import_module("src.flows.deploy")


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
    for artifact in [*d.AUX_FILES, d.MODEL_INFO_FILE]:
        assert artifact.relative_to(d.ROOT).as_posix() in config["include"]


def _aux_tags_and_files(tmp_path, d, pre_populate=True):
    """Return local artifact files and matching lineage tags."""
    src_dir = tmp_path / "remote"
    src_dir.mkdir()
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    specs = [
        ("base_linear_scaler_uri", "base_linear_scaler_hash", "linear_scaler.pkl"),
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

    return specs, tags, downloaded, artifacts_dir


def test_download_aux_artifacts_reuses_matching_local_files(monkeypatch, tmp_path):
    """An existing local artifact whose hash matches the pin is reused: no download."""
    d = _deploy()
    _specs, tags, downloaded, artifacts_dir = _aux_tags_and_files(tmp_path, d, pre_populate=True)
    monkeypatch.setattr(d, "DEPLOY_ARTIFACTS", artifacts_dir)

    temp = d._download_aux_artifacts(None, tags)

    assert downloaded == []  # all model artifacts reused
    assert temp == 1.0  # no calibration tags -> no-op temperature
    assert sorted(p.name for p in artifacts_dir.iterdir()) == sorted(name for _, _, name in _specs)


def test_download_aux_artifacts_requires_no_navigation_tags(monkeypatch, tmp_path):
    """Navigation artifacts are rebuilt from the snapshot, never downloaded."""
    d = _deploy()
    _specs, tags, _downloaded, artifacts_dir = _aux_tags_and_files(tmp_path, d, pre_populate=True)
    monkeypatch.setattr(d, "DEPLOY_ARTIFACTS", artifacts_dir)

    assert not any("similarity" in tag or "directory" in tag for tag in tags)
    d._download_aux_artifacts(None, tags)
    assert not (artifacts_dir / "player_directory.json").exists()


# --- Calibration artifact materialization (calibration_uri / calibration_hash tags) ---


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
    tags["calibration_uri"] = f"runs:/run-aux/{c.CALIBRATION_ARTIFACT}"
    tags["calibration_hash"] = _deploy()._file_hash(src)
    return src


def test_download_aux_artifacts_legacy_champion_writes_noop_calibration(monkeypatch, tmp_path):
    """A legacy champion with no calibration tags resolves to the explicit no-op
    temperature (1.0) — never a failure, never a stale temperature, and no
    separate calibration file is written."""
    import src.constants as c

    d = _deploy()
    _specs, tags, _downloaded, artifacts_dir = _aux_tags_and_files(tmp_path, d, pre_populate=True)
    monkeypatch.setattr(d, "DEPLOY_ARTIFACTS", artifacts_dir)
    assert "calibration_uri" not in tags
    assert "calibration_hash" not in tags

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
    _specs, tags, _downloaded, artifacts_dir = _aux_tags_and_files(tmp_path, d, pre_populate=True)
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
    _specs, tags, downloaded, artifacts_dir = _aux_tags_and_files(tmp_path, d, pre_populate=True)
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


def test_calibration_rejects_malformed_verified_artifact(monkeypatch, tmp_path):
    """A hash-verified calibration file with malformed JSON is rejected."""
    import src.constants as c

    d = _deploy()
    _specs, tags, _downloaded, artifacts_dir = _aux_tags_and_files(tmp_path, d, pre_populate=True)
    _pin_calibration(tmp_path, tags)
    src = tmp_path / "remote" / c.CALIBRATION_ARTIFACT
    src.write_text("{not json")
    tags["calibration_hash"] = d._file_hash(src)
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
    _specs, tags, _downloaded, artifacts_dir = _aux_tags_and_files(tmp_path, d, pre_populate=True)
    _pin_calibration(tmp_path, tags)
    src = tmp_path / "remote" / c.CALIBRATION_ARTIFACT
    src.write_text(json.dumps({"temperature": 0.0}))
    tags["calibration_hash"] = d._file_hash(src)
    (artifacts_dir / c.CALIBRATION_ARTIFACT).write_text(json.dumps({"temperature": 0.0}))
    monkeypatch.setattr(d, "DEPLOY_ARTIFACTS", artifacts_dir)

    import pytest

    with pytest.raises(RuntimeError, match="invalid calibration file"):
        d._download_aux_artifacts(None, tags)


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


def test_build_lineage_tags_flattens_exact_pins():
    """build_lineage_tags turns base pins into the champion tag set and emits no
    aux tags — only base-model lineage is pinned."""
    import src.constants as c

    base_pins = {
        "linear": {
            "registered_model_name": "linear_best",
            "version": "3",
            "run_id": "run-linear",
            "model_uri": "runs:/run-linear/linear_model",
            "scaler_uri": "runs:/run-linear/linear_scaler.pkl",
            "scaler_hash": "aaa",
            "selected_framework": "logistic_regression",
            "selection_metrics": {
                "logistic_regression": {
                    "inner_cv_log_loss": 0.60,
                    "outer_val_log_loss": 0.61,
                    "outer_val_roc_auc": 0.72,
                },
                "gaussian_naive_bayes": {
                    "inner_cv_log_loss": 0.62,
                    "outer_val_log_loss": 0.63,
                    "outer_val_roc_auc": 0.70,
                },
                "sgd_classifier": {
                    "inner_cv_log_loss": 0.61,
                    "outer_val_log_loss": 0.63,
                    "outer_val_roc_auc": 0.71,
                },
            },
        },
        "gbdt": {
            "registered_model_name": "gbdt_best",
            "version": "2",
            "run_id": "run-gbdt",
            "model_uri": "runs:/run-gbdt/gbdt_model",
            "selected_framework": "xgboost",
            "selection_metrics": {
                "xgboost": {
                    "inner_cv_log_loss": 0.59,
                    "outer_val_log_loss": 0.60,
                    "outer_val_roc_auc": 0.72,
                },
                "lightgbm": {
                    "inner_cv_log_loss": 0.60,
                    "outer_val_log_loss": 0.61,
                    "outer_val_roc_auc": 0.71,
                },
            },
        },
        "nn": {
            "registered_model_name": "nn_best",
            "version": "1",
            "run_id": "run-nn",
            "model_uri": "runs:/run-nn/nn_model",
            "selected_framework": "symmetric_gru",
            "selection_metrics": {
                "symmetric_gru": {
                    "outer_val_log_loss": 0.62,
                    "outer_val_roc_auc": 0.70,
                }
            },
        },
    }
    tags = c.build_lineage_tags(base_pins)

    assert tags["base_linear_version"] == "3"
    assert tags["base_linear_scaler_uri"] == "runs:/run-linear/linear_scaler.pkl"
    assert "base_gbdt_scaler_uri" not in tags  # only linear has a scaler
    assert tags["base_gbdt_run_id"] == "run-gbdt"
    assert tags["base_nn_model_uri"] == "runs:/run-nn/nn_model"
    # The champion records which framework won per base; candidate metrics stay
    # on the pinned base model versions so it stays inference-lean.
    assert tags["base_linear_selected_framework"] == "logistic_regression"
    assert tags["base_gbdt_selected_framework"] == "xgboost"
    assert tags["base_nn_selected_framework"] == "symmetric_gru"
    for metric_key in (
        "base_linear_logistic_regression_outer_val_log_loss",
        "base_linear_sgd_classifier_inner_cv_log_loss",
        "base_gbdt_lightgbm_outer_val_roc_auc",
        "base_nn_symmetric_gru_outer_val_log_loss",
    ):
        assert metric_key not in tags
    # Navigation artifacts (similarity index/metadata, player directory) are not
    # model lineage: build_lineage_tags never emits their tags.
    for nav_key in (
        "aux_similarity_index_uri",
        "aux_similarity_metadata_uri",
        "aux_player_directory_uri",
    ):
        assert nav_key not in tags


def test_lineage_pins_prefers_selected_framework_and_keeps_legacy_fallback():
    from types import SimpleNamespace

    import src.flows.deploy as deploy

    tags = {
        "feature_cols": json.dumps(list(deploy.FEATURE_COLS)),
        "feature_cols_hash": deploy._feature_cols_hash(list(deploy.FEATURE_COLS)),
    }
    for name in deploy.BASE_BENTO_NAMES:
        tags |= {
            f"base_{name}_registered_model_name": f"{name}_best",
            f"base_{name}_version": "1",
            f"base_{name}_run_id": f"run-{name}",
            f"base_{name}_model_uri": f"models:/{name}_best/1",
        }
    tags["base_gbdt_framework"] = "lightgbm"
    tags["base_gbdt_selected_framework"] = "xgboost"

    class Client:
        def get_model_version(self, _name, _version):
            return SimpleNamespace(tags=tags)

    production = SimpleNamespace(version="1", run_id="production-run")
    pins = deploy._lineage_pins(Client(), production)
    assert pins["gbdt"]["framework"] == "xgboost"

    del tags["base_gbdt_selected_framework"]
    assert deploy._lineage_pins(Client(), production)["gbdt"]["framework"] == "lightgbm"


# --- Navigation boundary contracts ---


# --- --no-cache flag: argparse, command construction, and propagation ---


def test_parse_deploy_args_recognizes_no_cache_flag():
    d = _deploy()
    assert d.parse_deploy_args(["--no-cache"]).no_cache is True


def test_buildx_build_cmd_default_has_no_no_cache():
    d = _deploy()
    cmd = d._buildx_build_cmd(
        builder="b", containerfile=Path("/c"), context=Path("/ctx"), image="img:dev"
    )
    assert "--no-cache" not in cmd
    assert cmd[:3] == ["docker", "buildx", "build"]


def test_buildx_build_cmd_includes_no_cache_when_set():
    d = _deploy()
    cmd = d._buildx_build_cmd(
        builder="b",
        containerfile=Path("/c"),
        context=Path("/ctx"),
        image="img:dev",
        no_cache=True,
    )
    assert "--no-cache" in cmd
    # --no-cache sits right after the build subcommand.
    assert cmd[cmd.index("build") + 1] == "--no-cache"


def test_generate_similarity_artifacts_no_cache_forces_rebuild(monkeypatch, tmp_path):
    """Matching inputs/sources and existing artifacts are reused by default, but
    no_cache forces a rebuild."""
    import pandas as pd

    import src.db.training as training

    d = _deploy()
    index = tmp_path / "player_similarity.index"
    meta = tmp_path / "player_metadata.json"
    index.write_bytes(b"x")
    meta.write_bytes(b"y")
    monkeypatch.setattr(d, "SIMILARITY_INDEX", index)
    monkeypatch.setattr(d, "SIMILARITY_METADATA", meta)
    monkeypatch.setattr(d, "_similarity_inputs_hash", lambda *_a, **_k: "ihash")
    monkeypatch.setattr(d, "_similarity_source_hash", lambda: "shash")
    monkeypatch.setattr(d, "_read_similarity_inputs_hash", lambda: "ihash")
    monkeypatch.setattr(d, "_read_similarity_source_hash", lambda: "shash")
    monkeypatch.setattr("src.serving.directory.PLAYERS_SQL", "select 1")
    monkeypatch.setattr("src.training.similarity.PLAYER_LIFETIME_SQL", "select 1")
    monkeypatch.setattr("src.serving.directory.directory_players", lambda _profiles: [])
    monkeypatch.setattr(
        training, "to_dataframe", lambda *_args, **_kwargs: pd.DataFrame([{"player_id": 1}])
    )

    class _FakeSim:
        def __init__(self):
            self.players: list = []

        def build(self, **kwargs):
            built["ran"] = True
            kwargs["index_path"].write_bytes(b"new-index")
            kwargs["metadata_path"].write_bytes(b"new-meta")

    monkeypatch.setattr("src.training.similarity.PlayerSimilarity", lambda: _FakeSim())
    monkeypatch.setattr(d, "_write_similarity_state", lambda *_args, **_kwargs: None)
    built = {}

    d.generate_similarity_artifacts()
    assert built.get("ran") is None  # reused
    d.generate_similarity_artifacts(no_cache=True)
    assert built["ran"] is True  # forced rebuild


# --- Task 7: quiet Bento Buildx output during deploy_bento ---


def _fake_mlflow(monkeypatch):
    """Disable MLflow side effects used by the success path of deploy_bento."""
    import mlflow

    monkeypatch.setattr(mlflow, "set_experiment", lambda *_a, **_k: None)
    monkeypatch.setattr(mlflow, "set_experiment_tag", lambda *_a, **_k: None)
    monkeypatch.setattr(mlflow, "log_param", lambda *_a, **_k: None)
    monkeypatch.setattr(mlflow, "log_artifact", lambda *_a, **_k: None)
    monkeypatch.setattr(mlflow, "start_run", lambda **_k: contextlib.nullcontext(_FakeRun()))
    monkeypatch.setattr(
        "mlflow.tracking.client.MlflowClient",
        lambda: _MlflowClientStub(),
    )


class _FakeRun:
    info = type("Info", (), {"run_id": "test-run-id"})()


class _MlflowClientStub:
    def set_model_version_tag(self, *_args, **_kwargs):
        return None


def test_run_teed_streams_to_console_when_not_quiet(capsys):
    """A direct (non-deploy) caller keeps the current console-streaming behavior."""
    d = _deploy()
    marker = "DIRECT_BUILDX_LINE"
    d._run_teed(["echo", marker], quiet=False, log=None)
    assert marker in capsys.readouterr().out


def test_run_teed_quiet_writes_log_only(tmp_path, capsys):
    """In quiet mode child output lands fully in the log and stays off the console."""
    d = _deploy()
    marker = "QUIET_BUILDX_LINE"
    log = tmp_path / "build.log"
    with log.open("w") as f:
        d._run_teed(["echo", marker], quiet=True, log=f)
    assert marker not in capsys.readouterr().out
    assert marker in log.read_text()


def test_deploy_bento_quiets_buildx_but_logs_it(tmp_path, monkeypatch, capsys):
    """During deploy the routine Buildx stream is absent from the console but
    present completely in the timestamped deploy log, while progress messages
    stay visible on the console."""
    d = _deploy()
    monkeypatch.setattr(d, "LOGS", tmp_path / "logs")
    monkeypatch.setattr(d, "STATE_FILE", tmp_path / "state.json")
    _fake_mlflow(monkeypatch)
    monkeypatch.setattr(d, "_docker_login", lambda: None)

    marker = "BUILDX_PROGRESS_LINE_123"

    def fake_build(no_cache=False, quiet=False, log=None):  # noqa: ARG001
        d._run_teed(["echo", marker], quiet=quiet, log=log)
        return "img:dev", 7

    monkeypatch.setattr(d, "build_bento_image", fake_build)

    d.deploy_bento()

    out = capsys.readouterr().out
    assert marker not in out  # routine Buildx output absent from console
    assert "Published" in out  # deployment progress message stays visible

    logs = list((tmp_path / "logs").glob("deploy_*.log"))
    assert len(logs) == 1
    assert marker in logs[0].read_text()  # present completely in the deploy log


def test_deploy_bento_failure_raises_and_logs_diagnostic(tmp_path, monkeypatch, capsys):
    """A Buildx failure still raises, keeps the child output in the deploy log,
    and records the failure diagnostic both on the console and in the log."""
    d = _deploy()
    monkeypatch.setattr(d, "LOGS", tmp_path / "logs")
    monkeypatch.setattr(d, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(d, "_docker_login", lambda: None)

    def fake_build_fail(no_cache=False, quiet=False, log=None):  # noqa: ARG001
        d._run_teed(["sh", "-c", "echo CHILD_ERR_LINE; exit 3"], quiet=quiet, log=log)
        return "img:dev", 7

    monkeypatch.setattr(d, "build_bento_image", fake_build_fail)

    with pytest.raises(subprocess.CalledProcessError):
        d.deploy_bento()

    out = capsys.readouterr().out
    assert "Deploy step failed" in out  # diagnostic visible on console

    logs = list((tmp_path / "logs").glob("deploy_*.log"))
    assert logs
    log_text = logs[0].read_text()
    assert "CHILD_ERR_LINE" in log_text  # child output retained in the log
    assert "Deploy step failed" in log_text  # failure diagnostic logged too


# --- GRU (nn) preprocessing artifact: validation and download ---


def _good_gru_preprocessing_artifact():
    """Build a notebook-shaped ``nn_preprocessing.json`` matching the producer."""
    import src.constants as c
    import src.features.nn_inference as ni

    artifact = {
        "artifact_name": c.NN_PREPROCESSING_ARTIFACT,
        "history_len": ni.HISTORY_LEN,
        "n_raw_features": ni.N_RAW,
        "n_context_features": len(ni.GRU_CONTEXT_NAMES),
        "raw_feature_names": list(ni.GRU_RAW_NAMES),
        "context_feature_names": list(ni.GRU_CONTEXT_NAMES),
        "fill": [0.0] * ni.N_RAW,
        "history_mean": [0.0] * ni.N_RAW,
        "history_scale": [1.0] * ni.N_RAW,
        "context_mean": [0.0] * len(ni.GRU_CONTEXT_NAMES),
        "context_scale": [1.0] * len(ni.GRU_CONTEXT_NAMES),
    }
    artifact["content_hash"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in artifact.items() if k != "artifact_name"}, sort_keys=True
        ).encode()
    ).hexdigest()
    return artifact


def test_load_gru_preprocessing_roundtrip_and_rejects_bad_schema(tmp_path):
    """A valid artifact loads; schema/name mismatches are rejected."""
    import src.constants as c
    from src.features.nn_inference import load_gru_preprocessing

    good = _good_gru_preprocessing_artifact()
    path = tmp_path / c.NN_PREPROCESSING_ARTIFACT
    path.write_text(json.dumps(good))

    loaded = load_gru_preprocessing(path)
    assert list(loaded.raw_names) == good["raw_feature_names"]
    assert list(loaded.context_names) == good["context_feature_names"]

    bad2 = dict(good)
    bad2["context_feature_names"] = ["wrong"]
    bad2_path = tmp_path / "bad2.json"
    bad2_path.write_text(json.dumps(bad2))
    with pytest.raises(RuntimeError, match="context_feature_names"):
        load_gru_preprocessing(bad2_path)


def test_materialize_nn_preprocessing_downloads_and_validates(monkeypatch, tmp_path):
    """A tagged champion preprocessing artifact is downloaded, hash-verified,
    and schema-validated before packaging."""
    import mlflow.artifacts

    import src.constants as c

    d = _deploy()
    monkeypatch.setattr(d, "DEPLOY_ARTIFACTS", tmp_path)

    good = _good_gru_preprocessing_artifact()
    src_file = tmp_path / "remote" / c.NN_PREPROCESSING_ARTIFACT
    src_file.parent.mkdir(parents=True, exist_ok=True)
    src_file.write_text(json.dumps(good))
    tags = {
        "base_nn_preprocessing_uri": f"runs:/run-nn/{c.NN_PREPROCESSING_ARTIFACT}",
        "base_nn_preprocessing_hash": d._file_hash(src_file),
    }

    downloaded = []

    def fake_download(uri, dst_path=None):
        target = Path(dst_path or ".") / c.NN_PREPROCESSING_ARTIFACT
        target.write_bytes(src_file.read_bytes())
        downloaded.append(uri)
        return str(target)

    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", fake_download)
    d._materialize_nn_preprocessing(None, tags)
    assert downloaded == [tags["base_nn_preprocessing_uri"]]
    assert (tmp_path / c.NN_PREPROCESSING_ARTIFACT).exists()


def test_materialize_nn_preprocessing_requires_tags(monkeypatch, tmp_path):
    """A GRU build without the preprocessing lineage tags fails fast."""
    import src.constants as c

    d = _deploy()
    monkeypatch.setattr(d, "DEPLOY_ARTIFACTS", tmp_path)
    with pytest.raises(RuntimeError, match="preprocessing lineage tags"):
        d._materialize_nn_preprocessing(None, {"calibration_uri": "x"})
