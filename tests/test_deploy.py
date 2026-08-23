"""Local deployment artifact and manifest tests."""

import importlib
from pathlib import Path


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
    import src.constants as c

    d = _deploy()
    _specs, tags, _downloaded, artifacts_dir = _aux_tags_and_files(tmp_path, d, pre_populate=True)
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
    _specs, tags, _downloaded, artifacts_dir = _aux_tags_and_files(tmp_path, d, pre_populate=True)
    _pin_calibration(tmp_path, tags)
    src = tmp_path / "remote" / c.CALIBRATION_ARTIFACT
    src.write_text(json.dumps({"temperature": 0.0}))
    tags[c.CALIBRATION_HASH_TAG] = d._file_hash(src)
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
    # No model-lineage aux artifacts remain; an empty map is expected.
    aux_pins: dict[str, str] = {}
    tags = c.build_lineage_tags(base_pins, aux_pins)

    assert tags["base_linear_version"] == "3"
    assert tags["base_linear_scaler_uri"] == "runs:/run-linear/linear_scaler.pkl"
    assert "base_gbdt_scaler_uri" not in tags  # only linear has a scaler
    assert tags["base_gbdt_run_id"] == "run-gbdt"
    assert tags["base_nn_model_uri"] == "runs:/run-nn/nn_model"
    # No aux lineage tags are produced.
    assert not any(key.startswith("aux_") for key in tags)
    # Navigation artifacts (similarity index/metadata, player directory) are not
    # model lineage: build_lineage_tags never emits their tags.
    for nav_key in (
        "aux_similarity_index_uri",
        "aux_similarity_metadata_uri",
        "aux_player_directory_uri",
    ):
        assert nav_key not in tags


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
