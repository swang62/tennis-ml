"""Build the promoted Bento and publish it to Docker Hub for multiple platforms."""

import contextlib
import hashlib
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import pandas as pd

from src.config import suppress_insecure_tls_warning
from src.constants import (
    AUX_TAG_PREFIX,
    BASE_TAG_PREFIX,
    CALIBRATION_ARTIFACT,
    CALIBRATION_HASH_TAG,
    CALIBRATION_URI_TAG,
    CHAMPION_ALIAS,
    DATA_PROCESSED,
    DEPLOY_ARTIFACTS,
    FEATURE_COLS_HASH_TAG,
    FEATURE_COLS_TAG,
    FRAMEWORK_KEY,
    FROZEN_ARTIFACTS,
    IMAGE_NAME,
    LINEAGE_AUX_KEYS,
    LINEAGE_BASE_KEYS,
    LINEAGE_MODEL_NAME_KEY,
    LINEAGE_MODEL_URI_KEY,
    LINEAGE_RUN_ID_KEY,
    LINEAGE_SCALER_KEYS,
    LINEAGE_VERSION_KEY,
    LOGS,
    PRODUCTION_MODEL,
    ROOT,
    SIM_BIO_PCA_DIM,
    SIM_BIO_WEIGHT,
    SIM_EXPERIENCE_K,
    SIM_IDENTITY_WEIGHT,
    SIM_PLAYSTYLE_WEIGHT,
    SIM_RANK_SCALE,
    SIM_REPUTATION_WEIGHT,
    SIM_SURFACE_SHRINK_K,
    SIM_SURFACE_WEIGHT,
    load_env,
)
from src.features.columns import FEATURE_COLS

# --- Deploy-only paths and names ---
TEMPLATE_BENTOFILE = ROOT / "bentofile.yaml"
SERVICE_FILE = ROOT / "src" / "serving" / "service.py"
PINNED_BENTOFILE = DATA_PROCESSED / "bentofile.pinned.yaml"
BENTO_TAG_FILE = DATA_PROCESSED / "bento_tag.txt"
STATE_FILE = DATA_PROCESSED / "bento_build_state.json"
# Snapshot-input hash and source/config fingerprint of the last staged
# similarity build (FAISS index + metadata). Kept separate from the bento
# state so similarity reuse never mixes with model lineage. Missing,
# invalid, or legacy (no source fingerprint) -> treated as a rebuild.
SIMILARITY_STATE_FILE = DATA_PROCESSED / "similarity_artifacts_state.json"

# Repository-controlled sources whose exact contents shape the similarity
# outputs without changing snapshot data: the player-list SQL/shaping that
# supplies the index rows (serving/directory.py), the similarity vector
# construction, embedding/PCA, weights (models/similarity.py), and the
# canonical country/player mapping (countries.py). The similarity-tuning
# constants live in constants.py next to unrelated settings, so only their
# exact values are fingerprinted, not the whole file.
SIMILARITY_SOURCE_FILES = [
    ROOT / "src" / "serving" / "directory.py",
    ROOT / "src" / "training" / "similarity.py",
    ROOT / "src" / "utils" / "countries.py",
]

# Multi-architecture publishing: one Docker Hub manifest list for both platforms.
MULTIARCH_PLATFORMS = ("linux/amd64", "linux/arm64")
# Named docker-container Buildx builder reused across deploys (keeps its cache).
BUILDX_BUILDER = "tennis-multiarch"
# Containerfile rendered from the current Bento, written fresh on every deploy.
BENTO_CONTAINERFILE = DEPLOY_ARTIFACTS / "Containerfile.bento"

# nn_best is exported to ONNX and included as an artifact, not a BentoModel.
BASE_BENTO_NAMES = {"linear": "linear_best", "gbdt": "gbdt_best", "nn": "nn_best"}

# BentoML model-metadata keys; only deploy.py writes them, so they stay local
# (the shared FRAMEWORK_KEY comes from src.constants for the serving manifest).
MLFLOW_URI_META_KEY = "mlflow_uri"
MLFLOW_VERSION_META_KEY = "mlflow_version"

# Model artifacts are materialized into DEPLOY_ARTIFACTS from champion lineage;
# similarity artifacts are freshly rebuilt there from the training snapshot.
NN_ONNX_FILE = DEPLOY_ARTIFACTS / "nn_best.onnx"
# Packaged index for offline /similar_players requests.
SIMILARITY_INDEX = DEPLOY_ARTIFACTS / "player_similarity.index"
SIMILARITY_METADATA = DEPLOY_ARTIFACTS / "player_metadata.json"
# Baked into the image; written at build time from the champion's exact lineage
# tags (see _write_model_info). Excluded from the build-input fingerprint.
MODEL_INFO_FILE = DEPLOY_ARTIFACTS / "model_info.json"
AUX_FILES = [
    *[DEPLOY_ARTIFACTS / name for name in FROZEN_ARTIFACTS],
    SIMILARITY_INDEX,
    SIMILARITY_METADATA,
    NN_ONNX_FILE,
]

# Files whose content is a build input but is NOT pinned in champion lineage.
# Lineage-pinned artifacts (bases, scaler, and embeddings) enter the fingerprint
# through the champion's exact tags. nn_best.onnx is a deploy-time export of the
# pinned nn version and model_info.json is generated from the fingerprint itself,
# so both are excluded too. The packaged runtime feature inputs below
# can change predictions without changing service.py, so they are fingerprinted
# directly.
SOURCE_FINGERPRINT_FILES = [
    TEMPLATE_BENTOFILE,
    SERVICE_FILE,
    # Added: these packaged runtime inputs can change predictions without changing service.py.
    ROOT / "src" / "features" / "columns.py",
    ROOT / "src" / "features" / "inference.py",
    ROOT / "src" / "features" / "tour_averages.py",
    ROOT / "src" / "constants.py",
    # Rest of the runtime source closure shipped in the Bento image.
    ROOT / "src" / "utils" / "countries.py",
    ROOT / "src" / "db" / "client.py",
    ROOT / "src" / "db" / "migrate_db.py",
    ROOT / "infra" / "postgres" / "schema.sql",
    ROOT / "src" / "utils" / "config.py",
    ROOT / "src" / "training" / "similarity.py",
    ROOT / "src" / "training" / "nn.py",
]

# .env is loaded by src.constants before the inline settings are read.
load_env()
suppress_insecure_tls_warning()

# Application image tag comes from root .env (same DOCKER_TAG the compose
# stack and `just deploy` web build use); MLflow pins determine the packaged
# model versions.
DOCKER_REPO = os.getenv("DOCKER_REPO", "swang62")
DOCKER_TAG = os.getenv("DOCKER_TAG", "dev")


def _log(category: str, message: str) -> None:
    colors = {"bento": "\033[32m", "similarity": "\033[35m"}
    color = (
        colors.get(category, "")
        if sys.stdout.isatty() or os.getenv("COURTSIDE_COLOR") == "1"
        else ""
    )
    reset = "\033[0m" if color else ""
    print(f"{color}[{category}]{reset} {message}")


def _latest_production_version(client: Any) -> Any:
    """Return the version resolved by `ensemble_lr_model@champion`, if any."""
    from mlflow.exceptions import MlflowException

    try:
        return client.get_model_version_by_alias(PRODUCTION_MODEL, CHAMPION_ALIAS)
    except MlflowException:
        return None


def _read_state() -> dict[str, Any]:
    try:
        return json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_state(state: dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state) + "\n")


def _similarity_inputs_hash(
    profiles: pd.DataFrame,
    lifetime: pd.DataFrame,
) -> str:
    """SHA-256 over every snapshot input the similarity build reads."""
    payload = {
        # Row order of profiles shapes the artifacts (index row order) and is
        # preserved; lifetime rows only feed player-keyed merges, so they are
        # sorted for a canonical hash regardless of snapshot row order.
        "profiles": json.loads(profiles.to_json(orient="records")),
        "lifetime": _frame_records(lifetime),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Canonical JSON records of a frame, sorted by player_id when present."""
    if "player_id" in frame.columns:
        frame = frame.sort_values("player_id")
    return json.loads(frame.to_json(orient="records"))


def _read_similarity_inputs_hash() -> str | None:
    """The persisted inputs hash of the last staged similarity build, else None."""
    try:
        return json.loads(SIMILARITY_STATE_FILE.read_text())["inputs_hash"]
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        return None


def _similarity_source_hash() -> str:
    """SHA-256 over every repository-controlled source/config input that can
    change the similarity outputs without changing snapshot data.

    Hashes the exact contents of SIMILARITY_SOURCE_FILES (relative path + SHA-256
    per file, mirroring the bento source fingerprint) plus the exact values of
    the similarity-tuning constants. A missing allowlisted file aborts loudly
    rather than silently weakening the fingerprint.
    """
    hasher = hashlib.sha256()
    for path in SIMILARITY_SOURCE_FILES:
        hasher.update(f"{path.relative_to(ROOT)}:{_file_hash(path)}\n".encode())
    constants = {
        "sim": {
            "identity_weight": SIM_IDENTITY_WEIGHT,
            "playstyle_weight": SIM_PLAYSTYLE_WEIGHT,
            "surface_weight": SIM_SURFACE_WEIGHT,
            "reputation_weight": SIM_REPUTATION_WEIGHT,
            "bio_weight": SIM_BIO_WEIGHT,
            "bio_pca_dim": SIM_BIO_PCA_DIM,
            "surface_shrink_k": SIM_SURFACE_SHRINK_K,
            "rank_scale": SIM_RANK_SCALE,
            "experience_k": SIM_EXPERIENCE_K,
        }
    }
    hasher.update(json.dumps(constants, sort_keys=True, separators=(",", ":")).encode())
    return hasher.hexdigest()


def _read_similarity_source_hash() -> str | None:
    """The persisted source fingerprint of the last staged build, else None.

    A state written before the source fingerprint existed has no
    ``source_hash`` key and so reads as None, forcing a rebuild.
    """
    try:
        return json.loads(SIMILARITY_STATE_FILE.read_text())["source_hash"]
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        return None


def _write_similarity_state(inputs_hash: str, source_hash: str) -> None:
    SIMILARITY_STATE_FILE.write_text(
        json.dumps({"inputs_hash": inputs_hash, "source_hash": source_hash}) + "\n"
    )


def _lineage_pins(client: Any, production: Any) -> dict[str, dict[str, str]]:
    """Resolve exact base pins from the champion model-version lineage tags.

    05_evaluate tags the promoted ensemble version with the exact registered
    name, version, run ID, and model URI of every base model, plus immutable
    scaler/embedding artifact URIs and content hashes. Base models
    carry no aliases — these tags are the only resolution authority.
    """
    version = client.get_model_version(PRODUCTION_MODEL, production.version)
    tags = dict(version.tags)
    tagged_features = tags.get(FEATURE_COLS_TAG)
    tagged_hash = tags.get(FEATURE_COLS_HASH_TAG)
    current_hash = _feature_cols_hash(list(FEATURE_COLS))
    try:
        tagged_columns = json.loads(tagged_features) if tagged_features is not None else None
    except (TypeError, ValueError, json.JSONDecodeError):
        tagged_columns = None
    if (
        tagged_columns is None
        or tagged_hash != current_hash
        or tagged_columns != list(FEATURE_COLS)
    ):
        # Stale/missing feature contracts used to block the build; warn and keep
        # going so the Docker/Bento image can still be built from the pins.
        _log(
            "bento",
            f"champion {PRODUCTION_MODEL} v{production.version} feature contract "
            "is missing, malformed, or stale versus serving FEATURE_COLS; packaging "
            f"its pinned feature columns. expected contract hash {current_hash}"
            + (f", tagged {tagged_hash}" if tagged_hash else ""),
        )
    missing = [
        f"{BASE_TAG_PREFIX}{cls}_{key}"
        for cls in BASE_BENTO_NAMES
        for key in LINEAGE_BASE_KEYS
        if f"{BASE_TAG_PREFIX}{cls}_{key}" not in tags
    ]
    if missing:
        raise RuntimeError(
            f"champion {PRODUCTION_MODEL} v{production.version} is missing lineage "
            f"tags {sorted(missing)} — promote through 05_evaluate first. Nothing to build."
        )
    pins: dict[str, dict[str, str]] = {
        "production": {
            LINEAGE_MODEL_NAME_KEY: PRODUCTION_MODEL,
            LINEAGE_VERSION_KEY: str(production.version),
            LINEAGE_RUN_ID_KEY: production.run_id,
            LINEAGE_MODEL_URI_KEY: f"models:/{PRODUCTION_MODEL}/{production.version}",
        }
    }
    for cls in BASE_BENTO_NAMES:
        pins[cls] = {key: tags[f"{BASE_TAG_PREFIX}{cls}_{key}"] for key in LINEAGE_BASE_KEYS}
        for key in LINEAGE_SCALER_KEYS:
            tag_key = f"{BASE_TAG_PREFIX}{cls}_{key}"
            if tag_key in tags:
                pins[cls][key] = tags[tag_key]
    # The GBDT framework picks the native serving adapter (bentoml.xgboost vs
    # bentoml.lightgbm); detect it once here so the manifest and the
    # materialization step both know it.
    cached_framework = _cached_gbdt_framework(pins["gbdt"][LINEAGE_VERSION_KEY])
    pins["gbdt"][FRAMEWORK_KEY] = cached_framework or _gbdt_framework(
        pins["gbdt"][LINEAGE_MODEL_URI_KEY]
    )
    return pins


def _write_model_info(
    client: Any,
    production: Any,
    pins: dict[str, dict[str, str]],
    fingerprint: str,
    calibration_temperature: float = 1.0,
) -> Path:
    """Write the immutable canonical champion manifest baked into the Bento.

    Built directly from the champion's exact lineage tags plus the
    non-circular build-input fingerprint: champion identity and creation time,
    exact base and auxiliary-artifact pins, and the fingerprint. It never
    contains the Bento tag, Docker identity, or any hash that includes the
    generated manifest itself. The calibrated temperature is embedded (1.0 for
    legacy champions) so serving reads it straight from the manifest.
    """
    version = client.get_model_version(PRODUCTION_MODEL, production.version)
    tags = version.tags
    manifest = {
        "champion": {
            LINEAGE_MODEL_NAME_KEY: PRODUCTION_MODEL,
            LINEAGE_VERSION_KEY: str(production.version),
            LINEAGE_RUN_ID_KEY: production.run_id,
            LINEAGE_MODEL_URI_KEY: f"models:/{PRODUCTION_MODEL}/{production.version}",
            "creation_timestamp_ms": int(version.creation_timestamp),
        },
        "bases": {cls: pins[cls] for cls in BASE_BENTO_NAMES},
        "aux_artifacts": {key: tags[f"{AUX_TAG_PREFIX}{key}"] for key in LINEAGE_AUX_KEYS},
        "build_input_fingerprint": fingerprint,
    }
    if CALIBRATION_URI_TAG in tags and CALIBRATION_HASH_TAG in tags:
        manifest["calibration"] = {
            "uri": tags.get(CALIBRATION_URI_TAG),
            "sha256": tags.get(CALIBRATION_HASH_TAG),
            "temperature": calibration_temperature,
        }
    if FEATURE_COLS_TAG in tags and FEATURE_COLS_HASH_TAG in tags:
        manifest["feature_contract"] = {
            "columns": json.loads(tags[FEATURE_COLS_TAG]),
            "sha256": tags[FEATURE_COLS_HASH_TAG],
        }
    MODEL_INFO_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODEL_INFO_FILE.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote canonical manifest: {MODEL_INFO_FILE}")
    return MODEL_INFO_FILE


def _download_aux_artifacts(client: Any, tags: dict[str, str]) -> float:
    """Download the champion's lineage-pinned aux artifacts into DEPLOY_ARTIFACTS.

    Returns the resolved calibration temperature (pinned value for a tagged
    champion, 1.0 for a legacy champion) so the caller can embed it in
    model_info.json.

    Three model-affecting artifacts are pinned on the champion model version via
    URI+hash lineage tags. A local copy is reused when its content hash
    already matches the pin; otherwise the artifact is downloaded from its
    exact URI (one retry on a transient download failure) and verified
    against its content hash before the build proceeds. A champion missing
    any required tag is not deployable and must be re-promoted from a full
    training run. The temperature-scaling calibration artifact is materialized
    separately by _materialize_calibration (optional for legacy champions).
    Similarity artifacts are rebuilt from the DuckDB snapshot at
    deploy time and are never champion-pinned.
    """
    import mlflow

    specs = [
        ("base_linear_scaler_uri", "base_linear_scaler_hash", "linear_scaler.pkl"),
        ("aux_embeddings_uri", "aux_embeddings_hash", "bio_embeddings.npz"),
        ("aux_bio_feature_cols_uri", "aux_bio_feature_cols_hash", "bio_feature_cols.json"),
    ]
    DEPLOY_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for uri_tag, hash_tag, filename in specs:
        missing = [tag for tag in (uri_tag, hash_tag) if tag not in tags]
        if missing:
            raise RuntimeError(
                f"champion {PRODUCTION_MODEL} is missing lineage tags {missing} — "
                "re-promote from a full training run (`just train`)"
            )
        local = DEPLOY_ARTIFACTS / filename
        if local.exists() and _file_hash(local) == tags[hash_tag]:
            if filename == "linear_scaler.pkl":
                _validate_scaler_file(local)
            print(f"Reusing {filename} (hash ok)")
            continue
        download_error: Exception | None = None
        for attempt in range(2):
            try:
                local = Path(
                    mlflow.artifacts.download_artifacts(  # type: ignore[reportPrivateImportUsage]  # conditional export
                        tags[uri_tag], dst_path=str(DEPLOY_ARTIFACTS)
                    )
                )
                break
            except Exception as exc:
                download_error = exc
                if attempt == 0:
                    print(f"Download of {filename} from {tags[uri_tag]} failed; retrying once")
        else:
            raise RuntimeError(
                f"failed to download {filename} from {tags[uri_tag]}: {download_error}"
            ) from download_error
        actual = _file_hash(local)
        if actual != tags[hash_tag]:
            raise RuntimeError(
                f"sha256 mismatch for {filename}: expected {tags[hash_tag]}, got {actual}"
            )
        if filename == "linear_scaler.pkl":
            _validate_scaler_file(local)
        print(f"Downloaded {filename} from {tags[uri_tag]} (sha256 ok)")
    return _materialize_calibration(client, tags)


def _validate_calibration_file(path: Path) -> float:
    """Parse calibration_t.json and return the temperature, raising on invalid content."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"invalid calibration file {path}: {exc}") from exc
    temperature = payload.get("temperature") if isinstance(payload, dict) else None
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise RuntimeError(
            f"invalid calibration file {path}: expected a JSON object with a strictly "
            "positive, finite numeric 'temperature'"
        )
    return float(temperature)


def _materialize_calibration(client: Any, tags: dict[str, str]) -> float:  # noqa: ARG001 — client kept for caller contract
    """Resolve and verify the calibration temperature from the champion's tags.

    New champions carry CALIBRATION_URI_TAG/CALIBRATION_HASH_TAG lineage tags
    pinning the temperature-scaling artifact; it is downloaded and hash-verified
    like the aux artifacts, and its temperature is returned for embedding in
    model_info.json. Legacy champions carry no tags: return the explicit no-op
    1.0 — never a failure, never an invented champion pin.
    """
    import mlflow

    uri = tags.get(CALIBRATION_URI_TAG)
    hash_tag = tags.get(CALIBRATION_HASH_TAG)
    if uri is None or hash_tag is None:
        # A legacy champion cannot pin calibration; any stale local file from a
        # previous champion's deploy must not leak, so remove it and use 1.0.
        DEPLOY_ARTIFACTS.mkdir(parents=True, exist_ok=True)
        stale = DEPLOY_ARTIFACTS / CALIBRATION_ARTIFACT
        if stale.exists():
            stale.unlink()
        _log("bento", "champion lineage has no calibration tag — serving no-op calibration (t=1.0)")
        return 1.0
    local = DEPLOY_ARTIFACTS / CALIBRATION_ARTIFACT
    DEPLOY_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if local.exists() and _file_hash(local) == hash_tag:
        temperature = _validate_calibration_file(local)
        print("Reusing calibration_t.json (hash ok)")
        return temperature
    download_error: Exception | None = None
    for attempt in range(2):
        try:
            local = Path(
                mlflow.artifacts.download_artifacts(  # type: ignore[reportPrivateImportUsage]  # conditional export
                    uri, dst_path=str(DEPLOY_ARTIFACTS)
                )
            )
            break
        except Exception as exc:
            download_error = exc
            if attempt == 0:
                print(f"Download of {CALIBRATION_ARTIFACT} from {uri} failed; retrying once")
    else:
        raise RuntimeError(
            f"failed to download {CALIBRATION_ARTIFACT} from {uri}: {download_error}"
        ) from download_error
    if _file_hash(local) != hash_tag:
        raise RuntimeError(
            f"sha256 mismatch for {CALIBRATION_ARTIFACT}: expected {hash_tag}, got {_file_hash(local)}"
        )
    temperature = _validate_calibration_file(local)
    print(f"Downloaded {CALIBRATION_ARTIFACT} from {uri} (sha256 ok)")
    return temperature


def _detect_gbdt_framework(raw: Any) -> str:
    """Map a raw GBDT estimator to its native BentoML adapter name."""
    import lightgbm
    import xgboost

    if isinstance(raw, xgboost.XGBModel):
        return "xgboost"
    if isinstance(raw, lightgbm.LGBMModel):
        return "lightgbm"
    raise RuntimeError(f"gbdt_best model is neither XGBoost nor LightGBM: {type(raw).__name__}")


def _gbdt_framework(model_uri: str) -> str:
    """Detect the pinned GBDT framework by loading the MLflow model briefly."""
    import mlflow

    raw = mlflow.pyfunc.load_model(model_uri).get_raw_model()
    framework = _detect_gbdt_framework(raw)
    _log("bento", f"detected GBDT framework: {framework} ({model_uri})")
    return framework


def _cached_gbdt_framework(version: str) -> str | None:
    """Return the cached GBDT framework when it matches the pinned version."""
    try:
        import bentoml

        stored = bentoml.models.get(BASE_BENTO_NAMES["gbdt"])
    except Exception:
        return None
    metadata = stored.info.metadata
    if metadata.get(MLFLOW_VERSION_META_KEY) != str(version):
        return None
    framework = metadata.get(FRAMEWORK_KEY)
    return (
        framework if isinstance(framework, str) and framework in {"xgboost", "lightgbm"} else None
    )


def _is_sklearn_estimator(model: Any) -> bool:
    """Cheap duck-type check: fitted sklearn estimators expose both."""
    return hasattr(model, "get_params") and hasattr(model, "predict")


def _materialize_native_model(
    pin: dict[str, Any], framework: str | None = None
) -> tuple[Any, str | None]:
    """Save a pinned MLflow version as a native BentoML model.

    Linear and the ensemble (sklearn estimators) go through
    bentoml.sklearn.save_model; the GBDT goes through bentoml.xgboost or
    bentoml.lightgbm depending on the detected framework. Returns the saved
    BentoModel (reused when the pinned version is already materialized) and
    the GBDT framework ("xgboost"/"lightgbm", else None).
    """
    import bentoml
    import mlflow

    registered_name = pin[LINEAGE_MODEL_NAME_KEY]
    version = str(pin[LINEAGE_VERSION_KEY])
    uri = f"models:/{registered_name}/{version}"
    try:
        stored = bentoml.models.get(registered_name)
    except Exception:
        stored = None
    metadata = stored.info.metadata if stored is not None else {}
    same_version = metadata.get(MLFLOW_VERSION_META_KEY) == version
    same_framework = (
        registered_name != BASE_BENTO_NAMES["gbdt"] or metadata.get(FRAMEWORK_KEY) == framework
    )
    if stored is not None and same_version and same_framework:
        _log("bento", f"reusing {registered_name} ({stored.tag}, MLflow v{version})")
        return stored, framework

    raw = mlflow.pyfunc.load_model(uri).get_raw_model()
    if registered_name == BASE_BENTO_NAMES["gbdt"]:
        framework = framework or _detect_gbdt_framework(raw)
        metadata = {
            MLFLOW_URI_META_KEY: uri,
            MLFLOW_VERSION_META_KEY: version,
            FRAMEWORK_KEY: framework,
        }
        save = bentoml.xgboost.save_model if framework == "xgboost" else bentoml.lightgbm.save_model
        return save(registered_name, raw, metadata=metadata), framework
    if not _is_sklearn_estimator(raw):
        raise RuntimeError(f"{registered_name} is not an sklearn estimator: {type(raw).__name__}")
    return (
        bentoml.sklearn.save_model(
            registered_name,
            raw,
            metadata={MLFLOW_URI_META_KEY: uri, MLFLOW_VERSION_META_KEY: version},
        ),
        None,
    )


def _mlflow_import_or_reuse(pin: dict[str, Any]) -> Any:
    """Import or reuse a pinned MLflow version by exact version, never alias."""
    import bentoml

    registered_name = pin[LINEAGE_MODEL_NAME_KEY]
    version = str(pin[LINEAGE_VERSION_KEY])
    uri = f"models:/{registered_name}/{version}"
    try:
        stored = bentoml.models.get(registered_name)
    except Exception:
        stored = None
    if stored is not None and stored.info.metadata.get(MLFLOW_VERSION_META_KEY) == version:
        _log("bento", f"reusing {registered_name} ({stored.tag}, MLflow v{version})")
        return stored
    stored = bentoml.mlflow.import_model(
        registered_name,
        uri,
        metadata={MLFLOW_URI_META_KEY: uri, MLFLOW_VERSION_META_KEY: version},
    )
    _log("bento", f"imported {registered_name} ({uri} -> {stored.tag})")
    return stored


def _import_or_reuse(pin: dict[str, Any], framework: str | None = None) -> Any:
    """Materialize a pinned MLflow version as a native BentoML model.

    The nn_best pin is the exception: it is exported to ONNX and never becomes
    a BentoModel, so it still takes the MLflow import path that
    _materialize_nn_onnx relies on.
    """
    if pin[LINEAGE_MODEL_NAME_KEY] == BASE_BENTO_NAMES["nn"]:
        return _mlflow_import_or_reuse(pin)
    return _materialize_native_model(pin, framework)[0]


def _materialize_nn_onnx(nn_pin: dict[str, Any]) -> None:
    """Export the pinned PyTorch nn_best model to the service's ONNX artifact."""
    import logging
    import warnings

    import bentoml
    import torch

    # Needed so torch.load can resolve the class — torch.save records the path.
    from src.training.nn import TabularBioMLP  # type: ignore[reportUnusedImport]

    # The MLP uses standard ONNX ops; suppress unrelated exporter warnings.
    logging.getLogger("torch.onnx").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*LeafSpec.*")

    _log(
        "bento",
        f"materializing nn_best ONNX from models:/"
        f"{nn_pin[LINEAGE_MODEL_NAME_KEY]}/{nn_pin[LINEAGE_VERSION_KEY]}",
    )
    stored = _import_or_reuse(nn_pin)

    pyfunc = bentoml.mlflow.load_model(stored.tag)
    raw = pyfunc.get_raw_model()  # TabularBioMLP
    raw.eval()

    tab_dim = raw.tab_mlp[0].in_features
    bio_dim = raw.bio_mlp[0].in_features
    dummy_tab = torch.zeros(1, tab_dim, dtype=torch.float32)
    dummy_bio_p = torch.zeros(1, bio_dim, dtype=torch.float32)
    dummy_bio_o = torch.zeros(1, bio_dim, dtype=torch.float32)

    NN_ONNX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            raw,
            (dummy_tab, dummy_bio_p, dummy_bio_o),
            str(NN_ONNX_FILE),
            input_names=["tab", "bio_p", "bio_o"],
            output_names=["logit"],
            opset_version=18,
            dynamic_axes={
                "tab": {0: "batch"},
                "bio_p": {0: "batch"},
                "bio_o": {0: "batch"},
                "logit": {0: "batch"},
            },
            # Single-file artifact: keep weights inline so ONNX Runtime needs
            # no nn_best.onnx.data sidecar in the image.
            external_data=False,
        )
    # Drop any stale external-data sidecar from an older export so local
    # rebuilds never pick up an orphan nn_best.onnx.data.
    stale_data = NN_ONNX_FILE.with_suffix(".onnx.data")
    if stale_data.exists():
        stale_data.unlink()
    _log("bento", f"wrote nn_best ONNX: {NN_ONNX_FILE} ({NN_ONNX_FILE.stat().st_size} bytes)")


def _reuse_or_materialize_nn_onnx(state: dict[str, Any], nn_pin: dict[str, Any]) -> bool:
    """Reuse an ONNX export only when it came from the exact pinned NN model."""
    if NN_ONNX_FILE.exists() and state.get("nn_onnx_model_uri") == nn_pin[LINEAGE_MODEL_URI_KEY]:
        _log("bento", f"reusing nn_best ONNX for {nn_pin[LINEAGE_MODEL_URI_KEY]}")
        return True
    _materialize_nn_onnx(nn_pin)
    return False


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _feature_cols_hash(columns: list[str]) -> str:
    payload = json.dumps(columns, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _validate_feature_contract(estimator: Any, artifact_name: str) -> None:
    """Reject a fitted artifact whose named inputs differ from serving."""
    fitted = getattr(estimator, "feature_names_in_", None)
    if fitted is None:
        return
    actual = [str(name) for name in fitted]
    expected = list(FEATURE_COLS)
    if actual != expected:
        missing = [name for name in actual if name not in expected]
        added = [name for name in expected if name not in actual]
        raise RuntimeError(
            f"{artifact_name} feature contract mismatch: fitted={len(actual)}, "
            f"serving={len(expected)}, missing_from_serving={missing}, "
            f"missing_from_model={added}; retrain before deploy"
        )


def _validate_scaler_file(path: Path) -> None:
    try:
        with path.open("rb") as artifact:
            _validate_feature_contract(pickle.load(artifact), path.name)
    except (EOFError, pickle.UnpicklingError):
        # Mocked tests may use arbitrary bytes for download/hash behavior.
        return


def build_input_fingerprint(client: Any, production: Any) -> str:
    """Canonical, non-circular fingerprint of every Bento build input.

    Includes the champion's exact lineage tags (base versions, run IDs, model
    URIs, and scaler/embedding artifact URIs + content hashes) and
    hashes of on-disk source/artifact build inputs. Excludes everything the
    build generates or records after the fact — the pinned bentofile, the
    nn_best.onnx export, the Bento tag, the Docker image identity,
    timestamps, and deploy state — so the fingerprint never depends on its
    own output.
    """
    version = client.get_model_version(PRODUCTION_MODEL, production.version)
    parts = [f"{PRODUCTION_MODEL}@{CHAMPION_ALIAS}=v{production.version}"]
    parts.append("lineage:")
    parts.extend(f"  {key}={version.tags[key]}" for key in sorted(version.tags))
    parts.append("sources:")
    parts.extend(
        f"  {path.relative_to(ROOT)}:{_file_hash(path)}" for path in SOURCE_FINGERPRINT_FILES
    )
    return "\n".join(parts)


def _import_models(pins: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Materialize each pinned MLflow version as a native BentoModel; return tags.

    Reuse is version-keyed via `_import_or_reuse` — the BentoML store name
    alone cannot gate reuse, so the pinned MLflow version is stored in the
    saved model's metadata.
    """
    tags: dict[str, str] = {}
    for key, pin in pins.items():
        framework = pin.get(FRAMEWORK_KEY) if key == "gbdt" else None
        tags[key] = str(_import_or_reuse(pin, framework).tag)
    return tags


def _write_pinned_bentofile(tags: dict[str, str]) -> Path:
    """Template bentofile with the exact imported model tags, for this build."""
    import yaml

    config = yaml.safe_load(TEMPLATE_BENTOFILE.read_text())
    # nn_best is not a BentoModel dep anymore — served via data/processed/nn_best.onnx.
    config["models"] = [tags[key] for key in ["linear", "gbdt", "production"]]
    PINNED_BENTOFILE.parent.mkdir(parents=True, exist_ok=True)
    PINNED_BENTOFILE.write_text(yaml.safe_dump(config, sort_keys=False))
    return PINNED_BENTOFILE


def _docker_login() -> None:
    """Authenticate `docker push` to Docker Hub before pushing.

    Reads DOCKER_TOKEN from the environment and logs in via
    `docker login --username <user> --password-stdin` so the token is passed
    through stdin and never appears in argv, logs, or raised exceptions. The
    username comes from DOCKER_USERNAME, else the DOCKER_REPO owner (default
    `swang62`). When DOCKER_TOKEN is unset, skip login and rely on an
    already-authenticated Docker CLI (the prior convention).
    """
    token = os.getenv("DOCKER_TOKEN")
    if not token:
        print("DOCKER_TOKEN unset — relying on an already-authenticated Docker CLI.")
        return
    username = os.getenv("DOCKER_USERNAME") or DOCKER_REPO or "swang62"
    print(f"Authenticating Docker Hub as {username} (token read via stdin).")
    proc = subprocess.run(
        ["docker", "login", "--username", username, "--password-stdin"],
        input=token + "\n",
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"docker login failed (exit {proc.returncode})")


class _Tee:
    """Write to two streams: the real console and a capture file."""

    def __init__(self, console: TextIO, log: TextIO) -> None:
        self.console = console
        self.log = log

    def write(self, s: str) -> int:
        self.log.write(s)
        return self.console.write(s)

    def flush(self) -> None:
        self.console.flush()
        self.log.flush()

    def isatty(self) -> bool:
        """Preserve terminal detection while stdout/stderr are redirected."""
        return self.console.isatty()


def _run_teed(
    cmd: list[str], log: TextIO | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[bytes]:
    """Run a deploy subprocess, streaming its output to the console AND a log file.

    With `log` None the output still reaches the console — and, inside
    deploy_bento's redirect, the deploy log — via sys.stdout.
    """
    proc = subprocess.Popen(
        cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        text = line.decode(errors="replace")
        # Docker push/buildx re-renders the full layer status table on every
        # change, spamming `Waiting` for layers that are already cached; drop those.
        if text.strip().endswith(": Waiting"):
            continue
        sys.stdout.write(text)
        sys.stdout.flush()
        if log is not None:
            log.write(text)
            log.flush()
    returncode = proc.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)
    return subprocess.CompletedProcess(cmd, returncode)


def _buildx_build_cmd(*, builder: str, containerfile: Path, context: Path, image: str) -> list[str]:
    """The exact `docker buildx build` invocation publishing the image.

    The Buildx image is tagged with DOCKER_TAG (default dev) and pushed via
    `--push` as part of the build, so the image is never separately tagged or
    pushed afterwards.
    """
    return [
        "docker",
        "buildx",
        "build",
        "--builder",
        builder,
        "--file",
        str(containerfile),
        "--platform",
        ",".join(MULTIARCH_PLATFORMS),
        "--tag",
        image,
        "--push",
        str(context),
    ]


def _ensure_buildx_builder() -> str:
    """Return the docker-container Buildx builder, creating it on first use.

    Multi-platform builds require the docker-container driver; the default
    docker driver cannot build or push multi-arch manifests. Reusing the
    named builder keeps its layer cache across deploys.
    """
    if (
        subprocess.run(
            ["docker", "buildx", "inspect", BUILDX_BUILDER],
            cwd=ROOT,
            capture_output=True,
        ).returncode
        != 0
    ):
        print(f"Creating Buildx builder {BUILDX_BUILDER} (docker-container driver)...")
        subprocess.run(
            [
                "docker",
                "buildx",
                "create",
                "--name",
                BUILDX_BUILDER,
                "--driver",
                "docker-container",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
    return BUILDX_BUILDER


def _write_bento_containerfile(bento: Any) -> Path:
    """Render BentoML's Containerfile and install the serving uv group."""
    import bentoml

    BENTO_CONTAINERFILE.parent.mkdir(parents=True, exist_ok=True)
    bentoml.container.get_containerfile(str(bento.tag), output_path=str(BENTO_CONTAINERFILE))
    containerfile = BENTO_CONTAINERFILE.read_text()
    generated_install = (
        "RUN  --mount=type=cache,sharing=locked,target=/root/.cache/ if [ -d ./src ]; "
        'then INSTALL_ROOT="./src"; '
        'else INSTALL_ROOT="./env/python"; '
        "fi; uv --directory $INSTALL_ROOT pip install -r $BENTO_PATH/env/python/requirements.txt"
    )
    uv_install = (
        "COPY --chown=bentoml:bentoml pyproject.toml uv.lock ./\n"
        "RUN  --mount=type=cache,sharing=locked,target=/root/.cache/ "
        "uv export --only-group inference --no-dev --no-emit-project --no-hashes "
        "-o /tmp/requirements.txt && uv pip install -r /tmp/requirements.txt"
    )
    if generated_install not in containerfile:
        raise RuntimeError("BentoML Containerfile format changed; cannot configure uv export")
    BENTO_CONTAINERFILE.write_text(containerfile.replace(generated_install, uv_install))
    print(f"Wrote Containerfile: {BENTO_CONTAINERFILE}")
    return BENTO_CONTAINERFILE


@contextlib.contextmanager
def _buildx_context(bento: Any):
    """Yield a portable build context: the Bento content plus its models.

    A stored Bento keeps its models in the global model store, not under
    bento.path/models, so the Containerfile's `COPY . ./` would miss them.
    Mirror BentoML's construct_containerfile: copy the Bento and materialize
    each model into the context so it really contains the whole Bento.
    """
    import bentoml

    with tempfile.TemporaryDirectory(prefix="bento-buildx-") as tmp:
        context = Path(tmp)
        shutil.copytree(bento.path, context, dirs_exist_ok=True)
        shutil.copy2(ROOT / "pyproject.toml", context / "pyproject.toml")
        shutil.copy2(ROOT / "uv.lock", context / "uv.lock")
        models_dir = context / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        for model in bento.info.all_models:
            stored = bentoml.models.get(model.tag)
            target = models_dir / str(stored.tag.name) / str(stored.tag.version)
            shutil.copytree(stored.path, target, dirs_exist_ok=True)
        yield context


def build_bento_image() -> tuple[str, int]:
    """Build the promoted Bento and publish the multi-arch Docker Hub image."""
    import bentoml
    from mlflow.tracking.client import MlflowClient

    client = MlflowClient()
    production = _latest_production_version(client)
    if production is None:
        raise RuntimeError("No promoted 'ensemble_lr_model' — nothing to build.")

    state = _read_state()
    pins = _lineage_pins(client, production)
    _reuse_or_materialize_nn_onnx(state, pins["nn"])
    version_tags = client.get_model_version(PRODUCTION_MODEL, production.version).tags
    calibration_temperature = _download_aux_artifacts(client, version_tags)
    generate_similarity_artifacts()
    fingerprint = build_input_fingerprint(client, production)
    _write_model_info(client, production, pins, fingerprint, calibration_temperature)

    tags = _import_models(pins)
    pinned = _write_pinned_bentofile(tags)
    bento = bentoml.bentos.build_bentofile(bentofile=str(pinned), build_ctx=str(ROOT))
    tag = str(bento.tag)
    BENTO_TAG_FILE.write_text(tag)
    _write_state(
        {
            **state,
            "fingerprint": fingerprint,
            "tag": tag,
            "nn_onnx_model_uri": pins["nn"][LINEAGE_MODEL_URI_KEY],
        }
    )
    print(f"Built {tag} from {pinned}")

    image = f"{DOCKER_REPO}/{IMAGE_NAME}:{DOCKER_TAG}"
    containerfile = _write_bento_containerfile(bento)
    builder = _ensure_buildx_builder()
    print(f"Containerizing {tag} -> {image} with Docker Buildx...")
    with _buildx_context(bento) as context:
        _run_teed(
            _buildx_build_cmd(
                builder=builder, containerfile=containerfile, context=context, image=image
            )
        )
    return image, int(production.version)


def generate_similarity_artifacts() -> Path:
    """Build and stage the FAISS similarity assets from the local DuckDB snapshot.

    Stages ``data/deploy/player_similarity.index`` and
    ``data/deploy/player_metadata.json`` from the snapshot — never downloaded
    from MLflow or pinned on the champion. A missing snapshot aborts the
    deploy with the snapshot helper's actionable error.

    When every snapshot input that shapes the similarity outputs (profiles,
    gold lifetime stats) and every repository-controlled source that shapes
    them (player-list SQL/shaping, similarity vector/embedding/PCA/weights,
    country mapping) is unchanged since the last staging and the staged
    similarity artifacts still exist, the artifacts are reused. Any input
    change, source/config change, missing artifact, or missing/legacy state
    rebuilds them.
    """
    from src.db import training
    from src.serving.directory import PLAYERS_SQL, directory_players
    from src.training.similarity import PLAYER_LIFETIME_SQL, PlayerSimilarity

    profiles = training.to_dataframe(PLAYERS_SQL)
    if profiles.empty:
        raise RuntimeError("training snapshot has no player profiles; refresh it before deploy")
    inputs_hash = _similarity_inputs_hash(profiles, training.to_dataframe(PLAYER_LIFETIME_SQL))
    source_hash = _similarity_source_hash()
    if (
        inputs_hash == _read_similarity_inputs_hash()
        and source_hash == _read_similarity_source_hash()
        and all(path.exists() for path in (SIMILARITY_INDEX, SIMILARITY_METADATA))
    ):
        _log(
            "similarity",
            "snapshot inputs and shaping sources unchanged; reusing staged similarity artifacts",
        )
        return SIMILARITY_INDEX
    _log(
        "similarity",
        f"snapshot inputs or sources changed: rebuilding similarity index for "
        f"{len(profiles)} players (bio embedding — no output until it finishes)",
    )
    similarity = PlayerSimilarity()
    similarity.build(
        query=training.to_dataframe,
        profiles=profiles,
        index_path=SIMILARITY_INDEX,
        metadata_path=SIMILARITY_METADATA,
    )
    players = directory_players(profiles)
    metadata = {str(player["player_id"]): player for player in similarity.players}
    if set(metadata) != {str(player["player_id"]) for player in players}:
        raise RuntimeError(
            "similarity index and directory player IDs differ in the training snapshot"
        )
    _write_similarity_state(inputs_hash, source_hash)
    _log(
        "similarity",
        f"staged snapshot-backed similarity artifacts: {SIMILARITY_INDEX}, {SIMILARITY_METADATA}",
    )
    return SIMILARITY_INDEX


def deploy_bento() -> None:
    """Authenticate, then build and publish the multi-arch image with Buildx.

    Build output streams to the console and to a single deploy_<timestamp>.log
    opened before the build, so a failed build still leaves a log of the
    attempt. Docker login runs before the build because Buildx's `--push` is
    part of the build command.
    """
    LOGS.mkdir(parents=True, exist_ok=True)
    deploy_log = LOGS / f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    try:
        with (
            deploy_log.open("w") as log,
            redirect_stdout(_Tee(sys.stdout, log)),
            redirect_stderr(_Tee(sys.stderr, log)),
        ):
            _docker_login()
            _image, production_version = build_bento_image()
    except subprocess.CalledProcessError as exc:
        print(
            f"Deploy step failed ({exc}); image was not published: "
            f"{DOCKER_REPO}/{IMAGE_NAME}:{DOCKER_TAG}"
        )
        raise

    image = f"{DOCKER_REPO}/{IMAGE_NAME}:{DOCKER_TAG}"
    state = _read_state()
    _write_state({**state, "deployed_version": production_version, "deployed_image": image})
    import mlflow
    from mlflow.tracking.client import MlflowClient

    mlflow.set_experiment("model-deployment")
    mlflow.set_experiment_tag("pipeline", "deploy")
    with mlflow.start_run(
        run_name="deploy-champion-model",
        tags={"pipeline": "deploy"},
    ) as run:
        mlflow.log_param("production_model", PRODUCTION_MODEL)
        mlflow.log_param("production_version", production_version)
        mlflow.log_param("image", image)
        mlflow.log_artifact(str(deploy_log))
        client = MlflowClient()
        client.set_model_version_tag(
            PRODUCTION_MODEL,
            str(production_version),
            "pipeline_last_deploy_run_id",
            run.info.run_id,
        )
        client.set_model_version_tag(
            PRODUCTION_MODEL, str(production_version), "pipeline_last_deploy_image", image
        )
    print(f"Published {image} to Docker Hub")


if __name__ == "__main__":
    deploy_bento()
