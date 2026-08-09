"""Build the promoted Bento image locally and push it to Docker Hub."""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from prefect import flow

from src.constants import (
    CHAMPION_ALIAS,
    DATA_PROCESSED,
    IMAGE_NAME,
    LOGS,
    PRODUCTION_MODEL,
    ROOT,
    load_env,
)

# --- Deploy-only paths and names ---
TEMPLATE_BENTOFILE = ROOT / "bentofile.yaml"
SERVICE_FILE = ROOT / "src" / "serving" / "service.py"
PINNED_BENTOFILE = DATA_PROCESSED / "bentofile.pinned.yaml"
BENTO_TAG_FILE = DATA_PROCESSED / "bento_tag.txt"
STATE_FILE = DATA_PROCESSED / "bento_build_state.json"

# .env is loaded by src.constants before the inline settings are read.
load_env()

assert IMAGE_NAME is not None, "IMAGE_NAME not set in env; load_env() must be called first"
# Docker Hub uses only `latest`; MLflow and Bento versions remain cache state.
DOCKER_REPO = os.getenv("DOCKER_REPO", "swang62")

# nn_best is exported to ONNX and included as an artifact, not a BentoModel.
BASE_BENTO_NAMES = {"linear": "linear_best", "gbdt": "gbdt_best", "nn": "nn_best"}

# Auxiliary-artifact tag names (champion lineage, `aux_<key>` prefix) baked into
# the canonical manifest.
_AUX_TAG_KEYS = (
    "embeddings_uri",
    "embeddings_hash",
    "bio_feature_cols_uri",
    "bio_feature_cols_hash",
)

# Packaged artifacts; serving reads PostgreSQL live, never training data.
NN_ONNX_FILE = DATA_PROCESSED / "nn_best.onnx"
# Packaged index for offline /similar_players requests.
SIMILARITY_INDEX = DATA_PROCESSED / "player_similarity.index"
SIMILARITY_METADATA = DATA_PROCESSED / "player_metadata.json"
# Baked into the image; written at build time from the champion's exact lineage
# tags (see _write_model_info). Excluded from the build-input fingerprint.
MODEL_INFO_FILE = DATA_PROCESSED / "model_info.json"
AUX_FILES = [
    DATA_PROCESSED / "linear_scaler.pkl",
    DATA_PROCESSED / "bio_embeddings.parquet",
    DATA_PROCESSED / "bio_feature_cols.json",
    NN_ONNX_FILE,
    SIMILARITY_INDEX,
    SIMILARITY_METADATA,
]

# Files whose content is a build input but is NOT pinned in champion lineage.
# Lineage-pinned artifacts (bases, scaler, embeddings, feature contract) enter
# the fingerprint through the champion's exact tags instead of their mutable
# data/processed copies. nn_best.onnx is a deploy-time export of the pinned nn
# version and model_info.json is generated from the fingerprint itself, so both
# are excluded too.
SOURCE_FINGERPRINT_FILES = [
    TEMPLATE_BENTOFILE,
    SERVICE_FILE,
    SIMILARITY_INDEX,
    SIMILARITY_METADATA,
]


@flow(log_prints=True)
def deploy_flow(force: bool = False) -> None:
    """Deploy the promoted model; force bypasses the Bento and image caches."""
    deploy_bento(force=force)


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


def _lineage_pins(client: Any, production: Any) -> dict[str, dict[str, str]]:
    """Resolve exact base pins from the champion model-version lineage tags.

    05_evaluate tags the promoted ensemble version with the exact registered
    name, version, run ID, and model URI of every base model, plus immutable
    scaler/embedding/feature artifact URIs and content hashes. Base models
    carry no aliases — these tags are the only resolution authority.
    """
    version = client.get_model_version(PRODUCTION_MODEL, production.version)
    tags = dict(version.tags)
    missing = [
        f"base_{cls}_{key}"
        for cls in BASE_BENTO_NAMES
        for key in ("registered_model_name", "version", "run_id", "model_uri")
        if f"base_{cls}_{key}" not in tags
    ]
    if missing:
        raise RuntimeError(
            f"champion {PRODUCTION_MODEL} v{production.version} is missing lineage "
            f"tags {sorted(missing)} — promote through 05_evaluate first. Nothing to build."
        )
    pins: dict[str, dict[str, str]] = {
        "production": {
            "registered_model_name": PRODUCTION_MODEL,
            "version": str(production.version),
            "run_id": production.run_id,
            "model_uri": f"models:/{PRODUCTION_MODEL}/{production.version}",
        }
    }
    for cls in BASE_BENTO_NAMES:
        pins[cls] = {
            key: tags[f"base_{cls}_{key}"]
            for key in ("registered_model_name", "version", "run_id", "model_uri")
        }
        for key in ("scaler_uri", "scaler_hash"):
            tag_key = f"base_{cls}_{key}"
            if tag_key in tags:
                pins[cls][key] = tags[tag_key]
    return pins


def _write_model_info(
    client: Any, production: Any, pins: dict[str, dict[str, str]], fingerprint: str
) -> Path:
    """Write the immutable canonical champion manifest baked into the Bento.

    Built directly from the champion's exact lineage tags (Task 2) plus the
    non-circular build-input fingerprint: champion identity and creation time,
    exact base and auxiliary-artifact pins, the feature-contract version/schema
    hash, and the fingerprint. It never contains the Bento tag, Docker identity,
    or any hash that includes the generated manifest itself.
    """
    version = client.get_model_version(PRODUCTION_MODEL, production.version)
    tags = version.tags
    manifest = {
        "champion": {
            "registered_model_name": PRODUCTION_MODEL,
            "version": str(production.version),
            "run_id": production.run_id,
            "model_uri": f"models:/{PRODUCTION_MODEL}/{production.version}",
            "creation_timestamp_ms": int(version.creation_timestamp),
        },
        "bases": {cls: pins[cls] for cls in BASE_BENTO_NAMES},
        "aux_artifacts": {key: tags[f"aux_{key}"] for key in _AUX_TAG_KEYS},
        "feature_contract": {
            "version": tags["aux_features_uri"],
            "schema_hash": tags["aux_feature_cols_hash"],
        },
        "build_input_fingerprint": fingerprint,
    }
    MODEL_INFO_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODEL_INFO_FILE.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote canonical manifest: {MODEL_INFO_FILE}")
    return MODEL_INFO_FILE


def _check_aux_files() -> None:
    """The serving artifacts packaged into the Bento must exist on disk."""
    missing = [str(p.relative_to(ROOT)) for p in AUX_FILES if not p.exists()]
    if missing:
        raise RuntimeError(
            "required serving artifacts missing: "
            + ", ".join(missing)
            + " — run the training pipeline (00/02 write these to data/processed; "
            "eda/01_player_similarity builds the similarity index)"
        )


def _import_or_reuse(pin: dict[str, Any]) -> Any:
    """Import or reuse a pinned MLflow version by exact version, never alias."""
    import bentoml

    registered_name = pin["registered_model_name"]
    version = str(pin["version"])
    uri = f"models:/{registered_name}/{version}"
    try:
        stored = bentoml.models.get(registered_name)
    except Exception:
        stored = None
    if stored is not None and stored.info.metadata.get("mlflow_version") == version:
        print(f"Reusing {stored.tag} — already imported (MLflow v{version})")
        return stored
    stored = bentoml.mlflow.import_model(
        registered_name, uri, metadata={"mlflow_uri": uri, "mlflow_version": version}
    )
    print(f"Imported {uri} -> {stored.tag}")
    return stored


def _materialize_nn_onnx(nn_pin: dict[str, Any]) -> None:
    """Export the pinned PyTorch nn_best model to the service's ONNX artifact."""
    import logging
    import warnings

    import bentoml
    import torch

    # Needed so torch.load can resolve the class — torch.save records the path.
    from src.models.nn import TabularBioMLP  # type: ignore[reportUnusedImport]

    # The MLP uses standard ONNX ops; suppress unrelated exporter warnings.
    logging.getLogger("torch.onnx").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*LeafSpec.*")

    print(
        f"Materializing ONNX from models:/{nn_pin['registered_model_name']}/{nn_pin['version']} "
        f"(v{nn_pin['version']})"
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
    print(f"Wrote ONNX: {NN_ONNX_FILE} ({NN_ONNX_FILE.stat().st_size} bytes)")


def _file_hash(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_input_fingerprint(client: Any, production: Any) -> str:
    """Canonical, non-circular fingerprint of every Bento build input.

    Includes the champion's exact lineage tags (base versions, run IDs, model
    URIs, and scaler/embedding/feature artifact URIs + content hashes) and
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
    """Import each pinned MLflow version into the BentoML store; return tags.

    Reuse is version-keyed via `_import_or_reuse` — the BentoML store name
    alone cannot gate reuse, so the pinned MLflow version is stored in the
    imported model's metadata.
    """
    tags: dict[str, str] = {}
    for key, pin in pins.items():
        tags[key] = str(_import_or_reuse(pin).tag)
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


def _cached_tag(state: dict[str, Any], fingerprint: str) -> str | None:
    """Return the previously built tag when the state is unchanged, else None."""
    if state.get("fingerprint") != fingerprint:
        return None
    tag = state.get("tag")
    if not tag:
        return None
    import bentoml

    try:
        bentoml.bentos.get(tag)
    except Exception:
        return None
    return tag


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


def _image_exists(image: str) -> bool:
    return (
        subprocess.run(
            ["docker", "image", "inspect", image],
            cwd=ROOT,
            capture_output=True,
        ).returncode
        == 0
    )


def _run_teed(
    cmd: list[str], log: TextIO, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    """Run a deploy subprocess, streaming its output to the console AND a log file."""
    proc = subprocess.Popen(
        cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        text = line.decode(errors="replace")
        sys.stdout.write(text)
        sys.stdout.flush()
        log.write(text)
        log.flush()
    returncode = proc.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)
    return subprocess.CompletedProcess(cmd, returncode)


def build_bento_image(force: bool = False) -> tuple[str, int]:
    """Build the promoted Bento as `${IMAGE_NAME}:latest`; force skips caches."""
    import bentoml
    from mlflow.tracking.client import MlflowClient

    client = MlflowClient()
    production = _latest_production_version(client)
    if production is None:
        raise RuntimeError("No promoted 'ensemble_lr_model' — nothing to build.")

    state = _read_state()
    pins = _lineage_pins(client, production)
    _materialize_nn_onnx(pins["nn"])
    _check_aux_files()
    fingerprint = build_input_fingerprint(client, production)
    _write_model_info(client, production, pins, fingerprint)

    if force:
        print("Force: rebuilding Bento and image regardless of cache.")
        tag = None
    else:
        tag = _cached_tag(state, fingerprint)
    rebuilt = tag is None
    if rebuilt:
        tags = _import_models(pins)
        pinned = _write_pinned_bentofile(tags)
        bento = bentoml.bentos.build_bentofile(bentofile=str(pinned), build_ctx=str(ROOT))
        tag = str(bento.tag)
        BENTO_TAG_FILE.write_text(tag)
        _write_state({**state, "fingerprint": fingerprint, "tag": tag})
        print(f"Built {tag} from {pinned}")
    else:
        print(f"No Bento rebuild needed — reusing {tag}.")

    image = f"{IMAGE_NAME}:latest"
    containerize = force or rebuilt or not _image_exists(image)
    if containerize:
        print(f"Containerizing {tag} -> {image} with BentoML...")
        bentoml.container.build(tag, backend="docker", image_tag=(image,))
    else:
        print(f"Local image already exists — reusing {image}.")
    return image, int(production.version)


def deploy_bento(force: bool = False) -> None:
    """Build then push the promoted image; token login uses stdin."""
    local_image, production_version = build_bento_image(force=force)

    latest = f"{DOCKER_REPO}/{IMAGE_NAME}:latest"
    _docker_login()
    LOGS.mkdir(parents=True, exist_ok=True)
    deploy_log = LOGS / f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    try:
        with deploy_log.open("w") as log:
            _run_teed(["docker", "push", latest], log)
    except subprocess.CalledProcessError as exc:
        print(f"Deploy step failed ({exc}) — skipping publish; local image is ready: {local_image}")
        return

    state = _read_state()
    _write_state({**state, "deployed_version": production_version, "deployed_image": latest})
    print(f"Published {latest} to Docker Hub")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build and push the production Bento serving image."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild + redeploy of the latest ensemble_lr_model, bypassing the Bento pinned cache.",
    )
    args = parser.parse_args()
    deploy_flow(force=args.force)
