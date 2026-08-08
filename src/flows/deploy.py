"""Build the promoted Bento serving image and push it to Docker Hub.

Deploy is host-executed and independent of k3d: build_bento_image() builds the
Bento into the local Docker engine as `${IMAGE_NAME}:latest`, then
deploy_bento() pushes it to Docker Hub as `${DOCKER_REPO}/${IMAGE_NAME}:latest`.
Docker Compose is NOT part of the deploy flow; the Compose stack is a separate
manual/test workflow (`pnpm docker`).

Usage:
    uv run python src/flows/deploy.py
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from prefect import flow

from src.constants import (
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
# Docker Hub repo: DOCKER_REPO/IMAGE_NAME. Only the moving `latest` Docker
# image tag is ever built or pushed; MLflow version numbers and Bento tags
# stay in state for internal cache/production tracking only.
DOCKER_REPO = os.getenv("DOCKER_REPO", "swang62")

# The BentoML model name each pinned MLflow registered model maps to —
# exactly the names the service references via `bentoml.models.BentoModel`.
# `nn_best` is imported into the BentoML store (for the deploy-time ONNX
# export) but its tag is NOT added to the pinned bentofile's models: the
# service consumes it via the onnx artifact under `include:` instead.
BASE_BENTO_NAMES = {"linear": "linear_best", "gbdt": "gbdt_best", "nn": "nn_best"}

# Serving artifacts packaged into the Bento (written by pipeline notebooks,
# materialized by this flow, or built by the similarity EDA notebook); content
# changes trigger a rebuild. The database is NOT packaged here — production
# serving reads PostgreSQL live through psycopg; training data never enters
# the image.
NN_ONNX_FILE = DATA_PROCESSED / "nn_best.onnx"
# Similarity index + metadata (built by notebooks/eda/01_player_similarity.ipynb),
# packaged so the /similar_players endpoint serves the same index offline.
SIMILARITY_INDEX = DATA_PROCESSED / "player_similarity.index"
SIMILARITY_METADATA = DATA_PROCESSED / "player_metadata.json"
AUX_FILES = [
    DATA_PROCESSED / "linear_scaler.pkl",
    DATA_PROCESSED / "bio_embeddings.parquet",
    DATA_PROCESSED / "bio_feature_cols.json",
    NN_ONNX_FILE,
    SIMILARITY_INDEX,
    SIMILARITY_METADATA,
]

# Files whose content changes should trigger a rebuild.
FINGERPRINT_FILES = [
    TEMPLATE_BENTOFILE,
    SERVICE_FILE,
    *AUX_FILES,
]


@flow(log_prints=True)
def deploy_flow(force: bool = False) -> None:
    """Deploy-only flow: run the single deployment path for the promoted model.

    No-op unless 05 promoted a production version newer than the last
    deployed one. Evaluation/promotion already ran in the training pipeline;
    no notebook runs here. With force=True, bypass the Bento/image cache and
    rebuild + redeploy regardless.
    """
    deploy_bento(force=force)


def _latest_production_version(client: Any) -> Any:
    """Return the version `models:/ensemble_lr_model@champion` resolves to.

    MLflow's `champion` alias points at the promoted production version;
    mirror that exactly instead of guessing. Returns None when no version
    has been aliased yet (no champion).
    """
    from mlflow.exceptions import MlflowException

    try:
        return client.get_model_version_by_alias(PRODUCTION_MODEL, "champion")
    except MlflowException:
        return None


def _read_state() -> dict[str, Any]:
    try:
        return json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_state(state: dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state) + "\n")


def _resolve_pins(client: Any, production: Any) -> dict[str, dict[str, Any]]:
    """Resolve the exact MLflow identities (name, alias, version) for all four models.

    Source of truth is the source run of `ensemble_lr_model`'s latest version
    — the promoted 04 candidate run that logged the three base pins. The
    recorded values are used as-is (no validation); a missing pin just means
    there is nothing pinned to build from. Each pin also carries the resolved
    MLflow `version` (the int the alias currently points at) so downstream
    reuse/cache checks can compare on version, not on the alias URI string —
    `@best` / `@champion` stay constant when repointed, so the URI alone
    cannot detect a stale local BentoML import.
    """
    run = client.get_run(production.run_id)

    def _pin(registered_name: str, alias: str) -> dict[str, Any]:
        version = client.get_model_version_by_alias(registered_name, alias).version
        return {
            "registered_model_name": registered_name,
            "alias": alias,
            "version": int(version),
        }

    pins: dict[str, dict[str, Any]] = {"production": _pin(PRODUCTION_MODEL, "champion")}
    for cls in BASE_BENTO_NAMES:
        registered_name = run.data.params.get(f"base_{cls}_registered_name")
        alias = run.data.params.get(f"base_{cls}_alias")
        if not registered_name or not alias:
            raise RuntimeError(
                f"production run {run.info.run_id} has no recorded "
                f"base_{cls}_registered_name / base_{cls}_alias params — "
                "run the training pipeline first. Nothing to build."
            )
        pins[cls] = _pin(registered_name, alias)
    return pins


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
    """Import the pinned MLflow version into the BentoML store, or reuse an
    existing local import when it's the SAME MLflow version.

    Reuse is keyed on the resolved MLflow `version` (stamped into the Bento store
    metadata on import), NOT on the alias URI string: `@best`/`@champion` stay
    constant when repointed to a new version, so a URI match would happily
    reuse a stale local import. The version is the only identity that actually
    moves.
    """
    import bentoml

    registered_name = pin["registered_model_name"]
    uri = f"models:/{registered_name}@{pin['alias']}"
    version = pin["version"]
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
    print(f"Imported {uri} (MLflow v{version}) -> {stored.tag}")
    return stored


def _materialize_nn_onnx(nn_pin: dict[str, Any]) -> None:
    """Materialize the pinned nn_best MLflow model to ONNX at deploy time.

    Training logs nn_best as a PyTorch TabularBioMLP to MLflow. Serving runs it
    through ONNX Runtime instead of torch (smaller image, faster build). Pulls
    the pinned nn version from MLflow into the local BentoML store (version-keyed
    reuse via `_import_or_reuse`), loads it, exports to ONNX with the three
    inputs the service expects (tab, bio_p, bio_o).
    """
    import logging
    import warnings

    import bentoml
    import torch

    # Needed so torch.load can resolve the class — torch.save records the path.
    from src.models.nn import TabularBioMLP  # type: ignore[reportUnusedImport]

    # Silence torch's ONNX exporter noise: torchvision operator-skip warnings
    # (torchvision is not a serving dep), opset version hints, dynamic axes
    # deprecation, and onnxscript version converter logs. None affect the
    # exported model — our MLP uses only standard ops (Linear, ReLU, Concat).
    logging.getLogger("torch.onnx").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*LeafSpec.*")

    print(
        f"Materializing ONNX from models:/{nn_pin['registered_model_name']}@{nn_pin['alias']} (v{nn_pin['version']})"
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


def _state_fingerprint(production_version: int) -> str:
    """Fingerprint of what gets baked into the Bento.

    Keys on the promoted `ensemble_lr_model@champion` VERSION NUMBER (not its
    alias string — `@champion` is constant, so the alias alone would never
    signal a change). When the champion version increments, the base models'
    `@best` aliases now point at the matching new MLflow versions, so they are
    pulled in fresh on the rebuild. The constant base-model alias strings are
    deliberately NOT in the fingerprint: a `@best` repoint without a champion
    promotion doesn't affect the deployed ensemble, so it must not trigger a
    rebuild. File hashes of the baked-in artifacts (ONNX, scaler, embeddings)
    cover the rest.
    """
    parts = [f"ensemble_lr_model@champion=v{production_version}"]
    parts.extend(f"{path.relative_to(ROOT)}:{_file_hash(path)}" for path in FINGERPRINT_FILES)
    return "\n".join(parts)


def _import_models(pins: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Import each pinned MLflow version into the BentoML store; return tags.

    Reuse is version-keyed via `_import_or_reuse` — see its docstring for why
    the alias URI string alone cannot gate reuse.
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
    """Build the promoted Bento into the local Docker image `${IMAGE_NAME}:latest`.

    force=True skips the fingerprint and image cache checks so the Bento and
    local image are always rebuilt. Only the moving `latest` Docker image tag
    is produced; the Bento's own tag and the MLflow production version stay in
    state (and BENTO_TAG_FILE) for cache invalidation.
    """
    import bentoml
    from mlflow.tracking.client import MlflowClient

    client = MlflowClient()
    production = _latest_production_version(client)
    if production is None:
        raise RuntimeError("No promoted 'ensemble_lr_model' — nothing to build.")

    state = _read_state()
    pins = _resolve_pins(client, production)
    _materialize_nn_onnx(pins["nn"])
    _check_aux_files()
    fingerprint = _state_fingerprint(int(production.version))

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
    """Build the promoted Bento image locally, then push it to Docker Hub.

    Builds the Bento into the local Docker engine tagged `${IMAGE_NAME}:latest`
    and pushes it as `${DOCKER_REPO}/${IMAGE_NAME}:latest`. Docker Hub
    authentication: if DOCKER_TOKEN is set, log in via `docker login
    --password-stdin` (token never touches argv/logs) using DOCKER_USERNAME or
    the DOCKER_REPO owner; otherwise rely on an already-authenticated Docker
    CLI. force=True rebuilds the Bento and image (no cache) even when cached.
    No Compose stack is started or stopped.
    """
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
