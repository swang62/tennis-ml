"""Build and deploy the production Bento serving image.

Deploy-only: deploy_bento() just invokes `build_bento_image()` which builds in the local Docker engine first
and never depends on k3d. If the cluster is up, it should push that image to the k3d-managed local
registry and force rollout.

Usage:
    uv run python src/flows/deploy.py
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from prefect import flow

from src.constants import ARTIFACTS, DATA_PROCESSED, PRODUCTION_MODEL, ROOT
from src.utils import load_env

# --- Deploy-only paths and names ---
TEMPLATE_BENTOFILE = ROOT / "bentofile.yaml"
SERVICE_FILE = ROOT / "src" / "serving" / "service.py"
PINNED_BENTOFILE = DATA_PROCESSED / "bentofile.pinned.yaml"
BENTO_TAG_FILE = DATA_PROCESSED / "bento_tag.txt"
STATE_FILE = DATA_PROCESSED / "bento_build_state.json"
LOCAL_BENTO_REPOSITORY = "bento-serving"

K3D_CLUSTER = "tennis-ml"
REGISTRY_NAME = "tennis-ml-registry"
REGISTRY_PUSH_REPOSITORY = "localhost:5000/bento-serving"  # from host
REGISTRY_PULL_REPOSITORY = "tennis-ml-registry:5000/bento-serving"  # inside k3d
BENTO_MANIFEST = ROOT / "infra" / "manifests" / "deploy" / "bentoml.yaml"

# The BentoML model name each pinned MLflow registered model maps to —
# exactly the names the service references via `bentoml.models.BentoModel`.
# `nn_best` is imported into the BentoML store (for the deploy-time ONNX
# export) but its tag is NOT added to the pinned bentofile's models: the
# service consumes it via the onnx artifact under `include:` instead.
BASE_BENTO_NAMES = {"linear": "linear_best", "gbdt": "gbdt_best", "nn": "nn_best"}

# Serving artifacts packaged into the Bento (written by pipeline notebooks or
# materialized by this flow); content changes trigger a rebuild.
NN_ONNX_FILE = DATA_PROCESSED / "nn_best.onnx"
AUX_FILES = [
    DATA_PROCESSED / "linear_scaler.pkl",
    DATA_PROCESSED / "bio_embeddings.parquet",
    DATA_PROCESSED / "bio_feature_cols.json",
    NN_ONNX_FILE,
    ROOT / "data" / "tennis.duckdb",
]

# Files whose content changes should trigger a rebuild.
FINGERPRINT_FILES = [
    TEMPLATE_BENTOFILE,
    SERVICE_FILE,
    *AUX_FILES,
]

load_env()


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
    """Resolve the exact MLflow identities for all four models.

    Source of truth is the source run of `ensemble_lr_model`'s latest version
    — the promoted 04 candidate run that logged the three base pins. The
    recorded values are used as-is (no validation); a missing pin just means
    there is nothing pinned to build from.
    """
    run = client.get_run(production.run_id)

    pins: dict[str, dict[str, Any]] = {
        "production": {"registered_model_name": PRODUCTION_MODEL, "alias": "champion"}
    }
    for cls in BASE_BENTO_NAMES:
        registered_name = run.data.params.get(f"base_{cls}_registered_name")
        alias = run.data.params.get(f"base_{cls}_alias")
        if not registered_name or not alias:
            raise RuntimeError(
                f"production run {run.info.run_id} has no recorded "
                f"base_{cls}_registered_name / base_{cls}_alias params — "
                "run the training pipeline first. Nothing to build."
            )
        pins[cls] = {"registered_model_name": registered_name, "alias": alias}
    return pins


def _check_aux_files() -> None:
    """The serving artifacts packaged into the Bento must exist on disk."""
    missing = [str(p.relative_to(ROOT)) for p in AUX_FILES if not p.exists()]
    if missing:
        raise RuntimeError(
            "required serving artifacts missing: "
            + ", ".join(missing)
            + " — run the training pipeline (00/02 write these to data/processed)"
        )


def _materialize_nn_onnx(nn_pin: dict[str, Any]) -> None:
    """Materialize the pinned nn_best MLflow model to ONNX at deploy time.

    Training logs nn_best as a PyTorch TabularBioMLP to MLflow. Serving runs it
    through ONNX Runtime instead of torch (smaller image, faster build). Pulls
    the pinned nn version from MLflow into the local BentoML store (idempotent
    on the mlflow_uri), loads it, exports to ONNX with the three inputs the
    service expects (tab, bio_p, bio_o).
    """
    import logging
    import warnings

    import bentoml
    import torch

    # Needed so torch.load can resolve the class — torch.save records the path.
    from src.models.nn import TabularBioMLP

    # Silence torch's ONNX exporter noise: torchvision operator-skip warnings
    # (torchvision is not a serving dep), opset version hints, dynamic axes
    # deprecation, and onnxscript version converter logs. None affect the
    # exported model — our MLP uses only standard ops (Linear, ReLU, Concat).
    logging.getLogger("torch.onnx").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*LeafSpec.*")

    uri = f"models:/{nn_pin['registered_model_name']}@{nn_pin['alias']}"
    print(f"Materializing ONNX from {uri}")

    # Import (or reuse) the pinned MLflow model into the local BentoML store so
    # bentoml.mlflow.load_model works against the local tag.
    registered_name = nn_pin["registered_model_name"]
    try:
        stored = bentoml.models.get(registered_name)
        if stored.info.metadata.get("mlflow_uri") != uri:
            stored = bentoml.mlflow.import_model(registered_name, uri, metadata={"mlflow_uri": uri})
    except Exception:
        stored = bentoml.mlflow.import_model(registered_name, uri, metadata={"mlflow_uri": uri})

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


def _state_fingerprint(pins: dict[str, dict[str, Any]]) -> str:
    """Hash the model pins plus the content of everything baked into the Bento."""
    parts = [
        f"{key}={pins[key]['registered_model_name']}@{pins[key]['alias']}"
        for key in ["linear", "gbdt", "nn", "production"]
    ]
    parts.extend(f"{path.relative_to(ROOT)}:{_file_hash(path)}" for path in FINGERPRINT_FILES)
    return "\n".join(parts)


def _import_models(pins: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Import each pinned MLflow version into the BentoML store; return tags."""
    import bentoml

    tags: dict[str, str] = {}
    for key, pin in pins.items():
        uri = f"models:/{pin['registered_model_name']}@{pin['alias']}"
        try:
            stored = bentoml.models.get(pin["registered_model_name"])
        except Exception:
            stored = None
        if stored is not None and stored.info.metadata.get("mlflow_uri") == uri:
            tag = stored.tag
            print(f"Reusing {tag} — already imported from {uri}")
        else:
            tag = bentoml.mlflow.import_model(
                pin["registered_model_name"], uri, metadata={"mlflow_uri": uri}
            ).tag
            print(f"Imported {uri} -> {tag}")
        tags[key] = str(tag)
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


def _cluster_running() -> bool:
    try:
        out = subprocess.run(
            ["k3d", "cluster", "list", "-o", "json"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        clusters = json.loads(out)
        return any(
            c.get("name") == K3D_CLUSTER and c.get("serversRunning", 0) > 0 for c in clusters
        )
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return False


def _registry_running() -> bool:
    try:
        registries = json.loads(
            subprocess.run(
                ["k3d", "registry", "list", "-o", "json"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        return any(
            registry.get("name") == REGISTRY_NAME and registry.get("State", {}).get("Running")
            for registry in registries
        )
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return False


def _image_exists(image: str) -> bool:
    return (
        subprocess.run(
            ["docker", "image", "inspect", image],
            cwd=ROOT,
            capture_output=True,
        ).returncode
        == 0
    )


def build_bento_image(force: bool = False) -> tuple[str, int]:
    """Build the promoted Bento into local Docker without requiring k3d.

    force=True skips the fingerprint and image cache checks so the Bento and
    local image are always rebuilt.
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
    fingerprint = _state_fingerprint(pins)

    if force:
        print("Force: rebuilding Bento and image regardless of cache.")
        tag = None
    else:
        tag = _cached_tag(state, fingerprint)
    if tag is None:
        tags = _import_models(pins)
        pinned = _write_pinned_bentofile(tags)
        bento = bentoml.bentos.build_bentofile(bentofile=str(pinned), build_ctx=str(ROOT))
        tag = str(bento.tag)
        BENTO_TAG_FILE.write_text(tag)
        _write_state({**state, "fingerprint": fingerprint, "tag": tag})
        print(f"Built {tag} from {pinned}")
    else:
        print(f"No Bento rebuild needed — reusing {tag}.")

    image = f"{LOCAL_BENTO_REPOSITORY}:{tag.split(':', 1)[1]}"
    if force:
        print(f"Force: containerizing {tag} -> {image} regardless of cache.")
        containerize = True
    else:
        containerize = not _image_exists(image)
    if containerize:
        print(f"Containerizing {tag} -> {image} with BentoML...")
        bentoml.container.build(tag, backend="docker", image_tag=(image,))
    else:
        print(f"Local image already exists — reusing {image}.")
    subprocess.run(["docker", "tag", image, f"{LOCAL_BENTO_REPOSITORY}:latest"], check=True)
    return image, int(production.version)


def deploy_bento(force: bool = False) -> None:
    """Build locally, then push and roll out through the k3d registry.

    A stopped or absent cluster does not prevent the local image build. A
    running cluster must have the registry declared in infra/k3d/config.yaml.
    force=True rebuilds the Bento and image even when cached.
    """
    local_image, production_version = build_bento_image(force=force)

    if not _cluster_running():
        print(f"k3d cluster {K3D_CLUSTER} is not running — local image is ready: {local_image}")
        return
    if not _registry_running():
        raise RuntimeError(
            f"k3d registry {REGISTRY_NAME} is not running — recreate the cluster with "
            "infra/k3d/config.yaml"
        )

    if force:
        print("Force: re-pushing image and re-applying manifest in the cluster.")
    image_version = local_image.split(":", 1)[1]
    push_image = f"{REGISTRY_PUSH_REPOSITORY}:{image_version}"
    push_latest = f"{REGISTRY_PUSH_REPOSITORY}:latest"
    for target in [push_image, push_latest]:
        subprocess.run(["docker", "tag", local_image, target], cwd=ROOT, check=True)
        subprocess.run(["docker", "push", target], cwd=ROOT, check=True)

    # In-cluster pull reference resolves to the same registry/repo as the push.
    # Render the manifest with the pinned tag and apply once: applying the
    # manifest's :latest then `set image` to the pinned tag creates an extra
    # ReplicaSet (and the 'old replicas pending termination' noise) every deploy.
    pull_image = f"{REGISTRY_PULL_REPOSITORY}:{image_version}"
    manifest = BENTO_MANIFEST.read_text()
    rendered = manifest.replace(f"{REGISTRY_PULL_REPOSITORY}:latest", pull_image)
    if rendered == manifest:
        raise RuntimeError(
            f"manifest {BENTO_MANIFEST} does not reference "
            f"{REGISTRY_PULL_REPOSITORY}:latest — cannot pin the deploy image"
        )
    subprocess.run(["kubectl", "apply", "-f", "-"], input=rendered, cwd=ROOT, check=True, text=True)
    subprocess.run(
        ["kubectl", "rollout", "status", "deployment/bento-serving", "--timeout=180s"],
        cwd=ROOT,
        check=True,
    )
    state = _read_state()
    _write_state({**state, "deployed_version": production_version, "deployed_image": pull_image})
    print(f"Deployed production {pull_image} to {K3D_CLUSTER}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build and deploy the production Bento serving image."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild + redeploy of the latest ensemble_lr_model, bypassing the Bento/image cache.",
    )
    args = parser.parse_args()
    deploy_flow(force=args.force)
