"""Build and deploy the production Bento serving image.

Deploy-only: no notebook runs here. The training pipeline already ran 05
(evaluation/promotion) as part of its notebook chain; this flow just invokes
`build_bento_image()` builds in the local Docker engine and never depends on
k3d. `deploy_bento()` additionally pushes that image to the k3d-managed local
registry and updates Kubernetes when the cluster is running.

Usage:
    uv run python src/flows/deploy.py
    just bento-build
    just deploy-bento
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from prefect import flow

from src.constants import ARTIFACTS, DATA_PROCESSED, PRODUCTION_MODEL, ROOT

# --- Deploy-only paths and names ---
TEMPLATE_BENTOFILE = ROOT / "bentofile.yaml"
SERVICE_FILE = ROOT / "src" / "serving" / "service.py"
PINNED_BENTOFILE = DATA_PROCESSED / "bentofile.pinned.yaml"
BENTO_TAG_FILE = DATA_PROCESSED / "bento_tag.txt"
STATE_FILE = DATA_PROCESSED / "bento_build_state.json"
LOCAL_BENTO_REPOSITORY = "bento-serving"
REGISTRY_NAME = "tennis-ml-registry"
# Host-side push endpoint: Docker trusts localhost as an insecure registry by
# default, so `localhost:5000` needs no daemon config. The cluster-internal
# endpoint below is what k8s pulls; the registry stores by repository path
# (bento-serving), so a push to localhost:5000 is pullable as the in-cluster
# hostname. See infra/manifests/deploy/bentoml.yaml for the pull-side image.
REGISTRY_PUSH_REPOSITORY = "localhost:5000/bento-serving"
REGISTRY_PULL_REPOSITORY = "tennis-ml-registry:5000/bento-serving"
K3D_CLUSTER = "tennis-ml"
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
]

# Files whose content changes should trigger a rebuild.
FINGERPRINT_FILES = [
    TEMPLATE_BENTOFILE,
    SERVICE_FILE,
    *AUX_FILES,
]

# The deploy gate reads the promoted `production_model` from MLflow; local
# runs without an explicit tracking URI default to a repo-root mlruns/, so
# redirect those to the shared artifacts/ area. Deployed workers (k8s
# config-map sets MLFLOW_TRACKING_URI=http://mlflow:5000) are untouched.
if not os.environ.get("MLFLOW_TRACKING_URI"):
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{ARTIFACTS / 'mlflow.db'}"
    os.environ["_MLFLOW_SERVER_ARTIFACT_ROOT"] = str(ARTIFACTS / "mlruns")


@flow(log_prints=True)
def deploy_flow() -> None:
    """Deploy-only flow: run the single deployment path for the promoted model.

    No-op unless 05 promoted a production version newer than the last
    deployed one. Evaluation/promotion already ran in the training pipeline;
    no notebook runs here.
    """
    deploy_bento()


def _latest_production_version(client: Any) -> Any:
    """Return the version `models:/production_model/latest` resolves to.

    MLflow's `latest` alias means the most recent version in stage "None";
    mirror that exactly instead of guessing. Returns None when there is no
    such version.
    """
    versions = client.search_model_versions(f"name = '{PRODUCTION_MODEL}'")
    staged = [v for v in versions if v.current_stage == "None"]
    if not staged:
        return None
    return max(staged, key=lambda v: int(v.version))


def _read_state() -> dict[str, Any]:
    try:
        return json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_state(state: dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state) + "\n")


def _resolve_pins(client: Any, production: Any) -> dict[str, dict[str, Any]]:
    """Resolve the exact MLflow identities for all four models.

    Source of truth is the source run of `production_model`'s latest version
    — the promoted 04 candidate run that logged the three base pins. The
    recorded values are used as-is (no validation); a missing pin just means
    there is nothing pinned to build from.
    """
    run = client.get_run(production.run_id)

    pins: dict[str, dict[str, Any]] = {
        "production": {"registered_model_name": PRODUCTION_MODEL, "version": production.version}
    }
    for cls in BASE_BENTO_NAMES:
        registered_name = run.data.params.get(f"base_{cls}_registered_name")
        version = run.data.params.get(f"base_{cls}_version")
        if not registered_name or not version:
            raise RuntimeError(
                f"production run {run.info.run_id} has no recorded "
                f"base_{cls}_registered_name / base_{cls}_version params — "
                "run the training pipeline first. Nothing to build."
            )
        pins[cls] = {"registered_model_name": registered_name, "version": version}
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

    uri = f"models:/{nn_pin['registered_model_name']}/{nn_pin['version']}"
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
        )
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
        f"{key}={pins[key]['registered_model_name']}:{pins[key]['version']}"
        for key in ["linear", "gbdt", "nn", "production"]
    ]
    parts.extend(f"{path.relative_to(ROOT)}:{_file_hash(path)}" for path in FINGERPRINT_FILES)
    return "\n".join(parts)


def _import_models(pins: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Import each pinned MLflow version into the BentoML store; return tags."""
    import bentoml

    tags: dict[str, str] = {}
    for key, pin in pins.items():
        uri = f"models:/{pin['registered_model_name']}/{pin['version']}"
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


def build_bento_image() -> tuple[str, int]:
    """Build the promoted Bento into local Docker without requiring k3d."""
    import bentoml
    from mlflow.tracking.client import MlflowClient

    client = MlflowClient()
    production = _latest_production_version(client)
    if production is None:
        raise RuntimeError("No promoted 'production_model' — nothing to build.")

    state = _read_state()
    pins = _resolve_pins(client, production)
    _materialize_nn_onnx(pins["nn"])
    _check_aux_files()
    fingerprint = _state_fingerprint(pins)

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
    if _image_exists(image):
        print(f"Local image already exists — reusing {image}.")
    else:
        print(f"Containerizing {tag} -> {image} with BentoML's native v2 image spec...")
        bentoml.container.build(tag, backend="docker", image_tag=(image,))
    subprocess.run(["docker", "tag", image, f"{LOCAL_BENTO_REPOSITORY}:latest"], check=True)
    return image, int(production.version)


def deploy_bento() -> None:
    """Build locally, then push and roll out through the k3d registry.

    A stopped or absent cluster does not prevent the local image build. A
    running cluster must have the registry declared in infra/k3d/config.yaml.
    """
    local_image, production_version = build_bento_image()

    if not _cluster_running():
        print(f"k3d cluster {K3D_CLUSTER} is not running — local image is ready: {local_image}")
        return
    if not _registry_running():
        raise RuntimeError(
            f"k3d registry {REGISTRY_NAME} is not running — recreate the cluster with "
            "infra/k3d/config.yaml"
        )

    image_version = local_image.split(":", 1)[1]
    push_image = f"{REGISTRY_PUSH_REPOSITORY}:{image_version}"
    push_latest = f"{REGISTRY_PUSH_REPOSITORY}:latest"
    for target in [push_image, push_latest]:
        subprocess.run(["docker", "tag", local_image, target], cwd=ROOT, check=True)
        subprocess.run(["docker", "push", target], cwd=ROOT, check=True)

    # In-cluster pull reference resolves to the same registry/repo as the push.
    pull_image = f"{REGISTRY_PULL_REPOSITORY}:{image_version}"
    subprocess.run(["kubectl", "apply", "-f", str(BENTO_MANIFEST)], cwd=ROOT, check=True)
    subprocess.run(
        ["kubectl", "set", "image", "deployment/bento-serving", f"bentoml={pull_image}"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        ["kubectl", "rollout", "status", "deployment/bento-serving", "--timeout=180s"],
        cwd=ROOT,
        check=True,
    )
    state = _read_state()
    _write_state({**state, "deployed_version": production_version, "deployed_image": pull_image})
    print(f"Deployed {pull_image} (production v{production_version}) to {K3D_CLUSTER}")


if __name__ == "__main__":
    deploy_flow()
