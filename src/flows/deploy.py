"""Prefect flow: the manual deploy stage — `just deploy-bento`.

Deploy-only: no notebook runs here. The training pipeline already ran 05
(evaluation/promotion) as part of its notebook chain; this flow just invokes
the single deployment path, `deploy_bento()` — the same implementation behind
the `just deploy-bento` target — which gates on a newer promoted
`production_model`, builds the Bento, containerizes it, and imports it into
k3d only when the promoted version is newer than the last deployed one.

Usage:
    uv run python src/flows/deploy.py
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
BENTO_IMAGE = "bento-serving:latest"
K3D_CLUSTER = "tennis-ml"

# The BentoML model name each pinned MLflow registered model maps to —
# exactly the names the service references via `bentoml.models.BentoModel`.
BASE_BENTO_NAMES = {"linear": "linear_best", "gbdt": "gbdt_best", "nn": "nn_best"}

# Serving artifacts packaged into the Bento (written by pipeline notebooks);
# content changes trigger a rebuild.
AUX_FILES = [
    DATA_PROCESSED / "linear_scaler.pkl",
    DATA_PROCESSED / "bio_embeddings.parquet",
    DATA_PROCESSED / "bio_feature_cols.json",
]

# Files whose content changes should trigger a rebuild.
FINGERPRINT_FILES = [TEMPLATE_BENTOFILE, SERVICE_FILE, *AUX_FILES]

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
    config["models"] = [tags[key] for key in ["linear", "gbdt", "nn", "production"]]
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
        return any(c.get("serversRunning", 0) > 0 for c in clusters)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return False


def deploy_bento() -> None:
    """Single deployment path: gate, build, and deploy the serving Bento.

    Invoked by `just deploy-bento` and by the deploy flow in
    `src/flows/deploy.py`. No-ops when `production_model` has no version
    newer than the last deployed one (recorded in
    `data/processed/bento_build_state.json`). Otherwise imports the pinned
    models, builds the Bento (cached by state fingerprint), containerizes it,
    and imports the image into the k3d cluster when it is running.
    """
    from mlflow.tracking.client import MlflowClient

    client = MlflowClient()
    production = _latest_production_version(client)
    if production is None:
        print("No promoted 'production_model' — nothing to deploy.")
        return

    state = _read_state()
    deployed_version = state.get("deployed_version")
    if deployed_version is not None and int(production.version) <= int(deployed_version):
        print(
            f"No newer promoted production model "
            f"(production v{production.version} <= deployed v{deployed_version}) — "
            "nothing to deploy."
        )
        return

    pins = _resolve_pins(client, production)
    _check_aux_files()
    fingerprint = _state_fingerprint(pins)

    tag = _cached_tag(state, fingerprint)
    if tag is None:
        tags = _import_models(pins)
        pinned = _write_pinned_bentofile(tags)

        import bentoml

        bento = bentoml.bentos.build_bentofile(bentofile=str(pinned), build_ctx=str(ROOT))
        tag = str(bento.tag)
        BENTO_TAG_FILE.write_text(tag)
        _write_state({"fingerprint": fingerprint, "tag": tag})
        print(f"Built {bento.tag} from {pinned}")
    else:
        print(f"No rebuild needed — Bento {tag} already built for this state.")

    subprocess.run(
        ["uv", "run", "bentoml", "containerize", tag, "-t", BENTO_IMAGE],
        cwd=ROOT,
        check=True,
    )

    if _cluster_running():
        subprocess.run(
            ["k3d", "image", "import", BENTO_IMAGE, "-c", K3D_CLUSTER],
            cwd=ROOT,
            check=True,
        )
        _write_state(
            {"fingerprint": fingerprint, "tag": tag, "deployed_version": int(production.version)}
        )
        print(f"Deployed {tag} (production v{production.version}) to k3d cluster {K3D_CLUSTER}")
    else:
        print(f"k3d cluster {K3D_CLUSTER} is not running — image built but not imported.")
        print("Start the cluster and re-run `just deploy-bento` (it will reuse the built Bento).")


if __name__ == "__main__":
    deploy_flow()
