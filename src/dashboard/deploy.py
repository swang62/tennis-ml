"""Build and deploy the Panel dashboard image to the k3d cluster.

Final step after the local dashboard is verified (`just dashboard-local`):
builds the docker image, pushes it to the k3d-managed registry, applies the
deploy manifest plus the ingress host, and forces a rollout so the new image
is pulled. Plain script, not a Prefect flow — the dashboard has no model
artifacts or promotion state.

Usage:
    uv run python src/dashboard/deploy.py
"""

import json
import os
import subprocess

from src.constants import REPO_NAME, ROOT
from src.utils import load_env

DASHBOARD_IMAGE = "tennis-dashboard"
DASHBOARD_DOCKERFILE = ROOT / "infra" / "manifests" / "deploy" / "Dockerfile"
DASHBOARD_MANIFEST = ROOT / "infra" / "manifests" / "deploy" / "dashboard.yaml"
INGRESS_MANIFEST = ROOT / "infra" / "manifests" / "default" / "ingress.yaml"

# Registry URL is composed from the registry host + the dashboard repo name,
# same pattern as src/flows/deploy.py: push goes
# host -> caddy (https) -> traefik -> registry.
REGISTRY_NAME = REPO_NAME + "-registry"

load_env()
REGISTRY_PUSH_URL = os.getenv("REGISTRY_PUSH_URL")
REGISTRY_PUSH_REPOSITORY = f"{REGISTRY_PUSH_URL}/{DASHBOARD_IMAGE}"


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
        return any(c.get("name") == REPO_NAME and c.get("serversRunning", 0) > 0 for c in clusters)
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


def deploy_dashboard() -> None:
    """Build locally, push through the k3d registry, apply manifests, rollout.

    A stopped or absent cluster does not prevent the local image build; if the
    registry is missing or the push fails, deployment is skipped (logged) and
    the local image is left ready.
    """
    print("Building dashboard image...")
    subprocess.run(
        [
            "docker",
            "build",
            "-t",
            DASHBOARD_IMAGE,
            "-f",
            str(DASHBOARD_DOCKERFILE),
            str(ROOT),
        ],
        cwd=ROOT,
        check=True,
    )

    if not _cluster_running():
        print(f"k3d cluster {REPO_NAME} is not running — local image is ready: {DASHBOARD_IMAGE}")
        return
    if not _registry_running():
        print(
            f"k3d registry {REGISTRY_NAME} is not running — skipping deployment; "
            f"local image is ready: {DASHBOARD_IMAGE}"
        )
        return
    if not REGISTRY_PUSH_URL:
        print("REGISTRY_PUSH_URL is not set — skipping deployment; local image is ready.")
        return

    push_latest = f"{REGISTRY_PUSH_REPOSITORY}:latest"
    try:
        subprocess.run(["docker", "tag", DASHBOARD_IMAGE, push_latest], cwd=ROOT, check=True)
        subprocess.run(["docker", "push", push_latest], cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        print(
            f"Registry push failed ({exc}) — skipping deployment; "
            f"local image is ready: {DASHBOARD_IMAGE}"
        )
        return

    # Deploy manifest pins :latest; rollout restart forces new pods to pull it
    # (imagePullPolicy: Always), since an unchanged image ref alone would
    # leave the old ReplicaSet untouched. Ingress apply enables
    # dashboard.macsteve.lan (idempotent).
    subprocess.run(["kubectl", "apply", "-f", str(DASHBOARD_MANIFEST)], cwd=ROOT, check=True)
    subprocess.run(["kubectl", "apply", "-f", str(INGRESS_MANIFEST)], cwd=ROOT, check=True)
    subprocess.run(
        ["kubectl", "rollout", "restart", "deployment/tennis-dashboard"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        ["kubectl", "rollout", "status", "deployment/tennis-dashboard", "--timeout=180s"],
        cwd=ROOT,
        check=True,
    )
    print(f"Deployed {push_latest} to {REPO_NAME} — https://dashboard.macsteve.lan")


if __name__ == "__main__":
    deploy_dashboard()
