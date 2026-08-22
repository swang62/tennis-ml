"""Host workflow probe for `just probe` / `just full-pipeline`.

Fail-fast, non-mutating preflight: required commands, Docker daemon, root
`.env` configuration, host PostgreSQL reachability, Prefect readiness, MLflow
reachability (HTTP URLs only; the local filesystem store is never contacted),
and `docker compose config -q`. It never writes infrastructure or data and
never prints credentials.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import psycopg
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"

# Commands the data pipeline shells out to; anything else is checked later by
# the step that needs it.
REQUIRED_COMMANDS = ("uv", "docker", "kubectl", "k3d", "serviceman")
# Host workflow contract: host PostgreSQL, host Prefect worker, Compose stack.
REQUIRED_ENV = ("DATABASE_URL", "PREFECT_API_URL", "POSTGRES_PASSWORD", "BENTO_API_KEY")

HTTP_TIMEOUT = 5  # seconds


# Terminal-aware colors matching scripts/dev.sh: ANSI only when stdout is a
# TTY, plain text when piped or tested. Blue for database, green for
# serving/readiness, muted for generic preflight.
COLOR_BLUE = "\033[34m"
COLOR_GREEN = "\033[32m"
COLOR_MUTED = "\033[90m"
COLOR_RESET = "\033[0m"


def _emit(color: str, message: str) -> None:
    """Print a `[probe]` line; colorize the tag only on a TTY, like dev.sh."""
    tag = f"{color}[probe]{COLOR_RESET}" if sys.stdout.isatty() else "[probe]"
    print(f"{tag} {message}")


def fail(message: str) -> None:
    print(f"[probe] error: {message}", file=sys.stderr)
    sys.exit(1)


def run(cmd: list[str]) -> int:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, timeout=60).returncode


def http_is_ok(url: str) -> bool:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "tennis-ml-probe/1.0"},
            timeout=HTTP_TIMEOUT,
        )
    except requests.RequestException:
        return False
    else:
        return response.status_code == 200


def _safe_url(url: str) -> str:
    """scheme://host[:port] only — drops credentials, path, and query."""
    parts = urlparse(url)
    netloc = parts.hostname or parts.netloc
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    return f"{parts.scheme}://{netloc}"


def _prefect_health_url(url: str) -> str:
    """Build the health URL whether the API URL includes the `/api` prefix."""
    base = url.rstrip("/")
    path = urlparse(base).path.rstrip("/")
    return f"{base}/health" if path.endswith("/api") else f"{base}/api/health"


def db_ok(url: str) -> str | None:
    """Return None when reachable, else a host:port-only error message."""
    try:
        with psycopg.connect(url, connect_timeout=HTTP_TIMEOUT) as conn:
            conn.execute("SELECT 1")
    except psycopg.Error:
        return f"cannot connect to PostgreSQL at {_safe_url(url)}"
    return None


def _probe() -> None:
    for command in REQUIRED_COMMANDS:
        if shutil.which(command) is None:
            fail(f"required command not found: {command}")
        _emit(COLOR_MUTED, f"ok: {command} available")

    if run(["docker", "info"]) != 0:
        fail("Docker daemon is not running (docker info failed)")
    _emit(COLOR_GREEN, "ok: Docker daemon reachable")

    if not ENV_PATH.is_file():
        fail(f"root .env not found at {ENV_PATH} (copy .env.example)")
    load_dotenv(ENV_PATH, override=True)
    for name in REQUIRED_ENV:
        if not os.getenv(name):
            fail(f"missing {name} in root .env (see .env.example)")
        _emit(COLOR_MUTED, f"ok: {name} configured")

    database_url = os.getenv("DATABASE_URL") or ""
    error = db_ok(database_url)
    if error:
        fail(f"{error} (check DATABASE_URL)")
    _emit(COLOR_BLUE, "ok: host PostgreSQL reachable")

    prefect_url = os.getenv("PREFECT_API_URL") or ""
    prefect_health_url = _prefect_health_url(prefect_url)
    if not http_is_ok(prefect_health_url):
        fail(
            f"Prefect not ready at {_safe_url(prefect_health_url)}{urlparse(prefect_health_url).path}"
        )
    _emit(COLOR_GREEN, "ok: Prefect ready")

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri and urlparse(mlflow_uri).scheme in ("http", "https"):
        if not http_is_ok(mlflow_uri):
            fail(f"MLflow not reachable at {_safe_url(mlflow_uri)}")
        _emit(COLOR_GREEN, "ok: MLflow reachable")
    else:
        _emit(COLOR_MUTED, "ok: MLflow uses the local filesystem store (no HTTP check)")

    if run(["docker", "compose", "config", "-q"]) != 0:
        fail("docker compose config failed (invalid compose.yaml or missing interpolation)")
    _emit(COLOR_MUTED, "ok: docker compose config valid")


def main() -> None:
    _emit(COLOR_MUTED, "host workflow preflight")
    _probe()
    _emit(COLOR_GREEN, "all checks passed")


if __name__ == "__main__":
    main()
