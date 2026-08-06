"""Offline tests for the host deployment path (src/flows/deploy.py).

These test command construction, secure credential handling, and force
behavior WITHOUT a Docker daemon, Docker Hub login/push, Docker Compose, or a
model deployment. build_bento_image / deploy_bento are exercised with their
heavy dependencies (MLflow, BentoML, Docker) mocked away.
"""

import importlib
import sys
from types import SimpleNamespace


def _deploy():
    return importlib.import_module("src.flows.deploy")


# --- Secure credential handling ---


def test_docker_login_passes_token_via_stdin_only(monkeypatch):
    d = _deploy()
    monkeypatch.setenv("DOCKER_TOKEN", "super-secret-token")
    monkeypatch.setenv("DOCKER_USERNAME", "hubuser")
    monkeypatch.setattr(d, "DOCKER_REPO", "acme")

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["input"] = kwargs.get("input")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)

    d._docker_login()

    assert "--password-stdin" in captured["cmd"]
    assert "hubuser" in captured["cmd"]
    # Token must travel only via stdin, never in argv.
    assert "super-secret-token" not in captured["cmd"]
    assert captured["input"] == "super-secret-token\n"


def test_docker_login_username_wins_over_repo_owner(monkeypatch):
    d = _deploy()
    monkeypatch.setenv("DOCKER_TOKEN", "tok")
    monkeypatch.setenv("DOCKER_USERNAME", "explicit-user")
    monkeypatch.setattr(d, "DOCKER_REPO", "acme")

    captured = {}

    def fake_run(cmd, **_kwargs):
        captured["cmd"] = list(cmd)
        return SimpleNamespace(returncode=0)


def test_docker_login_username_falls_back_to_repo_owner(monkeypatch):
    d = _deploy()
    monkeypatch.setenv("DOCKER_TOKEN", "tok")
    monkeypatch.delenv("DOCKER_USERNAME", raising=False)
    monkeypatch.setattr(d, "DOCKER_REPO", "acme")

    captured = {}

    def fake_run(cmd, **_kwargs):
        captured["cmd"] = list(cmd)
        return SimpleNamespace(returncode=0)


def test_docker_login_skips_when_token_unset(monkeypatch):
    d = _deploy()
    monkeypatch.delenv("DOCKER_TOKEN", raising=False)

    called = []

    def fake_run(*_args, **_kwargs):
        called.append(_args)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)

    d._docker_login()

    assert called == []


def test_docker_login_raises_on_failure(monkeypatch):
    d = _deploy()
    monkeypatch.setenv("DOCKER_TOKEN", "tok")
    monkeypatch.setattr("subprocess.run", lambda *_a, **_k: SimpleNamespace(returncode=1))

    import pytest

    with pytest.raises(RuntimeError):
        d._docker_login()


# --- Command construction (push latest + compose up) ---


def _stub_subprocess(monkeypatch):
    """Replace subprocess.run with a no-op returning success."""
    monkeypatch.setattr("subprocess.run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))


def test_deploy_bento_pushes_only_latest_and_composes_up(monkeypatch, tmp_path):
    d = _deploy()
    monkeypatch.setattr(d, "DOCKER_REPO", "acme")
    monkeypatch.setattr(d, "IMAGE_NAME", "tennis-ml")
    monkeypatch.setattr(d, "LOGS", tmp_path)
    monkeypatch.setattr(d, "build_bento_image", lambda **_kwargs: ("acme/tennis-ml:abc", 5))
    monkeypatch.setattr(d, "_docker_login", lambda: None)
    monkeypatch.setattr(d, "_read_state", lambda: {})
    monkeypatch.setattr(d, "_write_state", lambda _s: None)
    monkeypatch.setenv("POSTGRES_PASSWORD", "super-secret-deploy-pass")
    _stub_subprocess(monkeypatch)

    calls = []

    def fake_run_teed(cmd, _log, env=None):
        calls.append((list(cmd), dict(env or {})))
        return None

    monkeypatch.setattr(d, "_run_teed", fake_run_teed)

    d.deploy_bento(force=False)

    pushes = [c for c in calls if c[0][:2] == ["docker", "push"]]
    assert len(pushes) == 1
    assert pushes[0][0] == ["docker", "push", "acme/tennis-ml:latest"]

    composes = [c for c in calls if c[0][:2] == ["docker", "compose"]]
    # Two compose calls, ordered: build web, then up.
    assert len(composes) == 2
    build_cmd, _ = composes[0]
    assert build_cmd[-1] == "web"
    assert "build" in build_cmd
    assert "--no-cache" not in build_cmd  # only forced rebuilds skip the cache
    assert "-f" in build_cmd
    up_cmd, up_env = composes[1]
    assert up_cmd[-4:] == ["up", "-d", "--pull", "always"]
    assert "-f" in up_cmd
    assert up_env["DOCKER_IMAGE"] == "acme/tennis-ml"


def test_deploy_bento_force_builds_web_no_cache(monkeypatch, tmp_path):
    d = _deploy()
    monkeypatch.setattr(d, "DOCKER_REPO", "acme")
    monkeypatch.setattr(d, "IMAGE_NAME", "tennis-ml")
    monkeypatch.setattr(d, "LOGS", tmp_path)
    monkeypatch.setattr(d, "build_bento_image", lambda **_kwargs: ("acme/tennis-ml:abc", 5))
    monkeypatch.setattr(d, "_docker_login", lambda: None)
    monkeypatch.setattr(d, "_read_state", lambda: {})
    monkeypatch.setattr(d, "_write_state", lambda _s: None)
    monkeypatch.setenv("POSTGRES_PASSWORD", "super-secret-deploy-pass")
    _stub_subprocess(monkeypatch)

    calls = []

    def fake_run_teed(cmd, _log, env=None):
        calls.append((list(cmd), dict(env or {})))
        return None

    monkeypatch.setattr(d, "_run_teed", fake_run_teed)

    d.deploy_bento(force=True)

    composes = [c for c in calls if c[0][:2] == ["docker", "compose"]]
    assert len(composes) == 2
    build_cmd, _ = composes[0]
    assert build_cmd[-1] == "web"
    assert build_cmd[build_cmd.index("build") + 1] == "--no-cache"


def test_deploy_bento_requires_postgres_password(monkeypatch, tmp_path):
    """Production deploy must fail fast when POSTGRES_PASSWORD is unset — never
    boot Compose without the PostgreSQL secret."""
    import pytest

    d = _deploy()
    monkeypatch.setattr(d, "DOCKER_REPO", "acme")
    monkeypatch.setattr(d, "IMAGE_NAME", "tennis-ml")
    monkeypatch.setattr(d, "LOGS", tmp_path)
    monkeypatch.setattr(d, "build_bento_image", lambda **_kwargs: ("acme/tennis-ml:abc", 5))
    monkeypatch.setattr(d, "_docker_login", lambda: None)
    monkeypatch.setattr(d, "_read_state", lambda: {})
    monkeypatch.setattr(d, "_write_state", lambda _s: None)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    _stub_subprocess(monkeypatch)

    with pytest.raises(RuntimeError, match="POSTGRES_PASSWORD is required"):
        d.deploy_bento(force=False)


# --- Force behavior ---


def _stub_bento_build(monkeypatch):
    """Patch the MLflow/BentoML/Docker machinery behind build_bento_image."""
    d = _deploy()
    from pathlib import Path as _Path

    monkeypatch.setattr(d, "IMAGE_NAME", "tennis-ml")
    monkeypatch.setattr(d, "BENTO_TAG_FILE", _Path("/tmp/bento_tag.txt"))
    monkeypatch.setattr(d, "STATE_FILE", _Path("/tmp/state.json"))
    monkeypatch.setattr(
        d,
        "_latest_production_version",
        lambda _client: SimpleNamespace(version="7", run_id="r"),
    )
    monkeypatch.setattr(
        d,
        "_resolve_pins",
        lambda _client, _prod: {k: {} for k in ("nn", "linear", "gbdt", "production")},
    )
    monkeypatch.setattr(d, "_materialize_nn_onnx", lambda _nn: None)
    monkeypatch.setattr(d, "_check_aux_files", lambda: None)
    monkeypatch.setattr(d, "_state_fingerprint", lambda _v: "fp")
    monkeypatch.setattr(d, "_read_state", lambda: {"fingerprint": "fp"})
    # A cache HIT: _cached_tag would return a tag when not forced.
    monkeypatch.setattr(d, "_cached_tag", lambda _state, _fp: "bento:abc")
    monkeypatch.setattr(
        d, "_import_models", lambda _pins: {"linear": "l", "gbdt": "g", "production": "p"}
    )
    monkeypatch.setattr(d, "_write_pinned_bentofile", lambda _tags: "pinned")
    monkeypatch.setattr(d, "_write_state", lambda _s: None)
    monkeypatch.setattr(d, "_image_exists", lambda _image: True)
    _stub_subprocess(monkeypatch)

    built = {"bento": 0, "container": 0}
    bentoml = {
        "bentoml": SimpleNamespace(
            bentos=SimpleNamespace(get=lambda _tag: (_ for _ in ()).throw(Exception())),
            container=SimpleNamespace(),
        )
    }
    monkeypatch.setitem(sys.modules, "bentoml", bentoml["bentoml"])
    monkeypatch.setitem(
        sys.modules,
        "mlflow.tracking.client",
        SimpleNamespace(MlflowClient=lambda: None),
    )

    def build_bentofile(**_kwargs):
        built["bento"] += 1
        return SimpleNamespace(tag="bento:abc")

    def container_build(*_args, **_kwargs):
        built["container"] += 1

    bentoml["bentoml"].bentos.build_bentofile = build_bentofile
    bentoml["bentoml"].container.build = container_build
    return d, built


def test_build_force_rebuilds_over_cache_hit(monkeypatch):
    d, built = _stub_bento_build(monkeypatch)

    # force=True: rebuild the Bento and containerize even though the cache hits.
    d.build_bento_image(force=True)
    assert built["bento"] == 1
    assert built["container"] == 1


def test_build_no_force_reuses_cache_hit(monkeypatch):
    d, built = _stub_bento_build(monkeypatch)

    # force=False: cache hit -> no rebuild, and the local image exists -> no
    # containerization.
    d.build_bento_image(force=False)
    assert built["bento"] == 0
    assert built["container"] == 0


def test_deploy_flow_forwards_force(monkeypatch):
    d = _deploy()
    seen = {}

    def fake_deploy_bento(force=False):
        seen["force"] = force

    monkeypatch.setattr(d, "deploy_bento", fake_deploy_bento)

    d.deploy_flow(force=True)
    assert seen["force"] is True
    d.deploy_flow(force=False)
    assert seen["force"] is False


# --- Task 1: host-managed PostgreSQL + shared connection contract ---


def _compose():
    import yaml

    return yaml.safe_load(_deploy().COMPOSE_FILE.read_text())


def test_compose_image_has_safe_default_when_docker_image_unset():
    """Manual `docker compose config` with DOCKER_IMAGE unset must not render the
    invalid `:latest` image. The interpolation falls back to DOCKER_REPO/IMAGE_NAME
    — the same derivation deploy.py uses — and an explicit DOCKER_IMAGE still wins
    (deploy.py's override)."""
    import os
    import subprocess

    import pytest

    image = _compose()["services"]["bento"]["image"]
    # Offline: the interpolation must carry a DOCKER_IMAGE fallback that derives
    # from DOCKER_REPO/IMAGE_NAME, not a bare ${DOCKER_IMAGE}.
    assert "${DOCKER_IMAGE:-" in image
    assert "${DOCKER_REPO" in image
    assert "${IMAGE_NAME" in image

    # Real check through the compose CLI when available (config needs no daemon).
    if subprocess.run(["docker", "compose", "version"], capture_output=True).returncode != 0:
        pytest.skip("docker compose CLI not available")

    d = _deploy()
    base_env = {
        **os.environ,
        "DOCKER_REPO": "acme",
        "IMAGE_NAME": "tennis-ml",
        "POSTGRES_PASSWORD": "test-only",
    }
    base_env.pop("DOCKER_IMAGE", None)

    def rendered_image(env):
        proc = subprocess.run(
            ["docker", "compose", "-f", str(d.COMPOSE_FILE), "config"],
            cwd=d.ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        for line in proc.stdout.splitlines():
            if line.strip().startswith("image:"):
                return line.strip().split("image:", 1)[1].strip()
        raise AssertionError("no image line in compose config output")

    assert rendered_image(base_env) == "acme/tennis-ml:latest"
    assert rendered_image({**base_env, "DOCKER_IMAGE": "acme/other"}) == "acme/other:latest"


def test_compose_has_pinned_postgres_service():
    """PostgreSQL runs as a Compose service on the pinned pgduckdb image:
    host port 6543 -> container 5432, named volume, init.sql under
    /docker-entrypoint-initdb.d/, and a real healthcheck."""
    cfg = _compose()
    assert "postgres" in cfg["services"]
    svc = cfg["services"]["postgres"]
    assert svc["image"] == "pgduckdb/pgduckdb:17-v1.1.1"
    assert "6543:5432" in svc["ports"]
    assert "healthcheck" in svc
    assert "postgres-data" in cfg.get("volumes", {})
    assert any("/var/lib/postgresql/data" in v for v in svc["volumes"])
    assert any(
        "infra/postgres/init.sql" in v and "docker-entrypoint-initdb.d/init.sql" in v
        for v in svc["volumes"]
    )


def test_compose_postgres_requires_shared_secret():
    """The postgres service authenticates with the same operator credential the
    host commands use; the secret is never defaulted."""
    env = _compose()["services"]["postgres"]["environment"]
    assert env["POSTGRES_USER"] == "${POSTGRES_USER:-postgres}"
    assert env["POSTGRES_DB"] == "${POSTGRES_DB:-tennis}"
    assert env["POSTGRES_PASSWORD"] == "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD must be set}"


def test_compose_bento_reaches_postgres_via_compose_dns():
    """The Bento reaches the pgduckdb Compose service over DNS (postgres:5432),
    not a host gateway, and shares the operator credential."""
    bento_env = _compose()["services"]["bento"]["environment"]
    assert bento_env["POSTGRES_HOST"] == "postgres"
    assert bento_env["POSTGRES_PORT"] == "5432"
    assert bento_env["POSTGRES_USER"] == "${POSTGRES_USER:-postgres}"
    assert bento_env["POSTGRES_DB"] == "${POSTGRES_DB:-tennis}"
    assert bento_env["POSTGRES_PASSWORD"] == "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD must be set}"


def test_compose_bento_depends_on_postgres_healthy():
    """Bento starts only after the pgduckdb service is healthy."""
    deps = _compose()["services"]["bento"]["depends_on"]
    assert deps["postgres"]["condition"] == "service_healthy"


def test_compose_bento_readiness_runs_authenticated_postgres_query():
    """Bento readiness must execute a REAL authenticated PostgreSQL query, not a
    process-liveness HTTP probe: it connects over Compose DNS (postgres:5432),
    authenticates with the shared POSTGRES_PASSWORD, and runs SELECT 1."""
    cfg = _compose()
    assert "healthcheck" in cfg["services"]["bento"]
    test = cfg["services"]["bento"]["healthcheck"]["test"]
    assert test[0] == "CMD-SHELL"
    cmd = test[-1]
    assert "psycopg" in cmd  # the driver is in the serving image
    assert "connect(" in cmd
    assert "postgres" in cmd  # Compose service DNS, not a host gateway
    assert "5432" in cmd  # the container-side PostgreSQL port
    assert "POSTGRES_PASSWORD" in cmd  # real credential, never a liveness-only probe
    assert "SELECT 1" in cmd  # executes an actual query, not a bare URL read


def test_web_image_uses_exactly_one_nginx_worker():
    """The web image must pin nginx to one worker in the MAIN config, not in the
    server-only conf.d file: nginx:alpine's `worker_processes auto` would start
    one worker per visible CPU (10 on this host) for a static SPA that needs
    one. worker_processes is main-context, so it can never live in
    web/nginx.conf (conf.d/default.conf, http context) — this test locks both
    the pin and the placement."""
    root = _deploy().ROOT
    dockerfile = (root / "web" / "Dockerfile").read_text()
    nginx_conf = (root / "web" / "nginx.conf").read_text()

    # The Dockerfile patches the distro main config to exactly one worker and
    # fails the build loudly if the upstream line changes shape.
    assert "worker_processes" in dockerfile
    assert "worker_processes  1;" in dockerfile
    assert "sed -i" in dockerfile
    assert "/etc/nginx/nginx.conf" in dockerfile
    # The server-only conf.d file must stay free of main-context directives.
    assert "worker_processes" not in nginx_conf


def test_build_database_url_host_and_container_share_credential(monkeypatch):
    """Host (127.0.0.1:6543) and Bento-network (postgres:5432) endpoints
    authenticate with the same secret; only host/port differ."""
    import src.constants as c

    monkeypatch.setattr(c, "DATABASE_URL", None)
    monkeypatch.setattr(c, "POSTGRES_USER", "tennis")
    monkeypatch.setattr(c, "POSTGRES_PASSWORD", "s3cret")
    monkeypatch.setattr(c, "POSTGRES_DB", "tennis")
    monkeypatch.setattr(c, "POSTGRES_HOST", "127.0.0.1")
    monkeypatch.setattr(c, "POSTGRES_PORT", "6543")

    assert c.build_database_url() == "postgresql://tennis:s3cret@127.0.0.1:6543/tennis"
    # Inside the Bento container we override POSTGRES_HOST/POSTGRES_PORT.
    monkeypatch.setattr(c, "POSTGRES_PORT", "5432")
    assert c.build_database_url(host="postgres") == (
        "postgresql://tennis:s3cret@postgres:5432/tennis"
    )


def test_build_database_url_explicit_url_overrides_components(monkeypatch):
    import src.constants as c

    monkeypatch.setattr(c, "DATABASE_URL", "postgresql://u:p@db:5432/tennis?sslmode=disable")
    monkeypatch.setattr(c, "POSTGRES_HOST", "127.0.0.1")
    assert c.build_database_url() == "postgresql://u:p@db:5432/tennis?sslmode=disable"
    assert (
        c.build_database_url(host="anything") == "postgresql://u:p@db:5432/tennis?sslmode=disable"
    )
