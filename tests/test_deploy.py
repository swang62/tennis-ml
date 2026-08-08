"""Offline tests for the host deployment path (src/flows/deploy.py).

These test command construction, secure credential handling, and force
behavior WITHOUT a Docker daemon, Docker Hub login/push, Docker Compose, or a
model deployment. build_bento_image / deploy_bento are exercised with their
heavy dependencies (MLflow, BentoML, Docker) mocked away. The deploy flow is
push-only: it never invokes Docker Compose or builds the web image.
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

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["input"] = kwargs.get("input")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)

    d._docker_login()

    assert "--password-stdin" in captured["cmd"]
    assert "explicit-user" in captured["cmd"]
    assert "acme" not in captured["cmd"]  # repo owner is NOT the username here
    assert captured["input"] == "tok\n"


def test_docker_login_username_falls_back_to_repo_owner(monkeypatch):
    d = _deploy()
    monkeypatch.setenv("DOCKER_TOKEN", "tok")
    monkeypatch.delenv("DOCKER_USERNAME", raising=False)
    monkeypatch.setattr(d, "DOCKER_REPO", "acme")

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["input"] = kwargs.get("input")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)

    d._docker_login()

    assert "--password-stdin" in captured["cmd"]
    assert "acme" in captured["cmd"]  # DOCKER_REPO owner is the fallback username
    assert captured["input"] == "tok\n"


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


# --- Command construction (push latest only; no Compose, no web build) ---


def _stub_subprocess(monkeypatch):
    """Replace subprocess.run with a no-op returning success."""
    monkeypatch.setattr("subprocess.run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))


def test_deploy_bento_pushes_only_latest_no_compose_no_web(monkeypatch, tmp_path):
    d = _deploy()
    monkeypatch.setattr(d, "DOCKER_REPO", "acme")
    monkeypatch.setattr(d, "IMAGE_NAME", "tennis-ml")
    monkeypatch.setattr(d, "LOGS", tmp_path)
    monkeypatch.setattr(d, "build_bento_image", lambda **_kwargs: ("tennis-ml:latest", 5))
    monkeypatch.setattr(d, "_docker_login", lambda: None)
    monkeypatch.setattr(d, "_read_state", lambda: {})
    written = {}
    monkeypatch.setattr(d, "_write_state", lambda s: written.update(s))
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    _stub_subprocess(monkeypatch)

    calls = []

    def fake_run_teed(cmd, _log):
        calls.append(list(cmd))
        return None

    monkeypatch.setattr(d, "_run_teed", fake_run_teed)

    d.deploy_bento(force=False)

    # The ONLY command is the push of the Docker Hub `latest` image: no Compose
    # build/up, no web build, no `docker tag` of a versioned image.
    assert calls == [["docker", "push", "acme/tennis-ml:latest"]]
    assert written["deployed_version"] == 5
    assert written["deployed_image"] == "acme/tennis-ml:latest"


def test_deploy_bento_does_not_require_postgres_password(monkeypatch, tmp_path):
    """Deploy pushes the image with no PostgreSQL credential involved — the
    POSTGRES_PASSWORD gate existed only for the removed Compose boot."""
    d = _deploy()
    monkeypatch.setattr(d, "DOCKER_REPO", "acme")
    monkeypatch.setattr(d, "IMAGE_NAME", "tennis-ml")
    monkeypatch.setattr(d, "LOGS", tmp_path)
    monkeypatch.setattr(d, "build_bento_image", lambda **_kwargs: ("tennis-ml:latest", 5))
    monkeypatch.setattr(d, "_docker_login", lambda: None)
    monkeypatch.setattr(d, "_read_state", lambda: {})
    monkeypatch.setattr(d, "_write_state", lambda _s: None)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    pushed = []
    monkeypatch.setattr(d, "_run_teed", lambda cmd, _log: pushed.append(list(cmd)))
    _stub_subprocess(monkeypatch)

    d.deploy_bento(force=False)  # must not raise

    assert pushed == [["docker", "push", "acme/tennis-ml:latest"]]


def test_deploy_bento_forwards_force_to_build(monkeypatch, tmp_path):
    d = _deploy()
    seen = {}

    def fake_build(force=False):
        seen["force"] = force
        return ("tennis-ml:latest", 5)

    monkeypatch.setattr(d, "build_bento_image", fake_build)
    monkeypatch.setattr(d, "LOGS", tmp_path)
    monkeypatch.setattr(d, "_docker_login", lambda: None)
    monkeypatch.setattr(d, "_read_state", lambda: {})
    monkeypatch.setattr(d, "_write_state", lambda _s: None)
    monkeypatch.setattr(d, "_run_teed", lambda _cmd, _log: None)
    _stub_subprocess(monkeypatch)

    d.deploy_bento(force=True)
    assert seen["force"] is True


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
    docker_calls = []
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **_k: (
            (docker_calls.append(list(a[0])) if a else None) or SimpleNamespace(returncode=0)
        ),
    )

    built = {"bento": 0, "container": 0, "image_tags": [], "docker_calls": docker_calls}
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
        built["image_tags"].append(_kwargs.get("image_tag"))

    bentoml["bentoml"].bentos.build_bentofile = build_bentofile
    bentoml["bentoml"].container.build = container_build
    return d, built


def test_build_force_rebuilds_over_cache_hit(monkeypatch):
    d, built = _stub_bento_build(monkeypatch)

    # force=True: rebuild the Bento and containerize even though the cache hits.
    image, version = d.build_bento_image(force=True)
    assert built["bento"] == 1
    assert built["container"] == 1
    # Only the moving local `latest` Docker image tag is ever produced; no
    # versioned image tag and no `docker tag` retagging.
    assert built["image_tags"] == [("tennis-ml:latest",)]
    assert image == "tennis-ml:latest"
    assert version == 7
    assert built["docker_calls"] == []


def test_build_no_force_reuses_cache_hit(monkeypatch):
    d, built = _stub_bento_build(monkeypatch)

    # force=False: cache hit -> no rebuild, and the local image exists -> no
    # containerization.
    d.build_bento_image(force=False)
    assert built["bento"] == 0
    assert built["container"] == 0
    assert built["image_tags"] == []


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
# compose.yaml is the separate manual `pnpm docker` workflow — deploy.py no
# longer references it, so these tests read the file directly.


def _compose():
    import yaml

    return yaml.safe_load((_deploy().ROOT / "compose.yaml").read_text())


def test_compose_image_uses_repository_and_name():
    """The Bento image is derived from the required repository and image name."""
    import os
    import subprocess

    import pytest

    image = _compose()["services"]["bento"]["image"]
    assert image == "${DOCKER_REPO}/${IMAGE_NAME}:latest"

    # Real check through the compose CLI when available (config needs no daemon).
    if subprocess.run(["docker", "compose", "version"], capture_output=True).returncode != 0:
        pytest.skip("docker compose CLI not available")

    d = _deploy()
    base_env = {
        **os.environ,
        "DOCKER_REPO": "acme",
        "IMAGE_NAME": "tennis-ml",
    }

    def rendered_image(env):
        proc = subprocess.run(
            ["docker", "compose", "-f", str(d.ROOT / "compose.yaml"), "config"],
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


def test_compose_has_pinned_postgres_service():
    """PostgreSQL runs as a Compose service on the pinned postgres:18.4 image:
    host port 6543 -> container 5432, named volume at the version-18 parent
    path (never the old version-17 data dir), init.sql under
    /docker-entrypoint-initdb.d/, and a readiness-only healthcheck that does
    not probe any extension."""
    cfg = _compose()
    assert "postgres" in cfg["services"]
    svc = cfg["services"]["postgres"]
    assert svc["image"] == "postgres:18.4"
    assert "6543:5432" in svc["ports"]
    assert "healthcheck" in svc
    assert "postgres-data-18" in cfg.get("volumes", {})
    assert "postgres-data" not in cfg.get("volumes", {})
    assert any(v.endswith(":/var/lib/postgresql") for v in svc["volumes"])
    assert not any(":/var/lib/postgresql/data" in v for v in svc["volumes"])
    assert any(
        "infra/postgres/init.sql" in v and "docker-entrypoint-initdb.d/init.sql" in v
        for v in svc["volumes"]
    )
    healthcheck = svc["healthcheck"]["test"]
    assert healthcheck[0] == "CMD-SHELL"
    assert len(healthcheck) == 2  # readiness-only: a single pg_isready command
    assert "pg_isready" in healthcheck[1]


def test_compose_postgres_uses_fixed_local_dev_credential():
    """The Compose postgres service is password-authenticated with the fixed
    local-dev credential baked into tracked compose.yaml — never a .env
    interpolation, never trust/passwordless."""
    env = _compose()["services"]["postgres"]["environment"]
    assert env["POSTGRES_USER"] == "postgres"
    assert env["POSTGRES_DB"] == "tennis"
    assert env["POSTGRES_PASSWORD"] == "password"
    assert "POSTGRES_HOST_AUTH_METHOD" not in env


def test_compose_bento_gets_single_database_url():
    """The Bento receives exactly one application connection variable — the
    password-bearing Compose DATABASE_URL over postgres service DNS. No
    POSTGRES_* auth variables remain."""
    bento_env = _compose()["services"]["bento"]["environment"]
    assert bento_env == {"DATABASE_URL": "postgresql://postgres:password@postgres:5432/tennis"}


def test_compose_bento_depends_on_postgres_healthy():
    """Bento starts only after the postgres service is healthy."""
    deps = _compose()["services"]["bento"]["depends_on"]
    assert deps["postgres"]["condition"] == "service_healthy"


def test_compose_bento_readiness_runs_authenticated_postgres_query():
    """Bento readiness must execute a REAL authenticated PostgreSQL query, not a
    process-liveness HTTP probe: it connects over Compose DNS (postgres:5432)
    using the single DATABASE_URL and runs SELECT 1."""
    cfg = _compose()
    assert "healthcheck" in cfg["services"]["bento"]
    test = cfg["services"]["bento"]["healthcheck"]["test"]
    assert test[0] == "CMD-SHELL"
    cmd = test[-1]
    assert "psycopg" in cmd  # the driver is in the serving image
    assert "connect(" in cmd
    assert "DATABASE_URL" in cmd  # connects via the single URL (postgres:5432, password)
    assert "POSTGRES_PASSWORD" not in cmd  # no separate auth variable
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


def test_web_dockerfile_builds_with_pnpm_frozen_lockfile():
    """The web image must build with pnpm (Corepack) against the FROZEN
    dashboard lockfile — npm (package-lock.json / npm ci) must
    not reappear, and the install layer must be keyed on the manifests only so
    source edits don't blow the dependency cache."""
    dockerfile = (_deploy().ROOT / "web" / "Dockerfile").read_text()
    assert "corepack enable" in dockerfile
    assert "pnpm install --frozen-lockfile" in dockerfile
    assert "pnpm-lock.yaml" in dockerfile
    # npm is gone: no npm ci, no package-lock.json, no npm run build.
    assert "npm ci" not in dockerfile
    assert "package-lock.json" not in dockerfile
    assert "npm run" not in dockerfile


def test_compose_bento_has_no_host_published_port():
    """nginx (web) is the only public API entrypoint — the Bento must not
    publish a host port (no `3000:3000`)."""
    cfg = _compose()
    assert "ports" not in cfg["services"]["bento"]


def test_nginx_proxies_only_allowlisted_routes():
    """nginx must proxy exactly the allowlisted SPA routes to the Bento and
    reject everything else: model-only /predict, unknown /api paths, and any
    /api/ path that is not allowlisted."""
    conf = (_deploy().ROOT / "web" / "nginx.conf").read_text()
    # Allowlisted GET routes (stripped: /api/players -> /players).
    for route in (
        "players",
        "player_profile",
        "rank_history",
        "match_history",
        "head_to_head",
        "similar_players",
    ):
        assert f"location /api/{route}" in conf
        assert f"proxy_pass http://bento:3000/{route};" in conf
    # Allowlisted POST route.
    assert "location /api/predict_from_ids" in conf
    assert "proxy_pass http://bento:3000/predict_from_ids;" in conf
    # The broad /api/ catch-all proxy is gone.
    assert "proxy_pass http://bento:3000/;" not in conf
    # Model-only /predict is explicitly rejected; unknown /api paths 404.
    assert "location /api/predict" in conf
    assert "return 403" in conf
    assert "return 404" in conf


def test_nginx_hardens_prediction_and_limits():
    """Prediction hardening: POST-only, JSON-only, bounded body, per-IP
    rate/connection limits, bounded proxy timeouts, and generic gateway
    errors."""
    conf = (_deploy().ROOT / "web" / "nginx.conf").read_text()
    # Method + content-type + size guards on the prediction route.
    assert "limit_except POST" in conf
    assert "content-type must be application/json" in conf
    assert "client_max_body_size" in conf
    # Per-IP rate + connection limits.
    assert "limit_req_zone" in conf
    assert "limit_conn_zone" in conf
    assert "limit_req zone=api_req" in conf
    assert "limit_conn api_conn" in conf
    # Bounded proxy timeouts.
    assert "proxy_connect_timeout" in conf
    assert "proxy_read_timeout" in conf
    assert "proxy_send_timeout" in conf
    # Generic gateway errors (intercept + JSON error page).
    assert "proxy_intercept_errors on" in conf
    assert "error_page 502 504" in conf
    assert '"ok":false' in conf


def test_build_database_url_returns_passwordless_local_url(monkeypatch):
    """Homebrew trust path: the passwordless DATABASE_URL is returned verbatim."""
    import src.constants as c

    monkeypatch.setattr(c, "DATABASE_URL", "postgresql://steve@127.0.0.1:5432/postgres")
    assert c.build_database_url() == "postgresql://steve@127.0.0.1:5432/postgres"


def test_build_database_url_returns_password_bearing_compose_url(monkeypatch):
    """Compose path: the password-bearing DATABASE_URL is returned verbatim."""
    import src.constants as c

    monkeypatch.setattr(c, "DATABASE_URL", "postgresql://postgres:password@postgres:5432/tennis")
    assert c.build_database_url() == "postgresql://postgres:password@postgres:5432/tennis"


def test_build_database_url_missing_contract_fails_fast(monkeypatch):
    """No DATABASE_URL means no application connection: fail fast rather than
    silently target some default database."""
    import pytest

    import src.constants as c

    monkeypatch.setattr(c, "DATABASE_URL", None)
    with pytest.raises(RuntimeError, match="missing PostgreSQL configuration"):
        c.build_database_url()
