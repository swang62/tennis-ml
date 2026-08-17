"""Hermetic contract tests for GET /health and the nginx transparent /api proxy."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
from starlette.testclient import TestClient

from src.serving.service import DATA_APP, TennisPredictor

NGINX_TEMPLATE = Path(__file__).resolve().parents[1] / "web" / "nginx.conf.template"

client = TestClient(DATA_APP)


def test_health_ok_when_database_reachable():
    """Healthy envelope only when the authenticated SELECT 1 succeeds."""
    with patch(
        "src.serving.service.execute_df",
        return_value=pd.DataFrame({"?column?": [1]}),
    ) as exec:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "data": {"status": "healthy"}}
    assert exec.call_args_list[0].args[0] == "SELECT 1"


def test_health_503_when_database_unreachable():
    """DB failure -> ok:false 503 with a static body; no exception detail leaks."""
    with patch("src.serving.service.execute_df", side_effect=RuntimeError("boom")):
        resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.json() == {"ok": False, "error": "database unavailable"}
    assert "boom" not in resp.text


def test_nginx_transparent_proxy_and_gated_routes():
    """Public /api/ is a transparent proxy to Bento; model_info and bulk stay
    API-key-gated; model-only /predict is explicitly blocked."""
    conf = NGINX_TEMPLATE.read_text()
    # Transparent proxy present.
    assert "location /api/ {" in conf
    assert "proxy_pass ${BENTO_API_URL}/;" in conf
    # Auth-gated model_info.
    assert "location /api/model_info {" in conf
    assert "proxy_pass ${BENTO_API_URL}/model_info;" in conf
    # Auth-gated bulk.
    assert "location /api/predict_from_ids_bulk {" in conf
    assert "proxy_pass ${BENTO_API_URL}/predict_from_ids_bulk;" in conf
    # API-key guard present (used by both gated routes).
    assert conf.count('$http_x_api_key != "${BENTO_API_KEY}"') == 2
    # Model-only /predict blocked by an exact match.
    assert "location = /api/predict {" in conf
    # No legacy /api/internal route names.
    assert "/api/internal" not in conf
    # No bespoke public mappings survive.
    assert "proxy_pass ${BENTO_API_URL}/health;" not in conf
    # No bespoke OpenAPI location (the transparent proxy exposes /api/docs.json).
    assert "location = /api/openapi.json" not in conf


def test_service_declares_two_workers():
    """Bento runs two worker processes; one max pooled connection each caps
    the app at two database connections, none held while idle."""
    assert TennisPredictor.config.get("workers") == 2
