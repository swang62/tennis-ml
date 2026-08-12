"""Hermetic contract tests for GET /health and its nginx allowlist entry."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
from starlette.testclient import TestClient

from src.serving.service import DATA_APP

client = TestClient(DATA_APP)

NGINX_TEMPLATE = Path(__file__).resolve().parents[1] / "web" / "nginx.conf.template"


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


def test_nginx_allowlists_health_and_nothing_else():
    """GET /api/health is the only public health route: no readyz, no bare
    model_info, no internal prediction paths on the public allowlist."""
    conf = NGINX_TEMPLATE.read_text()
    assert "location /api/health {" in conf
    assert "proxy_pass ${BENTO_API_URL}/health;" in conf
    assert "${BENTO_API_URL}/readyz" not in conf
    assert "location /api/model_info" not in conf
    assert "location /api/predict_from_ids_bulk" not in conf
    # Model-only /predict stays explicitly blocked.
    assert "location /api/predict {" in conf
