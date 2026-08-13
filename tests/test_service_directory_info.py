from pathlib import Path

import pandas as pd
from starlette.requests import Request

from src.serving import service

NGINX_TEMPLATE = Path(__file__).resolve().parents[1] / "web" / "nginx.conf.template"


def test_nginx_allowlists_directory_info():
    """GET /api/directory_info must be proxied, not caught by the /api/ 404."""
    conf = NGINX_TEMPLATE.read_text()
    assert "location /api/directory_info {" in conf
    assert "proxy_pass ${BENTO_API_URL}/directory_info;" in conf


def test_directory_info_returns_latest_bronze_match_date(monkeypatch):
    monkeypatch.setattr(
        service,
        "execute_df",
        lambda _sql: pd.DataFrame({"latest_match_date": ["2026-07-12"]}),
    )

    request = Request({"type": "http", "method": "GET", "path": "/directory_info"})
    response = bytes(service._directory_info(request).body).decode()

    assert "2026-07-12" in response
