import pandas as pd
from starlette.requests import Request

from src.serving import service


def test_directory_info_returns_latest_bronze_match_date(monkeypatch):
    monkeypatch.setattr(
        service,
        "execute_df",
        lambda _sql: pd.DataFrame({"latest_match_date": ["2026-07-12"]}),
    )

    request = Request({"type": "http", "method": "GET", "path": "/directory_info"})
    response = bytes(service._directory_info(request).body).decode()

    assert "2026-07-12" in response
