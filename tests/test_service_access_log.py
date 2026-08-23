"""Hermetic tests for production request access logging."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pandas as pd
from bentoml._internal.server.http.access import AccessLogMiddleware
from starlette.testclient import TestClient

from src.serving.service import DATA_APP, _effective_log_level

# Mirror production's BentoML middleware wrapper for hermetic in-process assertions.
client = TestClient(
    AccessLogMiddleware(
        DATA_APP,
        has_request_content_length=True,
        has_request_content_type=True,
        has_response_content_length=True,
        has_response_content_type=True,
    )
)


def test_effective_log_level_maps_log_level_case_insensitively(monkeypatch):
    for raw, expected in (
        ("INFO", logging.INFO),
        ("info", logging.INFO),  # case-insensitive, existing behavior
        ("Warning", logging.WARNING),
        ("DEBUG", logging.DEBUG),
        ("bogus", logging.INFO),  # unknown values fall back to INFO
        (None, logging.INFO),  # unset -> INFO
    ):
        if raw is None:
            monkeypatch.delenv("LOG_LEVEL", raising=False)
        else:
            monkeypatch.setenv("LOG_LEVEL", raw)
        assert _effective_log_level() == expected


def test_info_level_emits_one_concise_access_log_per_request(caplog):
    with (
        patch("src.serving.service.execute_df", return_value=pd.DataFrame()),
        caplog.at_level(logging.INFO, logger="bentoml.access"),
    ):
        resp = client.get("/rank_history?player_id=p1")

    assert resp.status_code == 200
    lines = [r for r in caplog.records if r.name == "bentoml.access"]
    assert len(lines) == 1


def test_warning_level_suppresses_info_access_logs(caplog):
    with (
        patch("src.serving.service.execute_df", return_value=pd.DataFrame()),
        caplog.at_level(logging.WARNING, logger="bentoml.access"),
    ):
        resp = client.get("/rank_history?player_id=p1")

    assert resp.status_code == 200
    assert not [r for r in caplog.records if r.name == "bentoml.access"]


def test_error_responses_still_get_an_access_log_line(caplog):
    with caplog.at_level(logging.INFO, logger="bentoml.access"):
        resp = client.get("/rank_history")  # missing player_id -> 400

    assert resp.status_code == 400
    lines = [r for r in caplog.records if r.name == "bentoml.access"]
    assert len(lines) == 1
    assert "status=400" in lines[0].getMessage()
