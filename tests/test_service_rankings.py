"""Focused contract tests for official ranking serving.

/rank_history, /players, and /player_profile all read their rank values from
bronze.rankings (weekly official ATP top-200 rows) — never from match rows.
"""

from unittest.mock import patch

import pandas as pd
import pytest
from starlette.testclient import TestClient

from src.db.client import get_conn
from src.serving.service import DATA_APP

client = TestClient(DATA_APP)

# A seeded player id used by the live contract test; its seeded profile exists
# in gold.player_profiles (Sebastian Baez, ARG).
_SEEDED_PLAYER = "B0BI"
# A weekly ranking Monday well after the deterministic seed matches (2026).
_TEST_DATE = "2026-08-03"
_TEST_RANK = 37


# ── Unit (mocked DB) ────────────────────────────────────────────────────────


def test_rank_history_preserves_chronological_order():
    """History returns bronze.rankings rows in SQL order, week by week."""
    df = pd.DataFrame(
        [
            {"ranking_date": "2024-01-01", "rank": 100, "points": 400},
            {"ranking_date": "2024-01-08", "rank": 98, "points": 420},
            {"ranking_date": "2024-01-15", "rank": 95, "points": 460},
        ]
    )
    with patch("src.serving.service.execute_df", return_value=df) as exec:
        resp = client.get("/rank_history?player_id=p1")
    assert resp.status_code == 200
    sql, params = exec.call_args_list[0].args
    assert params == ["p1"]
    # The history is sourced from bronze.rankings only — never match rows.
    assert "bronze.rankings" in sql
    assert "match_events" not in sql
    assert resp.json()["data"]["rank_history"] == [
        {"rank_date": "2024-01-01", "rank": 100},
        {"rank_date": "2024-01-08", "rank": 98},
        {"rank_date": "2024-01-15", "rank": 95},
    ]


def test_rank_history_empty_for_player_without_official_rows():
    """A player with no approved official ranking rows gets an empty history,
    not a fallback derived from match rows."""
    df = pd.DataFrame(columns=["ranking_date", "rank", "points"])
    with patch("src.serving.service.execute_df", return_value=df) as exec:
        resp = client.get("/rank_history?player_id=zzz")
    assert resp.status_code == 200
    sql, params = exec.call_args_list[0].args
    assert params == ["zzz"]
    assert "bronze.rankings" in sql
    assert resp.json()["data"]["rank_history"] == []


def test_players_current_rank_comes_from_bronze_rankings():
    """The /players current_rank is the dbt-materialized column from
    gold.player_profiles (official ranking with match-time fallback)."""
    df = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "display_name": "A",
                "matches_played": 5,
                "latest_rank_points": 1000.0,
                "ioc": "ESP",
                "current_rank": 12,
            }
        ]
    )
    with patch("src.serving.service.execute_df", return_value=df) as exec:
        resp = client.get("/players")
    assert resp.status_code == 200
    players = resp.json()["data"]["players"]
    assert players[0]["current_rank"] == 12
    sql = exec.call_args_list[0].args[0]
    assert "current_rank" in sql
    assert "FROM gold.player_profiles" in sql


def test_rank_history_requires_player_id():
    resp = client.get("/rank_history")
    assert resp.status_code == 400
    assert "player_id" in resp.json()["error"]


def test_rank_history_database_error_returns_500():
    with patch("src.serving.service.execute_df", side_effect=RuntimeError("boom")):
        resp = client.get("/rank_history?player_id=p1")
    assert resp.status_code == 500
    assert "rank history query failed" in resp.json()["error"]


# ── Live database contract ─────────────────────────────────────────────────


def test_live_official_rankings_drive_api_rank_values(postgres_ready, gold_ready):  # noqa: ARG001
    """An inserted bronze.rankings row is what /rank_history reports; /players and
    /player_profile read current_rank from dbt-materialized gold.player_profiles."""
    with get_conn().cursor() as cur:
        cur.execute(
            "DELETE FROM bronze.rankings WHERE player_id = %s AND ranking_date = %s",
            (_SEEDED_PLAYER, _TEST_DATE),
        )
        cur.execute(
            "INSERT INTO bronze.rankings (ranking_date, player_id, rank, points) "
            "VALUES (%s, %s, %s, %s)",
            (_TEST_DATE, _SEEDED_PLAYER, _TEST_RANK, 1500),
        )
    try:
        history = client.get(f"/rank_history?player_id={_SEEDED_PLAYER}").json()["data"][
            "rank_history"
        ]
        assert {"rank_date": _TEST_DATE, "rank": _TEST_RANK} in history
    finally:
        with get_conn().cursor() as cur:
            cur.execute(
                "DELETE FROM bronze.rankings WHERE player_id = %s AND ranking_date = %s",
                (_SEEDED_PLAYER, _TEST_DATE),
            )
