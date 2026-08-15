"""Focused tests for the player_profile route's single point query."""

from unittest.mock import patch

import pandas as pd
import pytest
from starlette.testclient import TestClient

from src.serving.service import DATA_APP

client = TestClient(DATA_APP)


def _profile_row(**overrides) -> pd.DataFrame:
    """One bronze metadata row joined to the gold aggregates and the tour
    singleton.

    Mirrors the columns the consolidated profile query returns: the bronze
    metadata the handler reads plus every materialized gold aggregate plus the
    ten tour benchmark columns.
    """
    row = {
        # identity
        "player_id": "p1",
        "display_name": "Test Player",
        "atp_name": "T. Player",
        "birthdate": "1998-04-01",
        "weight": 80,
        "height": 185,
        "turned_pro": 2010,
        "birthplace": "Test",
        "coaches": "Coach",
        "handedness": "R",
        "backhand": "2h",
        "ioc": "ESP",
        "summary": "A test player.",
        # career
        "match_count": 50,
        "latest_match_date": "2026-01-15",
        # rank
        "latest_rank_points": 1200.0,
        "earliest_rank_points": 800.0,
        "earliest_rank_points_date": "2024-01-01",
        "latest_rank_points_date": "2026-01-15",
        "rank_points_delta": 400.0,
        "current_rank": 7,
        # serve
        "first_serve_in_pct": 0.62,
        "aces_per_first_serve": 0.09,
        "first_serve_points_won_pct": 0.75,
        "second_serve_points_won_pct": 0.55,
        "overall_serve_points_won_pct": 0.65,
        "double_faults_per_serve_point": 0.03,
        "aces_per_service_game": 0.45,
        "break_points_saved_pct": 0.60,
        # return
        "return_points_won_pct": 0.42,
        "first_serve_return_points_won_pct": 0.30,
        "second_serve_return_points_won_pct": 0.52,
        "break_point_conversion_pct": 0.40,
        "break_point_opportunities_per_return_game": 0.55,
        # surface
        "hard_matches": 30,
        "clay_matches": 10,
        "grass_matches": 10,
        "hard_win_rate": 0.60,
        "clay_win_rate": 0.50,
        "grass_win_rate": 0.70,
        # form
        "recent_snapshot_date": "2026-01-10",
        "win_rate_10": 0.8,
        # tour benchmarks (singleton side of the cross join)
        "tour_first_serve_win_pct": 0.72,
        "tour_second_serve_win_pct": 0.52,
        "tour_ace_rate": 0.08,
        "tour_first_serve_pct": 0.60,
        "tour_break_points_saved_pct": 0.58,
        "tour_serve_win_pct": 0.63,
        "tour_return_points_won_pct": 0.40,
        "tour_df_rate": 0.04,
        "tour_aces_per_svc_game": 0.40,
        "tour_break_point_opportunities_per_return_game": 0.50,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_profile_single_query_returns_full_contract():
    """One DB call returns identity, career, serve, return, surface, form, rank."""
    with patch("src.serving.service.execute_df", side_effect=[_profile_row()]) as exec:
        resp = client.get("/player_profile?player_id=p1")
    assert resp.status_code == 200
    assert exec.call_count == 1
    data = resp.json()["data"]

    # identity (existing flat keys plus the new materialized columns)
    assert data["player_id"] == "p1"
    assert data["display_name"] == "Test Player"
    assert data["atp_name"] == "T. Player"
    assert data["birthdate"] == "1998-04-01"
    assert data["weight"] == 80
    assert data["handedness"] == "R"
    assert data["backhand"] == "2h"
    assert data["ioc"] == "ESP"
    assert data["iso2"] == "ES"
    assert data["country_name"] == "Spain"

    # career
    assert data["career"] == {"matches_played": 50, "latest_match_date": "2026-01-15"}

    # serve: all eight materialized metrics
    assert data["serve"] == {
        "first_serve_in_pct": 0.62,
        "aces_per_first_serve": 0.09,
        "first_serve_points_won_pct": 0.75,
        "second_serve_points_won_pct": 0.55,
        "overall_serve_points_won_pct": 0.65,
        "double_faults_per_serve_point": 0.03,
        "aces_per_service_game": 0.45,
        "break_points_saved_pct": 0.60,
    }

    # return: all five materialized metrics
    assert data["return"] == {
        "return_points_won_pct": 0.42,
        "first_serve_return_points_won_pct": 0.30,
        "second_serve_return_points_won_pct": 0.52,
        "break_point_conversion_pct": 0.40,
        "break_point_opportunities_per_return_game": 0.55,
    }

    # surface: three fixed surfaces, unplayed surfaces show 0 matches/null rate
    assert data["surface_rates"] == [
        {"surface": "clay", "matches": 10, "win_rate": 0.50},
        {"surface": "grass", "matches": 10, "win_rate": 0.70},
        {"surface": "hard", "matches": 30, "win_rate": 0.60},
    ]

    # form and rank trend preserved
    assert data["recent_form"] == {"snapshot_date": "2026-01-10", "last_10_win_rate": 0.8}
    assert data["rank_points_trend"] == {"earliest": 800.0, "latest": 1200.0, "delta": 400.0}

    # new rank object with the full materialized set
    assert data["rank"] == {
        "current_rank": 7,
        "latest_rank_points": 1200.0,
        "earliest_rank_points": 800.0,
        "earliest_rank_points_date": "2024-01-01",
        "latest_rank_points_date": "2026-01-15",
        "rank_points_delta": 400.0,
    }

    # existing tour averages object preserved
    assert data["tour_averages"] == {"first_serve_win_pct": 0.72, "second_serve_win_pct": 0.52}


def test_profile_tour_comparisons_are_player_minus_tour_deltas():
    """Deltas computed in Python; return-side benchmarks derive from serve complements."""
    with patch("src.serving.service.execute_df", side_effect=[_profile_row()]):
        resp = client.get("/player_profile?player_id=p1")
    assert resp.status_code == 200
    tc = resp.json()["data"]["tour_comparisons"]

    assert tc["first_serve_in_pct"] == pytest.approx(0.62 - 0.60)
    assert tc["aces_per_first_serve"] == pytest.approx(0.09 - 0.08)
    assert tc["first_serve_points_won_pct"] == pytest.approx(0.75 - 0.72)
    assert tc["second_serve_points_won_pct"] == pytest.approx(0.55 - 0.52)
    assert tc["overall_serve_points_won_pct"] == pytest.approx(0.65 - 0.63)
    assert tc["double_faults_per_serve_point"] == pytest.approx(0.03 - 0.04)
    assert tc["aces_per_service_game"] == pytest.approx(0.45 - 0.40)
    assert tc["break_points_saved_pct"] == pytest.approx(0.60 - 0.58)
    assert tc["return_points_won_pct"] == pytest.approx(0.42 - 0.40)
    # return-side benchmarks are 1 - the serve benchmark
    assert tc["first_serve_return_points_won_pct"] == pytest.approx(0.30 - (1 - 0.72))
    assert tc["second_serve_return_points_won_pct"] == pytest.approx(0.52 - (1 - 0.52))
    assert tc["break_point_conversion_pct"] == pytest.approx(0.40 - (1 - 0.58))
    assert tc["break_point_opportunities_per_return_game"] == pytest.approx(0.55 - 0.50)


def test_profile_null_benchmark_flows_through():
    """NULL tour benchmark -> null tour average and null delta (never false zero)."""
    with patch(
        "src.serving.service.execute_df",
        side_effect=[_profile_row(tour_first_serve_win_pct=None)],
    ):
        resp = client.get("/player_profile?player_id=p1")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["tour_averages"]["first_serve_win_pct"] is None
    assert data["tour_comparisons"]["first_serve_points_won_pct"] is None
    assert data["tour_comparisons"]["first_serve_return_points_won_pct"] is None


def test_profile_zero_match_player():
    """Zero-match players keep identity, zero counts, and null rates."""
    row = {
        "match_count": 0,
        "latest_match_date": None,
        "latest_rank_points": None,
        "earliest_rank_points": None,
        "earliest_rank_points_date": None,
        "latest_rank_points_date": None,
        "rank_points_delta": None,
        "current_rank": None,
        "first_serve_in_pct": None,
        "aces_per_first_serve": None,
        "first_serve_points_won_pct": None,
        "second_serve_points_won_pct": None,
        "overall_serve_points_won_pct": None,
        "double_faults_per_serve_point": None,
        "aces_per_service_game": None,
        "break_points_saved_pct": None,
        "return_points_won_pct": None,
        "first_serve_return_points_won_pct": None,
        "second_serve_return_points_won_pct": None,
        "break_point_conversion_pct": None,
        "break_point_opportunities_per_return_game": None,
        "hard_matches": 0,
        "clay_matches": 0,
        "grass_matches": 0,
        "hard_win_rate": None,
        "clay_win_rate": None,
        "grass_win_rate": None,
        "recent_snapshot_date": None,
        "win_rate_10": None,
    }
    with patch("src.serving.service.execute_df", side_effect=[_profile_row(**row)]):
        resp = client.get("/player_profile?player_id=p1")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["career"]["matches_played"] == 0
    assert data["career"]["latest_match_date"] is None
    assert data["recent_form"] is None
    assert data["rank_points_trend"] is None
    assert data["rank"]["current_rank"] is None
    assert data["rank"]["latest_rank_points"] is None
    assert all(s["win_rate"] is None and s["matches"] == 0 for s in data["surface_rates"])
    assert all(v is None for v in data["serve"].values())
    assert all(v is None for v in data["return"].values())


def test_profile_uses_parameterized_point_query():
    """The handler binds exactly one player_id to a single parameterized query."""
    with patch("src.serving.service.execute_df", side_effect=[_profile_row()]) as exec:
        resp = client.get("/player_profile?player_id=p1")
    assert resp.status_code == 200
    sql, params = exec.call_args_list[0].args
    assert params == ["p1"]
    assert "%s" in sql
    assert "FROM bronze.player_profiles" in sql
    assert "JOIN gold.player_profiles" in sql
    assert "gold.tour_averages" in sql
    # Ownership join keys are player_id on both sides (bronze PK in schema.sql;
    # gold PK re-applied by the dbt post-hook) — a direct equality probe.
    assert "LEFT JOIN gold.player_profiles gp ON gp.player_id = bp.player_id" in sql


def test_profile_country_metadata_unk_fallback():
    """Missing/invalid IOC resolves to the UNK country row, never a raw guess."""
    for bad_ioc in ("TST", None):
        with patch(
            "src.serving.service.execute_df",
            side_effect=[_profile_row(ioc=bad_ioc)],
        ):
            resp = client.get("/player_profile?player_id=p1")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["ioc"] == "UNK"
        assert data["iso2"] == ""
        assert data["country_name"] == "Country unknown"


def test_profile_unknown_player_404():
    """Empty result (unknown player) -> 404, no further queries."""
    with patch("src.serving.service.execute_df", side_effect=[pd.DataFrame()]) as exec:
        resp = client.get("/player_profile?player_id=nobody")
    assert resp.status_code == 404
    assert "unknown player_id" in resp.json()["error"]
    assert exec.call_count == 1


def test_profile_requires_player_id():
    resp = client.get("/player_profile")
    assert resp.status_code == 400
    assert "player_id" in resp.json()["error"]


def test_profile_database_error_returns_500():
    with patch("src.serving.service.execute_df", side_effect=RuntimeError("connection lost")):
        resp = client.get("/player_profile?player_id=p1")
    assert resp.status_code == 500
    assert "profile query failed" in resp.json()["error"]


def test_profile_fewer_than_10_matches_valid_win_rate():
    """Players with fewer than 10 matches still expose a valid last-10 win rate."""
    with patch(
        "src.serving.service.execute_df",
        side_effect=[_profile_row(match_count=3, win_rate_10=2 / 3)],
    ):
        resp = client.get("/player_profile?player_id=p1")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["career"]["matches_played"] == 3
    assert data["recent_form"] == {"snapshot_date": "2026-01-10", "last_10_win_rate": 2 / 3}


def test_profile_zero_denominators_null_not_false_zero():
    """Metrics whose denominators are missing are null, never a false zero."""
    with patch(
        "src.serving.service.execute_df",
        side_effect=[
            _profile_row(
                aces_per_service_game=None,  # no service_games
                break_points_saved_pct=None,  # no break_points_faced
                break_point_conversion_pct=None,  # no opp break_points_faced
                break_point_opportunities_per_return_game=None,  # no opp service_games
            )
        ],
    ):
        resp = client.get("/player_profile?player_id=p1")
    assert resp.status_code == 200
    data = resp.json()["data"]
    # zero-denominator metrics stay null...
    assert data["serve"]["aces_per_service_game"] is None
    assert data["serve"]["break_points_saved_pct"] is None
    assert data["return"]["break_point_conversion_pct"] is None
    assert data["return"]["break_point_opportunities_per_return_game"] is None
    # ...while the remaining metrics keep their values
    assert data["serve"]["first_serve_in_pct"] == 0.62
    assert data["serve"]["overall_serve_points_won_pct"] == 0.65
    assert data["return"]["second_serve_return_points_won_pct"] == 0.52


def test_profile_matches_without_snapshot_recent_form_null():
    """A player with matches but no rolling snapshot has a null recent form."""
    with patch(
        "src.serving.service.execute_df",
        side_effect=[_profile_row(match_count=5, recent_snapshot_date=None, win_rate_10=None)],
    ):
        resp = client.get("/player_profile?player_id=p1")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["career"]["matches_played"] == 5
    assert data["recent_form"] is None
