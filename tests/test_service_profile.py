"""Focused tests for the player_profile route including tour_averages."""

from unittest.mock import patch

import pandas as pd
from starlette.testclient import TestClient

from src.serving.service import DATA_APP

client = TestClient(DATA_APP)


def _bio_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": "p1",
                "display_name": "Test Player",
                "handedness": "R",
                "backhand": "2h",
                "height": 185,
                "turned_pro": 2010,
                "birthplace": "Test",
                "summary": "A test player.",
            }
        ]
    )


def _career_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "matches_played": 50,
                "win_rate": 0.6,
                "first_serve_win_pct": 0.75,
                "second_serve_win_pct": 0.55,
                "serve_win_pct": 0.65,
                "break_points_saved_pct": 0.60,
            }
        ]
    )


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame()


def _profile_query_results() -> list[pd.DataFrame]:
    """Results for the five profile queries, in call order."""
    return [_bio_df(), _career_df(), _empty_df(), _empty_df(), _empty_df()]


def _tour_averages_df(**overrides) -> pd.DataFrame:
    """Return a minimal valid tour_averages singleton row as a DataFrame."""
    row = {
        "singleton_id": 1,
        "pool_as_of_date": "2026-01-01",
        "snapshot_pool_rows": 100,
        "snapshot_pool_players": 10,
        "profile_rows": 10,
        "player_match_rows": 200,
        "tour_first_serve_win_pct": 0.72,
        "tour_second_serve_win_pct": 0.52,
        "tour_ace_rate": None,
        "tour_first_serve_pct": None,
        "tour_break_points_saved_pct": None,
        "tour_serve_win_pct": None,
        "tour_return_points_won_pct": None,
        "tour_df_rate": None,
        "tour_aces_per_svc_game": None,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_profile_includes_tour_averages():
    """Profile response includes a tour_averages object with the two exposed rates."""
    with (
        patch("src.serving.service.execute_df", side_effect=_profile_query_results()),
        patch("src.features.tour_averages.execute_df", return_value=_tour_averages_df()),
    ):
        resp = client.get("/player_profile?player_id=p1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    career = data["data"]["career"]
    assert career["first_serve_win_pct"] == 0.75  # player career rate unchanged
    ta = data["data"]["tour_averages"]
    assert ta["first_serve_win_pct"] == 0.72
    assert ta["second_serve_win_pct"] == 0.52


def test_profile_tour_averages_null_when_benchmark_missing():
    """NULL tour benchmark serializes as null in the JSON response."""
    with (
        patch("src.serving.service.execute_df", side_effect=_profile_query_results()),
        patch(
            "src.features.tour_averages.execute_df",
            return_value=_tour_averages_df(
                tour_first_serve_win_pct=None, tour_second_serve_win_pct=None
            ),
        ),
    ):
        resp = client.get("/player_profile?player_id=p1")
    assert resp.status_code == 200
    ta = resp.json()["data"]["tour_averages"]
    assert ta["first_serve_win_pct"] is None
    assert ta["second_serve_win_pct"] is None


def test_profile_performs_exactly_one_singleton_lookup():
    """A single profile request triggers exactly one load_tour_averages() call."""
    with (
        patch("src.serving.service.execute_df", side_effect=_profile_query_results()) as svc_exec,
        patch("src.features.tour_averages.execute_df", return_value=_tour_averages_df()) as ta_exec,
    ):
        resp = client.get("/player_profile?player_id=p1")
    assert resp.status_code == 200
    assert ta_exec.call_count == 1
    assert svc_exec.call_count == 5  # bio, career, surface, form, rank points


def test_profile_errors_on_missing_tour_averages():
    """Clear error when gold.tour_averages is absent or invalid."""
    with (
        patch("src.serving.service.execute_df", side_effect=_profile_query_results()),
        patch("src.features.tour_averages.execute_df", return_value=_empty_df()),
    ):
        resp = client.get("/player_profile?player_id=p1")
    assert resp.status_code == 500
    error = resp.json()["error"].lower()
    assert "tour_averages" in error or "empty" in error
