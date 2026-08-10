"""Contract tests for /players, /rank_history, /match_history, and /head_to_head."""

from unittest.mock import patch

import pandas as pd
import pytest
from starlette.testclient import TestClient

from src.db.client import execute_df
from src.serving.service import DATA_APP

client = TestClient(DATA_APP)


# ── /players ────────────────────────────────────────────────────────────────


def _players_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": "p2",
                "display_name": "B Player",
                "matches_played": 40,
                "latest_rank_points": 1500.0,
                "ioc": "ESP",
                "current_rank": 1,
            },
            {
                "player_id": "p1",
                "display_name": "A Player",
                "matches_played": 20,
                "latest_rank_points": None,  # never had positive points
                "ioc": "UNK",
                "current_rank": None,
            },
            {
                "player_id": "p3",
                "display_name": "C Player",
                "matches_played": 60,
                "latest_rank_points": 900.0,
                "ioc": "ARG",
                "current_rank": 2,
            },
        ]
    )


def test_players_returns_materialized_directory():
    """Directory entries preserve query order and expose rank data, nulls included."""
    with patch("src.serving.service.execute_df", return_value=_players_df()) as exec:
        resp = client.get("/players")
    assert resp.status_code == 200
    assert exec.call_count == 1
    players = resp.json()["data"]["players"]
    assert [p["player_id"] for p in players] == ["p2", "p1", "p3"]
    assert players[0] == {
        "player_id": "p2",
        "display_name": "B Player",
        "matches_played": 40,
        "latest_rank_points": 1500.0,
        "ioc": "ESP",
        "iso2": "ES",
        "country_name": "Spain",
        "current_rank": 1,
    }
    # unranked players keep the entry with null rank data and UNK country
    assert players[1]["latest_rank_points"] is None
    assert players[1]["current_rank"] is None
    assert players[1]["ioc"] == "UNK"
    assert players[1]["iso2"] == ""
    assert players[1]["country_name"] == "Country unknown"
    assert players[2]["iso2"] == "AR"
    assert players[2]["country_name"] == "Argentina"


def test_players_sql_reads_profiles_without_match_aggregation():
    """One ordered read from gold.player_profiles; current_rank is dbt-
    materialized (no per-query join to bronze.rankings or bronze.match_events)."""
    with patch("src.serving.service.execute_df", return_value=_players_df()) as exec:
        resp = client.get("/players")
    assert resp.status_code == 200
    sql = exec.call_args_list[0].args[0]
    assert "FROM gold.player_profiles" in sql
    assert "bronze.rankings" not in sql
    assert "bronze.match_events" not in sql
    assert "ORDER BY current_rank NULLS LAST, display_name, player_id" in sql


def test_players_empty():
    with patch("src.serving.service.execute_df", return_value=pd.DataFrame()):
        resp = client.get("/players")
    assert resp.status_code == 200
    assert resp.json()["data"]["players"] == []


def test_players_database_error_returns_500():
    with patch("src.serving.service.execute_df", side_effect=RuntimeError("boom")):
        resp = client.get("/players")
    assert resp.status_code == 500
    assert "players query failed" in resp.json()["error"]


def test_players_tie_ordering_by_player_id_deterministic():
    """Equal current ranks are ordered deterministically by the SQL order."""
    df = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "display_name": "A Player",
                "matches_played": 5,
                "latest_rank_points": 1000.0,
                "ioc": "ESP",
                "current_rank": 1,
            },
            {
                "player_id": "p3",
                "display_name": "C Player",
                "matches_played": 5,
                "latest_rank_points": 1000.0,
                "ioc": "ESP",
                "current_rank": 2,
            },
            {
                "player_id": "p2",
                "display_name": "B Player",
                "matches_played": 5,
                "latest_rank_points": None,
                "ioc": "UNK",
                "current_rank": None,
            },
        ]
    )
    with patch("src.serving.service.execute_df", return_value=df) as exec:
        resp = client.get("/players")
    assert resp.status_code == 200
    players = resp.json()["data"]["players"]
    # query order preserved (handler never re-sorts)
    assert [p["player_id"] for p in players] == ["p1", "p3", "p2"]
    sql = exec.call_args_list[0].args[0]
    assert "ORDER BY current_rank NULLS LAST, display_name, player_id" in sql


def test_players_null_rank_ordered_last():
    """Profiles without a bronze.rankings row keep a null current_rank and sort last."""
    df = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "display_name": "A",
                "matches_played": 5,
                "latest_rank_points": 900.0,
                "ioc": "ESP",
                "current_rank": 1,
            },
            {
                "player_id": "p2",
                "display_name": "B",
                "matches_played": 2,
                "latest_rank_points": None,
                "ioc": "UNK",
                "current_rank": None,
            },
        ]
    )
    with patch("src.serving.service.execute_df", return_value=df):
        resp = client.get("/players")
    assert resp.status_code == 200
    players = resp.json()["data"]["players"]
    assert [p["player_id"] for p in players] == ["p1", "p2"]
    assert players[1]["current_rank"] is None
    assert players[1]["latest_rank_points"] is None


def test_players_zero_match_profile_present():
    """Profiles with zero matches are still present in the directory."""
    df = pd.DataFrame(
        [
            {
                "player_id": "p0",
                "display_name": "No Match",
                "matches_played": 0,
                "latest_rank_points": None,
                "ioc": "UNK",
                "current_rank": None,
            }
        ]
    )
    with patch("src.serving.service.execute_df", return_value=df):
        resp = client.get("/players")
    assert resp.status_code == 200
    players = resp.json()["data"]["players"]
    assert len(players) == 1
    assert players[0]["matches_played"] == 0
    assert players[0]["player_id"] == "p0"


def test_players_stable_ordering_repeated_calls():
    """Repeated calls with identical data produce identical responses."""
    with patch("src.serving.service.execute_df", return_value=_players_df()):
        first = client.get("/players").json()["data"]["players"]
        second = client.get("/players").json()["data"]["players"]
    assert first == second
    assert [p["player_id"] for p in first] == [p["player_id"] for p in second]


def test_players_ignores_unexpected_query_params():
    """The players query is static; unexpected params never reach SQL."""
    with patch("src.serving.service.execute_df", return_value=_players_df()) as exec:
        resp = client.get("/players?x=1%3BDROP%20TABLE%20gold.player_profiles")
    assert resp.status_code == 200
    args = exec.call_args_list[0].args
    assert len(args) == 1  # players query takes no params
    assert "%s" not in args[0]


def test_players_live_directory_contains_all_profiles(postgres_ready, gold_ready):  # noqa: ARG001
    """Every profile is present (including zero-match players) and no served
    rank point is a false zero: only positive or null."""
    resp = client.get("/players")
    assert resp.status_code == 200
    players = resp.json()["data"]["players"]
    expected_row = execute_df("SELECT COUNT(*) FROM gold.player_profiles").iloc[0, 0]
    expected: int = int(expected_row)  # type: ignore[arg-type]
    assert len(players) == expected
    assert any(p["matches_played"] == 0 for p in players)
    for p in players:
        assert p["latest_rank_points"] is None or p["latest_rank_points"] > 0


def test_players_latest_rank_points_ignore_newer_zero_observations(postgres_ready, gold_ready):  # noqa: ARG001
    """gold.latest_rank_points honors the latest POSITIVE observation: a newer
    zero/null rank-points observation must not override it. Cross-checked
    against the silver-derived ARRAY_AGG FILTER contract and the live endpoint."""
    mismatches_row = execute_df(
        """
            WITH expected AS (
                SELECT player_id,
                    (ARRAY_AGG(player_rank_points ORDER BY match_date DESC, match_id DESC)
                        FILTER (WHERE player_rank_points > 0))[1] AS exp_latest
                FROM silver.player_matches
                GROUP BY player_id
            )
            SELECT COUNT(*) FROM expected e
            JOIN gold.player_profiles g ON g.player_id = e.player_id
            WHERE g.latest_rank_points IS DISTINCT FROM e.exp_latest
            """
    ).iloc[0, 0]
    mismatches: int = int(mismatches_row)  # type: ignore[arg-type]
    assert mismatches == 0

    resp = client.get("/players")
    served = {p["player_id"]: p["latest_rank_points"] for p in resp.json()["data"]["players"]}
    gold = execute_df("SELECT player_id, latest_rank_points FROM gold.player_profiles")
    for _, row in gold.iterrows():
        served_val = served[str(row["player_id"])]
        expected_val = row["latest_rank_points"]
        if pd.isna(expected_val):
            assert served_val is None
        else:
            assert served_val == pytest.approx(float(expected_val))


def test_players_live_null_rank_ordered_last(postgres_ready, gold_ready):  # noqa: ARG001
    """Unranked players (no bronze.rankings row) sort after all ranked players."""
    players = client.get("/players").json()["data"]["players"]
    ranks = [p["current_rank"] for p in players]
    ranked = [r for r in ranks if r is not None]
    unranked = [r for r in ranks if r is None]
    assert ranks == ranked + unranked
    assert ranked == sorted(ranked)  # ascending, distinct via player_id tie-break


def test_players_live_stable_ordering(postgres_ready, gold_ready):  # noqa: ARG001
    a = client.get("/players").json()["data"]["players"]
    b = client.get("/players").json()["data"]["players"]
    assert a == b


# ── /rank_history ───────────────────────────────────────────────────────────


def _rank_history_df() -> pd.DataFrame:
    # bronze.rankings rows, chronological (the query orders by ranking_date).
    return pd.DataFrame(
        [
            {"ranking_date": "2024-01-01", "rank": 100, "points": 400},
            {"ranking_date": "2024-02-05", "rank": 90, "points": 500},
            {"ranking_date": "2024-03-04", "rank": 88, "points": 530},
        ]
    )


def test_rank_history_shape_and_parameter_binding():
    with patch("src.serving.service.execute_df", return_value=_rank_history_df()) as exec:
        resp = client.get("/rank_history?player_id=p1")
    assert resp.status_code == 200
    sql, params = exec.call_args_list[0].args
    assert params == ["p1"]
    assert "%s" in sql
    data = resp.json()["data"]
    assert data["player_id"] == "p1"
    assert data["rank_history"] == [
        {"rank_date": "2024-01-01", "rank": 100},
        {"rank_date": "2024-02-05", "rank": 90},
        {"rank_date": "2024-03-04", "rank": 88},
    ]


def test_rank_history_reads_bronze_rankings_not_match_events():
    """History is weekly official rows from bronze.rankings, never match ranks."""
    with patch("src.serving.service.execute_df", return_value=pd.DataFrame()) as exec:
        resp = client.get("/rank_history?player_id=p1")
    assert resp.status_code == 200
    sql = exec.call_args_list[0].args[0]
    assert "FROM bronze.rankings" in sql
    assert "ORDER BY ranking_date" in sql
    assert "bronze.match_events" not in sql


def test_rank_history_empty_for_player_without_rankings():
    """A player with no bronze.rankings rows gets an empty history, not a 404."""
    with patch("src.serving.service.execute_df", return_value=pd.DataFrame()):
        resp = client.get("/rank_history?player_id=p1")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["player_id"] == "p1"
    assert data["rank_history"] == []


def test_rank_history_requires_player_id():
    resp = client.get("/rank_history")
    assert resp.status_code == 400
    assert "player_id" in resp.json()["error"]


# ── /match_history ──────────────────────────────────────────────────────────


def _match_history_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m2",
                "match_date": "2026-01-20",
                "tournament": "grand_slam",
                "tournament_name": "Australian Open",
                "surface": "hard",
                "round": "r32",
                "opponent_id": "p9",
                "opponent_name": "Rival",
                "player_ranking": 5,
                "match_won": 0,
                "aces": 12,
                "double_faults": 3,
                "first_serve_points_won": 50,
                "second_serve_points_won": 20,
                "total_serve_points": 100,
                "service_games": 14,
                "break_points_saved": 4,
                "break_points_faced": 6,
            },
            {
                "match_id": "m1",
                "match_date": "2026-01-10",
                "tournament": "masters",
                "tournament_name": None,
                "surface": "hard",
                "round": "qf",
                "opponent_id": "p8",
                "opponent_name": None,
                "player_ranking": 4,
                "match_won": 1,
                "aces": 8,
                "double_faults": 1,
                "first_serve_points_won": 40,
                "second_serve_points_won": 15,
                "total_serve_points": 80,
                "service_games": 12,
                "break_points_saved": 2,
                "break_points_faced": 3,
            },
        ]
    )


def test_match_history_shape_and_default_limit():
    with patch("src.serving.service.execute_df", return_value=_match_history_df()) as exec:
        resp = client.get("/match_history?player_id=p1")
    assert resp.status_code == 200
    sql, params = exec.call_args_list[0].args
    assert params == ["p1", 20]  # default limit
    assert "%s" in sql
    matches = resp.json()["data"]["matches"]
    assert len(matches) == 2
    assert matches[0]["match_id"] == "m2"
    assert matches[0]["result"] == "lost"
    assert matches[0]["opponent_name"] == "Rival"
    assert matches[1]["result"] == "won"
    assert matches[1]["tournament_name"] is None
    assert matches[1]["opponent_name"] is None


def test_match_history_limit_clamped():
    for raw, expected in (("500", 100), ("0", 1), ("7", 7)):
        with patch("src.serving.service.execute_df", return_value=pd.DataFrame()) as exec:
            resp = client.get(f"/match_history?player_id=p1&limit={raw}")
        assert resp.status_code == 200
        assert exec.call_args_list[0].args[1] == ["p1", expected]


def test_match_history_invalid_limit_returns_400():
    resp = client.get("/match_history?player_id=p1&limit=abc")
    assert resp.status_code == 400
    assert "limit" in resp.json()["error"]


def test_match_history_requires_player_id():
    resp = client.get("/match_history")
    assert resp.status_code == 400
    assert "player_id" in resp.json()["error"]


# ── /head_to_head ───────────────────────────────────────────────────────────


def _h2h_df() -> pd.DataFrame:
    # Meetings newest-first (the query orders match_date DESC, match_id DESC).
    rows = [
        ("m6", "2026-02-01", "a", "b", "a"),
        ("m5", "2026-01-01", "a", "b", "b"),
        ("m4", "2025-12-01", "a", "b", "a"),
        ("m3", "2025-11-01", "a", "b", "b"),
        ("m2", "2025-10-01", "a", "b", "b"),
        ("m1", "2025-09-01", "a", "b", "a"),
    ]
    return pd.DataFrame(
        [
            {
                "match_id": mid,
                "match_date": date,
                "tournament": "masters",
                "round": "qf",
                "surface": "hard",
                "player1_id": p1,
                "player2_id": p2,
                "winner_id": winner,
            }
            for mid, date, p1, p2, winner in rows
        ]
    )


def test_head_to_head_canonical_orientation_both_param_conventions():
    """Response uses the lower id on the player1 side regardless of request order."""
    for params in (
        {"player1_id": "b", "player2_id": "a"},
        {"player_id": "b", "opponent_id": "a"},
        {"player1_id": "a", "player2_id": "b"},
    ):
        with patch("src.serving.service.execute_df", return_value=_h2h_df()) as exec:
            resp = client.get("/head_to_head", params=params)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["player1_id"] == "a"
        assert data["player2_id"] == "b"
        sql, bound = exec.call_args_list[0].args
        assert bound[:4] == ["a", "b", "b", "a"]
        assert bound[4] == 100  # default limit
        assert "ORDER BY match_date DESC, match_id DESC" in sql
        assert "LIMIT %s" in sql


def test_head_to_head_meetings_and_summary():
    with patch("src.serving.service.execute_df", return_value=_h2h_df()):
        resp = client.get("/head_to_head?player1_id=a&player2_id=b")
    assert resp.status_code == 200
    data = resp.json()["data"]
    meetings = data["meetings"]
    assert len(meetings) == 6
    assert [m["match_id"] for m in meetings] == ["m6", "m5", "m4", "m3", "m2", "m1"]
    assert meetings[0]["player1_won"] is True
    assert meetings[0]["winner_id"] == "a"
    assert meetings[0]["loser_id"] == "b"
    assert meetings[1]["player1_won"] is False
    assert meetings[1]["loser_id"] == "a"
    assert "match_id" in meetings[0]
    assert data["summary"] == {
        "meetings": 6,
        "player1_wins": 3,
        "player2_wins": 3,
        "player1_win_rate": 0.5,
        # last-5 mirror: rows m6..m2 -> a wins m6 and m4 -> 2/5
        "last5_player1_win_rate": 0.4,
    }


def test_head_to_head_zero_meetings():
    with patch("src.serving.service.execute_df", return_value=pd.DataFrame()):
        resp = client.get("/head_to_head?player1_id=a&player2_id=b")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["meetings"] == []
    assert data["summary"] == {
        "meetings": 0,
        "player1_wins": 0,
        "player2_wins": 0,
        "player1_win_rate": 0.5,
        "last5_player1_win_rate": 0.5,
    }


def test_head_to_head_sql_is_direct_bronze_pair_read():
    """One bronze query per request; no silver expansion, grouping, or dedup."""
    with patch("src.serving.service.execute_df", return_value=pd.DataFrame()) as exec:
        resp = client.get("/head_to_head?player1_id=a&player2_id=b")
    assert resp.status_code == 200
    sql = exec.call_args_list[0].args[0]
    assert "FROM bronze.match_events" in sql
    assert "silver" not in sql
    assert "GROUP BY" not in sql


def test_head_to_head_limit_clamped():
    with patch("src.serving.service.execute_df", return_value=pd.DataFrame()) as exec:
        resp = client.get("/head_to_head?player1_id=a&player2_id=b&limit=500")
    assert resp.status_code == 200
    assert exec.call_args_list[0].args[1][4] == 100
    with patch("src.serving.service.execute_df", return_value=pd.DataFrame()) as exec:
        resp = client.get("/head_to_head?player1_id=a&player2_id=b&limit=0")
    assert resp.status_code == 200
    assert exec.call_args_list[0].args[1][4] == 1


def test_head_to_head_validation_errors():
    resp = client.get("/head_to_head")
    assert resp.status_code == 400
    resp = client.get("/head_to_head?player1_id=a")
    assert resp.status_code == 400
    resp = client.get("/head_to_head?player1_id=a&player2_id=b&limit=abc")
    assert resp.status_code == 400
    assert "limit" in resp.json()["error"]


def test_head_to_head_database_error_returns_500():
    with patch("src.serving.service.execute_df", side_effect=RuntimeError("boom")):
        resp = client.get("/head_to_head?player1_id=a&player2_id=b")
    assert resp.status_code == 500
    assert "head-to-head query failed" in resp.json()["error"]
