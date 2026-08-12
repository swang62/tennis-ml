"""Contract tests for /players, /rank_history, /match_history, and /head_to_head."""

from unittest.mock import patch

import pandas as pd
from psycopg.errors import UndefinedColumn, UndefinedTable
from starlette.testclient import TestClient

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
    """Directory joins bronze metadata with dbt-derived gold aggregates; no
    per-query join to bronze.rankings or bronze.match_events."""
    with patch("src.serving.service.execute_df", return_value=_players_df()) as exec:
        resp = client.get("/players")
    assert resp.status_code == 200
    sql = exec.call_args_list[0].args[0]
    assert "FROM bronze.player_profiles" in sql
    assert "JOIN gold.player_profiles" in sql
    assert "bronze.rankings" not in sql
    assert "bronze.match_events" not in sql
    assert "ORDER BY gp.current_rank NULLS LAST, bp.display_name, bp.player_id" in sql


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


def test_players_empty_when_dbt_relations_missing():
    """Pre-ETL (missing dbt-created gold relations) /players renders empty, not a 500."""
    with patch(
        "src.serving.service.execute_df",
        side_effect=UndefinedTable('relation "gold.player_profiles" does not exist'),
    ):
        resp = client.get("/players")
    assert resp.status_code == 200
    assert resp.json()["data"]["players"] == []


def test_players_other_db_errors_still_500():
    """Only missing relations are masked — bad columns/connections stay 500."""
    with patch(
        "src.serving.service.execute_df",
        side_effect=UndefinedColumn("column bp.display_name does not exist"),
    ):
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
    assert "ORDER BY gp.current_rank NULLS LAST, bp.display_name, bp.player_id" in sql


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
    assert len(args) == 2  # _safe_query(sql, params=None)
    assert "%s" not in args[0]  # query string never parameterised from URL
    assert args[1] is None


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
                "opponent_ranking": 5,
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
                "opponent_ranking": 4,
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
    assert matches[0]["opponent_ranking"] == 5
    assert matches[1]["result"] == "won"
    assert matches[1]["tournament_name"] is None
    assert matches[1]["opponent_name"] is None
    assert matches[1]["opponent_ranking"] == 4
    # The rank shown is the opponent's, never the profile player's, and the
    # ambiguous old `ranking` field is gone.
    assert "ranking" not in matches[0]
    assert "player_ranking" not in matches[0]
    # Rows are individual matches sorted deterministically; the limit is a
    # parameter, never interpolated.
    assert "opponent_ranking" in sql
    assert "COALESCE(pm.opponent_ranking, historical_rank.rank)" in sql
    assert "ranking_date <= pm.match_date" in sql
    assert "ORDER BY pm.match_date DESC, pm.match_id DESC" in sql
    assert "LIMIT %s" in sql


def test_match_history_returns_individual_matches_not_grouped_by_tournament():
    """Same-occurrence rounds (Rome final + earlier rounds) all come back as
    individual rows, newest first, up to the limit — no ROW_NUMBER dedup."""
    rows = [
        ("m4", "2026-05-25", "grand_slam", "Roland Garros", "r128", "p9"),
        ("m3", "2026-05-17", "masters", "Rome", "f", "p8"),
        ("m2", "2026-05-17", "masters", "Rome", "sf", "p7"),
        ("m1", "2026-05-17", "masters", "Rome", "r16", "p6"),
    ]
    df = pd.DataFrame(
        [
            {
                "match_id": mid,
                "match_date": date,
                "tournament": tier,
                "tournament_name": name,
                "surface": "clay",
                "round": rnd,
                "opponent_id": opp,
                "opponent_name": opp,
                "opponent_ranking": i,
                "match_won": 1,
                "aces": 0,
                "double_faults": 0,
                "first_serve_points_won": 0,
                "second_serve_points_won": 0,
                "total_serve_points": 0,
                "service_games": 0,
                "break_points_saved": 0,
                "break_points_faced": 0,
            }
            for i, (mid, date, tier, name, rnd, opp) in enumerate(rows, start=1)
        ]
    )
    with patch("src.serving.service.execute_df", return_value=df) as exec:
        resp = client.get("/match_history?player_id=p1&limit=20")
    assert resp.status_code == 200
    sql, params = exec.call_args_list[0].args
    assert params == ["p1", 20]
    # No tournament-dedup constructs remain; order is deterministic and the
    # limit still applies in SQL.
    assert "ROW_NUMBER" not in sql
    assert "PARTITION BY" not in sql
    assert "rn = 1" not in sql
    assert "round_depth" not in sql
    assert "ORDER BY pm.match_date DESC, pm.match_id DESC" in sql
    assert "LIMIT %s" in sql
    # All three Rome rounds survive as individual matches, newest first.
    matches = resp.json()["data"]["matches"]
    assert [m["match_id"] for m in matches] == ["m4", "m3", "m2", "m1"]
    rome = [m for m in matches if m["tournament_name"] == "Rome"]
    assert [m["round"] for m in rome] == ["f", "sf", "r16"]


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
