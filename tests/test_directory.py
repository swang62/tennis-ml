"""Hermetic tests for the shared player-directory contract and its deploy artifact.

No live database: the query result is a fixture DataFrame and the deploy
generation path patches execute_df at the module boundary.
"""

import importlib
import json
from datetime import date

import numpy as np
import pandas as pd

from src.serving.directory import (
    LATEST_MATCH_DATE_SQL,
    PLAYERS_SQL,
    directory_players,
    latest_match_date,
)


def _directory_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": "p2",
                "display_name": "B Player",
                "matches_played": np.int64(40),
                "latest_rank_points": np.float64(1500.0),
                "ioc": "ESP",
                "current_rank": np.int64(1),
            },
            {
                "player_id": "p1",
                "display_name": "A Player",
                "matches_played": np.int64(20),
                "latest_rank_points": None,  # never had positive points
                "ioc": "UNK",
                "current_rank": None,
            },
            {
                "player_id": "p3",
                "display_name": "C Player",
                "matches_played": np.int64(60),
                "latest_rank_points": np.float64(900.0),
                "ioc": "ARG",
                "current_rank": np.int64(2),
            },
        ]
    )


# ── Shared directory contract (used by /players and the deploy artifact) ────


def test_directory_players_matches_players_contract():
    players = directory_players(_directory_df())
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


def test_directory_players_converts_numpy_scalars_to_json_native():
    players = directory_players(_directory_df())
    assert all(isinstance(p["matches_played"], int) for p in players)
    assert isinstance(players[0]["latest_rank_points"], float)
    # the full list is JSON-serializable (the deploy artifact is raw JSON)
    json.dumps(players)


def test_directory_players_preserves_sql_row_order():
    players = directory_players(_directory_df())
    assert [p["player_id"] for p in players] == ["p2", "p1", "p3"]


def test_directory_players_deterministic():
    assert directory_players(_directory_df()) == directory_players(_directory_df())


def test_directory_players_empty_df():
    assert directory_players(pd.DataFrame()) == []


def test_directory_players_unknown_ioc_normalizes_to_unk():
    df = pd.DataFrame(
        [
            {
                "player_id": "x",
                "display_name": "X",
                "matches_played": np.int64(1),
                "latest_rank_points": None,
                "ioc": "ZZZ",
                "current_rank": None,
            }
        ]
    )
    players = directory_players(df)
    assert players[0]["ioc"] == "UNK"
    assert players[0]["iso2"] == ""
    assert players[0]["country_name"] == "Country unknown"


def test_players_sql_is_the_single_directory_source():
    """The one directory read: bronze metadata joined to dbt-derived gold
    aggregates; no per-query rankings or match_events aggregation."""
    assert "FROM bronze.player_profiles" in PLAYERS_SQL
    assert "JOIN gold.player_profiles" in PLAYERS_SQL
    assert "bronze.rankings" not in PLAYERS_SQL
    assert "bronze.match_events" not in PLAYERS_SQL
    assert "ORDER BY gp.current_rank NULLS LAST, bp.display_name, bp.player_id" in PLAYERS_SQL


def test_players_sql_columns_cover_the_player_contract():
    for field in (
        "player_id",
        "display_name",
        "ioc",
        "AS matches_played",
        "latest_rank_points",
        "current_rank",
    ):
        assert field in PLAYERS_SQL


# ── Deploy-time artifact generation ─────────────────────────────────────────


def _deploy():
    return importlib.import_module("src.flows.deploy")


def _latest_date_df() -> pd.DataFrame:
    """The MAX(match_date) result as psycopg returns it: a datetime.date."""
    return pd.DataFrame({"latest_match_date": [date(2026, 7, 12)]})


def _fake_execute(sql: str) -> pd.DataFrame:
    """Hermetic stand-in for execute_df: fixed players and a fixed latest
    match date, so artifact generation is deterministic for a database state."""
    if sql == _deploy().LATEST_MATCH_DATE_SQL:
        return _latest_date_df()
    if sql == _deploy().PLAYERS_SQL:
        return _directory_df()
    raise AssertionError(f"unexpected SQL: {sql}")


def test_generate_directory_artifact_writes_players_json(monkeypatch, tmp_path):
    d = _deploy()
    out = tmp_path / "web" / "public" / "player-directory.json"
    monkeypatch.setattr(d, "WEB_DIRECTORY_ARTIFACT", out)
    monkeypatch.setattr(d, "execute_df", _fake_execute)

    path = d.generate_directory_artifact()

    assert path == out
    data = json.loads(out.read_text())
    players = data["players"]
    assert [p["player_id"] for p in players] == ["p2", "p1", "p3"]
    assert players[0]["display_name"] == "B Player"
    assert players[0]["country_name"] == "Spain"
    assert players[1]["current_rank"] is None
    assert players[1]["country_name"] == "Country unknown"


def test_generate_directory_artifact_includes_latest_match_date(monkeypatch, tmp_path):
    """The artifact carries the database's MAX(match_date) value — a fixed
    fixture here, so it is provably never the deployment time."""
    d = _deploy()
    out = tmp_path / "web" / "public" / "player-directory.json"
    monkeypatch.setattr(d, "WEB_DIRECTORY_ARTIFACT", out)
    monkeypatch.setattr(d, "execute_df", _fake_execute)

    d.generate_directory_artifact()

    data = json.loads(out.read_text())
    assert data["latest_match_date"] == "2026-07-12"
    assert [p["player_id"] for p in data["players"]] == ["p2", "p1", "p3"]


def test_latest_match_date_returns_the_database_value():
    assert latest_match_date(_latest_date_df()) == "2026-07-12"
    # MAX over an empty match table is NULL, not the deployment time.
    assert latest_match_date(pd.DataFrame({"latest_match_date": [None]})) is None
    assert latest_match_date(pd.DataFrame()) is None


def test_latest_match_date_sql_is_the_database_max_query():
    """The latest match date comes from bronze match rows, never now()/deploy
    time, so every deploy bakes in the same value a database query returns."""
    assert "SELECT MAX(match_date) AS latest_match_date" in LATEST_MATCH_DATE_SQL
    assert "FROM bronze.match_events" in LATEST_MATCH_DATE_SQL
    assert "now()" not in LATEST_MATCH_DATE_SQL.lower()
    assert "current_timestamp" not in LATEST_MATCH_DATE_SQL.lower()


def test_generate_directory_artifact_is_deterministic(monkeypatch, tmp_path):
    d = _deploy()
    out = tmp_path / "web" / "public" / "player-directory.json"
    monkeypatch.setattr(d, "WEB_DIRECTORY_ARTIFACT", out)
    monkeypatch.setattr(d, "execute_df", _fake_execute)

    d.generate_directory_artifact()
    first = out.read_bytes()
    d.generate_directory_artifact()
    assert out.read_bytes() == first


def test_generate_directory_artifact_uses_canonical_queries(monkeypatch, tmp_path):
    d = _deploy()
    monkeypatch.setattr(d, "WEB_DIRECTORY_ARTIFACT", tmp_path / "player-directory.json")
    called = []
    monkeypatch.setattr(d, "execute_df", lambda sql: called.append(sql) or _fake_execute(sql))

    d.generate_directory_artifact()

    assert called == [d.PLAYERS_SQL, d.LATEST_MATCH_DATE_SQL]


def test_generate_directory_artifact_raises_and_writes_nothing_on_query_failure(
    monkeypatch, tmp_path
):
    d = _deploy()
    out = tmp_path / "web" / "public" / "player-directory.json"

    def boom(_sql):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(d, "execute_df", boom)
    monkeypatch.setattr(d, "WEB_DIRECTORY_ARTIFACT", out)

    import pytest

    with pytest.raises(RuntimeError, match="connection refused"):
        d.generate_directory_artifact()
    assert not out.exists()


def test_generate_directory_artifact_raises_and_writes_nothing_when_date_query_fails(
    monkeypatch, tmp_path
):
    d = _deploy()
    out = tmp_path / "web" / "public" / "player-directory.json"

    def boom(sql):
        if sql == d.LATEST_MATCH_DATE_SQL:
            raise RuntimeError("match date query failed")
        return _directory_df()

    monkeypatch.setattr(d, "execute_df", boom)
    monkeypatch.setattr(d, "WEB_DIRECTORY_ARTIFACT", out)

    import pytest

    with pytest.raises(RuntimeError, match="match date query failed"):
        d.generate_directory_artifact()
    assert not out.exists()


def test_generate_directory_artifact_raises_when_write_fails(monkeypatch, tmp_path):
    """A write failure also aborts the artifact (and therefore the deploy)."""
    d = _deploy()
    monkeypatch.setattr(d, "execute_df", _fake_execute)
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    monkeypatch.setattr(d, "WEB_DIRECTORY_ARTIFACT", blocked / "player-directory.json")

    import pytest

    with pytest.raises(OSError):
        d.generate_directory_artifact()
