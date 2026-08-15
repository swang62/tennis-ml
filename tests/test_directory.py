"""Hermetic tests for the shared player-directory contract and its deploy artifact.

No live database: the query result is a fixture DataFrame and the deploy
generation path patches execute_df at the module boundary.
"""

import importlib
import io
import json

import numpy as np
import pandas as pd

from src.serving.directory import PLAYERS_SQL, directory_players


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


# ── Shared directory contract (used by the deploy artifact) ─────────────────


def test_directory_players_matches_players_contract():
    players = directory_players(_directory_df())
    assert players[0] == {
        "player_id": "p2",
        "display_name": "B Player",
        "matches_played": 40,
        "ioc": "ESP",
        "iso2": "ES",
        "current_rank": 1,
    }
    # unranked players keep the entry with null rank data and UNK country
    assert players[1]["current_rank"] is None
    assert players[1]["ioc"] == "UNK"
    assert players[1]["iso2"] == ""


def test_directory_players_converts_numpy_scalars_to_json_native():
    players = directory_players(_directory_df())
    assert all(isinstance(p["matches_played"], int) for p in players)
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
        "current_rank",
    ):
        assert field in PLAYERS_SQL


# ── Deploy-time artifact generation ─────────────────────────────────────────


def _deploy():
    return importlib.import_module("src.flows.deploy")


def _fake_execute(sql: str) -> pd.DataFrame:
    """Hermetic stand-in for the directory query."""
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
    assert players[1]["current_rank"] is None


def test_deploy_tee_preserves_console_isatty():
    d = _deploy()

    class Console:
        def isatty(self):
            return True

    tee = d._Tee(Console(), io.StringIO())

    assert tee.isatty() is True


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

    assert called == [d.PLAYERS_SQL]


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
