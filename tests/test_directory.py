"""Hermetic tests for the shared player-directory contract and its deploy artifact.

No live database: the query result is a fixture DataFrame and the deploy
generation path patches the DuckDB snapshot query helper at the module
boundary.
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
        "cluster_label": None,  # no clustering artifacts: no archetype yet
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


def test_directory_players_use_supplied_cluster_labels():
    players = directory_players(_directory_df(), {"p2": "Big Server", "p1": "Big Server"})

    assert players[0]["cluster_label"] == "Big Server"  # p2
    assert players[1]["cluster_label"] == "Big Server"  # p1
    assert players[2]["cluster_label"] is None  # p3 has no assignment
    assert [p["player_id"] for p in players] == ["p2", "p1", "p3"]  # row order kept


def test_directory_players_without_cluster_labels_use_null():
    players = directory_players(_directory_df())
    assert all(p["cluster_label"] is None for p in players)


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


# ── Deployment staging ──────────────────────────────────────────────────────


def _deploy():
    return importlib.import_module("src.flows.deploy")


def test_generate_navigation_artifacts_builds_from_snapshot(monkeypatch, tmp_path):
    """Deploy builds matching directory and similarity metadata from one snapshot read."""
    d = _deploy()
    out = tmp_path / "web" / "public" / "player-directory.json"
    monkeypatch.setattr(d, "WEB_DIRECTORY_ARTIFACT", out)
    monkeypatch.setattr(d, "SIMILARITY_INDEX", tmp_path / "player_similarity.index")
    monkeypatch.setattr(d, "SIMILARITY_METADATA", tmp_path / "player_metadata.json")
    calls = []
    monkeypatch.setattr(
        "src.db.training.to_dataframe", lambda sql: calls.append(sql) or _directory_df()
    )
    assignments = pd.DataFrame({"player_id": ["p1", "p2"], "cluster_id": ["0", "1"]})
    monkeypatch.setattr(
        "src.models.similarity.load_cluster_artifacts",
        lambda: (assignments, {"0": "Big Server", "1": "Counterpuncher"}),
    )

    class FakeSimilarity:
        def build(self, **kwargs):
            self.players = [
                {"player_id": "p2", "cluster_label": "Counterpuncher"},
                {"player_id": "p1", "cluster_label": "Big Server"},
                {"player_id": "p3", "cluster_label": None},
            ]
            kwargs["index_path"].write_bytes(b"index")
            kwargs["metadata_path"].write_text(
                json.dumps(
                    [
                        {"player_id": "p2", "cluster_label": "Counterpuncher"},
                        {"player_id": "p1", "cluster_label": "Big Server"},
                    ]
                )
            )

    monkeypatch.setattr("src.models.similarity.PlayerSimilarity", FakeSimilarity)

    path = d.generate_navigation_artifacts()

    assert path == out
    artifact = json.loads(out.read_text())
    assert [p["player_id"] for p in artifact["players"]] == ["p2", "p1", "p3"]
    assert artifact["players"][0]["cluster_label"] == "Counterpuncher"
    assert calls == [PLAYERS_SQL]
    # The web loader contract: every player carries the static-picker fields.
    for field in (
        "player_id",
        "display_name",
        "matches_played",
        "current_rank",
        "ioc",
        "iso2",
        "cluster_label",
    ):
        assert all(field in player for player in artifact["players"])
    assert [p["player_id"] for p in artifact["players"]] == ["p2", "p1", "p3"]


def test_generate_navigation_artifacts_requires_snapshot(monkeypatch, tmp_path):
    """Without the training snapshot the deploy fails with the actionable
    snapshot-missing error and stages nothing."""
    d = _deploy()
    out = tmp_path / "web" / "public" / "player-directory.json"
    monkeypatch.setattr(d, "WEB_DIRECTORY_ARTIFACT", out)

    def no_snapshot(_sql):
        raise FileNotFoundError(
            "training snapshot not found at data/processed/training_snapshot.duckdb; "
            "run `just snapshot` first"
        )

    monkeypatch.setattr("src.db.training.to_dataframe", no_snapshot)

    import pytest

    with pytest.raises(FileNotFoundError, match="training snapshot not found"):
        d.generate_navigation_artifacts()
    assert not out.exists()


def test_deploy_tee_preserves_console_isatty():
    d = _deploy()

    class Console:
        def isatty(self):
            return True

    tee = d._Tee(Console(), io.StringIO())

    assert tee.isatty() is True


def test_generate_navigation_artifacts_raises_when_write_fails(monkeypatch, tmp_path):
    """A staging failure also aborts the artifact (and therefore the deploy)."""
    d = _deploy()
    monkeypatch.setattr("src.db.training.to_dataframe", lambda _sql: _directory_df())
    monkeypatch.setattr(
        "src.models.similarity.load_cluster_artifacts", lambda: (pd.DataFrame(), {})
    )

    def no_build(self, *_args, **_kwargs):
        self.players = [
            {"player_id": player_id, "cluster_label": None} for player_id in ("p2", "p1", "p3")
        ]
        return None

    monkeypatch.setattr("src.models.similarity.PlayerSimilarity.build", no_build)
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    monkeypatch.setattr(d, "WEB_DIRECTORY_ARTIFACT", blocked / "player-directory.json")

    import pytest

    with pytest.raises(OSError):
        d.generate_navigation_artifacts()
