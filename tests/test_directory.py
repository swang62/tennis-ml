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


def test_directory_players_cluster_labels_from_artifacts(monkeypatch, tmp_path):
    """When the clustering artifacts exist, each entry carries its archetype
    label; players without an assignment stay null."""
    import src.serving.directory as directory

    (tmp_path / "cluster_assignments.parquet").parent.mkdir(exist_ok=True)
    pd.DataFrame({"player_id": ["p2", "p1"], "cluster_id": ["0", "0"]}).to_parquet(
        tmp_path / "cluster_assignments.parquet"
    )
    (tmp_path / "cluster_descriptions.json").write_text(
        json.dumps({"0": "Big Server", "1": "Counterpuncher"})
    )
    monkeypatch.setattr(directory, "DEFAULT_CLUSTERS", tmp_path / "cluster_assignments.parquet")
    monkeypatch.setattr(directory, "DEFAULT_CLUSTER_LABELS", tmp_path / "cluster_descriptions.json")

    players = directory_players(_directory_df())

    assert players[0]["cluster_label"] == "Big Server"  # p2
    assert players[1]["cluster_label"] == "Big Server"  # p1
    assert players[2]["cluster_label"] is None  # p3 has no assignment
    assert [p["player_id"] for p in players] == ["p2", "p1", "p3"]  # row order kept


def test_directory_players_malformed_cluster_artifacts_are_ignored(monkeypatch, tmp_path):
    """A corrupt artifact must not break the directory: it behaves as absent."""
    import src.serving.directory as directory

    (tmp_path / "cluster_descriptions.json").write_text("{not json")
    monkeypatch.setattr(directory, "DEFAULT_CLUSTER_LABELS", tmp_path / "cluster_descriptions.json")
    monkeypatch.setattr(directory, "DEFAULT_CLUSTERS", tmp_path / "missing.parquet")

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


def test_generate_directory_artifact_builds_from_snapshot(monkeypatch, tmp_path):
    """The deploy-time artifact is generated from the DuckDB training snapshot —
    no champion-pinned directory file, no MLflow download."""
    d = _deploy()
    out = tmp_path / "web" / "public" / "player-directory.json"
    monkeypatch.setattr(d, "WEB_DIRECTORY_ARTIFACT", out)
    monkeypatch.setattr("src.db.training.to_dataframe", lambda _sql: _directory_df())

    path = d.generate_directory_artifact()

    assert path == out
    artifact = json.loads(out.read_text())
    assert artifact == {"players": directory_players(_directory_df())}
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


def test_generate_directory_artifact_requires_snapshot(monkeypatch, tmp_path):
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
        d.generate_directory_artifact()
    assert not out.exists()


def test_deploy_tee_preserves_console_isatty():
    d = _deploy()

    class Console:
        def isatty(self):
            return True

    tee = d._Tee(Console(), io.StringIO())

    assert tee.isatty() is True


def test_generate_directory_artifact_raises_when_write_fails(monkeypatch, tmp_path):
    """A staging failure also aborts the artifact (and therefore the deploy)."""
    d = _deploy()
    monkeypatch.setattr("src.db.training.to_dataframe", lambda _sql: _directory_df().head(0))
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    monkeypatch.setattr(d, "WEB_DIRECTORY_ARTIFACT", blocked / "player-directory.json")

    import pytest

    with pytest.raises(OSError):
        d.generate_directory_artifact()
