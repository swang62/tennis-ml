"""/similar_players tests using an in-memory finder."""

import faiss
import numpy as np
import pytest
from starlette.testclient import TestClient

import src.serving.service as service
from src.training.similarity import PlayerSimilarity


def _hand_built_finder() -> PlayerSimilarity:
    """5 players; P1 is the query player with descending scores P2..P5."""
    finder = PlayerSimilarity()
    finder.index = faiss.IndexFlatIP(4)
    finder.index.add(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],  # P1 (query)
                [0.9, 0.2, 0.0, 0.0],  # P2
                [0.8, 0.2, 0.0, 0.0],  # P3
                [0.7, 0.2, 0.0, 0.0],  # P4
                [0.6, 0.2, 0.0, 0.0],  # P5
            ],
            dtype=np.float32,
        )
    )
    finder.players = [
        {"player_id": "P1", "display_name": "Alice"},
        {"player_id": "P2", "display_name": "Bob"},
        {"player_id": "P3", "display_name": "Carol"},
        {"player_id": "P4", "display_name": "Dave"},
        {"player_id": "P5", "display_name": "Eve"},
    ]
    finder.player_ids = ["P1", "P2", "P3", "P4", "P5"]
    return finder


@pytest.fixture
def setup(monkeypatch):
    """A TestClient over DATA_APP with the module finder replaced in-memory."""
    finder = _hand_built_finder()
    monkeypatch.setattr(service, "_get_similarity_finder", lambda: finder)
    return TestClient(service.DATA_APP), finder


def test_similar_players_returns_top_3_sorted_self_excluded(setup):
    client, _ = setup
    res = client.get("/similar_players", params={"player_id": "P1", "limit": 3})
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    data = body["data"]
    assert data["player_id"] == "P1"
    similar = data["similar_players"]
    assert [r["player_id"] for r in similar] == ["P2", "P3", "P4"]
    assert all(r["player_id"] != "P1" for r in similar)
    scores = [float(r["score"]) for r in similar]
    assert scores == sorted(scores, reverse=True)


def test_similar_players_limit_clamped_to_three(setup):
    """Limit is capped at the default maximum of three results."""
    client, _ = setup
    for params in ({"player_id": "P1", "limit": 10}, {"player_id": "P1"}):
        res = client.get("/similar_players", params=params)
        assert res.status_code == 200
        assert len(res.json()["data"]["similar_players"]) == 3


def test_similar_players_response_keys_are_static_player_fields_only(setup):
    """Results expose the static player fields (id, name) and the numeric
    score — never dynamic profile data like rank."""
    client, _ = setup
    res = client.get("/similar_players", params={"player_id": "P1", "limit": 3})
    assert res.status_code == 200
    for entry in res.json()["data"]["similar_players"]:
        assert set(entry.keys()) == {"player_id", "display_name", "score"}


def test_similar_players_requires_player_id(setup):
    client, _ = setup
    res = client.get("/similar_players")
    assert res.status_code == 400
    assert "player_id" in res.json()["error"]


def test_similar_players_rejects_non_integer_limit(setup):
    client, _ = setup
    res = client.get("/similar_players", params={"player_id": "P1", "limit": "abc"})
    assert res.status_code == 400
    assert "limit" in res.json()["error"]


def test_similar_players_unknown_player_returns_empty(setup):
    client, _ = setup
    res = client.get("/similar_players", params={"player_id": "Nobody"})
    assert res.status_code == 200
    assert res.json()["data"]["similar_players"] == []
