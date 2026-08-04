"""Hermetic tests for src/models/similarity.py (no network, no DuckDB).

fastembed.TextEmbedding is stubbed with a fake that yields fixed 4-dim ones
vectors, and src.db.client.to_dataframe is stubbed with small in-memory frames
dispatched by table name (player profiles vs. latest rolling-features
snapshots). Tests mirror the real implementation: bio_* embedding frames,
case-insensitive name lookup, FAISS self-exclusion and score formatting, and
the build()/load() disk round-trip via patched DEFAULT_INDEX/DEFAULT_METADATA.
"""

import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import pytest

from src.models import similarity
from src.models.similarity import PlayerData, PlayerSimilarity, embed_bio_summaries


class FakeTextEmbedding:
    """Stand-in for fastembed.TextEmbedding: fixed 4-dim ones vectors, no network."""

    def __init__(self, _model_name: str = "") -> None:
        self.embed_calls: list[list[str]] = []

    def embed(self, texts):
        self.embed_calls.append(list(texts))
        return np.ones((len(texts), 4), dtype=np.float32)


def _patch_embedding(monkeypatch: pytest.MonkeyPatch) -> FakeTextEmbedding:
    fake = FakeTextEmbedding()
    monkeypatch.setattr(similarity, "TextEmbedding", lambda _model_name: fake)
    return fake


def _profiles_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": ["P1", "P2", "P3", ""],
            "display_name": ["Alice", "Bob", "Carol", "Ghost"],
            "backhand": ["one", "two", "one", "one"],
            "handedness": ["right", "left", "right", "right"],
            "summary": ["Great server", "", "Solid returner", None],
        }
    )


# Same column order as the style_cols list in PlayerSimilarity.build.
STYLE_COLS = [
    "ace_rate_10",
    "first_serve_pct_10",
    "break_points_saved_pct_10",
    "clay_win_rate_10",
    "grass_win_rate_10",
    "hard_win_rate_10",
]


def _rolling_df() -> pd.DataFrame:
    """Latest snapshot per player. P2 never played clay (NULL surface rate);
    P3 has no snapshot at all, so its style stats must impute to 0.0."""
    return pd.DataFrame(
        {
            "player_id": ["P1", "P2"],
            "ace_rate_10": [0.15, 0.05],
            "first_serve_pct_10": [0.62, 0.58],
            "break_points_saved_pct_10": [0.41, 0.38],
            "clay_win_rate_10": [0.55, None],
            "grass_win_rate_10": [0.60, 0.70],
            "hard_win_rate_10": [0.50, 0.52],
        }
    )


def _stub_to_dataframe(sql: str) -> pd.DataFrame:
    """Stand-in for src.db.client.to_dataframe: dispatch by table in the SQL."""
    if "gold.rolling_features" in sql:
        return _rolling_df()
    return _profiles_df()


def _hand_built_finder() -> PlayerSimilarity:
    """PlayerSimilarity with a 3-vector FAISS index and matching metadata.

    P1 (row 0) is the query player in the search tests; its vector has the
    highest dot product with itself (1.0), then P2 (0.9), then P3 (0.5).
    """
    finder = PlayerSimilarity()
    finder.index = faiss.IndexFlatIP(4)
    finder.index.add(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],  # P1
                [0.9, 0.2, 0.0, 0.0],  # P2
                [0.5, 0.0, 0.0, 0.0],  # P3
            ],
            dtype=np.float32,
        )
    )
    finder.players = [
        PlayerData(player_id="P1", display_name="Alice"),
        PlayerData(player_id="P2", display_name="Bob"),
        PlayerData(player_id="P3", display_name="Carol"),
    ]
    finder.player_ids = ["P1", "P2", "P3"]
    return finder


def test_embed_bio_summaries_output_shape_and_empty_summary_handling(monkeypatch):
    fake = _patch_embedding(monkeypatch)
    profiles = pd.DataFrame(
        {"player_id": ["P1", "P2", "P3"], "summary": ["Great server", "", None]}
    )

    out = embed_bio_summaries(profiles)

    assert out.columns.tolist() == ["player_id", "bio_0", "bio_1", "bio_2", "bio_3"]
    assert out["player_id"].tolist() == ["P1", "P2", "P3"]
    assert len(out) == 3
    # Empty/None summaries are normalized to "" before embedding, so every row embeds.
    assert fake.embed_calls == [["Great server", "", ""]]


def test_find_by_name_exact_case_insensitive_and_unknown():
    finder = PlayerSimilarity()
    finder.players = [
        PlayerData(player_id="P1", display_name="Carlos Alcaraz"),
        PlayerData(player_id="P2", display_name="Jannik Sinner"),
    ]

    assert finder.find_by_name("Carlos Alcaraz") == "P1"
    assert finder.find_by_name("carlos alcaraz") == "P1"
    assert finder.find_by_name("Nobody") is None


def test_search_respects_top_k_and_excludes_query_player():
    finder = _hand_built_finder()

    top1 = finder.search("P1", top_k=1)

    assert len(top1) == 1
    assert [r["player_id"] for r in top1] == ["P2"]
    assert top1[0]["score"] == "0.900"

    all_ = finder.search("P1", top_k=5)

    assert [r["player_id"] for r in all_] == ["P2", "P3"]
    assert all(r["player_id"] != "P1" for r in all_)
    assert all(re.fullmatch(r"^0\.\d{3}$", r["score"]) for r in all_)


def test_search_by_display_name_query():
    finder = _hand_built_finder()

    results = finder.search("Bob", top_k=5)

    assert [r["player_id"] for r in results] == ["P1", "P3"]


def test_search_unknown_query_returns_empty():
    finder = _hand_built_finder()

    assert finder.search("Nobody") == []
    assert finder.search("unknown-id") == []


def test_search_single_player_index_returns_empty():
    finder = PlayerSimilarity()
    finder.index = faiss.IndexFlatIP(4)
    finder.index.add(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    finder.players = [PlayerData(player_id="P1", display_name="Alice")]
    finder.player_ids = ["P1"]

    assert finder.search("P1") == []


def test_search_empty_players_returns_empty():
    finder = PlayerSimilarity()
    finder.index = faiss.IndexFlatIP(4)
    finder.players = []
    finder.player_ids = []

    assert finder.search("P1") == []


def test_build_load_round_trip(tmp_path: Path, monkeypatch):
    _patch_embedding(monkeypatch)
    monkeypatch.setattr(similarity, "to_dataframe", _stub_to_dataframe)
    index_path = tmp_path / "idx"
    meta_path = tmp_path / "meta.json"
    monkeypatch.setattr(similarity, "DEFAULT_INDEX", index_path)
    monkeypatch.setattr(similarity, "DEFAULT_METADATA", meta_path)

    finder = PlayerSimilarity()
    finder.build()

    assert index_path.exists()
    assert meta_path.exists()

    loaded = PlayerSimilarity()
    loaded.load()

    # build() drops the empty-player_id row, so only P1/P2/P3 are indexed.
    assert loaded.player_ids == finder.player_ids == ["P1", "P2", "P3"]
    assert loaded.players == finder.players
    assert loaded.index is not None and finder.index is not None
    assert loaded.index.d == finder.index.d
    assert loaded.index.ntotal == finder.index.ntotal


def _build_with_fakes(tmp_path: Path, monkeypatch) -> PlayerSimilarity:
    """Build a PlayerSimilarity against the stubbed to_dataframe."""
    _patch_embedding(monkeypatch)
    monkeypatch.setattr(similarity, "to_dataframe", _stub_to_dataframe)
    monkeypatch.setattr(similarity, "DEFAULT_INDEX", tmp_path / "idx")
    monkeypatch.setattr(similarity, "DEFAULT_METADATA", tmp_path / "meta.json")
    finder = PlayerSimilarity()
    finder.build()
    return finder


def test_build_vector_contains_style_stats_and_bio(tmp_path: Path, monkeypatch):
    finder = _build_with_fakes(tmp_path, monkeypatch)
    index = finder.index
    assert index is not None

    # 4 one-hot (backhand x2, handedness x2) + 6 style stats + 4 bio dims.
    assert index.d == 4 + len(STYLE_COLS) + 4
    p1 = index.reconstruct(0)
    # Style stats and bio embeddings land in the vector after the one-hot block.
    assert np.all(p1[4 : 4 + len(STYLE_COLS)] > 0.0)
    assert np.all(p1[4 + len(STYLE_COLS) :] > 0.0)


def test_build_null_style_cells_filled_zero(tmp_path: Path, monkeypatch):
    finder = _build_with_fakes(tmp_path, monkeypatch)
    index = finder.index
    assert index is not None

    # P2 has no clay matches -> clay_win_rate_10 is NULL and must impute to 0.0.
    p2 = index.reconstruct(1)
    clay_idx = 4 + STYLE_COLS.index("clay_win_rate_10")
    ace_idx = 4 + STYLE_COLS.index("ace_rate_10")
    assert p2[clay_idx] == 0.0
    assert p2[ace_idx] > 0.0


def test_build_player_without_snapshot_still_indexed(tmp_path: Path, monkeypatch):
    finder = _build_with_fakes(tmp_path, monkeypatch)
    index = finder.index
    assert index is not None

    # P3 is in profiles but absent from rolling_features: still indexed, all
    # style stats zero-filled.
    assert finder.player_ids == ["P1", "P2", "P3"]
    p3 = index.reconstruct(2)
    assert np.all(p3[4 : 4 + len(STYLE_COLS)] == 0.0)
    assert np.all(p3[4 + len(STYLE_COLS) :] > 0.0)


def test_load_missing_index_raises(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(similarity, "DEFAULT_INDEX", tmp_path / "missing.index")
    monkeypatch.setattr(similarity, "DEFAULT_METADATA", tmp_path / "meta.json")

    with pytest.raises(FileNotFoundError):
        PlayerSimilarity().load()
