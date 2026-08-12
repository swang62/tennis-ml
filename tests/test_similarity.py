"""Similarity tests with fake embeddings and DuckDB fixtures."""

import re
import sys
from pathlib import Path

import duckdb
import faiss
import numpy as np
import pandas as pd
import pytest

from src.db import client, training
from src.models import similarity
from src.models.similarity import (
    LIFETIME_PLAYSTYLE_COLS,
    PlayerData,
    PlayerSimilarity,
    embed_bio_summaries,
)


class FakeTextEmbedding:
    """Stand-in for fastembed.TextEmbedding: fixed 4-dim ones vectors, no network."""

    def __init__(self, _model_name: str = "") -> None:
        self.embed_calls: list[list[str]] = []

    def embed(self, texts):
        self.embed_calls.append(list(texts))
        return np.ones((len(texts), 4), dtype=np.float32)


class _FakeFastembed:
    """Stand-in fastembed module whose TextEmbedding factory returns the fake."""

    def __init__(self, factory) -> None:
        self.TextEmbedding = factory


def _patch_embedding(monkeypatch: pytest.MonkeyPatch) -> FakeTextEmbedding:
    # Inject the fake module because fastembed is imported lazily.
    fake = FakeTextEmbedding()
    monkeypatch.setitem(sys.modules, "fastembed", _FakeFastembed(lambda _model_name: fake))
    return fake


# Included career lifetime playstyle aggregates (gold.player_profiles) plus
# the excluded identity/career/recent-form fields that must never enter a
# vector.
_GOLD_PROFILE_COLS = [
    "first_serve_in_pct",
    "aces_per_first_serve",
    "first_serve_points_won_pct",
    "second_serve_points_won_pct",
    "overall_serve_points_won_pct",
    "double_faults_per_serve_point",
    "aces_per_service_game",
    "break_points_saved_pct",
    "return_points_won_pct",
    "first_serve_return_points_won_pct",
    "second_serve_return_points_won_pct",
    "break_point_conversion_pct",
    "break_point_opportunities_per_return_game",
    "hard_win_rate",
    "clay_win_rate",
    "grass_win_rate",
    "match_count",
    "career_win_rate",
    "current_rank",
    "win_rate_10",
]


def _create_two_table_fixture(con: duckdb.DuckDBPyConnection) -> None:
    """Create profile fixtures: bronze metadata + gold lifetime aggregates."""
    con.execute("CREATE SCHEMA gold")
    con.execute("CREATE SCHEMA bronze")
    con.execute(
        """
        CREATE TABLE bronze.player_profiles (
            player_id VARCHAR,
            display_name VARCHAR,
            backhand VARCHAR,
            handedness VARCHAR,
            summary VARCHAR,
            height DOUBLE,
            turned_pro INTEGER,
            birthplace VARCHAR
        )
        """
    )
    con.execute(
        "CREATE TABLE gold.player_profiles (player_id VARCHAR, "
        + ", ".join(f'"{c}" DOUBLE' for c in _GOLD_PROFILE_COLS)
        + ")"
    )
    con.executemany(
        "INSERT INTO bronze.player_profiles VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("P1", "Alice", "1H", "L", "Great server", 185.0, 2010, "Spain"),
            ("P2", "Bob", "2H", "R", "", 190.0, 2015, "Italy"),
            ("P3", "Carol", "1H", "R", "Solid returner", 178.0, 2020, "USA"),
            ("P4", "Dave", "2H", "L", None, 195.0, 2012, "France"),
            ("", "Ghost", "1H", "R", "No id", 180.0, 2018, "UK"),
        ],
    )
    # P3 (and Ghost) have no match history: every playstyle cell is NULL and
    # must impute to 0.0, while the excluded career fields stay populated.
    con.executemany(
        "INSERT INTO gold.player_profiles VALUES ("
        + ", ".join(["?"] * (1 + len(_GOLD_PROFILE_COLS)))
        + ")",
        [
            (
                "P1",
                0.62,
                0.20,
                0.73,
                0.51,
                0.65,
                0.04,
                0.50,
                0.62,
                0.42,
                0.30,
                0.55,
                0.40,
                0.45,
                0.55,
                0.58,
                0.60,
                400,
                0.72,
                5,
                0.80,
            ),
            (
                "P2",
                0.60,
                0.15,
                0.70,
                0.53,
                0.63,
                0.06,
                0.35,
                0.58,
                0.44,
                0.33,
                0.53,
                0.42,
                0.48,
                0.52,
                0.60,
                0.35,
                300,
                0.68,
                8,
                0.60,
            ),
            (
                "P3",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                0,
                0.55,
                50,
                None,
            ),
            (
                "P4",
                0.58,
                0.12,
                0.68,
                0.55,
                0.61,
                0.08,
                0.25,
                0.50,
                0.46,
                0.36,
                0.51,
                0.44,
                0.52,
                0.45,
                0.30,
                0.50,
                350,
                0.70,
                20,
                0.45,
            ),
            (
                "",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                50,
                0.50,
                60,
                0.40,
            ),
        ],
    )


def _duck_query(con: duckdb.DuckDBPyConnection):
    """Query function reading from a DuckDB connection, like training.to_dataframe."""

    def query(sql: str) -> pd.DataFrame:
        return con.execute(sql).df()

    return query


def _build_with_fixture(tmp_path: Path, monkeypatch) -> PlayerSimilarity:
    """Build against an in-memory DuckDB two-table fixture (offline path)."""
    _patch_embedding(monkeypatch)
    monkeypatch.setattr(similarity, "DEFAULT_INDEX", tmp_path / "idx")
    monkeypatch.setattr(similarity, "DEFAULT_METADATA", tmp_path / "meta.json")
    con = duckdb.connect()
    try:
        _create_two_table_fixture(con)
        finder = PlayerSimilarity()
        finder.build(query=_duck_query(con))
    finally:
        con.close()
    return finder


# One-hot playstyle descriptors precede LIFETIME_PLAYSTYLE_COLS; bio
# embeddings follow.
ONE_HOT = 4
STYLE = LIFETIME_PLAYSTYLE_COLS


def _style_block(vector: object) -> np.ndarray:
    # faiss IndexFlatIP.reconstruct returns a plain np.ndarray.
    arr: np.ndarray = np.asarray(vector)
    return arr[ONE_HOT : ONE_HOT + len(STYLE)]


def _style_values(vector: object) -> dict[str, float]:
    block = _style_block(vector)
    return {name: float(block[i]) for i, name in enumerate(LIFETIME_PLAYSTYLE_COLS)}


def _assert_style(vector: object, expected: dict[str, float], reference: str) -> None:
    """Check playstyle ratios against a common reference (L2 normalization preserves them)."""
    values = _style_values(vector)
    ref = values[reference]
    assert ref > 0.0
    for name, expected_value in expected.items():
        if expected_value == 0.0:
            assert values[name] == 0.0, f"{name} should impute to 0.0"
        else:
            assert np.isclose(
                values[name] / ref, expected_value / expected[reference], atol=1e-6
            ), f"ratio mismatch for {name}"


def test_build_uses_lifetime_playstyle_aggregates(tmp_path: Path, monkeypatch):
    finder = _build_with_fixture(tmp_path, monkeypatch)
    index = finder.index
    assert index is not None
    # 4 one-hot + 16 lifetime playstyle stats + 4 bio dims.
    assert index.d == ONE_HOT + len(STYLE) + 4

    # The vector carries each player's career gold.player_profiles aggregates,
    # not any recent rolling form.
    _assert_style(
        index.reconstruct(finder.player_ids.index("P1")),
        {
            "first_serve_in_pct": 0.62,
            "aces_per_first_serve": 0.20,
            "first_serve_points_won_pct": 0.73,
            "second_serve_points_won_pct": 0.51,
            "overall_serve_points_won_pct": 0.65,
            "double_faults_per_serve_point": 0.04,
            "aces_per_service_game": 0.50,
            "break_points_saved_pct": 0.62,
            "return_points_won_pct": 0.42,
            "first_serve_return_points_won_pct": 0.30,
            "second_serve_return_points_won_pct": 0.55,
            "break_point_conversion_pct": 0.40,
            "break_point_opportunities_per_return_game": 0.45,
            "hard_win_rate": 0.55,
            "clay_win_rate": 0.58,
            "grass_win_rate": 0.60,
        },
        reference="overall_serve_points_won_pct",
    )

    _assert_style(
        index.reconstruct(finder.player_ids.index("P4")),
        {
            "first_serve_in_pct": 0.58,
            "aces_per_first_serve": 0.12,
            "first_serve_points_won_pct": 0.68,
            "second_serve_points_won_pct": 0.55,
            "overall_serve_points_won_pct": 0.61,
            "double_faults_per_serve_point": 0.08,
            "aces_per_service_game": 0.25,
            "break_points_saved_pct": 0.50,
            "return_points_won_pct": 0.46,
            "first_serve_return_points_won_pct": 0.36,
            "second_serve_return_points_won_pct": 0.51,
            "break_point_conversion_pct": 0.44,
            "break_point_opportunities_per_return_game": 0.52,
            "hard_win_rate": 0.45,
            "clay_win_rate": 0.30,
            "grass_win_rate": 0.50,
        },
        reference="overall_serve_points_won_pct",
    )


def test_build_players_on_either_side_included_exactly_once(tmp_path: Path, monkeypatch):
    finder = _build_with_fixture(tmp_path, monkeypatch)
    # Ghost (empty player_id) is dropped; P4 (opponent-only) is included.
    assert finder.player_ids == ["P1", "P2", "P3", "P4"]
    assert len(finder.player_ids) == len(set(finder.player_ids))


def test_build_null_and_cold_start_style_cells_imputed_zero(tmp_path: Path, monkeypatch):
    finder = _build_with_fixture(tmp_path, monkeypatch)
    index = finder.index
    assert index is not None

    # P3 has a profile but no matches: every playstyle stat imputes to 0.0.
    p3 = _style_block(index.reconstruct(finder.player_ids.index("P3")))
    assert np.all(p3 == 0.0)

    # Players with match history carry non-zero lifetime playstyle stats.
    p1 = _style_block(index.reconstruct(finder.player_ids.index("P1")))
    assert np.all(p1 > 0.0)


def test_build_embeds_bio_and_one_hot_identity(tmp_path: Path, monkeypatch):
    fake = _patch_embedding(monkeypatch)
    monkeypatch.setattr(similarity, "DEFAULT_INDEX", tmp_path / "idx")
    monkeypatch.setattr(similarity, "DEFAULT_METADATA", tmp_path / "meta.json")
    con = duckdb.connect()
    try:
        _create_two_table_fixture(con)
        finder = PlayerSimilarity()
        finder.build(query=_duck_query(con))
    finally:
        con.close()
    index = finder.index
    assert index is not None
    # Every profiled player's summary is embedded (empty/None normalized to "").
    assert fake.embed_calls == [["Great server", "", "Solid returner", ""]]
    p2 = index.reconstruct(finder.player_ids.index("P2"))
    # Bio block is ones (normalized); lifetime playstyle stats are non-zero.
    assert np.all(p2[ONE_HOT + len(STYLE) :] > 0.0)
    assert np.all(_style_block(p2) > 0.0)


def test_lifetime_playstyle_cols_are_playstyle_only():
    """LIFETIME_PLAYSTYLE_COLS is exactly the lifetime playstyle signal set:
    no recent rolling form and no identity/career attributes."""
    assert LIFETIME_PLAYSTYLE_COLS == [
        "first_serve_in_pct",
        "first_serve_points_won_pct",
        "second_serve_points_won_pct",
        "overall_serve_points_won_pct",
        "aces_per_first_serve",
        "aces_per_service_game",
        "double_faults_per_serve_point",
        "break_points_saved_pct",
        "return_points_won_pct",
        "first_serve_return_points_won_pct",
        "second_serve_return_points_won_pct",
        "break_point_conversion_pct",
        "break_point_opportunities_per_return_game",
        "hard_win_rate",
        "clay_win_rate",
        "grass_win_rate",
    ]
    excluded_names = {
        # Identity/bio metadata.
        "player_id",
        "display_name",
        "name",
        "height",
        "age",
        "player_age",
        "turned_pro",
        "years_pro",
        "birthplace",
        "birthdate",
        "weight",
        "ioc",
        "country",
        # Career/achievement attributes.
        "ranking",
        "player_ranking",
        "rank_points",
        "current_rank",
        "latest_rank_points",
        "rank_points_delta",
        "match_count",
        "matches_played",
        "career_win_rate",
        "hard_matches",
        "clay_matches",
        "grass_matches",
        "latest_match_date",
        "recent_snapshot_date",
        # Recent rolling match performance.
        "win_rate_10",
        "weighted_form_10",
        "streak",
        "ace_rate_10",
        "df_rate_10",
        "aces_per_svc_game_10",
        "first_serve_pct_10",
        "first_serve_win_pct_10",
        "second_serve_win_pct_10",
        "serve_win_pct_10",
        "break_points_saved_pct_10",
        "return_points_won_pct_10",
        "surface_win_rate_10",
        "clay_win_rate_10",
        "grass_win_rate_10",
        "hard_win_rate_10",
    }
    for col in LIFETIME_PLAYSTYLE_COLS:
        assert col not in excluded_names, col


def test_build_is_deterministic_same_data_identical_vectors(tmp_path: Path, monkeypatch):
    """Rebuilding from identical data yields bit-identical vectors and layout."""
    _patch_embedding(monkeypatch)
    monkeypatch.setattr(similarity, "DEFAULT_INDEX", tmp_path / "idx")
    monkeypatch.setattr(similarity, "DEFAULT_METADATA", tmp_path / "meta.json")
    con = duckdb.connect()
    try:
        _create_two_table_fixture(con)
        first = PlayerSimilarity()
        first.build(query=_duck_query(con))
        second = PlayerSimilarity()
        second.build(query=_duck_query(con))
    finally:
        con.close()
    assert first.player_ids == second.player_ids
    assert first.players == second.players
    assert first.index is not None and second.index is not None
    assert first.index.d == second.index.d
    for i in range(first.index.ntotal):
        assert np.array_equal(first.index.reconstruct(i), second.index.reconstruct(i))


def test_build_one_hot_layout_fixed_when_category_missing(tmp_path: Path, monkeypatch):
    """Missing one-hot categories must not change the vector dimension."""
    _patch_embedding(monkeypatch)
    monkeypatch.setattr(similarity, "DEFAULT_INDEX", tmp_path / "idx")
    monkeypatch.setattr(similarity, "DEFAULT_METADATA", tmp_path / "meta.json")
    con = duckdb.connect()
    try:
        _create_two_table_fixture(con)
        finder = PlayerSimilarity()
        finder.build(query=_duck_query(con))
        assert finder.index is not None
        baseline_dim = finder.index.d

        # No left-handed players and no one-handed backhands anywhere.
        con.execute("UPDATE bronze.player_profiles SET handedness = 'R', backhand = '2H'")
        finder2 = PlayerSimilarity()
        finder2.build(query=_duck_query(con))
        assert finder2.index is not None
        assert finder2.index.d == baseline_dim
    finally:
        con.close()


def test_build_always_rebuilds_index_from_fresh_data(tmp_path: Path, monkeypatch):
    """build() replaces any previously built index; stale vectors never persist."""
    _patch_embedding(monkeypatch)
    monkeypatch.setattr(similarity, "DEFAULT_INDEX", tmp_path / "idx")
    monkeypatch.setattr(similarity, "DEFAULT_METADATA", tmp_path / "meta.json")
    con = duckdb.connect()
    try:
        _create_two_table_fixture(con)
        finder = PlayerSimilarity()
        finder.build(query=_duck_query(con))
        assert finder.index is not None
        old_index = finder.index
        base = np.array(old_index.reconstruct(0))

        con.execute(
            "UPDATE gold.player_profiles SET first_serve_points_won_pct = 0.99 "
            "WHERE player_id = 'P1'"
        )
        finder.build(query=_duck_query(con))
        assert finder.index is not old_index  # a fresh index, not a cached one
        assert not np.array_equal(base, np.array(finder.index.reconstruct(0)))
    finally:
        con.close()


def test_build_excludes_physical_and_resume_metrics(tmp_path: Path, monkeypatch):
    """Excluded identity/career/recent-form fields leave vectors unchanged;
    lifetime playstyle fields do not."""
    _patch_embedding(monkeypatch)
    monkeypatch.setattr(similarity, "DEFAULT_INDEX", tmp_path / "idx")
    monkeypatch.setattr(similarity, "DEFAULT_METADATA", tmp_path / "meta.json")
    con = duckdb.connect()
    try:
        _create_two_table_fixture(con)
        finder = PlayerSimilarity()
        finder.build(query=_duck_query(con))
        assert finder.index is not None
        base = [np.array(finder.index.reconstruct(i)) for i in range(finder.index.ntotal)]

        # Mutate only excluded signals: physical/bio identity (height, turned_pro,
        # birthplace, display_name), career achievements (match_count,
        # career_win_rate, current_rank), and recent rolling form (win_rate_10).
        con.execute(
            "UPDATE bronze.player_profiles SET height = height + 5, "
            "turned_pro = turned_pro - 3, birthplace = 'X', display_name = 'X'"
        )
        con.execute(
            "UPDATE gold.player_profiles SET match_count = match_count + 999, "
            "career_win_rate = 0.99, current_rank = 1, win_rate_10 = 0.99"
        )
        finder2 = PlayerSimilarity()
        finder2.build(query=_duck_query(con))
        assert finder2.index is not None
        for i in range(finder2.index.ntotal):
            assert np.array_equal(base[i], finder2.index.reconstruct(i))

        # A lifetime playstyle signal (career first-serve points won) must
        # change the vectors.
        con.execute(
            "UPDATE gold.player_profiles SET first_serve_points_won_pct = 0.99 "
            "WHERE player_id = 'P1'"
        )
        finder3 = PlayerSimilarity()
        finder3.build(query=_duck_query(con))
        assert finder3.index is not None
        assert any(
            not np.array_equal(base[i], finder3.index.reconstruct(i))
            for i in range(finder3.index.ntotal)
        )
    finally:
        con.close()


def test_search_returns_sorted_top_k_from_built_index(tmp_path: Path, monkeypatch):
    finder = _build_with_fixture(tmp_path, monkeypatch)
    results = finder.search("P1", top_k=3)
    assert len(results) <= 3
    assert all(r["player_id"] != "P1" for r in results)
    scores = [float(r["score"]) for r in results]
    assert scores == sorted(scores, reverse=True)


def test_find_by_name_exact_case_insensitive_and_unknown():
    finder = PlayerSimilarity()
    finder.players = [
        PlayerData(player_id="P1", display_name="Carlos Alcaraz"),
        PlayerData(player_id="P2", display_name="Jannik Sinner"),
    ]

    assert finder.find_by_name("Carlos Alcaraz") == "P1"
    assert finder.find_by_name("carlos alcaraz") == "P1"
    assert finder.find_by_name("Nobody") is None


def _hand_built_finder() -> PlayerSimilarity:
    """Build a three-vector FAISS fixture with descending P1 similarity."""
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
    monkeypatch.setattr(similarity, "DEFAULT_INDEX", tmp_path / "idx")
    monkeypatch.setattr(similarity, "DEFAULT_METADATA", tmp_path / "meta.json")
    con = duckdb.connect()
    try:
        _create_two_table_fixture(con)
        finder = PlayerSimilarity()
        finder.build(query=_duck_query(con))
    finally:
        con.close()

    assert (tmp_path / "idx").exists()
    assert (tmp_path / "meta.json").exists()

    loaded = PlayerSimilarity()
    loaded.load()

    assert loaded.player_ids == finder.player_ids
    assert loaded.players == finder.players
    assert loaded.index is not None and finder.index is not None
    assert loaded.index.d == finder.index.d
    assert loaded.index.ntotal == finder.index.ntotal


def test_load_missing_index_raises(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(similarity, "DEFAULT_INDEX", tmp_path / "missing.index")
    monkeypatch.setattr(similarity, "DEFAULT_METADATA", tmp_path / "meta.json")

    with pytest.raises(FileNotFoundError):
        PlayerSimilarity().load()


def test_build_defaults_to_live_postgresql_client():
    """The default query is the operational PostgreSQL client; offline builds
    pass src.db.training.to_dataframe explicitly."""
    assert similarity.to_dataframe is client.to_dataframe


def test_duck_and_snapshot_fixtures_produce_identical_vectors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The same SQL + builder yield bit-identical vectors through a direct
    DuckDB query and the two-table DuckDB snapshot (offline training path)."""
    _patch_embedding(monkeypatch)
    monkeypatch.setattr(similarity, "DEFAULT_INDEX", tmp_path / "idx")
    monkeypatch.setattr(similarity, "DEFAULT_METADATA", tmp_path / "meta.json")

    con = duckdb.connect()
    try:
        _create_two_table_fixture(con)
        direct = PlayerSimilarity()
        direct.build(query=_duck_query(con))
    finally:
        con.close()

    # The offline training path reads the same data from a snapshot file.
    snap = tmp_path / "parity.duckdb"
    con = duckdb.connect(str(snap))
    try:
        _create_two_table_fixture(con)
    finally:
        con.close()
    monkeypatch.setattr(training, "SNAPSHOT_PATH", snap)
    training.close()
    try:
        offline = PlayerSimilarity()
        offline.build(query=training.to_dataframe)
    finally:
        training.close()

    assert direct.player_ids == offline.player_ids
    assert direct.players == offline.players
    assert direct.index is not None and offline.index is not None
    assert direct.index.ntotal == offline.index.ntotal
    for i in range(direct.index.ntotal):
        assert np.array_equal(direct.index.reconstruct(i), offline.index.reconstruct(i))
