"""Similarity tests with DuckDB fixtures (no bio embeddings)."""

import re
from pathlib import Path

import duckdb
import faiss
import numpy as np
import pandas as pd
import pytest

from src.db import training
from src.training import similarity
from src.training.similarity import (
    DOMINANCE_COLS,
    LIFETIME_PLAYSTYLE_COLS,
    REPUTATION_BLOCK_COLS,
    SURFACE_BLOCK_COLS,
    PlayerData,
    PlayerSimilarity,
    block_slices,
)

# Included gold aggregates are lifetime playstyle, surface, and reputation signals.
# Recent form, rank, height, and turned_pro remain excluded.
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
    "hard_matches",
    "clay_matches",
    "grass_matches",
    "match_count",
    "career_win_rate",
    "current_rank",
    "win_rate_10",
    "dominance",
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
            birthdate DATE,
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
        "INSERT INTO bronze.player_profiles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("P1", "Alice", "1H", "L", "Great server", 185.0, "1990-01-01", 2010, "Spain"),
            ("P2", "Bob", "2H", "R", "", 190.0, "1995-01-01", 2015, "Italy"),
            ("P3", "Carol", "1H", "R", "Solid returner", 178.0, "2000-01-01", 2020, "USA"),
            ("P4", "Dave", "2H", "L", None, 195.0, "1992-01-01", 2012, "France"),
            ("", "Ghost", "1H", "R", "No id", 180.0, "1998-01-01", 2018, "UK"),
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
                200,
                150,
                50,
                400,
                0.72,
                5,
                0.80,
                1.0,
            ),
            (
                "P2",
                0.60,
                1.0,
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
                180,
                100,
                20,
                300,
                0.68,
                8,
                0.60,
            ),
            (
                "P3",
                None,
                1.0,
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
                0,
                0,
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
                1.0,
                0.30,
                0.50,
                250,
                60,
                40,
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
                30,
                10,
                0,
                50,
                0.50,
                60,
                0.40,
                1.0,
            ),
        ],
    )


def _duck_query(con: duckdb.DuckDBPyConnection):
    """Return a training.to_dataframe-compatible DuckDB query function."""

    def query(sql: str) -> pd.DataFrame:
        return con.execute(sql).df()

    return query


def _build_with_fixture(tmp_path: Path, monkeypatch) -> PlayerSimilarity:
    """Build against an in-memory DuckDB two-table fixture (offline path)."""
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


# Vector layout: identity, lifetime playstyle, surface, then reputation blocks.
STYLE = LIFETIME_PLAYSTYLE_COLS
SURFACE = SURFACE_BLOCK_COLS
REPUTATION = REPUTATION_BLOCK_COLS


def _block(vector: object, name: str) -> np.ndarray:
    """Slice one calibrated block out of a vector (see block_slices)."""
    arr: np.ndarray = np.asarray(vector)
    return arr[block_slices()[name]]


def _style_block(vector: object) -> np.ndarray:
    # faiss IndexFlatIP.reconstruct returns a plain np.ndarray.
    arr: np.ndarray = np.asarray(vector)
    return arr[block_slices()["playstyle"]]


def _style_values(vector: object) -> dict[str, float]:
    block = _style_block(vector)
    return {name: float(block[i]) for i, name in enumerate(LIFETIME_PLAYSTYLE_COLS)}


def _assert_style(vector: object, expected: dict[str, float], reference: str) -> None:
    """Check playstyle ratios against a common reference (block normalization preserves them)."""
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

    # Career playstyle aggregates and surface rates occupy separate blocks.
    _assert_style(
        index.reconstruct(finder.player_ids.index("P1")),
        {
            "first_serve_in_pct": 0.62,
            "overall_serve_points_won_pct": 0.65,
            "aces_per_service_game": 0.50,
            "return_points_won_pct": 0.42,
            "break_point_conversion_pct": 0.40,
        },
        reference="overall_serve_points_won_pct",
    )

    _assert_style(
        index.reconstruct(finder.player_ids.index("P4")),
        {
            "first_serve_in_pct": 0.58,
            "overall_serve_points_won_pct": 0.61,
            "aces_per_service_game": 0.25,
            "return_points_won_pct": 0.46,
            "break_point_conversion_pct": 0.44,
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


def test_build_is_deterministic_same_data_identical_vectors(tmp_path: Path, monkeypatch):
    """Rebuilding from identical data yields bit-identical vectors and layout."""
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
            "UPDATE gold.player_profiles SET overall_serve_points_won_pct = 0.99 "
            "WHERE player_id = 'P1'"
        )
        finder.build(query=_duck_query(con))
        assert finder.index is not old_index  # a fresh index, not a cached one
        assert not np.array_equal(base, np.array(finder.index.reconstruct(0)))
    finally:
        con.close()


def test_build_excludes_identity_and_recent_form(tmp_path: Path, monkeypatch):
    """Excluded identity and recent-form fields do not change vectors."""
    monkeypatch.setattr(similarity, "DEFAULT_INDEX", tmp_path / "idx")
    monkeypatch.setattr(similarity, "DEFAULT_METADATA", tmp_path / "meta.json")
    con = duckdb.connect()
    try:
        _create_two_table_fixture(con)
        finder = PlayerSimilarity()
        finder.build(query=_duck_query(con))
        assert finder.index is not None
        base = [np.array(finder.index.reconstruct(i)) for i in range(finder.index.ntotal)]

        # Mutate only excluded signals: audit metadata and recent rolling form.
        con.execute(
            "UPDATE bronze.player_profiles SET birthplace = 'X', display_name = 'X', summary = 'X'"
        )
        con.execute("UPDATE gold.player_profiles SET win_rate_10 = 0.99")
        # current_rank is also excluded from the vector now.
        con.execute("UPDATE gold.player_profiles SET current_rank = 1")
        finder2 = PlayerSimilarity()
        finder2.build(query=_duck_query(con))
        assert finder2.index is not None
        for i in range(finder2.index.ntotal):
            assert np.array_equal(base[i], finder2.index.reconstruct(i))

        con.execute("UPDATE bronze.player_profiles SET height = height + 5")
        finder_profile = PlayerSimilarity()
        finder_profile.build(query=_duck_query(con))
        assert finder_profile.index is not None
        assert not np.array_equal(base[0], finder_profile.index.reconstruct(0))

        # The reputation signals (match_count, career_win_rate) are career-level
        # and included in the vector.
        con.execute(
            "UPDATE gold.player_profiles SET match_count = match_count + 999, "
            "career_win_rate = 0.99"
        )
        finder3 = PlayerSimilarity()
        finder3.build(query=_duck_query(con))
        assert finder3.index is not None
        assert any(
            not np.array_equal(base[i], finder3.index.reconstruct(i))
            for i in range(finder3.index.ntotal)
        )

        # So are lifetime playstyle and surface signals.
        con.execute(
            "UPDATE gold.player_profiles SET overall_serve_points_won_pct = 0.99, "
            "clay_win_rate = 0.85 WHERE player_id = 'P1'"
        )
        finder4 = PlayerSimilarity()
        finder4.build(query=_duck_query(con))
        assert finder4.index is not None
        assert any(
            not np.array_equal(base[i], finder4.index.reconstruct(i))
            for i in range(finder4.index.ntotal)
        )
    finally:
        con.close()


def test_build_profile_block_carries_bronze_metadata(tmp_path: Path, monkeypatch):
    """The profile block uses bronze metadata and responds to height changes."""
    monkeypatch.setattr(similarity, "DEFAULT_INDEX", tmp_path / "idx")
    monkeypatch.setattr(similarity, "DEFAULT_METADATA", tmp_path / "meta.json")
    con = duckdb.connect()
    try:
        _create_two_table_fixture(con)
        finder = PlayerSimilarity()
        finder.build(query=_duck_query(con))
        assert finder.index is not None
        p1_profile = _block(finder.index.reconstruct(finder.player_ids.index("P1")), "profile")
        # height (185.0), age, and years_pro are all populated and non-zero.
        assert np.all(p1_profile > 0.0)
        assert p1_profile[0] > 0.0  # height component

        # A height change in the bronze snapshot moves the profile block and
        # therefore the full vector.
        findex = finder.index
        assert findex is not None
        base = np.array(findex.reconstruct(finder.player_ids.index("P1")))
        con.execute("UPDATE bronze.player_profiles SET height = height + 5 WHERE player_id = 'P1'")
        taller = PlayerSimilarity()
        taller.build(query=_duck_query(con))
        tindex = taller.index
        assert tindex is not None
        assert not np.array_equal(base, np.array(tindex.reconstruct(finder.player_ids.index("P1"))))
    finally:
        con.close()


def test_surface_block_shrinks_rates_and_carries_exposure(tmp_path: Path, monkeypatch):
    """Surface rates are exposure-shrunk toward 0.5 and retain exposure counts."""
    finder = _build_with_fixture(tmp_path, monkeypatch)
    index = finder.index
    assert index is not None
    p1 = _block(index.reconstruct(finder.player_ids.index("P1")), "surface")
    # Block layout: [hard_shrunk, clay_shrunk, grass_shrunk, hard_exp,
    # clay_exp, grass_exp]; shrunk = 0.5 + (rate - 0.5) * n / (n + 30).
    expected_clay_hard = (0.5 + 0.08 * 150 / 180) / (0.5 + 0.05 * 200 / 230)
    assert np.isclose(p1[1] / p1[0], expected_clay_hard, rtol=1e-4)
    expected_clay_exp = (150 / 180) / (200 / 230)
    assert np.isclose(p1[4] / p1[3], expected_clay_exp, rtol=1e-4)
    assert np.all(p1[3:] > 0.0)  # exposure counts are present in the vector

    # Cold start (P3): exposure is zero and every rate sits at the neutral 0.5
    # prior — no surface-preference signal, but never a spurious extreme.
    p3 = _block(index.reconstruct(finder.player_ids.index("P3")), "surface")
    assert np.all(p3[3:] == 0.0)
    assert np.allclose(p3[:3], p3[0])


def test_changing_clay_win_rate_changes_vectors(tmp_path: Path, monkeypatch):
    """Changing one player's clay rate moves only that player's vector."""
    monkeypatch.setattr(similarity, "DEFAULT_INDEX", tmp_path / "idx")
    monkeypatch.setattr(similarity, "DEFAULT_METADATA", tmp_path / "meta.json")
    con = duckdb.connect()
    try:
        _create_two_table_fixture(con)
        finder = PlayerSimilarity()
        finder.build(query=_duck_query(con))
        assert finder.index is not None
        base = [np.array(finder.index.reconstruct(i)) for i in range(finder.index.ntotal)]

        con.execute("UPDATE gold.player_profiles SET clay_win_rate = 0.95 WHERE player_id = 'P1'")
        finder.build(query=_duck_query(con))
        assert finder.index is not None
        for i in range(finder.index.ntotal):
            if finder.player_ids[i] == "P1":
                assert not np.array_equal(base[i], finder.index.reconstruct(i))
            else:
                assert np.array_equal(base[i], finder.index.reconstruct(i))

        # P1's clay component rises relative to hard within the surface block.
        baseline_p1 = _block(base[finder.player_ids.index("P1")], "surface")
        p1 = _block(finder.index.reconstruct(finder.player_ids.index("P1")), "surface")
        assert p1[1] / p1[0] > baseline_p1[1] / baseline_p1[0]
    finally:
        con.close()


def _synthetic_profiles(player_ids: list[str]) -> pd.DataFrame:
    """Profiled players with identical identity/bio (only career stats vary)."""
    return pd.DataFrame(
        {
            "player_id": player_ids,
            "display_name": player_ids,
            "backhand": ["2H"] * len(player_ids),
            "handedness": ["R"] * len(player_ids),
        }
    )


def _stub_query(rows: list[dict[str, str | float]]):
    """Gold aggregate stub: one row per dict, missing career cells default 0.5."""
    cols = [
        "player_id",
        *LIFETIME_PLAYSTYLE_COLS,
        *SURFACE_BLOCK_COLS,
        *DOMINANCE_COLS,
        *REPUTATION_BLOCK_COLS,
    ]

    def query(_sql: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                col: [
                    r.get(col, 0.5) if col == "player_id" else float(r.get(col, 0.5)) for r in rows
                ]
                for col in cols
            }
        )

    return query


def _cos(a: object, b: object) -> float:
    return float(np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32))


def test_reputation_block_excludes_rank_and_includes_career(tmp_path, monkeypatch):
    """Reputation uses career count and win rate, but not current rank."""
    monkeypatch.setattr(similarity, "DEFAULT_INDEX", tmp_path / "idx")
    monkeypatch.setattr(similarity, "DEFAULT_METADATA", tmp_path / "meta.json")
    con = duckdb.connect()
    try:
        _create_two_table_fixture(con)
        finder = PlayerSimilarity()
        finder.build(query=_duck_query(con))
        assert finder.index is not None
        base = [np.array(finder.index.reconstruct(i)) for i in range(finder.index.ntotal)]

        # current_rank is no longer part of the vector.
        con.execute("UPDATE gold.player_profiles SET current_rank = 1 WHERE player_id = 'P1'")
        finder.build(query=_duck_query(con))
        assert np.array_equal(base[0], np.array(finder.index.reconstruct(0)))

        # match_count change does move the vector.
        con.execute(
            "UPDATE gold.player_profiles SET match_count = match_count + 999 WHERE player_id = 'P1'"
        )
        finder.build(query=_duck_query(con))
        assert not np.array_equal(base[0], np.array(finder.index.reconstruct(0)))
    finally:
        con.close()


def _surface_query(player_ids: list[str]):
    """Return gold aggregates with enough surface exposure to show rate changes."""

    def query(_sql: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "player_id": player_ids,
                **dict.fromkeys(LIFETIME_PLAYSTYLE_COLS, 0.4),
                "hard_win_rate": 0.55,
                "clay_win_rate": 0.50,
                "grass_win_rate": 0.50,
                "hard_matches": 150.0,
                "clay_matches": 200.0,
                "grass_matches": 50.0,
                "match_count": 200.0,
                "career_win_rate": 0.6,
            }
        )

    return query


def test_similar_style_players_more_comparable():
    """Similar clay-court styles rank closer than a dissimilar hard-court style."""
    clay_a = {
        "player_id": "A",
        "clay_win_rate": 0.76,
        "clay_matches": 320.0,
        "hard_win_rate": 0.55,
        "hard_matches": 190.0,
        "grass_win_rate": 0.50,
        "grass_matches": 40.0,
        "first_serve_in_pct": 0.62,
        "aces_per_first_serve": 0.05,
        "break_points_saved_pct": 0.65,
        "return_points_won_pct": 0.57,
        "second_serve_return_points_won_pct": 0.60,
        "break_point_conversion_pct": 0.48,
    }
    clay_b = dict(clay_a) | {"player_id": "B", "first_serve_in_pct": 0.60}
    hard_c = {
        "player_id": "C",
        "hard_win_rate": 0.78,
        "hard_matches": 350.0,
        "clay_win_rate": 0.40,
        "clay_matches": 30.0,
        "grass_win_rate": 0.45,
        "grass_matches": 40.0,
        "first_serve_in_pct": 0.58,
        "aces_per_first_serve": 0.32,
        "first_serve_points_won_pct": 0.78,
        "double_faults_per_serve_point": 0.05,
        "break_points_saved_pct": 0.60,
        "return_points_won_pct": 0.38,
        "break_point_conversion_pct": 0.34,
    }
    matrix = similarity.build_playstyle_matrix(
        _synthetic_profiles(["A", "B", "C"]),
        query=_stub_query([clay_a, clay_b, hard_c]),
    )
    v = matrix.to_numpy(np.float32)
    cos_ab = _cos(v[0], v[1])
    cos_ac = _cos(v[0], v[2])
    cos_bc = _cos(v[1], v[2])
    assert cos_ab > 0.99  # near-identical clay+style profiles are very close
    assert cos_ac < 0.99  # a hard-court attacker is clearly farther
    assert cos_ab > cos_ac and cos_ab > cos_bc


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
    monkeypatch.setattr(similarity, "DEFAULT_INDEX", tmp_path / "idx")
    monkeypatch.setattr(similarity, "DEFAULT_METADATA", tmp_path / "meta.json")
    # Point build and serving paths at the same temporary files for the round-trip.
    monkeypatch.setattr(similarity, "SERVING_INDEX", tmp_path / "idx")
    monkeypatch.setattr(similarity, "SERVING_METADATA", tmp_path / "meta.json")
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
    monkeypatch.setattr(similarity, "SERVING_INDEX", tmp_path / "missing.index")
    monkeypatch.setattr(similarity, "SERVING_METADATA", tmp_path / "meta.json")

    with pytest.raises(FileNotFoundError):
        PlayerSimilarity().load()


def test_duck_and_snapshot_fixtures_produce_identical_vectors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Direct and snapshot DuckDB queries produce identical vectors."""
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
