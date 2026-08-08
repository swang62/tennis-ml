"""Tests for src/models/similarity.py (no network, no live PostgreSQL needed).

fastembed.TextEmbedding is stubbed with a fake that yields fixed 4-dim ones
vectors. Player state is read from an in-memory DuckDB fixture holding the
two-table snapshot boundary — gold.match_features (the columns the builder
reads) and gold.player_profiles — through an explicit query function, exactly
as an offline build passes ``src.db.training.to_dataframe``. A live parity
test rebuilds the real two-table snapshot from PostgreSQL via
``src.db.snapshot.refresh_snapshot`` and asserts bit-identical FAISS vectors
between the live PostgreSQL and snapshot DuckDB query paths when the seeded
database is reachable.
"""

import re
import sys
from pathlib import Path

import duckdb
import faiss
import numpy as np
import pandas as pd
import pytest

from src.db import client, snapshot, training
from src.models import similarity
from src.models.similarity import STYLE_COLS, PlayerData, PlayerSimilarity, embed_bio_summaries


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
    # embed_bio_summaries imports fastembed lazily inside the function, so the
    # fake module is injected into sys.modules (never by patching a similarity
    # module attribute, which the lazy import would clobber).
    fake = FakeTextEmbedding()
    monkeypatch.setitem(sys.modules, "fastembed", _FakeFastembed(lambda _model_name: fake))
    return fake


def _create_two_table_fixture(con: duckdb.DuckDBPyConnection) -> None:
    """Create the gold.match_features + gold.player_profiles fixture tables.

    match_features declares the columns the builder's state query reads (the
    full 44-column contract is exercised by the live parity test) plus a set
    of physical/résumé columns the builder must NEVER read (age, rankings) so
    the exclusion test can prove they do not affect the vectors. Rows:
      m1 clay  P1 vs P4   m2 grass P1 vs P2   m3 hard  P2 vs P4
      m4 clay  P1 vs P2 (later than m1: latest weighted form / clay rate win
                         and the latest serve/return percentages for P1/P2)
    P4 appears only on the opponent side; P3 has a profile but no matches.
    """
    con.execute("CREATE SCHEMA gold")
    con.execute(
        """
        CREATE TABLE gold.match_features (
            match_id VARCHAR,
            match_date DATE,
            surface VARCHAR,
            player_id VARCHAR,
            opponent_id VARCHAR,
            player_weighted_form_10 DOUBLE,
            player_surface_win_rate_10 DOUBLE,
            opponent_weighted_form_10 DOUBLE,
            opponent_surface_win_rate_10 DOUBLE,
            player_first_serve_pct_10 DOUBLE,
            player_first_serve_win_pct_10 DOUBLE,
            player_second_serve_win_pct_10 DOUBLE,
            player_serve_win_pct_10 DOUBLE,
            player_return_points_won_pct_10 DOUBLE,
            opponent_first_serve_pct_10 DOUBLE,
            opponent_first_serve_win_pct_10 DOUBLE,
            opponent_second_serve_win_pct_10 DOUBLE,
            opponent_serve_win_pct_10 DOUBLE,
            opponent_return_points_won_pct_10 DOUBLE,
            player_age DOUBLE,
            player_ranking DOUBLE,
            opponent_age DOUBLE,
            opponent_ranking DOUBLE
        )
        """
    )
    con.execute(
        """
        CREATE TABLE gold.player_profiles (
            player_id VARCHAR,
            display_name VARCHAR,
            backhand VARCHAR,
            handedness VARCHAR,
            summary VARCHAR,
            height DOUBLE,
            turned_pro INTEGER,
            birthplace VARCHAR,
            matches_played INTEGER,
            career_win_rate DOUBLE
        )
        """
    )
    # Column order: match_id, match_date, surface, player_id, opponent_id,
    # pw, ps, ow, os, then per-side serve/return (player then opponent):
    # first_serve_pct, first_serve_win, second_serve_win, serve_win,
    # return_points_won; then excluded age/ranking (player then opponent).
    con.executemany(
        "INSERT INTO gold.match_features VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            # m1 P1 vs P4 clay
            (
                "m1",
                "2026-05-01",
                "clay",
                "P1",
                "P4",
                0.80,
                0.55,
                0.60,
                0.30,
                0.61,
                0.71,
                0.49,
                0.63,
                0.54,
                0.56,
                0.65,
                0.43,
                0.57,
                0.48,
                30.0,
                5.0,
                33.0,
                20.0,
            ),
            # m2 P1 vs P2 grass
            (
                "m2",
                "2026-06-01",
                "grass",
                "P1",
                "P2",
                0.78,
                0.60,
                0.40,
                0.52,
                0.60,
                0.69,
                0.47,
                0.61,
                0.52,
                0.59,
                0.68,
                0.46,
                0.60,
                0.51,
                30.0,
                5.0,
                25.0,
                8.0,
            ),
            # m3 P2 vs P4 hard
            (
                "m3",
                "2026-07-01",
                "hard",
                "P2",
                "P4",
                0.41,
                0.52,
                0.36,
                0.45,
                0.58,
                0.67,
                0.45,
                0.59,
                0.50,
                0.57,
                0.66,
                0.44,
                0.58,
                0.49,
                25.0,
                8.0,
                33.0,
                20.0,
            ),
            # m4 P1 vs P2 clay (latest for P1 and P2)
            (
                "m4",
                "2026-07-15",
                "clay",
                "P1",
                "P2",
                0.90,
                0.58,
                0.44,
                0.48,
                0.62,
                0.72,
                0.50,
                0.64,
                0.55,
                0.60,
                0.70,
                0.48,
                0.62,
                0.53,
                30.0,
                5.0,
                25.0,
                8.0,
            ),
        ],
    )
    con.executemany(
        "INSERT INTO gold.player_profiles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("P1", "Alice", "one", "right", "Great server", 185.0, 2010, "Spain", 400, 0.72),
            ("P2", "Bob", "two", "left", "", 190.0, 2015, "Italy", 300, 0.68),
            ("P3", "Carol", "one", "right", "Solid returner", 178.0, 2020, "USA", 100, 0.55),
            ("P4", "Dave", "two", "right", None, 195.0, 2012, "France", 350, 0.70),
            ("", "Ghost", "one", "right", "No id", 180.0, 2018, "UK", 50, 0.50),
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


# One-hot block (backhand x2, handedness x2) precedes the style stats; the bio
# block trails them. STYLE_COLS is imported so the layout stays in sync with
# the builder's vector contract.
ONE_HOT = 4
STYLE = STYLE_COLS


def _style_block(vector: object) -> np.ndarray:
    # faiss IndexFlatIP.reconstruct returns a plain np.ndarray.
    arr: np.ndarray = np.asarray(vector)
    return arr[ONE_HOT : ONE_HOT + len(STYLE)]


def test_build_uses_latest_pre_match_absolute_state(tmp_path: Path, monkeypatch):
    finder = _build_with_fixture(tmp_path, monkeypatch)
    index = finder.index
    assert index is not None
    # 4 one-hot + 9 style stats + 4 bio dims.
    assert index.d == ONE_HOT + len(STYLE) + 4

    p1 = index.reconstruct(finder.player_ids.index("P1"))
    p1_style = _style_block(p1)
    # P1's latest match (m4) supplies weighted form 0.90, clay rate 0.58 and
    # the serve/return percentages (fp .62 fw .72 sw .50 sv .64 rp .55), not
    # the older clay match's 0.80/0.55/...; grass comes from the only grass
    # match; hard was never played and stays 0.0. L2 normalization scales all
    # components by one factor, so within-vector ratios survive.
    assert np.isclose(p1_style[0] / p1_style[1], 0.90 / 0.58)
    assert np.isclose(p1_style[2], p1_style[1] * (0.60 / 0.58))
    assert p1_style[3] == 0.0
    assert np.isclose(p1_style[4] / p1_style[1], 0.62 / 0.58)
    assert np.isclose(p1_style[5] / p1_style[1], 0.72 / 0.58)
    assert np.isclose(p1_style[6] / p1_style[1], 0.50 / 0.58)
    assert np.isclose(p1_style[7] / p1_style[1], 0.64 / 0.58)
    assert np.isclose(p1_style[8] / p1_style[1], 0.55 / 0.58)

    p4 = index.reconstruct(finder.player_ids.index("P4"))
    p4_style = _style_block(p4)
    # P4 appears only on the opponent side: latest overall is the hard match
    # (weighted 0.36, hard 0.45, serve fp .57 fw .66 sw .44 sv .58 rp .49)
    # while clay comes from the only clay match (m1).
    assert np.isclose(p4_style[0] / p4_style[1], 0.36 / 0.30)
    assert np.isclose(p4_style[3], p4_style[1] * (0.45 / 0.30))
    assert p4_style[2] == 0.0
    assert np.isclose(p4_style[4] / p4_style[1], 0.57 / 0.30)
    assert np.isclose(p4_style[8] / p4_style[1], 0.49 / 0.30)


def test_build_players_on_either_side_included_exactly_once(tmp_path: Path, monkeypatch):
    finder = _build_with_fixture(tmp_path, monkeypatch)
    # Ghost (empty player_id) is dropped; P4 (opponent-only) is included.
    assert finder.player_ids == ["P1", "P2", "P3", "P4"]
    assert len(finder.player_ids) == len(set(finder.player_ids))


def test_build_null_and_cold_start_style_cells_imputed_zero(tmp_path: Path, monkeypatch):
    finder = _build_with_fixture(tmp_path, monkeypatch)
    index = finder.index
    assert index is not None

    # P3 has a profile but no matches: every style stat imputes to 0.0.
    p3 = _style_block(index.reconstruct(finder.player_ids.index("P3")))
    assert np.all(p3 == 0.0)

    # P2 never played hard on the winning side... its hard rate comes from m3;
    # P1 never played hard at all -> 0.0 while other rates stay non-zero.
    p1 = _style_block(index.reconstruct(finder.player_ids.index("P1")))
    assert p1[3] == 0.0
    assert np.all(p1[:3] > 0.0)


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
    # Bio block is ones (normalized); style stats are non-zero and present.
    assert np.all(p2[ONE_HOT + len(STYLE) :] > 0.0)
    assert np.all(_style_block(p2) > 0.0)


def test_build_excludes_physical_and_resume_metrics(tmp_path: Path, monkeypatch):
    """Age, height, turned_pro, birthplace, rankings and career totals never
    enter the similarity signal: mutating them across the tables leaves every
    vector bit-identical, while mutating a serve percentage changes it."""
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

        # Mutate only excluded signals: physical (age, height, birthplace),
        # résumé (turned_pro), rankings, and career totals/lifetime stats.
        con.execute(
            "UPDATE gold.match_features SET player_age = player_age + 10, "
            "opponent_age = opponent_age + 10, player_ranking = player_ranking + 50, "
            "opponent_ranking = opponent_ranking + 50"
        )
        con.execute(
            "UPDATE gold.player_profiles SET height = height + 5, "
            "turned_pro = turned_pro - 3, birthplace = 'X', "
            "matches_played = matches_played + 999, career_win_rate = 0.99"
        )
        finder2 = PlayerSimilarity()
        finder2.build(query=_duck_query(con))
        assert finder2.index is not None
        for i in range(finder2.index.ntotal):
            assert np.array_equal(base[i], finder2.index.reconstruct(i))

        # A real style signal (serve pct) must change the vectors.
        con.execute(
            "UPDATE gold.match_features SET player_first_serve_pct_10 = 0.99 WHERE match_id = 'm4'"
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


def test_postgres_and_snapshot_fixtures_produce_identical_vectors(
    postgres_ready,  # noqa: ARG001 — skip-gate fixture, unused in body
    gold_ready,  # noqa: ARG001 — skip-gate fixture, unused in body
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The same SQL + builder yield bit-identical vectors through live
    PostgreSQL and the two-table DuckDB snapshot (offline path)."""
    _patch_embedding(monkeypatch)
    monkeypatch.setattr(similarity, "DEFAULT_INDEX", tmp_path / "idx")
    monkeypatch.setattr(similarity, "DEFAULT_METADATA", tmp_path / "meta.json")

    snap = tmp_path / "parity.duckdb"
    snapshot.refresh_snapshot(snap)  # copies gold.match_features + player_profiles
    monkeypatch.setattr(training, "SNAPSHOT_PATH", snap)
    training.close()
    try:
        live = PlayerSimilarity()
        live.build(query=client.to_dataframe)
        offline = PlayerSimilarity()
        offline.build(query=training.to_dataframe)
    finally:
        training.close()

    assert live.player_ids == offline.player_ids
    assert live.players == offline.players
    assert live.index is not None and offline.index is not None
    assert live.index.ntotal == offline.index.ntotal
    for i in range(live.index.ntotal):
        assert np.array_equal(live.index.reconstruct(i), offline.index.reconstruct(i))
    assert live.search("P1") == offline.search("P1")
