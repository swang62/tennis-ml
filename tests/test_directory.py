"""Hermetic tests for the player directory and similarity staging."""

import importlib
import json

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.serving.directory import PLAYERS_SQL, directory_players
from src.training.similarity import PLAYER_LIFETIME_SQL


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


# ── Shared directory contract ───────────────────────────────────────────────


def _directory_sql_df() -> pd.DataFrame:
    """Run the real PLAYERS_SQL against in-memory bronze and gold fixtures."""
    con = duckdb.connect()
    con.execute("CREATE SCHEMA bronze")
    con.execute("CREATE SCHEMA gold")
    con.execute(
        """
        CREATE TABLE bronze.player_profiles (
            player_id VARCHAR, display_name VARCHAR, ioc VARCHAR,
            backhand VARCHAR, handedness VARCHAR, summary VARCHAR,
            height DOUBLE, birthdate DATE, turned_pro INTEGER
        )
        """
    )
    con.execute(
        "CREATE TABLE gold.player_profiles (player_id VARCHAR, match_count BIGINT, current_rank BIGINT)"
    )
    con.executemany(
        "INSERT INTO bronze.player_profiles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("p1", "A Player", "ESP", "1H", "R", "s1", 185.0, "1990-01-01", 2010),
            ("p2", "B Player", "ARG", "2H", "L", "s2", 190.0, "1992-01-01", 2012),
            ("p3", "C Player", "USA", "1H", "L", "s3", 178.0, "1994-01-01", 2014),
            (
                "p4",
                "D Player",
                "FRA",
                "2H",
                "R",
                "s4",
                195.0,
                "1996-01-01",
                2016,
            ),  # no gold row at all
            ("p5", "E Player", "AUS", "2H", "R", "s5", 180.0, "1998-01-01", 2018),
        ],
    )
    con.executemany(
        "INSERT INTO gold.player_profiles VALUES (?, ?, ?)",
        [
            ("p1", 1, 5),  # 1 physical match
            ("p2", 0, 10),  # zero-match gold row: retained
            ("p3", 30, None),  # 30 physical matches: unranked but retained
            ("p5", 5, None),  # 5 physical matches: retained as-is
        ],
    )
    return con.execute(PLAYERS_SQL).df()


def test_players_sql_returns_every_bronze_profile():
    """The real query, run hermetically: all bronze profiles are retained in
    SQL row order — zero-match gold rows and players with no gold row included
    (reporting 0 matches) — and the directory mapping carries all of them."""
    df = _directory_sql_df()
    assert list(df["player_id"]) == ["p1", "p2", "p3", "p4", "p5"]
    assert list(df["matches_played"]) == [1, 0, 30, 0, 5]
    assert pd.isna(df["current_rank"].iloc[2])  # p3 unranked sorts after ranked

    players = directory_players(df)
    assert [p["player_id"] for p in players] == ["p1", "p2", "p3", "p4", "p5"]
    assert [p["matches_played"] for p in players] == [1, 0, 30, 0, 5]
    assert players[0]["matches_played"] == 1


def test_players_sql_reports_gold_physical_count_directly():
    """matches_played preserves gold physical counts, including odd and zero counts."""
    df = _directory_sql_df()
    by_id = {row["player_id"]: row for _, row in df.iterrows()}
    assert by_id["p1"]["matches_played"] == 1
    assert by_id["p2"]["matches_played"] == 0
    assert by_id["p3"]["matches_played"] == 30
    assert by_id["p4"]["matches_played"] == 0  # no gold row: COALESCE 0
    assert by_id["p5"]["matches_played"] == 5


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
    # the full list is JSON-serializable (the directory endpoint response)
    json.dumps(players)


def test_directory_players_preserves_sql_row_order():
    players = directory_players(_directory_df())
    assert [p["player_id"] for p in players] == ["p2", "p1", "p3"]


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


# ── Deployment staging ──────────────────────────────────────────────────────


def _deploy():
    return importlib.import_module("src.flows.deploy")


def _similarity_df() -> pd.DataFrame:
    """Profiles-frame stand-in for the snapshot PLAYERS_SQL result."""
    return _directory_df()


def _similarity_lifetime_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"player_id": "p1", "first_serve_in_pct": np.float64(0.62)},
            {"player_id": "p2", "first_serve_in_pct": np.float64(0.64)},
            {"player_id": "p3", "first_serve_in_pct": np.float64(0.66)},
        ]
    )


class _FakeSimilarity:
    """Records the build and stages index/metadata over the given players."""

    def __init__(self, players):
        self.players = players
        self.built = 0

    def build(self, **kwargs):
        self.built += 1
        kwargs["index_path"].write_bytes(b"index")
        kwargs["metadata_path"].write_text(json.dumps(self.players))


def _patch_similarity_staging(monkeypatch, d, tmp_path, profiles, lifetime, similarity_players):
    """Point the deploy similarity paths at tmp and stub snapshot+similarity."""
    sim_index = tmp_path / "player_similarity.index"
    sim_meta = tmp_path / "player_metadata.json"
    state = tmp_path / "similarity_artifacts_state.json"
    monkeypatch.setattr(d, "SIMILARITY_INDEX", sim_index)
    monkeypatch.setattr(d, "SIMILARITY_METADATA", sim_meta)
    monkeypatch.setattr(d, "SIMILARITY_STATE_FILE", state)

    queries = {PLAYERS_SQL: profiles, PLAYER_LIFETIME_SQL: lifetime}
    monkeypatch.setattr("src.db.training.to_dataframe", lambda sql: queries[sql])
    fake = _FakeSimilarity(similarity_players)
    monkeypatch.setattr("src.training.similarity.PlayerSimilarity", lambda: fake)
    return sim_index, sim_meta, state, fake


def test_generate_similarity_artifacts_builds_index_and_metadata_from_snapshot(
    monkeypatch, tmp_path
):
    """Deploy builds the FAISS similarity index + metadata from one snapshot read."""
    d = _deploy()
    sim_index, sim_meta, _state, fake = _patch_similarity_staging(
        monkeypatch,
        d,
        tmp_path,
        _similarity_df(),
        _similarity_lifetime_df(),
        similarity_players=[{"player_id": "p2"}, {"player_id": "p1"}, {"player_id": "p3"}],
    )

    path = d.generate_similarity_artifacts()

    assert path == sim_index
    assert fake.built == 1
    assert sim_index.read_bytes() == b"index"
    assert [p["player_id"] for p in json.loads(sim_meta.read_text())] == ["p2", "p1", "p3"]
    # The similarity metadata is the snapshot-backed player list; the staged
    # state records the current inputs/source hashes.
    state = json.loads(_state.read_text())
    assert state["inputs_hash"] == d._similarity_inputs_hash(
        _similarity_df(), _similarity_lifetime_df()
    )
    assert state["source_hash"] == d._similarity_source_hash()


def test_generate_similarity_artifacts_cross_checks_directory_player_ids(monkeypatch, tmp_path):
    """The staged similarity players must match the directory player set."""
    d = _deploy()
    _sim_index, _sim_meta, _state, _fake = _patch_similarity_staging(
        monkeypatch,
        d,
        tmp_path,
        _similarity_df(),
        _similarity_lifetime_df(),
        similarity_players=[{"player_id": "p2"}, {"player_id": "p1"}],  # p3 missing
    )

    import pytest

    with pytest.raises(RuntimeError, match="player IDs differ"):
        d.generate_similarity_artifacts()


def test_generate_similarity_artifacts_requires_snapshot(monkeypatch, tmp_path):
    """Without the training snapshot the deploy fails with the actionable
    snapshot-missing error and stages nothing."""
    d = _deploy()
    sim_index, _sim_meta, _state, _fake = _patch_similarity_staging(
        monkeypatch,
        d,
        tmp_path,
        pd.DataFrame(),
        pd.DataFrame(),
        similarity_players=[],
    )

    def no_snapshot(_sql):
        raise FileNotFoundError(
            "training snapshot not found at data/processed/training_snapshot.duckdb; "
            "run `just snapshot` first"
        )

    monkeypatch.setattr("src.db.training.to_dataframe", no_snapshot)

    import pytest

    with pytest.raises(FileNotFoundError, match="training snapshot not found"):
        d.generate_similarity_artifacts()
    assert not sim_index.exists()


def test_generate_similarity_artifacts_requires_profile_rows(monkeypatch, tmp_path):
    """An empty snapshot profiles read aborts the deploy with the actionable
    refresh error, never staging half-built similarity assets."""
    d = _deploy()
    sim_index, _sim_meta, _state, _fake = _patch_similarity_staging(
        monkeypatch,
        d,
        tmp_path,
        pd.DataFrame(),
        pd.DataFrame(),
        similarity_players=[],
    )

    import pytest

    with pytest.raises(RuntimeError, match="no player profiles"):
        d.generate_similarity_artifacts()
    assert not sim_index.exists()


def test_generate_similarity_artifacts_raises_when_state_write_fails(monkeypatch, tmp_path):
    """A staging failure also aborts the artifact (and therefore the deploy)."""
    d = _deploy()
    sim_index, sim_meta, _state, _fake = _patch_similarity_staging(
        monkeypatch,
        d,
        tmp_path,
        _similarity_df(),
        _similarity_lifetime_df(),
        similarity_players=[{"player_id": "p2"}, {"player_id": "p1"}, {"player_id": "p3"}],
    )

    def fail_state_write(_inputs_hash, _source_hash):
        raise OSError("state write failed")

    monkeypatch.setattr(d, "_write_similarity_state", fail_state_write)

    import pytest

    with pytest.raises(OSError):
        d.generate_similarity_artifacts()
    # The artifacts were built before the failing state write.
    assert sim_index.read_bytes() == b"index"
    assert sim_meta.exists()


# ── Similarity artifact content-hash reuse ──────────────────────────────────


def _sim_fixtures():
    """One snapshot input set: profiles and gold lifetime stats."""
    profiles = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "display_name": "A Player",
                "matches_played": np.int64(20),
                "latest_rank_points": None,
                "ioc": "ESP",
                "current_rank": None,
            },
            {
                "player_id": "p2",
                "display_name": "B Player",
                "matches_played": np.int64(40),
                "latest_rank_points": np.float64(1500.0),
                "ioc": "ARG",
                "current_rank": np.int64(2),
            },
        ]
    )
    lifetime = pd.DataFrame(
        [
            {"player_id": "p1", "first_serve_in_pct": np.float64(0.62)},
            {"player_id": "p2", "first_serve_in_pct": np.float64(0.64)},
        ]
    )
    return profiles, lifetime


def _mutated(kind, profiles, lifetime):
    """Return a copy of the fixtures with exactly one input changed."""
    if kind == "profiles":
        profiles = profiles.copy()
        profiles.loc[0, "display_name"] = "A Player v2"
    else:
        lifetime = lifetime.copy()
        lifetime.loc[0, "first_serve_in_pct"] = np.float64(0.99)
    return profiles, lifetime


def _stage_sim_build(
    monkeypatch,
    tmp_path,
    d,
    profiles,
    lifetime,
    *,
    state_hash,
    state_source_hash=None,
):
    """Run similarity generation against staged in-memory fixtures and record rebuilds.
    (builds) list, so reuse decisions can be asserted.
    """
    sim_index = tmp_path / "player_similarity.index"
    sim_meta = tmp_path / "player_metadata.json"
    state = tmp_path / "similarity_artifacts_state.json"
    monkeypatch.setattr(d, "SIMILARITY_INDEX", sim_index)
    monkeypatch.setattr(d, "SIMILARITY_METADATA", sim_meta)
    monkeypatch.setattr(d, "SIMILARITY_STATE_FILE", state)
    if state_hash is not None:
        state.write_text(json.dumps({"inputs_hash": state_hash, "source_hash": state_source_hash}))
    for path, content in (
        (sim_index, "stale-index"),
        (sim_meta, "stale-metadata"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    queries = {PLAYERS_SQL: profiles, PLAYER_LIFETIME_SQL: lifetime}
    monkeypatch.setattr("src.db.training.to_dataframe", lambda sql: queries[sql])

    builds = []

    class _RecordingSimilarity:
        def __init__(self):
            self.players: list[dict[str, object]] = []

        def build(self, **kwargs):
            builds.append(1)
            self.players = [{"player_id": str(row["player_id"])} for _, row in profiles.iterrows()]
            kwargs["index_path"].write_bytes(b"fresh-index")
            kwargs["metadata_path"].write_text(json.dumps(self.players))

    monkeypatch.setattr("src.training.similarity.PlayerSimilarity", _RecordingSimilarity)
    return sim_index, sim_meta, state, builds


def test_similarity_inputs_hash_deterministic_and_row_semantics():
    """The hash is stable across calls; lifetime rows sort canonically but the
    profile row order is preserved because it shapes the index."""
    d = _deploy()
    profiles, lifetime = _sim_fixtures()
    first = d._similarity_inputs_hash(profiles, lifetime)
    assert first == d._similarity_inputs_hash(profiles, lifetime)
    # Lifetime rows feed only a player-keyed merge: shuffled rows hash the same.
    shuffled = lifetime.iloc[::-1].reset_index(drop=True)
    assert first == d._similarity_inputs_hash(profiles, shuffled)
    # Profile row order is part of the artifact contract (index row order).
    assert first != d._similarity_inputs_hash(profiles.iloc[::-1].reset_index(drop=True), lifetime)


def test_generate_similarity_artifacts_reuses_unchanged_inputs(monkeypatch, tmp_path):
    """Unchanged snapshot inputs, unchanged shaping sources, and present staged
    artifacts reuse the staged similarity — no expensive FAISS/embedding build."""
    d = _deploy()
    profiles, lifetime = _sim_fixtures()
    state_hash = d._similarity_inputs_hash(profiles, lifetime)
    sim_index, sim_meta, _state, _ = _stage_sim_build(
        monkeypatch,
        tmp_path,
        d,
        profiles,
        lifetime,
        state_hash=state_hash,
        state_source_hash=d._similarity_source_hash(),
    )

    assert d.generate_similarity_artifacts() == sim_index
    assert sim_index.read_text() == "stale-index"  # staged similarity untouched
    assert sim_meta.read_text() == "stale-metadata"


@pytest.mark.parametrize("mutate", ["profiles", "lifetime"])
def test_generate_similarity_artifacts_rebuilds_when_dependency_changes(
    monkeypatch, tmp_path, mutate
):
    """A change to any one of the hash inputs forces a full rebuild."""
    d = _deploy()
    profiles, lifetime = _sim_fixtures()
    stale_hash = d._similarity_inputs_hash(profiles, lifetime)
    changed = _mutated(mutate, profiles, lifetime)
    sim_index, _sim_meta, state, _ = _stage_sim_build(
        monkeypatch, tmp_path, d, *changed, state_hash=stale_hash
    )

    d.generate_similarity_artifacts()

    assert sim_index.read_bytes() == b"fresh-index"
    # The new inputs hash is persisted so the next deploy can reuse.
    assert json.loads(state.read_text())["inputs_hash"] == d._similarity_inputs_hash(*changed)


def test_generate_similarity_artifacts_rebuilds_when_output_missing(monkeypatch, tmp_path):
    """A matching hash and source fingerprint but a deleted staged output still
    rebuilds it."""
    d = _deploy()
    profiles, lifetime = _sim_fixtures()
    state_hash = d._similarity_inputs_hash(profiles, lifetime)
    _sim_index, sim_meta, _state, _ = _stage_sim_build(
        monkeypatch,
        tmp_path,
        d,
        profiles,
        lifetime,
        state_hash=state_hash,
        state_source_hash=d._similarity_source_hash(),
    )
    sim_meta.unlink()

    d.generate_similarity_artifacts()

    assert sim_meta.read_text() != "stale-metadata"


@pytest.mark.parametrize("state_content", [None, "not json", "[]"])
def test_generate_similarity_artifacts_rebuilds_on_invalid_state(
    monkeypatch, tmp_path, state_content
):
    """Missing or corrupt persisted state never reuses stale artifacts."""
    d = _deploy()
    profiles, lifetime = _sim_fixtures()
    sim_index, _sim_meta, state, _ = _stage_sim_build(
        monkeypatch, tmp_path, d, profiles, lifetime, state_hash=None
    )
    if state_content is not None:
        state.write_text(state_content)

    d.generate_similarity_artifacts()

    assert sim_index.read_bytes() == b"fresh-index"


# ── Similarity source/config fingerprint ────────────────────────────────────


# Allowlisted source and tuning inputs that can change similarity outputs.
_SIM_SOURCE_MUTATIONS = [
    *[("file", name) for name in ("directory.py", "similarity.py", "countries.py")],
    *[
        ("constant", name)
        for name in (
            "SIM_IDENTITY_WEIGHT",
            "SIM_PLAYSTYLE_WEIGHT",
            "SIM_SURFACE_WEIGHT",
            "SIM_REPUTATION_WEIGHT",
            "SIM_SURFACE_SHRINK_K",
            "SIM_EXPERIENCE_K",
        )
    ],
]


@pytest.mark.parametrize("kind,name", _SIM_SOURCE_MUTATIONS)
def test_generate_similarity_artifacts_rebuilds_when_source_changes(
    monkeypatch, tmp_path, kind, name
):
    """A change to any allowlisted source/config fingerprint input forces a
    full rebuild even when every snapshot input is unchanged."""
    d = _deploy()
    profiles, lifetime = _sim_fixtures()
    data_hash = d._similarity_inputs_hash(profiles, lifetime)
    # Old staged state carries the current data hash but an older source
    # fingerprint, computed with a per-file stub so no repo file is touched.
    file_hashes = {path.name: "old" for path in d.SIMILARITY_SOURCE_FILES}
    monkeypatch.setattr(d, "_file_hash", lambda path: file_hashes[path.name])
    old_source = d._similarity_source_hash()

    if kind == "constant":
        monkeypatch.setattr(d, name, 11 if name == "SIM_BIO_PCA_DIM" else 0.99)
    else:
        file_hashes[name] = "changed"
    sim_index, _sim_meta, state, _ = _stage_sim_build(
        monkeypatch,
        tmp_path,
        d,
        profiles,
        lifetime,
        state_hash=data_hash,
        state_source_hash=old_source,
    )
    d.generate_similarity_artifacts()

    assert sim_index.read_bytes() == b"fresh-index"
    staged = json.loads(state.read_text())
    assert staged["inputs_hash"] == data_hash
    assert staged["source_hash"] == d._similarity_source_hash()


def test_similarity_source_hash_legacy_state_without_fingerprint_rebuilds(monkeypatch, tmp_path):
    """A pre-fingerprint state (inputs_hash only) never reuses staged assets."""
    d = _deploy()
    profiles, lifetime = _sim_fixtures()
    data_hash = d._similarity_inputs_hash(profiles, lifetime)
    sim_index, _sim_meta, state, builds = _stage_sim_build(
        monkeypatch,
        tmp_path,
        d,
        profiles,
        lifetime,
        state_hash=data_hash,
    )
    state.write_text(json.dumps({"inputs_hash": data_hash}))  # legacy shape, no source_hash

    d.generate_similarity_artifacts()

    assert builds == [1]
    assert sim_index.read_bytes() == b"fresh-index"
    assert json.loads(state.read_text())["source_hash"] == d._similarity_source_hash()
