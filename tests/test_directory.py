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
import pytest

from src.constants import ROOT
from src.models.similarity import PLAYER_LIFETIME_SQL
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
    monkeypatch.setattr(d, "NAVIGATION_STATE_FILE", tmp_path / "nav_state.json")
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
        def __init__(self):
            self.players: list[dict[str, object]] = []

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
    # The full navigation input set is read from the snapshot: directory
    # profiles plus the gold lifetime stats the similarity vector consumes.
    assert calls == [PLAYERS_SQL, PLAYER_LIFETIME_SQL]
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


# ── Navigation artifact content-hash reuse ──────────────────────────────────


def _nav_fixtures():
    """One snapshot input set: profiles, gold lifetime stats, cluster artifacts."""
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
    assignments = pd.DataFrame({"player_id": ["p1", "p2"], "cluster_id": ["0", "1"]})
    labels = {"0": "Big Server", "1": "Counterpuncher"}
    return profiles, lifetime, assignments, labels


def _mutated(kind, profiles, lifetime, assignments, labels):
    """Return a copy of the fixtures with exactly one input changed."""
    if kind == "profiles":
        profiles = profiles.copy()
        profiles.loc[0, "display_name"] = "A Player v2"
    elif kind == "lifetime":
        lifetime = lifetime.copy()
        lifetime.loc[0, "first_serve_in_pct"] = np.float64(0.99)
    elif kind == "assignments":
        assignments = assignments.copy()
        assignments.loc[0, "cluster_id"] = "2"
    else:
        labels = dict(labels, **{"0": "Renamed Archetype"})
    return profiles, lifetime, assignments, labels


def _stage_nav_build(
    monkeypatch,
    tmp_path,
    d,
    profiles,
    lifetime,
    assignments,
    labels,
    *,
    state_hash,
    state_source_hash=None,
):
    """Run deploy's navigation generation against in-memory fixtures.

    Pre-stages stale outputs and the persisted ``state_hash`` (plus the
    optional source fingerprint), and records every rebuild in the returned
    (builds) list, so reuse decisions can be asserted.
    """
    out = tmp_path / "web" / "public" / "player-directory.json"
    sim_index = tmp_path / "player_similarity.index"
    sim_meta = tmp_path / "player_metadata.json"
    state = tmp_path / "nav_state.json"
    monkeypatch.setattr(d, "WEB_DIRECTORY_ARTIFACT", out)
    monkeypatch.setattr(d, "SIMILARITY_INDEX", sim_index)
    monkeypatch.setattr(d, "SIMILARITY_METADATA", sim_meta)
    monkeypatch.setattr(d, "NAVIGATION_STATE_FILE", state)
    if state_hash is not None:
        state.write_text(json.dumps({"inputs_hash": state_hash, "source_hash": state_source_hash}))
    for path, content in (
        (out, "stale-directory"),
        (sim_index, "stale-index"),
        (sim_meta, "stale-metadata"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    queries = {PLAYERS_SQL: profiles, PLAYER_LIFETIME_SQL: lifetime}
    monkeypatch.setattr("src.db.training.to_dataframe", lambda sql: queries[sql])
    monkeypatch.setattr(
        "src.models.similarity.load_cluster_artifacts", lambda: (assignments, labels)
    )

    builds = []

    class _RecordingSimilarity:
        def __init__(self):
            self.players: list[dict[str, object]] = []

        def build(self, **kwargs):
            builds.append(1)
            # Mirror production label derivation so the cross-check in
            # generate_navigation_artifacts sees agreeing labels.
            assignments = kwargs.get("cluster_assignments")
            if assignments is None:
                assignments = pd.DataFrame()
            labels = kwargs.get("cluster_labels")
            if labels is None:
                labels = {}
            self.players = [
                {
                    "player_id": str(row["player_id"]),
                    "cluster_label": labels.get(
                        str(row["cluster_id"]), f"cluster_{row['cluster_id']}"
                    ),
                }
                for _, row in assignments.iterrows()
            ]
            kwargs["index_path"].write_bytes(b"fresh-index")
            kwargs["metadata_path"].write_text(json.dumps(self.players))

    monkeypatch.setattr("src.models.similarity.PlayerSimilarity", _RecordingSimilarity)
    return out, sim_index, sim_meta, state, builds


def test_navigation_inputs_hash_deterministic_and_row_semantics():
    """The hash is stable across calls; lifetime rows sort canonically but the
    profile row order is preserved because it shapes the artifacts."""
    d = _deploy()
    profiles, lifetime, assignments, labels = _nav_fixtures()
    first = d._navigation_inputs_hash(profiles, lifetime, assignments, labels)
    assert first == d._navigation_inputs_hash(profiles, lifetime, assignments, labels)
    # Lifetime rows feed only a player-keyed merge: shuffled rows hash the same.
    shuffled = lifetime.iloc[::-1].reset_index(drop=True)
    assert first == d._navigation_inputs_hash(profiles, shuffled, assignments, labels)
    # Profile row order is part of the artifact contract (index/JSON row order).
    assert first != d._navigation_inputs_hash(
        profiles.iloc[::-1].reset_index(drop=True), lifetime, assignments, labels
    )


def test_generate_navigation_artifacts_reuses_unchanged_inputs(monkeypatch, tmp_path):
    """Unchanged inputs and unchanged shaping sources reuse the staged
    similarity artifacts and restage the raw directory — the web builder
    deletes it, so its presence is not required for reuse."""
    d = _deploy()
    profiles, lifetime, assignments, labels = _nav_fixtures()
    state_hash = d._navigation_inputs_hash(profiles, lifetime, assignments, labels)
    out, sim_index, sim_meta, _state, builds = _stage_nav_build(
        monkeypatch,
        tmp_path,
        d,
        profiles,
        lifetime,
        assignments,
        labels,
        state_hash=state_hash,
        state_source_hash=d._navigation_source_hash(),
    )

    assert d.generate_navigation_artifacts() == out
    assert builds == []  # the expensive FAISS/embedding build never ran
    assert sim_index.read_text() == "stale-index"  # staged similarity untouched
    assert sim_meta.read_text() == "stale-metadata"
    # The raw directory is deterministically restaged (was stale-directory).
    players = json.loads(out.read_text())["players"]
    assert [p["player_id"] for p in players] == ["p1", "p2"]
    assert players[0]["cluster_label"] == "Big Server"
    assert players[1]["cluster_label"] == "Counterpuncher"


def test_generate_navigation_artifacts_restages_directory_after_web_delete(monkeypatch, tmp_path):
    """build-player-index.mjs consumes (deletes) the raw directory after
    serializing it; an unchanged second generation must restage byte-identical
    raw without a similarity rebuild, so the builder's sourceHash cache
    reuses its hashed payloads."""
    d = _deploy()
    profiles, lifetime, assignments, labels = _nav_fixtures()
    out, sim_index, _sim_meta, state, builds = _stage_nav_build(
        monkeypatch, tmp_path, d, profiles, lifetime, assignments, labels, state_hash=None
    )

    # First generation: full build stages similarity + raw directory.
    assert d.generate_navigation_artifacts() == out
    assert builds == [1]
    first_raw = out.read_bytes()
    assert first_raw != b"stale-directory"

    # Simulate build-player-index.mjs consuming the raw directory.
    out.unlink()
    assert not out.exists()

    # Second generation with identical snapshot inputs: no similarity rebuild,
    # and the raw directory is restored byte-for-byte (MiniSearch sourceHash
    # reuse across dev reruns).
    assert d.generate_navigation_artifacts() == out
    assert builds == [1]  # no second similarity build
    assert out.read_bytes() == first_raw
    assert sim_index.read_bytes() == b"fresh-index"
    assert json.loads(state.read_text())["inputs_hash"] == d._navigation_inputs_hash(
        *_nav_fixtures()
    )


@pytest.mark.parametrize("mutate", ["profiles", "lifetime", "assignments", "labels"])
def test_generate_navigation_artifacts_rebuilds_when_dependency_changes(
    monkeypatch, tmp_path, mutate
):
    """A change to any one of the four hash inputs forces a full rebuild."""
    d = _deploy()
    profiles, lifetime, assignments, labels = _nav_fixtures()
    stale_hash = d._navigation_inputs_hash(profiles, lifetime, assignments, labels)
    changed = _mutated(mutate, profiles, lifetime, assignments, labels)
    out, sim_index, _sim_meta, state, builds = _stage_nav_build(
        monkeypatch, tmp_path, d, *changed, state_hash=stale_hash
    )

    d.generate_navigation_artifacts()

    assert builds == [1]
    assert sim_index.read_bytes() == b"fresh-index"
    assert out.read_text() != "stale-directory"
    # The new inputs hash is persisted so the next deploy can reuse.
    assert json.loads(state.read_text())["inputs_hash"] == d._navigation_inputs_hash(*changed)


def test_generate_navigation_artifacts_rebuilds_when_output_missing(monkeypatch, tmp_path):
    """A matching hash and source fingerprint but a deleted staged output still
    rebuilds it."""
    d = _deploy()
    profiles, lifetime, assignments, labels = _nav_fixtures()
    state_hash = d._navigation_inputs_hash(profiles, lifetime, assignments, labels)
    _out, sim_index, _sim_meta, _state, builds = _stage_nav_build(
        monkeypatch,
        tmp_path,
        d,
        profiles,
        lifetime,
        assignments,
        labels,
        state_hash=state_hash,
        state_source_hash=d._navigation_source_hash(),
    )
    sim_index.unlink()

    d.generate_navigation_artifacts()

    assert builds == [1]
    assert sim_index.read_bytes() == b"fresh-index"


@pytest.mark.parametrize("state_content", [None, "not json", "[]"])
def test_generate_navigation_artifacts_rebuilds_on_invalid_state(
    monkeypatch, tmp_path, state_content
):
    """Missing or corrupt persisted state never reuses stale artifacts."""
    d = _deploy()
    profiles, lifetime, assignments, labels = _nav_fixtures()
    out, sim_index, _sim_meta, state, builds = _stage_nav_build(
        monkeypatch, tmp_path, d, profiles, lifetime, assignments, labels, state_hash=None
    )
    if state_content is not None:
        state.write_text(state_content)

    d.generate_navigation_artifacts()

    assert builds == [1]
    assert sim_index.read_bytes() == b"fresh-index"
    assert out.read_text() != "stale-directory"


# ── Navigation source/config fingerprint ────────────────────────────────────


def test_navigation_source_hash_deterministic_over_real_sources():
    """The source fingerprint is stable across calls and every allowlisted
    source exists under the repo (a missing file would abort the deploy
    loudly rather than silently weaken the fingerprint)."""
    d = _deploy()
    assert d._navigation_source_hash() == d._navigation_source_hash()
    for path in d.NAVIGATION_SOURCE_FILES:
        assert path.is_file()
        assert path.resolve().is_relative_to(ROOT)


# Every allowlisted source/config input that can change the navigation
# outputs without changing snapshot data: the four shaping sources plus the
# exact values of the similarity-tuning constants.
_SOURCE_MUTATIONS = [
    *[("file", name) for name in ("deploy.py", "directory.py", "similarity.py", "countries.py")],
    *[
        ("constant", name)
        for name in (
            "SIM_IDENTITY_WEIGHT",
            "SIM_PLAYSTYLE_WEIGHT",
            "SIM_SURFACE_WEIGHT",
            "SIM_REPUTATION_WEIGHT",
            "SIM_BIO_WEIGHT",
            "SIM_CLUSTER_WEIGHT",
            "SIM_BIO_PCA_DIM",
            "SIM_SURFACE_SHRINK_K",
            "SIM_RANK_SCALE",
            "SIM_EXPERIENCE_K",
        )
    ],
]


@pytest.mark.parametrize("kind,name", _SOURCE_MUTATIONS)
def test_generate_navigation_artifacts_rebuilds_when_source_changes(
    monkeypatch, tmp_path, kind, name
):
    """A change to any allowlisted source/config fingerprint input forces a
    full rebuild even when every snapshot input is unchanged."""
    d = _deploy()
    profiles, lifetime, assignments, labels = _nav_fixtures()
    data_hash = d._navigation_inputs_hash(profiles, lifetime, assignments, labels)
    # Old staged state carries the current data hash but an older source
    # fingerprint, computed with a per-file stub so no repo file is touched.
    file_hashes = {path.name: "old" for path in d.NAVIGATION_SOURCE_FILES}
    monkeypatch.setattr(d, "_file_hash", lambda path: file_hashes[path.name])
    old_source = d._navigation_source_hash()

    if kind == "constant":
        monkeypatch.setattr(d, name, 11 if name == "SIM_BIO_PCA_DIM" else 0.99)
    else:
        file_hashes[name] = "changed"
    assert d._navigation_source_hash() != old_source  # the mutation is fingerprint-relevant

    out, sim_index, _sim_meta, state, builds = _stage_nav_build(
        monkeypatch,
        tmp_path,
        d,
        profiles,
        lifetime,
        assignments,
        labels,
        state_hash=data_hash,
        state_source_hash=old_source,
    )
    d.generate_navigation_artifacts()

    assert builds == [1]
    assert sim_index.read_bytes() == b"fresh-index"
    assert out.read_text() != "stale-directory"
    staged = json.loads(state.read_text())
    assert staged["inputs_hash"] == data_hash
    assert staged["source_hash"] == d._navigation_source_hash()


def test_navigation_source_hash_legacy_state_without_fingerprint_rebuilds(monkeypatch, tmp_path):
    """A pre-fingerprint state (inputs_hash only) never reuses staged assets."""
    d = _deploy()
    profiles, lifetime, assignments, labels = _nav_fixtures()
    data_hash = d._navigation_inputs_hash(profiles, lifetime, assignments, labels)
    _out, sim_index, _sim_meta, state, builds = _stage_nav_build(
        monkeypatch,
        tmp_path,
        d,
        profiles,
        lifetime,
        assignments,
        labels,
        state_hash=data_hash,
    )
    state.write_text(json.dumps({"inputs_hash": data_hash}))  # legacy shape, no source_hash

    d.generate_navigation_artifacts()

    assert builds == [1]
    assert sim_index.read_bytes() == b"fresh-index"
    assert json.loads(state.read_text())["source_hash"] == d._navigation_source_hash()
