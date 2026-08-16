"""Hermetic tests for the standalone training pipeline runner.

Only the player-similarity rebuild wiring is covered here (the notebook run
loop is out of scope). No live database, MLflow, or papermill execution.
"""

import json
import sys
from types import SimpleNamespace

from src.constants import (
    PLAYSTYLE_CLUSTER_LABELS,
    PLAYSTYLE_N_CLUSTERS,
    PLAYSTYLE_RANDOM_STATE,
)
from src.flows import pipeline
from src.models import similarity


def test_generate_cluster_artifacts_uses_snapshot_query_and_configured_config(monkeypatch):
    """The runner generates cluster artifacts from the fresh DuckDB snapshot
    with the configured count/labels/seed — never a live DB; the index build
    then consumes them."""

    class _FakeQuery:
        def __call__(self, _sql):
            return None

    fake_query = _FakeQuery()
    monkeypatch.setattr(pipeline.training, "to_dataframe", fake_query)
    captured: dict[str, object] = {}

    def _fake_build(n_clusters, labels, query=None, *, random_state=42):
        captured.update(
            n_clusters=n_clusters, labels=labels, query=query, random_state=random_state
        )

    monkeypatch.setattr(similarity, "build_cluster_artifacts", _fake_build)

    pipeline.generate_cluster_artifacts()

    assert captured["query"] is fake_query
    assert captured["n_clusters"] == PLAYSTYLE_N_CLUSTERS
    assert captured["labels"] == PLAYSTYLE_CLUSTER_LABELS
    assert captured["random_state"] == PLAYSTYLE_RANDOM_STATE


def test_build_similarity_index_rebuilds_with_snapshot_query(monkeypatch, tmp_path):
    """The runner rebuilds the index via PlayerSimilarity.build using the
    DuckDB snapshot query helper — it never loads a previously saved index —
    then logs the artifacts to MLflow and writes the similarity pins file."""
    captured: list[object] = []
    logged: list[str] = []

    class _FakeRun:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    fake_mlflow = SimpleNamespace(
        set_experiment=lambda _name: None,
        start_run=lambda: _FakeRun(),
        log_artifact=lambda path: logged.append(path),
        active_run=lambda: SimpleNamespace(info=SimpleNamespace(run_id="run-xyz")),
    )

    index_file = tmp_path / "player_similarity.index"
    metadata_file = tmp_path / "player_metadata.json"
    data_processed = tmp_path / "processed"

    class _FakePlayerSimilarity:
        def build(self, query=None) -> None:
            captured.append(query)
            index_file.write_bytes(b"index-bytes")
            metadata_file.write_bytes(b"metadata-bytes")

    class _FakeQuery:
        def __call__(self, _sql):
            return None

    fake_query = _FakeQuery()
    monkeypatch.setattr(pipeline.training, "to_dataframe", fake_query)
    monkeypatch.setattr(similarity, "PlayerSimilarity", _FakePlayerSimilarity)
    monkeypatch.setattr(similarity, "DEFAULT_INDEX", index_file)
    monkeypatch.setattr(similarity, "DEFAULT_METADATA", metadata_file)
    monkeypatch.setattr(pipeline, "DATA_PROCESSED", data_processed)
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    pipeline.build_similarity_index()

    assert captured == [fake_query]
    # Only the similarity index and metadata are logged/pinned; the web player
    # directory is generated at deploy time from the same snapshot.
    assert logged == [str(index_file), str(metadata_file)]
    assert not (data_processed / "player_directory.json").exists()

    pins = json.loads((data_processed / "similarity_pins.json").read_text())
    assert set(pins) == {
        "similarity_index_uri",
        "similarity_index_hash",
        "similarity_metadata_uri",
        "similarity_metadata_hash",
    }
    assert pins["similarity_index_uri"] == "runs:/run-xyz/player_similarity.index"
    assert pins["similarity_metadata_uri"] == "runs:/run-xyz/player_metadata.json"
    assert pins["similarity_index_hash"] == pipeline._file_hash(index_file)
    assert pins["similarity_metadata_hash"] == pipeline._file_hash(metadata_file)
