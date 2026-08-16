"""Hermetic tests for the standalone training pipeline runner.

Only the player-similarity rebuild wiring is covered here (the notebook run
loop is out of scope). No live database, MLflow, or papermill execution.
"""

from pathlib import Path

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


def test_pipeline_source_has_no_navigation_build_or_mlflow_pins():
    source = pipeline.__file__
    assert source is not None
    text = Path(source).read_text()
    assert "build_similarity_index" not in text
    assert "similarity_pins.json" not in text
    assert "mlflow" not in text
    notebook = Path("notebooks/parameters/03_train_ensemble.ipynb").read_text()
    assert "similarity_pins.json" not in notebook
