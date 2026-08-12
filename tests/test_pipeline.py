"""Hermetic tests for the standalone training pipeline runner.

Only the player-similarity rebuild wiring is covered here (the notebook run
loop is out of scope). No live database, MLflow, or papermill execution.
"""

from src.flows import pipeline
from src.models import similarity


def test_build_similarity_index_rebuilds_with_snapshot_query(monkeypatch):
    """The runner rebuilds the index via PlayerSimilarity.build using the
    DuckDB snapshot query helper — it never loads a previously saved index."""
    captured: list[object] = []

    class _FakePlayerSimilarity:
        def build(self, query=None) -> None:
            captured.append(query)

    fake_query = object()
    monkeypatch.setattr(pipeline.training, "to_dataframe", fake_query)
    monkeypatch.setattr(similarity, "PlayerSimilarity", _FakePlayerSimilarity)

    pipeline.build_similarity_index()

    assert captured == [fake_query]
