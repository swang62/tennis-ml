"""Shared test fixtures.

No live-database fixtures live here: the seeded_test_db / postgres_ready /
gold_ready fixtures were removed because the suite must be hermetic (see
AGENTS.md). Every test mocks the database boundary or uses an in-memory
fixture; the inference-builder suite keeps its own in-memory DuckDB stand-in
inside test_inference_features.py.
"""
