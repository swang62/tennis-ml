"""Offline tests for the Quack companion server logic (infra/duckdb/server.py).

No Docker daemon and no live Quack are required: mirror_views is exercised
against a real (in-memory/file) DuckDB, and the env/config guardrails are
checked without starting a server.
"""

import importlib.util
import sys
from types import ModuleType

import duckdb
import pytest


def _load_server() -> ModuleType:
    """Import infra/duckdb/server.py as a module (it has no import side effects
    that need a running server)."""
    spec = importlib.util.spec_from_file_location(
        "quack_server_under_test", "infra/duckdb/server.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_mirror_views_creates_main_schema_views(tmp_path):
    """Each production table must be mirrored as a main-schema view."""
    server = _load_server()
    db = tmp_path / "m.duckdb"
    conn = duckdb.connect(str(db))
    conn.execute("CREATE SCHEMA gold")
    conn.execute("CREATE TABLE gold.match_features (id INTEGER)")
    conn.execute("INSERT INTO gold.match_features VALUES (42)")

    server.mirror_views(conn)

    # The mirror view resolves the bare table name against main.
    assert conn.execute("SELECT id FROM match_features").fetchone() == (42,)
    # And the physical (schema-qualified) table is still reachable.
    assert conn.execute("SELECT id FROM gold.match_features").fetchone() == (42,)
    conn.close()


def test_mirror_views_is_idempotent(tmp_path):
    server = _load_server()
    db = tmp_path / "m.duckdb"
    conn = duckdb.connect(str(db))
    conn.execute("CREATE SCHEMA silver")
    conn.execute("CREATE TABLE silver.player_matches (id INTEGER)")

    server.mirror_views(conn)
    server.mirror_views(conn)  # CREATE OR REPLACE VIEW must not raise

    assert conn.execute("SELECT COUNT(*) FROM player_matches").fetchone() == (0,)
    conn.close()


def test_missing_data_file_fails_fast(monkeypatch, tmp_path):
    server = _load_server()
    monkeypatch.setattr(server, "TOKEN", "secret-token-1234")
    monkeypatch.setattr(server, "DATA_FILE", tmp_path / "nope.duckdb")

    with pytest.raises(FileNotFoundError):
        server.main()


def test_short_token_fails_fast(monkeypatch, tmp_path):
    server = _load_server()
    monkeypatch.setattr(server, "TOKEN", "abc")
    monkeypatch.setattr(server, "DATA_FILE", tmp_path / "x.duckdb")

    with pytest.raises(RuntimeError, match="at least 4"):
        server.main()
