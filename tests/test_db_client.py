from pathlib import Path

import duckdb
import pytest

import src.db.client as db_client

CREATE_TABLE_SQL = "CREATE TABLE t (id INTEGER, name VARCHAR)"
INSERT_SQL = "INSERT INTO t VALUES (1, 'Alice'), (2, 'O''Brien')"


@pytest.fixture(scope="module")
def _in_memory_db():
    """Swap client._conn for an in-memory DuckDB; never touch the real DB file."""
    conn = duckdb.connect(":memory:")
    conn.execute(CREATE_TABLE_SQL)
    conn.execute(INSERT_SQL)
    db_client._conn = conn
    yield
    conn.close()
    db_client._conn = None


def test_to_dataframe_returns_expected_columns_and_rows(_in_memory_db):
    df = db_client.to_dataframe("SELECT id, name FROM t ORDER BY id")

    assert list(df.columns) == ["id", "name"]
    assert len(df) == 2
    assert df.iloc[0].to_dict() == {"id": 1, "name": "Alice"}
    assert df.iloc[1].to_dict() == {"id": 2, "name": "O'Brien"}


def test_execute_df_with_placeholder_params(_in_memory_db):
    df = db_client.execute_df("SELECT name FROM t WHERE id = ?", [1])

    assert list(df.columns) == ["name"]
    assert df.to_dict(orient="records") == [{"name": "Alice"}]


def test_execute_df_param_binding_handles_literal_quote(_in_memory_db):
    # If params were interpolated into the SQL, the quote in O'Brien would break it.
    df = db_client.execute_df("SELECT id FROM t WHERE name = ?", ["O'Brien"])

    assert df.to_dict(orient="records") == [{"id": 2}]


def test_execute_df_without_params(_in_memory_db):
    df = db_client.execute_df("SELECT id, name FROM t ORDER BY id")

    assert list(df.columns) == ["id", "name"]
    assert len(df) == 2


def test_first_row_dict_returns_string_keys(_in_memory_db):
    df = db_client.execute_df("SELECT id, name FROM t ORDER BY id")

    row = db_client.first_row_dict(df)

    assert row == {"id": 1, "name": "Alice"}
    assert all(isinstance(key, str) for key in row)


# --- Mode selection & production guardrails (no Docker/Quack needed) ---


def test_dev_mode_opens_local_embedded_db(monkeypatch, tmp_path):
    """dev mode must open a local file-backed DuckDB, never contact Quack."""
    monkeypatch.setattr(db_client.constants, "ENVIRONMENT", "dev")
    db = tmp_path / "dev.duckdb"
    monkeypatch.setattr(db_client.constants, "TENNIS_DB_PATH", str(db))
    monkeypatch.setattr(db_client, "_conn", None)

    conn = db_client._connect_dev()

    assert Path(db).exists()
    conn.close()


def test_invalid_environment_fails_fast(monkeypatch):
    """An unknown ENVIRONMENT (or a None connection) must raise, never fall back."""
    monkeypatch.setattr(db_client.constants, "ENVIRONMENT", "staging")
    monkeypatch.setattr(db_client, "_conn", None)

    import pytest

    with pytest.raises(RuntimeError, match="invalid ENVIRONMENT"):
        db_client.get_conn()


def test_production_missing_config_fails_fast(monkeypatch):
    """ENVIRONMENT=production without QUACK_URI/QUACK_TOKEN must raise."""
    monkeypatch.setattr(db_client.constants, "ENVIRONMENT", "production")
    monkeypatch.setattr(db_client.constants, "QUACK_URI", None)
    monkeypatch.setattr(db_client.constants, "QUACK_TOKEN", None)
    monkeypatch.setattr(db_client, "_conn", None)

    import pytest

    with pytest.raises(RuntimeError, match="QUACK_URI and QUACK_TOKEN"):
        db_client._connect_production()


def test_production_configures_attach_and_default_catalog(monkeypatch):
    """Production must ATTACH the remote with the token and USE it as default."""
    monkeypatch.setattr(db_client.constants, "ENVIRONMENT", "production")
    monkeypatch.setattr(db_client.constants, "QUACK_URI", "quack:quack-db:9494")
    monkeypatch.setattr(db_client.constants, "QUACK_TOKEN", "secret-token")
    monkeypatch.setattr(db_client.constants, "QUACK_CATALOG", "tennis")
    monkeypatch.setattr(db_client, "_conn", None)

    calls = []

    class FakeConn:
        def execute(self, sql, params=None):
            calls.append((sql, params))
            return self

        def sql(self, _sql):
            return self

        def fetchdf(self):
            import pandas as pd

            return pd.DataFrame()

    monkeypatch.setattr(duckdb, "connect", lambda _path: FakeConn())

    conn = db_client._connect_production()

    attach_sql = next(s for s, _ in calls if s.startswith("ATTACH"))
    assert "quack:quack-db:9494" in attach_sql
    # The token must travel as a bound parameter, never interpolated into SQL.
    assert "secret-token" not in attach_sql
    attach_params = next(p for s, p in calls if s.startswith("ATTACH"))
    assert attach_params == ["secret-token"]
    assert any(s == "USE tennis" for s, _ in calls)
    assert isinstance(conn, FakeConn)


# --- No-package production DB boundary ---


def test_bentofile_does_not_package_production_db():
    """The DB must not be baked into the Bento image."""
    import yaml

    from src.constants import ROOT

    config = yaml.safe_load((ROOT / "bentofile.yaml").read_text())
    assert "data/tennis.duckdb" not in config["include"]


def test_deploy_aux_files_do_not_include_production_db():
    """deploy.py must not fingerprint/package the production DB."""
    from src.flows import deploy

    assert not any("tennis.duckdb" in str(p) for p in deploy.AUX_FILES)
    assert not any("tennis.duckdb" in str(p) for p in deploy.FINGERPRINT_FILES)
