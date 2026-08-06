"""Focused tests for the PostgreSQL operational client.

No live PostgreSQL server is required: `db_client._conn` is swapped for a
fake connection that mimics the minimal psycopg surface the client uses
(`cursor()`, `transaction()`, `execute`, `description`, `fetchall`). The fake
records every statement it receives so tests can assert that `%s` placeholders
are used, bound values travel as parameters (never interpolated), and
pg_duckdb is forced only inside the explicit analytical path.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

import src.db.client as db_client


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.description = None
        self._rows: list[tuple[object, ...]] = []

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        return False

    def execute(self, sql: str, params: object | None = None):
        # Record (sql, params, in-transaction) so tests can assert placeholder
        # use, parameter binding, and the transaction scope of SET LOCAL.
        self.conn.statements.append((sql, params, self.conn.in_tx))
        columns, rows = self.conn.results.get(sql, ([], []))
        self.description = [SimpleNamespace(name=name) for name in columns]
        self._rows = rows
        return self

    def fetchall(self) -> list[tuple[object, ...]]:
        return self._rows


class FakeTransaction:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        self.conn.tx_entered += 1
        self.conn.in_tx = True
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.conn.in_tx = False
        return False


class FakeConn:
    def __init__(self):
        self.results: dict[str, tuple[list[str], list[tuple[object, ...]]]] = {}
        self.statements: list[tuple[str, object | None, bool]] = []
        self.tx_entered = 0
        self.in_tx = False
        self.closed = False

    def cursor(self, row_factory=None):
        del row_factory  # fake accepts the psycopg signature but ignores it
        return FakeCursor(self)

    def transaction(self):
        return FakeTransaction(self)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_conn(monkeypatch):
    conn = FakeConn()
    conn.results["SELECT id, name FROM t ORDER BY id"] = (
        ["id", "name"],
        [(1, "Alice"), (2, "O'Brien")],
    )
    monkeypatch.setattr(db_client, "_conn", conn)
    return conn


# --- DataFrame conversion shapes ---


def test_to_dataframe_returns_expected_columns_and_rows(fake_conn):
    df = db_client.to_dataframe("SELECT id, name FROM t ORDER BY id")

    assert list(df.columns) == ["id", "name"]
    assert len(df) == 2
    assert df.iloc[0].to_dict() == {"id": 1, "name": "Alice"}
    assert df.iloc[1].to_dict() == {"id": 2, "name": "O'Brien"}
    assert fake_conn.statements[0][0] == "SELECT id, name FROM t ORDER BY id"


def test_execute_df_without_params(fake_conn):
    df = db_client.execute_df("SELECT id, name FROM t ORDER BY id")

    assert list(df.columns) == ["id", "name"]
    assert len(df) == 2
    assert fake_conn.statements[0][1] is None


def test_first_row_dict_returns_string_keys(fake_conn):
    df = db_client.execute_df("SELECT id, name FROM t ORDER BY id")

    row = db_client.first_row_dict(df)

    assert row == {"id": 1, "name": "Alice"}
    assert all(isinstance(key, str) for key in row)
    assert fake_conn.statements[0][1] is None


# --- %s placeholder binding (values never interpolated) ---


def test_execute_df_uses_placeholder_and_binds_params(fake_conn):
    sql = "SELECT id FROM t WHERE name = %s"
    db_client.execute_df(sql, ["O'Brien"])

    statement, params, _in_tx = fake_conn.statements[0]
    assert statement == sql  # SQL text untouched: the quote never enters it
    assert params == ["O'Brien"]  # value travels as a bound parameter


def test_execute_df_with_tuple_params(fake_conn):
    sql = "SELECT id FROM t WHERE name = %s AND id = %s"
    db_client.execute_df(sql, ("O'Brien", 2))

    statement, params, _in_tx = fake_conn.statements[0]
    assert statement == sql
    assert params == ("O'Brien", 2)


# --- Connection lifecycle & configuration guardrails ---


def test_get_conn_uses_shared_contract(monkeypatch):
    monkeypatch.setattr(db_client.constants, "DATABASE_URL", None)
    monkeypatch.setattr(db_client.constants, "POSTGRES_PASSWORD", "secret")
    monkeypatch.setattr(db_client.constants, "POSTGRES_USER", "postgres")
    monkeypatch.setattr(db_client.constants, "POSTGRES_DB", "tennis")
    monkeypatch.setattr(db_client.constants, "POSTGRES_HOST", "127.0.0.1")
    monkeypatch.setattr(db_client.constants, "POSTGRES_PORT", "6543")
    monkeypatch.setattr(db_client, "_conn", None)

    fake = FakeConn()
    monkeypatch.setattr(db_client.psycopg, "connect", lambda _url, **_kwargs: fake)
    conn = db_client.get_conn()

    assert conn is fake


def test_missing_config_fails_before_any_fallback(monkeypatch):
    monkeypatch.setattr(db_client.constants, "DATABASE_URL", None)
    monkeypatch.setattr(db_client.constants, "POSTGRES_PASSWORD", None)
    monkeypatch.setattr(db_client, "_conn", None)

    with pytest.raises(RuntimeError, match="missing PostgreSQL configuration"):
        db_client.get_conn()


def test_close_resets_connection(monkeypatch):
    conn = FakeConn()
    monkeypatch.setattr(db_client, "_conn", conn)

    db_client.close()

    assert conn.closed
    assert db_client._conn is None


# --- Analytical path: pg_duckdb forced only inside a transaction ---


def test_analytical_df_forces_pg_duckdb_inside_transaction(monkeypatch):
    conn = FakeConn()
    conn.results["SELECT count(*) FROM gold.match_features"] = (["cnt"], [(42,)])
    monkeypatch.setattr(db_client, "_conn", conn)

    df = db_client.analytical_df("SELECT count(*) FROM gold.match_features")

    assert df.iloc[0, 0] == 42
    assert conn.tx_entered == 1
    set_local, _params, in_tx = conn.statements[0]
    assert set_local == "SET LOCAL duckdb.force_execution = true"
    assert in_tx is True  # SET LOCAL is transaction-scoped
    numeric_set_local, _params, in_tx = conn.statements[1]
    assert numeric_set_local == "SET LOCAL duckdb.convert_unsupported_numeric_to_double = true"
    assert in_tx is True
    query, _params, _in_tx = conn.statements[2]
    assert query == "SELECT count(*) FROM gold.match_features"


def test_analytical_df_binds_params_inside_transaction(monkeypatch):
    conn = FakeConn()
    conn.results["SELECT surface FROM gold.match_features WHERE match_date > %s"] = (
        ["surface"],
        [("clay",)],
    )
    monkeypatch.setattr(db_client, "_conn", conn)

    df = db_client.analytical_df(
        "SELECT surface FROM gold.match_features WHERE match_date > %s",
        ["2026-01-01"],
    )

    assert df.iloc[0, 0] == "clay"
    _set_local, _params, _in_tx = conn.statements[0]
    _numeric_set_local, _params, _in_tx = conn.statements[1]
    query, params, in_tx = conn.statements[2]
    assert query == "SELECT surface FROM gold.match_features WHERE match_date > %s"
    assert params == ["2026-01-01"]
    assert in_tx is True


def test_normal_reads_never_force_pg_duckdb(fake_conn):
    db_client.execute_df("SELECT id, name FROM t ORDER BY id")
    db_client.to_dataframe("SELECT id, name FROM t ORDER BY id")

    assert fake_conn.tx_entered == 0
    assert not any("force_execution" in sql for sql, _p, _t in fake_conn.statements)
