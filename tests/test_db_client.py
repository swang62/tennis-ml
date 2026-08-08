"""PostgreSQL client tests using a recording psycopg-shaped fake."""

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
        # Record SQL and bound values for parameterization checks.
        self.conn.statements.append((sql, params))
        columns, rows = self.conn.results.get(sql, ([], []))
        self.description = [SimpleNamespace(name=name) for name in columns]
        self._rows = rows
        return self

    def fetchall(self) -> list[tuple[object, ...]]:
        return self._rows


class FakeConn:
    def __init__(self):
        self.results: dict[str, tuple[list[str], list[tuple[object, ...]]]] = {}
        self.statements: list[tuple[str, object | None]] = []
        self.closed = False

    def cursor(self, row_factory=None):
        del row_factory  # fake accepts the psycopg signature but ignores it
        return FakeCursor(self)

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

    statement, params = fake_conn.statements[0]
    assert statement == sql  # SQL text untouched: the quote never enters it
    assert params == ["O'Brien"]  # value travels as a bound parameter


def test_execute_df_with_tuple_params(fake_conn):
    sql = "SELECT id FROM t WHERE name = %s AND id = %s"
    db_client.execute_df(sql, ("O'Brien", 2))

    statement, params = fake_conn.statements[0]
    assert statement == sql
    assert params == ("O'Brien", 2)


# --- Connection lifecycle & configuration guardrails ---


def test_get_conn_uses_passwordless_local_database_url(monkeypatch):
    """Use the passwordless local DATABASE_URL verbatim."""
    monkeypatch.setattr(
        db_client.constants, "DATABASE_URL", "postgresql://steve@127.0.0.1:5432/postgres"
    )
    monkeypatch.setattr(db_client, "_conn", None)

    fake = FakeConn()
    urls = []
    monkeypatch.setattr(
        db_client.psycopg, "connect", lambda url, **_kwargs: urls.append(url) or fake
    )
    conn = db_client.get_conn()

    assert conn is fake
    assert urls == ["postgresql://steve@127.0.0.1:5432/postgres"]


def test_missing_config_fails_before_any_fallback(monkeypatch):
    monkeypatch.setattr(db_client.constants, "DATABASE_URL", None)
    monkeypatch.setattr(db_client, "_conn", None)

    with pytest.raises(RuntimeError, match="missing PostgreSQL configuration"):
        db_client.get_conn()


def test_get_conn_uses_password_bearing_database_url(monkeypatch):
    """Use the password-bearing Compose DATABASE_URL verbatim."""
    monkeypatch.setattr(
        db_client.constants,
        "DATABASE_URL",
        "postgresql://postgres:password@postgres:5432/tennis",
    )
    monkeypatch.setattr(db_client, "_conn", None)

    fake = FakeConn()
    urls = []
    monkeypatch.setattr(
        db_client.psycopg, "connect", lambda url, **_kwargs: urls.append(url) or fake
    )
    conn = db_client.get_conn()

    assert conn is fake
    assert urls == ["postgresql://postgres:password@postgres:5432/tennis"]


def test_close_resets_connection(monkeypatch):
    conn = FakeConn()
    monkeypatch.setattr(db_client, "_conn", conn)

    db_client.close()

    assert conn.closed
    assert db_client._conn is None
