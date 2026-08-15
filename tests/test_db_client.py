"""PostgreSQL client tests using a recording psycopg-pool-shaped fake.

The pool is faked at the module boundary (`db_client.ConnectionPool`), so no
live PostgreSQL server is ever contacted: checkout/return, health checking,
bounded concurrency, and broken-connection discard are all exercised against
the fake's documented psycopg-pool contract.
"""

import threading
from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast

import pandas as pd
import psycopg
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
        self.checks = 0
        self.closed = False
        self.broken = False

    def cursor(self, row_factory=None):
        del row_factory  # fake accepts the psycopg signature but ignores it
        return FakeCursor(self)

    def transaction(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        return False

    def close(self):
        self.closed = True


class BrokenCursor(FakeCursor):
    def __init__(self, conn, error):
        super().__init__(conn)
        self.error = error

    def execute(self, sql, params=None):
        # Record the attempt, then fail like a severed connection would:
        # a connection-level error marks the connection broken.
        self.conn.statements.append((sql, params))
        self.conn.broken = True
        raise self.error


class BrokenConn(FakeConn):
    def __init__(self, error):
        super().__init__()
        self.error = error

    def cursor(self, row_factory=None):
        del row_factory
        return BrokenCursor(self, self.error)


class FakePool:
    """ConnectionPool-shaped fake with a bounded free list.

    Mirrors psycopg-pool's observable contract: connections are health-checked
    at checkout, broken connections are discarded instead of reused, at most
    ``max_size`` connections are in use concurrently, and ``close()`` closes
    the pool.
    """

    def __init__(
        self,
        conninfo: str,
        *,
        min_size: int = 1,
        max_size: int = 2,
        kwargs: dict[str, object] | None = None,
        check=None,
        name: str | None = None,
        **_ignored,
    ):
        self.conninfo = conninfo
        self.min_size = min_size
        self.max_size = max_size
        self.kwargs = kwargs or {}
        self.check = check
        self.name = name
        self.waited = False
        self.closed = False
        self._free: list[FakeConn] = [FakeConn() for _ in range(max_size)]
        self._in_use: set[FakeConn] = set()
        self._cv = threading.Condition()

    @staticmethod
    def check_connection(conn: FakeConn) -> None:
        """Health probe mirroring psycopg_pool.ConnectionPool.check_connection."""
        conn.checks += 1
        if conn.broken:
            raise psycopg.OperationalError("connection is broken")

    def wait(self, timeout: float | None = None) -> None:
        del timeout  # fake: the pool is always ready
        self.waited = True

    def close(self) -> None:
        with self._cv:
            self.closed = True
            for conn in self._free:
                conn.close()
            self._cv.notify_all()

    @contextmanager
    def connection(self, timeout: float | None = None):
        del timeout  # fake: connections are always available within bounds
        conn = self._acquire()
        try:
            yield conn
        finally:
            self._release(conn)

    def _acquire(self) -> FakeConn:
        with self._cv:
            while not self.closed:
                # Discard broken connections instead of handing them out.
                for conn in list(self._free):
                    if conn.broken:
                        self._free.remove(conn)
                        conn.close()
                if not self._free:
                    self._free.append(FakeConn())  # replacement, as AddConnection would
                for conn in self._free:
                    if conn not in self._in_use:
                        if self.check is not None:
                            try:
                                self.check(conn)
                            except psycopg.Error:
                                self._free.remove(conn)
                                conn.close()
                                continue
                        self._in_use.add(conn)
                        return conn
                self._cv.wait(0.05)
            raise RuntimeError("pool is closed")

    def _release(self, conn: FakeConn) -> None:
        with self._cv:
            self._in_use.discard(conn)
            self._cv.notify_all()


@pytest.fixture
def fake_pool(monkeypatch):
    pool = FakePool("postgresql://test@localhost:5432/test", check=FakePool.check_connection)
    monkeypatch.setattr(db_client, "_pool", pool)
    return pool


# --- DataFrame conversion shapes ---


def test_to_dataframe_returns_expected_columns_and_rows(fake_pool):
    fake_pool._free[0].results["SELECT id, name FROM t ORDER BY id"] = (
        ["id", "name"],
        [(1, "Alice"), (2, "O'Brien")],
    )
    df = db_client.to_dataframe("SELECT id, name FROM t ORDER BY id")

    assert list(df.columns) == ["id", "name"]
    assert len(df) == 2
    assert df.iloc[0].to_dict() == {"id": 1, "name": "Alice"}
    assert df.iloc[1].to_dict() == {"id": 2, "name": "O'Brien"}
    assert fake_pool._free[0].statements[0][0] == "SELECT id, name FROM t ORDER BY id"


def test_execute_df_without_params(fake_pool):
    fake_pool._free[0].results["SELECT id, name FROM t ORDER BY id"] = (
        ["id", "name"],
        [(1, "Alice"), (2, "O'Brien")],
    )
    df = db_client.execute_df("SELECT id, name FROM t ORDER BY id")

    assert list(df.columns) == ["id", "name"]
    assert len(df) == 2
    assert fake_pool._free[0].statements[0][1] is None


def test_first_row_dict_returns_string_keys(fake_pool):
    fake_pool._free[0].results["SELECT id, name FROM t ORDER BY id"] = (
        ["id", "name"],
        [(1, "Alice"), (2, "O'Brien")],
    )
    df = db_client.execute_df("SELECT id, name FROM t ORDER BY id")

    row = db_client.first_row_dict(df)

    assert row == {"id": 1, "name": "Alice"}
    assert all(isinstance(key, str) for key in row)
    assert fake_pool._free[0].statements[0][1] is None


# --- %s placeholder binding (values never interpolated) ---


def test_execute_df_uses_placeholder_and_binds_params(fake_pool):
    sql = "SELECT id FROM t WHERE name = %s"
    db_client.execute_df(sql, ["O'Brien"])

    statement, params = fake_pool._free[0].statements[0]
    assert statement == sql  # SQL text untouched: the quote never enters it
    assert params == ["O'Brien"]  # value travels as a bound parameter


def test_execute_df_with_tuple_params(fake_pool):
    sql = "SELECT id FROM t WHERE name = %s AND id = %s"
    db_client.execute_df(sql, ("O'Brien", 2))

    statement, params = fake_pool._free[0].statements[0]
    assert statement == sql
    assert params == ("O'Brien", 2)


# --- Pool configuration & lifecycle guardrails ---


def test_get_pool_uses_passwordless_local_database_url(monkeypatch):
    """Use the passwordless local DATABASE_URL verbatim for the pool."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://steve@127.0.0.1:5432/postgres")
    monkeypatch.setattr(db_client, "_pool", None)
    monkeypatch.setattr(db_client, "ConnectionPool", FakePool)

    pool = cast(FakePool, db_client.get_pool())

    assert pool.conninfo == "postgresql://steve@127.0.0.1:5432/postgres"
    assert pool.min_size == 1
    assert pool.max_size == 2
    assert pool.kwargs == {"autocommit": True}
    assert pool.check is FakePool.check_connection  # health check configured
    assert pool.waited  # wait() called: first use surfaces an unreachable DB


def test_get_pool_uses_password_bearing_database_url(monkeypatch):
    """Use the password-bearing Compose DATABASE_URL verbatim for the pool."""
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://postgres:password@postgres:5432/tennis",
    )
    monkeypatch.setattr(db_client, "_pool", None)
    monkeypatch.setattr(db_client, "ConnectionPool", FakePool)

    pool = cast(FakePool, db_client.get_pool())

    assert pool.conninfo == "postgresql://postgres:password@postgres:5432/tennis"


def test_missing_config_fails_before_any_pool_creation(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(db_client, "_pool", None)
    created = []
    monkeypatch.setattr(
        db_client,
        "ConnectionPool",
        lambda *args, **kwargs: created.append(args) or FakePool(*args, **kwargs),
    )

    with pytest.raises(RuntimeError, match="DATABASE_URL not set"):
        db_client.get_pool()

    assert created == []  # fail fast: no pool is ever constructed


def test_get_pool_is_a_process_local_singleton(fake_pool):
    assert db_client.get_pool() is fake_pool
    assert db_client.get_pool() is fake_pool  # cached, not re-created


def test_close_closes_and_resets_the_pool(fake_pool):
    db_client.close()

    assert fake_pool.closed
    assert db_client._pool is None


# --- Checkout/return discipline ---


def test_execute_df_checks_out_and_returns_one_connection(fake_pool):
    fake_pool._free[0].results["SELECT 1"] = (["ok"], [(1,)])

    assert len(fake_pool._in_use) == 0
    df = db_client.execute_df("SELECT 1")
    assert df.iloc[0].to_dict() == {"ok": 1}
    assert len(fake_pool._in_use) == 0  # connection returned after the call
    assert fake_pool._free[0].checks == 1  # health probe ran at checkout


def test_transaction_holds_one_checked_out_connection(fake_pool):
    fake_pool._free[0].results["SELECT 1"] = (["ok"], [(1,)])

    with db_client.transaction() as cur:
        assert len(fake_pool._in_use) == 1  # one connection for the whole context
        cur.execute("SELECT 1")

    assert len(fake_pool._in_use) == 0  # returned on exit


def test_concurrent_callers_obtain_separate_connections(fake_pool):
    """Concurrent callers each check out their own pooled connection.

    All threads hold their connection simultaneously (barrier), so a shared
    serial connection would surface as a single distinct id.
    """
    assert db_client.get_pool() is fake_pool
    barrier = threading.Barrier(db_client.MAX_POOL_SIZE)
    observed: list[int] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            with db_client.connection() as conn:
                barrier.wait(timeout=5)
                observed.append(id(conn))
        except BaseException as exc:  # test thread reporter
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(db_client.MAX_POOL_SIZE)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert len(observed) == db_client.MAX_POOL_SIZE
    assert len(set(observed)) == db_client.MAX_POOL_SIZE  # distinct connections


# --- Broken connection resilience (regression: severed/cached conn) ---


def test_execute_df_skips_broken_pooled_connection(fake_pool):
    """A connection that died while idle is discarded at checkout, not reused."""
    stale = FakeConn()
    stale.broken = True  # server closed it while it sat in the pool
    fresh = FakeConn()
    fresh.results["SELECT 1"] = (["ok"], [(1,)])
    fake_pool._free = [stale, fresh]

    df = db_client.execute_df("SELECT 1")

    assert df.iloc[0].to_dict() == {"ok": 1}
    assert stale.closed  # discarded, never handed out
    assert stale.statements == []  # the statement was never replayed on it
    assert fresh.statements == [("SELECT 1", None)]


@pytest.mark.parametrize(
    "error",
    [
        psycopg.OperationalError("connection lost"),
        psycopg.InterfaceError("connection is closed"),
    ],
)
def test_execute_df_discards_broken_connection_then_reconnects(fake_pool, error):
    """A mid-query connection failure discards the connection and re-raises.

    The next call checks out a replacement; the failed statement is never
    replayed automatically.
    """
    broken = BrokenConn(error)
    fresh = FakeConn()
    fresh.results["SELECT 1"] = (["ok"], [(1,)])
    fake_pool._free = [broken, fresh]

    with pytest.raises(type(error)):
        db_client.execute_df("SELECT 1")

    assert broken.broken  # connection-level failure marks it broken
    assert broken.statements == [("SELECT 1", None)]  # attempted exactly once

    # Next request obtains a fresh connection; the failed statement is not replayed.
    df = db_client.execute_df("SELECT 1")
    assert df.iloc[0].to_dict() == {"ok": 1}
    assert broken.closed  # the broken connection was discarded
    assert fresh.statements == [("SELECT 1", None)]
