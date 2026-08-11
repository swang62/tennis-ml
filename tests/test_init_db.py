from types import SimpleNamespace

import src.db.init_db as init_db


class _Cursor:
    def __init__(self):
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, statement):
        self.statements.append(statement)


def test_init_bootstraps_configured_database_from_template1(monkeypatch, tmp_path):
    """Bootstrap derives the target and SSL settings from DATABASE_URL."""
    init_sql = tmp_path / "init.sql"
    init_sql.write_text("CREATE SCHEMA bronze;")
    maintenance_cursor = _Cursor()
    target_cursor = _Cursor()
    maintenance_connection = SimpleNamespace(cursor=lambda: maintenance_cursor)
    target_connection = SimpleNamespace(cursor=lambda: target_cursor)
    connections = []

    def connect(conninfo, **_kwargs):
        connections.append(conninfo)
        return maintenance_connection

    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://postgres:password@localhost:6543/tennis?sslmode=require"
    )
    monkeypatch.setattr(init_db, "INIT_SQL", init_sql)
    monkeypatch.setattr(init_db.psycopg, "connect", connect)
    monkeypatch.setattr(init_db, "get_conn", lambda: target_connection)

    init_db.init()

    assert "dbname=template1" in connections[0]
    assert "sslmode=require" in connections[0]
    assert "CREATE DATABASE" in repr(maintenance_cursor.statements[0])
    assert target_cursor.statements == ["CREATE SCHEMA bronze;"]
