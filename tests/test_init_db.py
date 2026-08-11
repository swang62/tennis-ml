import re
from types import SimpleNamespace

import src.db.init_db as init_db


def _init_ddl() -> str:
    """The actual bootstrap DDL this module applies (no DB involved)."""
    return init_db.INIT_SQL.read_text()


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


# ── Schema access-path contracts (static, hermetic: read init.sql / dbt config) ──


def test_init_sql_rankings_identity_pk_supports_idempotent_upsert():
    """bronze.rankings PK (ranking_date, player_id) is the exact conflict target
    of both rankings upserts (ingest_rankings seed, weekly scrape catch-up), so
    ON CONFLICT (ranking_date, player_id) always has an arbiter index."""
    ddl = _init_ddl()
    assert re.search(
        r"CREATE TABLE IF NOT EXISTS bronze\.rankings \(.*?"
        r"PRIMARY KEY \(ranking_date, player_id\)",
        ddl,
        re.S,
    ), "rankings identity PK missing — ON CONFLICT (ranking_date, player_id) would fail"


def test_init_sql_rankings_history_and_latest_rank_index():
    """idx_rankings_player_date (player_id, ranking_date) serves both named
    rank access paths: chronological history (WHERE player_id = %s ORDER BY
    ranking_date in /rank_history) and latest rank per player (gold
    player_profiles DISTINCT ON player_id, backward index scan)."""
    ddl = _init_ddl()
    assert re.search(
        r"CREATE INDEX IF NOT EXISTS idx_rankings_player_date\s+"
        r"ON bronze\.rankings \(player_id, ranking_date\)",
        ddl,
    ), "per-player (player_id, ranking_date) index missing"


def test_init_sql_profile_ownership_join_key_is_primary_key():
    """bronze.player_profiles.player_id is the PK: every ownership join/read
    (serving bp.player_id join, inference WHERE player_id, enrichment UPDATE)
    probes the primary key."""
    ddl = _init_ddl()
    assert re.search(
        r"CREATE TABLE IF NOT EXISTS bronze\.player_profiles \(.*?"
        r"player_id\s+VARCHAR PRIMARY KEY",
        ddl,
        re.S,
    ), "player_profiles PK on player_id missing"


def test_dbt_gold_player_profiles_join_key_is_primary_key():
    """dbt_project.yml re-applies the gold.player_profiles PK on player_id
    after each materialization — the gold-side key of the serving ownership
    join (gp.player_id = bp.player_id)."""
    config = (init_db.ROOT / "dbt" / "dbt_project.yml").read_text()
    assert "{{ ensure_primary_key_sql(this, 'pk_player_profiles', 'player_id') }}" in config, (
        "gold.player_profiles post-hook PK on player_id missing"
    )
