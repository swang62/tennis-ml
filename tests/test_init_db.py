import re
from contextlib import nullcontext
from types import SimpleNamespace

import src.db.migrate_db as migrate_db


def _schema_ddl() -> str:
    """The actual migration DDL this module applies (no DB involved)."""
    return migrate_db.SCHEMA_SQL.read_text()


class _Cursor:
    def __init__(self):
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, statement):
        self.statements.append(statement)


def test_migrate_bootstraps_configured_database_from_template1(monkeypatch, tmp_path):
    """Bootstrap derives the target and SSL settings from DATABASE_URL."""
    schema_sql = tmp_path / "schema.sql"
    schema_sql.write_text("CREATE SCHEMA bronze;")
    maintenance_cursor = _Cursor()
    target_cursor = _Cursor()
    maintenance_connection = SimpleNamespace(cursor=lambda: maintenance_cursor)
    target_connection = SimpleNamespace(cursor=lambda: target_cursor, transaction=lambda: _Cursor())
    connects = []

    def connect(conninfo, **_kwargs):
        connects.append((conninfo, _kwargs))
        return maintenance_connection

    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://postgres:password@localhost:6543/tennis?sslmode=require"
    )
    monkeypatch.setattr(migrate_db, "SCHEMA_SQL", schema_sql)
    monkeypatch.setattr(migrate_db.psycopg, "connect", connect)
    monkeypatch.setattr(migrate_db, "connection", lambda: nullcontext(target_connection))

    migrate_db.migrate()

    conninfo, kwargs = connects[0]
    assert "dbname=template1" in conninfo
    assert "sslmode=require" in conninfo
    assert kwargs == {"autocommit": True, "connect_timeout": migrate_db.CONNECT_TIMEOUT_S}
    assert "CREATE DATABASE" in repr(maintenance_cursor.statements[0])
    assert target_cursor.statements == ["CREATE SCHEMA bronze;"]


def test_migrate_prints_sanitized_progress_before_connecting(monkeypatch, capsys, tmp_path):
    """A host:port/db progress line precedes the first network call, with no
    URL, user, or password leaked."""
    schema_sql = tmp_path / "schema.sql"
    schema_sql.write_text("CREATE SCHEMA bronze;")
    maintenance_connection = SimpleNamespace(cursor=lambda: _Cursor())
    target_connection = SimpleNamespace(cursor=lambda: _Cursor(), transaction=lambda: _Cursor())
    output_at_connect = []

    def connect(_conninfo, **_kwargs):
        # Snapshot the output as it is when the first network call happens.
        output_at_connect.append(capsys.readouterr().out)
        return maintenance_connection

    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://user:secret@localhost:6543/tennis?sslmode=require"
    )
    monkeypatch.setattr(migrate_db, "SCHEMA_SQL", schema_sql)
    monkeypatch.setattr(migrate_db.psycopg, "connect", connect)
    monkeypatch.setattr(migrate_db, "connection", lambda: nullcontext(target_connection))

    migrate_db.migrate()

    assert output_at_connect[0] == "Connecting to localhost:6543/tennis...\n"
    out = capsys.readouterr().out
    assert "user" not in out
    assert "secret" not in out
    assert "postgresql://" not in out


# ── Schema access-path contracts (static, hermetic: read schema.sql / dbt config) ──


def test_init_sql_rankings_identity_pk_supports_idempotent_upsert():
    """bronze.rankings PK (ranking_date, player_id) is the exact conflict target
    of both rankings upserts (ingest_rankings seed, weekly scrape catch-up), so
    ON CONFLICT (ranking_date, player_id) always has an arbiter index."""
    ddl = _schema_ddl()
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
    ddl = _schema_ddl()
    assert re.search(
        r"CREATE INDEX IF NOT EXISTS idx_rankings_player_date\s+"
        r"ON bronze\.rankings \(player_id, ranking_date\)",
        ddl,
    ), "per-player (player_id, ranking_date) index missing"


def test_init_sql_latest_match_date_index():
    """A standalone date index supports the global MAX(match_date) footer query."""
    ddl = _schema_ddl()
    assert re.search(
        r"CREATE INDEX IF NOT EXISTS idx_match_events_date\s+"
        r"ON bronze\.match_events \(match_date\)",
        ddl,
    ), "standalone match_date index missing"


def test_init_sql_head_to_head_indexes_match_endpoint_ordering():
    """Both sides of the /head_to_head OR predicate have deterministic
    date/match-id ordered indexes, avoiding a full history sort."""
    ddl = _schema_ddl()
    for player_column in ("player1_id", "player2_id"):
        assert re.search(
            rf"CREATE INDEX IF NOT EXISTS idx_match_events_p[12]_date_match\s+"
            rf"ON bronze\.match_events \({player_column}, match_date DESC, match_id DESC\)",
            ddl,
        ), f"/head_to_head index missing for {player_column}"


def test_dbt_player_matches_index_serves_match_history_ordering():
    """/match_history filters by player then returns the newest match ids."""
    config = (migrate_db.ROOT / "dbt" / "dbt_project.yml").read_text()
    assert (
        "{{ ensure_index_sql(this, 'idx_player_matches_pid_date_match', "
        "'player_id, match_date DESC, match_id DESC') }}"
    ) in config


def test_init_sql_profile_ownership_join_key_is_primary_key():
    """bronze.player_profiles.player_id is the PK: every ownership join/read
    (serving bp.player_id join, inference WHERE player_id, enrichment UPDATE)
    probes the primary key."""
    ddl = _schema_ddl()
    assert re.search(
        r"CREATE TABLE IF NOT EXISTS bronze\.player_profiles \(.*?"
        r"player_id\s+VARCHAR PRIMARY KEY",
        ddl,
        re.S,
    ), "player_profiles PK on player_id missing"


def test_schema_sql_widens_match_counts_without_reset():
    ddl = _schema_ddl()
    assert "ALTER COLUMN player1_total_serve_points TYPE INTEGER" in ddl
    assert "DROP CONSTRAINT IF EXISTS match_events_check_integer_counts" in ddl
    assert "player1_total_serve_points >= 0" in ddl
    assert "SELECT pg_advisory_xact_lock(7910881)" in ddl


def test_dbt_gold_player_profiles_join_key_is_primary_key():
    """dbt_project.yml re-applies the gold.player_profiles PK on player_id
    after each materialization — the gold-side key of the serving ownership
    join (gp.player_id = bp.player_id)."""
    config = (migrate_db.ROOT / "dbt" / "dbt_project.yml").read_text()
    assert "{{ ensure_primary_key_sql(this, 'pk_player_profiles', 'player_id') }}" in config, (
        "gold.player_profiles post-hook PK on player_id missing"
    )
