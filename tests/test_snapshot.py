"""Tests for PostgreSQL-to-DuckDB training snapshots."""

import os
from collections.abc import Sequence

import duckdb
import pytest

from src.db import snapshot, training
from src.db.snapshot import EXPECTED_FEATURE_ORDER, SNAPSHOT_TABLES
from src.features.columns import FEATURE_COLS

# The metadata columns of a gold match_features row that are not in FEATURE_COLS.
_META = (
    "match_id",
    "match_date",
    "player_id",
    "opponent_id",
    "tournament",
    "round",
    "surface",
    "match_won",
)


def _feature_row(match_id, player_id, opponent_id, match_won, columns):
    """One full gold row in `columns` order; identity/label from args, all
    remaining columns 1 (non-null, finite)."""
    identity = {
        "match_id": match_id,
        "player_id": player_id,
        "opponent_id": opponent_id,
        "match_won": match_won,
    }
    return tuple(identity.get(c, 1) for c in columns)


def _reciprocal_rows(columns):
    """The two directional rows of one match: reciprocal ids, complementary labels."""
    return [
        _feature_row(1, 1, 2, 1, columns),  # player 1 beats player 2
        _feature_row(1, 2, 1, 0, columns),  # mirrored perspective
    ]


def _write_valid_snapshot(
    path,
    *,
    extra_table: str | None = None,
    bad_columns: bool = False,
    empty: bool = False,
    rows: Sequence[tuple[int, ...]] | None = None,
) -> None:
    """Write a DuckDB file that passes (or breaks) validate_snapshot."""
    if bad_columns:
        columns = (*_META, FEATURE_COLS[1], FEATURE_COLS[0], *FEATURE_COLS[2:])  # wrong order
    else:
        columns = EXPECTED_FEATURE_ORDER
    con = duckdb.connect(str(path))
    try:
        con.execute("CREATE SCHEMA gold")
        con.execute("CREATE SCHEMA bronze")
        col_sql = ", ".join(f'"{c}" INTEGER' for c in columns)
        con.execute(f"CREATE TABLE gold.match_features ({col_sql})")
        con.execute("CREATE TABLE gold.player_profiles (player_id VARCHAR PRIMARY KEY)")
        con.execute("CREATE TABLE bronze.player_profiles (player_id VARCHAR PRIMARY KEY)")
        if extra_table:
            con.execute(f"CREATE TABLE gold.{extra_table} (x INTEGER)")
        if not empty:
            placeholders = ", ".join(["?"] * len(columns))
            con.executemany(
                f"INSERT INTO gold.match_features VALUES ({placeholders})",
                rows if rows is not None else _reciprocal_rows(columns),
            )
        con.execute("INSERT INTO gold.player_profiles (player_id) VALUES ('p1')")
        con.execute("INSERT INTO bronze.player_profiles (player_id) VALUES ('p1')")
    finally:
        con.close()


def test_validate_accepts_valid_snapshot(tmp_path) -> None:
    p = tmp_path / "snap.duckdb"
    _write_valid_snapshot(p)
    snapshot.validate_snapshot(p)  # must not raise


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"extra_table": "rolling_features"}, "expected exactly"),
        ({"bad_columns": True}, "do not match"),
        ({"empty": True}, "is empty"),
    ],
)
def test_validate_rejects_malformed_snapshot(tmp_path, kw, message) -> None:
    p = tmp_path / "snap.duckdb"
    _write_valid_snapshot(p, **kw)
    with pytest.raises(snapshot.SnapshotError, match=message):
        snapshot.validate_snapshot(p)


def test_validate_rejects_not_exactly_two_rows_per_match(tmp_path) -> None:
    """A lone directional row per match_id fails the two-rows contract."""
    p = tmp_path / "snap.duckdb"
    _write_valid_snapshot(p, rows=[_feature_row(1, 1, 2, 1, EXPECTED_FEATURE_ORDER)])
    with pytest.raises(snapshot.SnapshotError, match="exactly 2"):
        snapshot.validate_snapshot(p)


def test_validate_rejects_three_rows_per_match(tmp_path) -> None:
    """More than two rows for one match_id also fails the two-rows contract."""
    p = tmp_path / "snap.duckdb"
    rows = [
        *_reciprocal_rows(EXPECTED_FEATURE_ORDER),
        _feature_row(1, 3, 1, 1, EXPECTED_FEATURE_ORDER),
    ]
    _write_valid_snapshot(p, rows=rows)
    with pytest.raises(snapshot.SnapshotError, match="exactly 2"):
        snapshot.validate_snapshot(p)


def test_validate_rejects_duplicate_directional_row(tmp_path) -> None:
    """Duplicate (match_id, player_id) is rejected even with two rows total."""
    p = tmp_path / "snap.duckdb"
    rows = [
        _feature_row(1, 1, 2, 1, EXPECTED_FEATURE_ORDER),
        _feature_row(1, 1, 3, 0, EXPECTED_FEATURE_ORDER),  # same player_id
    ]
    _write_valid_snapshot(p, rows=rows)
    with pytest.raises(snapshot.SnapshotError, match=r'duplicate "\(match_id, player_id\)"'):
        snapshot.validate_snapshot(p)


def test_validate_rejects_non_reciprocal_pairs(tmp_path) -> None:
    """Row A's opponent must be row B's player and vice versa."""
    p = tmp_path / "snap.duckdb"
    rows = [
        _feature_row(1, 1, 2, 1, EXPECTED_FEATURE_ORDER),
        _feature_row(1, 2, 3, 0, EXPECTED_FEATURE_ORDER),  # opponent not the other player
    ]
    _write_valid_snapshot(p, rows=rows)
    with pytest.raises(snapshot.SnapshotError, match="reciprocal"):
        snapshot.validate_snapshot(p)


def test_validate_rejects_non_complementary_labels(tmp_path) -> None:
    """Both orientations of a match cannot share the same label."""
    p = tmp_path / "snap.duckdb"
    rows = [
        _feature_row(1, 1, 2, 1, EXPECTED_FEATURE_ORDER),
        _feature_row(1, 2, 1, 1, EXPECTED_FEATURE_ORDER),  # same label
    ]
    _write_valid_snapshot(p, rows=rows)
    with pytest.raises(snapshot.SnapshotError, match="complementary"):
        snapshot.validate_snapshot(p)


def test_validate_rejects_null_model_features(tmp_path) -> None:
    """A NULL in any FEATURE_COLS cell fails the finalized contract (the exact
    regression this model-ready gold contract prevents)."""
    p = tmp_path / "snap.duckdb"
    _write_valid_snapshot(p)
    con = duckdb.connect(str(p))
    con.execute("UPDATE gold.match_features SET rank_diff = NULL")
    con.close()
    with pytest.raises(snapshot.SnapshotError, match="NULL or non-finite model feature"):
        snapshot.validate_snapshot(p)


def test_refresh_failure_preserves_previous_snapshot(tmp_path, monkeypatch) -> None:
    """A failed refresh leaves the previous snapshot byte-for-byte intact."""
    p = tmp_path / "snap.duckdb"
    _write_valid_snapshot(p)
    before = p.read_bytes()

    def boom(_tmp, _pg_url):
        raise ConnectionError("source unavailable")

    monkeypatch.setattr(snapshot, "_copy_tables", boom)
    with pytest.raises(ConnectionError, match="source unavailable"):
        snapshot.refresh_snapshot(p, pg_url="unused")

    assert p.read_bytes() == before  # untouched
    assert not list(tmp_path.glob(".*.tmp"))  # no leftover temp files


def test_refresh_failure_cleans_temp_when_validation_fails(tmp_path, monkeypatch) -> None:
    """An invalid copy is discarded, not installed."""
    p = tmp_path / "snap.duckdb"
    _write_valid_snapshot(p)
    before = p.read_bytes()

    def bad_copy(tmp: "os.PathLike[str]", _pg: str) -> None:
        # Write a snapshot with only one (non-profiles) table so validation fails.
        con = duckdb.connect(str(tmp))
        con.execute("CREATE SCHEMA gold")
        con.execute("CREATE TABLE gold.match_features (x INTEGER)")
        con.close()

    monkeypatch.setattr(snapshot, "_copy_tables", bad_copy)
    with pytest.raises(snapshot.SnapshotError):
        snapshot.refresh_snapshot(p, pg_url="unused")

    assert p.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_refresh_snapshot_copies_exactly_the_snapshot_tables(tmp_path, monkeypatch) -> None:
    """refresh_snapshot installs exactly the snapshot tables (two gold tables
    plus bronze profile metadata), exact column order, atomically."""
    p = tmp_path / "live_snap.duckdb"

    def fake_copy(tmp: "os.PathLike[str]", _pg_url: str) -> None:
        # Stand-in for the PostgreSQL copy: write a valid four-table snapshot.
        _write_valid_snapshot(tmp)

    monkeypatch.setattr(snapshot, "_copy_tables", fake_copy)
    snapshot.refresh_snapshot(p, pg_url="unused")
    assert p.exists()

    con = duckdb.connect(str(p), read_only=True)
    try:
        tables = {
            (s, t)
            for s, t in con.execute(
                "SELECT schema_name, table_name FROM duckdb_tables()"
            ).fetchall()
        }
        # silver and rolling_features must not leak into training data.
        assert tables == set(SNAPSHOT_TABLES)
        cols = tuple(
            c[0] for c in con.execute("SELECT * FROM gold.match_features LIMIT 0").description
        )
        assert cols == EXPECTED_FEATURE_ORDER
        count = con.execute("SELECT COUNT(*) FROM gold.match_features").fetchone()
        assert count is not None and count[0] > 0
    finally:
        con.close()
    assert not list(tmp_path.glob("*.tmp"))


def test_training_reads_snapshot_not_postgres(tmp_path, monkeypatch) -> None:
    """training.to_dataframe reads the local snapshot and fails fast if missing."""
    p = tmp_path / "snap.duckdb"
    _write_valid_snapshot(p)
    monkeypatch.setattr(training, "SNAPSHOT_PATH", p)
    training.close()
    assert training.to_dataframe("SELECT COUNT(*) AS n FROM gold.match_features").iloc[0, 0] == 2

    monkeypatch.setattr(training, "SNAPSHOT_PATH", tmp_path / "missing.duckdb")
    training.close()
    with pytest.raises(FileNotFoundError, match="snapshot"):
        training.connect()


def test_training_module_does_not_import_operational_client() -> None:
    """Notebooks must use the training helper, not the PostgreSQL client."""
    assert "client" not in vars(training)
    assert "psycopg" not in vars(training)


def test_redact_pg_url_masks_secrets_preserves_detail() -> None:
    """A log-safe source URL keeps username, host/port, path, and non-sensitive
    query parameters/fragments; passwords and sensitive parameter values become ****."""
    cases = [
        # The password in userinfo is masked; the username is retained.
        (
            "postgresql://alice:hunter2@db:5432/tennis",
            "postgresql://alice:****@db:5432/tennis",
        ),
        # Username-only userinfo and non-sensitive params survive.
        (
            "postgresql://alice@db:5432/tennis?sslmode=require",
            "postgresql://alice@db:5432/tennis?sslmode=require",
        ),
        # Sensitive query params (case-insensitive keys) are masked, others kept.
        (
            "postgresql://db:5432/tennis?token=xJ9&secret=abc&sslmode=require",
            "postgresql://db:5432/tennis?token=****&secret=****&sslmode=require",
        ),
        # Secret values hiding in a query-string-like fragment are masked too.
        (
            "postgresql://alice:hunter2@db:5432/tennis#access_token=xJ9",
            "postgresql://alice:****@db:5432/tennis#access_token=****",
        ),
        # A plain fragment is preserved.
        (
            "postgresql://alice:hunter2@db:5432/tennis?sslmode=require#view",
            "postgresql://alice:****@db:5432/tennis?sslmode=require#view",
        ),
        # The full acceptance example.
        (
            "postgresql://alice:hunter2@db:5432/tennis?sslmode=require&password=x#view",
            "postgresql://alice:****@db:5432/tennis?sslmode=require&password=****#view",
        ),
    ]
    for raw, expected in cases:
        redacted = snapshot._redact_pg_url(raw)
        assert redacted == expected
        assert "hunter2" not in redacted and "xJ9" not in redacted


def test_refresh_snapshot_logs_safe_diagnostics(tmp_path, monkeypatch, capsys) -> None:
    """A normal refresh logs the redacted source URL (secrets masked, useful
    detail retained), per-table counts, and the installed DuckDB file size /
    PRAGMA database_size."""
    p = tmp_path / "live_snap.duckdb"
    monkeypatch.setattr(snapshot, "_copy_tables", lambda tmp, _pg: _write_valid_snapshot(tmp))
    snapshot.refresh_snapshot(
        p,
        pg_url="postgresql://alice:hunter2@db:5432/tennis?sslmode=require&password=x#view",
    )

    out = capsys.readouterr().out
    assert "postgresql://alice:****@db:5432/tennis?sslmode=require&password=****#view" in out
    assert "hunter2" not in out and "password=x" not in out
    assert "copied gold.match_features: 2 rows" in out
    assert "copied gold.player_profiles: 1 rows" in out
    assert "copied bronze.player_profiles: 1 rows" in out
    assert "MB on disk" in out
    assert "database_size=" in out
    assert "total_blocks=" in out


def test_refresh_snapshot_counts_stay_useful_when_table_empty(
    tmp_path, monkeypatch, capsys
) -> None:
    """An empty copied table still logs a 0 count before validation rejects it."""
    p = tmp_path / "live_snap.duckdb"
    monkeypatch.setattr(
        snapshot,
        "_copy_tables",
        lambda tmp, _pg: _write_valid_snapshot(tmp, empty=True),
    )
    with pytest.raises(snapshot.SnapshotError, match="is empty"):
        snapshot.refresh_snapshot(p, pg_url="postgresql://db:5432/tennis")
    out = capsys.readouterr().out
    assert "copied gold.match_features: 0 rows" in out
