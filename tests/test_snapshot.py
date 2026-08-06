"""Tests for the PostgreSQL -> DuckDB training snapshot.

The pure-validation and atomicity behavior runs entirely on locally built
DuckDB fixtures (no PostgreSQL required). The live copy path
(``refresh_snapshot`` against the configured PostgreSQL) is exercised only
when the ``postgres_ready`` session fixture is available and otherwise skips,
mirroring the rest of the suite.
"""

import os

import duckdb
import pytest

from src.db import snapshot, training
from src.db.snapshot import EXPECTED_FEATURE_ORDER, META_COLS, SNAPSHOT_TABLES
from src.features.columns import FEATURE_COLS

# The one expected match_features metadata column that is not in FEATURE_COLS.
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


def _write_valid_snapshot(
    path, *, extra_table: str | None = None, bad_columns: bool = False, empty: bool = False
) -> None:
    """Write a DuckDB file that passes (or breaks) validate_snapshot."""
    if bad_columns:
        columns = (*_META, FEATURE_COLS[1], FEATURE_COLS[0], *FEATURE_COLS[2:])  # wrong order
    else:
        columns = EXPECTED_FEATURE_ORDER
    con = duckdb.connect(str(path))
    try:
        con.execute("CREATE SCHEMA gold")
        col_sql = ", ".join(f'"{c}" INTEGER' for c in columns)
        con.execute(f"CREATE TABLE gold.match_features ({col_sql})")
        con.execute("CREATE TABLE gold.player_profiles (player_id VARCHAR PRIMARY KEY)")
        if extra_table:
            con.execute(f"CREATE TABLE gold.{extra_table} (x INTEGER)")
        if not empty:
            con.execute(
                "INSERT INTO gold.match_features VALUES (" + ", ".join(["1"] * len(columns)) + ")"
            )
        con.execute("INSERT INTO gold.player_profiles (player_id) VALUES ('p1')")
    finally:
        con.close()


def test_meta_cols_precede_feature_cols() -> None:
    """The snapshot contract is exactly the 8 metadata columns then 36 features."""
    assert META_COLS == _META
    assert len(META_COLS) == 8
    assert len(FEATURE_COLS) == 36
    assert len(EXPECTED_FEATURE_ORDER) == 44
    assert EXPECTED_FEATURE_ORDER[:8] == _META


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


def test_validate_rejects_duplicate_match_ids(tmp_path) -> None:
    p = tmp_path / "snap.duckdb"
    _write_valid_snapshot(p)
    con = duckdb.connect(str(p))
    extra = ", ".join(["1"] * len(EXPECTED_FEATURE_ORDER))  # same match_id as first row
    con.execute(f"INSERT INTO gold.match_features VALUES ({extra})")
    con.close()
    with pytest.raises(snapshot.SnapshotError, match='duplicate "match_id" rows'):
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


def test_refresh_snapshot_copies_exactly_two_gold_tables(postgres_ready, tmp_path) -> None:  # noqa: ARG001 — skip-gate fixture, unused in body
    """Live refresh: snapshot has exactly the two gold tables, exact column order."""
    p = tmp_path / "live_snap.duckdb"
    snapshot.refresh_snapshot(p)
    assert p.exists()

    con = duckdb.connect(str(p), read_only=True)
    try:
        tables = {
            (s, t)
            for s, t in con.execute(
                "SELECT schema_name, table_name FROM duckdb_tables()"
            ).fetchall()
        }
        # bronze, silver, and rolling_features must not leak into training data.
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
    assert training.to_dataframe("SELECT COUNT(*) AS n FROM gold.match_features").iloc[0, 0] == 1

    monkeypatch.setattr(training, "SNAPSHOT_PATH", tmp_path / "missing.duckdb")
    training.close()
    with pytest.raises(FileNotFoundError, match="db-snapshot"):
        training.connect()


def test_training_module_does_not_import_operational_client() -> None:
    """Notebooks must use the training helper, not the PostgreSQL client."""
    assert "client" not in vars(training)
    assert "psycopg" not in vars(training)
