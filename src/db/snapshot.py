"""Build and atomically install a validated PostgreSQL-to-DuckDB training snapshot."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb

from src.constants import DATA_PROCESSED, build_database_url
from src.features.columns import FEATURE_COLS, SIMILARITY_COLS

# Training's single, atomically replaced local input.
SNAPSHOT_PATH = DATA_PROCESSED / "training_snapshot.duckdb"

# Training reads the gold feature/aggregate tables plus the bronze profile
# metadata (bio summaries, handedness) that similarity/embeddings consume.
SNAPSHOT_TABLES = (
    ("gold", "match_features"),
    ("gold", "player_profiles"),
    ("bronze", "player_profiles"),
)

# gold.match_features metadata columns preceding the 36 feature columns.
META_COLS = (
    "match_id",
    "match_date",
    "player_id",
    "opponent_id",
    "tournament",
    "round",
    "surface",
    "match_won",
)

# Snapshot order: metadata, model features, then similarity-only columns.
EXPECTED_FEATURE_ORDER = (*META_COLS, *FEATURE_COLS, *SIMILARITY_COLS)


class SnapshotError(RuntimeError):
    """Raised when a snapshot is missing, invalid, or could not be built."""


def _copy_tables(tmp_path: Path, pg_url: str) -> None:
    """Copy all snapshot tables in one transaction for a consistent source view."""
    con = duckdb.connect(str(tmp_path))
    try:
        con.execute(f"ATTACH '{pg_url}' AS pg (TYPE postgres)")
        con.execute("CREATE SCHEMA IF NOT EXISTS gold")
        con.execute("CREATE SCHEMA IF NOT EXISTS bronze")
        con.execute("BEGIN TRANSACTION")
        for schema, table in SNAPSHOT_TABLES:
            con.execute(
                f'CREATE TABLE "{schema}"."{table}" AS SELECT * FROM pg."{schema}"."{table}"'
            )
        con.execute("COMMIT")
    finally:
        con.close()


def validate_snapshot(path: Path) -> None:
    """Validate required tables, order, non-empty content, and unique keys."""
    con = duckdb.connect(str(path), read_only=True)
    try:
        tables = {
            (schema, table)
            for schema, table in con.execute(
                "SELECT schema_name, table_name FROM duckdb_tables()"
            ).fetchall()
        }
        if tables != set(SNAPSHOT_TABLES):
            raise SnapshotError(
                f"snapshot contains {sorted(tables)}; expected exactly {sorted(SNAPSHOT_TABLES)}"
            )

        columns = tuple(
            col[0] for col in con.execute("SELECT * FROM gold.match_features LIMIT 0").description
        )
        if columns != EXPECTED_FEATURE_ORDER:
            raise SnapshotError(
                f"gold.match_features columns do not match META_COLS + FEATURE_COLS "
                f"({len(EXPECTED_FEATURE_ORDER)} columns); got {len(columns)}"
            )

        for schema, table in SNAPSHOT_TABLES:
            count_row = con.execute(f'SELECT COUNT(*) FROM "{schema}"."{table}"').fetchone()
            assert count_row is not None
            if count_row[0] == 0:
                raise SnapshotError(f'snapshot table "{schema}"."{table}" is empty')

        for schema, table, key in (
            ("gold", "match_features", "match_id"),
            ("gold", "player_profiles", "player_id"),
            ("bronze", "player_profiles", "player_id"),
        ):
            dupes_row = con.execute(
                f'SELECT COUNT(*) FROM (SELECT "{key}" FROM "{schema}"."{table}" '
                f'GROUP BY "{key}" HAVING COUNT(*) > 1)'
            ).fetchone()
            assert dupes_row is not None
            if dupes_row[0]:
                raise SnapshotError(
                    f'snapshot table "{schema}"."{table}" has {dupes_row[0]} duplicate "{key}" rows'
                )

        # Contract: every FEATURE_COLS cell is non-null and finite. dbt already
        # enforces this in gold; the snapshot re-checks it so training can never
        # read NULL/NaN/Infinity model features (similarity columns excluded).
        col_checks = " OR ".join(
            f'"{c}" IS NULL OR isnan("{c}") OR isinf("{c}")' for c in FEATURE_COLS
        )
        bad_row = con.execute(
            f"SELECT COUNT(*) FROM gold.match_features WHERE {col_checks}"
        ).fetchone()
        assert bad_row is not None
        if bad_row[0]:
            raise SnapshotError(
                f"gold.match_features has {bad_row[0]} NULL or non-finite model feature cells"
            )
    finally:
        con.close()


def refresh_snapshot(path: Path = SNAPSHOT_PATH, pg_url: str | None = None) -> Path:
    """Build, validate, and atomically replace the training snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        _copy_tables(tmp, pg_url or build_database_url())
        validate_snapshot(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


if __name__ == "__main__":
    print(f"Training snapshot refreshed: {refresh_snapshot()}")
