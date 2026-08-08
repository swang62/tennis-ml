"""Atomic PostgreSQL -> DuckDB training snapshot.

Builds the local training database that the notebooks and PlayerSimilarity
read: a DuckDB file containing exactly two tables — ``gold.match_features``
and ``gold.player_profiles`` — copied from PostgreSQL through DuckDB's
``postgres`` extension. The copy runs inside one DuckDB transaction, which the
scanner maps to a single PostgreSQL transaction, so both tables come from one
consistent source snapshot. The temp file is then validated read-only
(exactly two tables, exact META_COLS + FEATURE_COLS column order, non-empty,
no duplicate keys) and only after validation is it atomically swapped over the
previous snapshot. Any failure deletes the temp file and leaves the previous
snapshot untouched, so training either reads a fully validated snapshot or
fails loudly — it never silently uses stale data.

The operational PostgreSQL client (``src.db.client``) is deliberately not used
here: this module talks to PostgreSQL only through the DuckDB postgres
scanner, and training code reads the result via ``src.db.training``.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb

from src.constants import DATA_PROCESSED, build_database_url
from src.features.columns import FEATURE_COLS, SIMILARITY_COLS

# The one file the training pipeline is allowed to read. Git-ignored and
# atomically replaced on every refresh; no retention or archives.
SNAPSHOT_PATH = DATA_PROCESSED / "training_snapshot.duckdb"

# The only tables copied into the snapshot. Bronze, silver, and
# silver.rolling_features stay operational-only and never enter training data.
SNAPSHOT_TABLES = (
    ("gold", "match_features"),
    ("gold", "player_profiles"),
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

# The exact ordered contract every snapshot must match: the 8 metadata
# columns, the 36 FEATURE_COLS, then the appended similarity-analysis serve/
# return columns (which are NOT model features — see columns.py).
EXPECTED_FEATURE_ORDER = (*META_COLS, *FEATURE_COLS, *SIMILARITY_COLS)


class SnapshotError(RuntimeError):
    """Raised when a snapshot is missing, invalid, or could not be built."""


def _copy_tables(tmp_path: Path, pg_url: str) -> None:
    """Copy the two gold tables from PostgreSQL into a fresh DuckDB file.

    Both CREATE TABLE AS SELECT statements run inside one DuckDB transaction,
    which the postgres scanner executes as a single PostgreSQL transaction, so
    the two reads see one consistent snapshot even if the source tables change
    mid-copy.
    """
    con = duckdb.connect(str(tmp_path))
    try:
        con.execute(f"ATTACH '{pg_url}' AS pg (TYPE postgres)")
        con.execute("CREATE SCHEMA IF NOT EXISTS gold")
        con.execute("BEGIN TRANSACTION")
        for schema, table in SNAPSHOT_TABLES:
            con.execute(
                f'CREATE TABLE "{schema}"."{table}" AS SELECT * FROM pg."{schema}"."{table}"'
            )
        con.execute("COMMIT")
    finally:
        con.close()


def validate_snapshot(path: Path) -> None:
    """Open a snapshot read-only and assert its structure and content.

    Checks: exactly the two expected user tables, gold.match_features columns
    exactly META_COLS + FEATURE_COLS in order, both tables non-empty, and no
    duplicate match_id / player_id rows. Raises SnapshotError on any mismatch.
    """
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
    finally:
        con.close()


def refresh_snapshot(path: Path = SNAPSHOT_PATH, pg_url: str | None = None) -> Path:
    """Build, validate, and atomically install a fresh training snapshot.

    Writes to a temp file next to ``path``, validates it read-only, then swaps
    it over the previous snapshot with os.replace. On any failure the temp
    file is removed and the previous snapshot (if any) is left untouched.
    """
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
