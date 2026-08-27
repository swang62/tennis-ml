"""Build and atomically install a validated PostgreSQL-to-DuckDB training snapshot."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

import duckdb

from src.constants import GOLD_MATCHES_TABLE, ROOT, get_database_url
from src.features.columns import FEATURE_COLS, SIMILARITY_COLS

SNAPSHOT_PATH = ROOT / "data" / "training_snapshot.duckdb"

SNAPSHOT_TABLES = (
    ("gold", "match_features"),
    ("gold", "player_profiles"),
    ("bronze", "player_profiles"),
    ("silver", "player_matches"),
)

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
        con.execute("CREATE SCHEMA IF NOT EXISTS silver")
        con.execute("BEGIN TRANSACTION")
        for schema, table in SNAPSHOT_TABLES:
            con.execute(
                f'CREATE TABLE "{schema}"."{table}" AS SELECT * FROM pg."{schema}"."{table}"'
            )
        con.execute("COMMIT")
    finally:
        con.close()


def validate_snapshot(path: Path) -> None:
    """Validate required tables, order, non-empty content, and the directional row contract."""
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
            col[0] for col in con.execute(f"SELECT * FROM {GOLD_MATCHES_TABLE} LIMIT 0").description
        )
        if columns != EXPECTED_FEATURE_ORDER:
            raise SnapshotError(
                f"{GOLD_MATCHES_TABLE} columns do not match META_COLS + FEATURE_COLS "
                f"({len(EXPECTED_FEATURE_ORDER)} columns); got {len(columns)}"
            )

        for schema, table in SNAPSHOT_TABLES:
            count_row = con.execute(f'SELECT COUNT(*) FROM "{schema}"."{table}"').fetchone()
            assert count_row is not None
            if count_row[0] == 0:
                raise SnapshotError(f'snapshot table "{schema}"."{table}" is empty')

        for schema, table, key in (
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

        # Each physical match must have two reciprocal, complementary rows.
        bad_groups = con.execute(
            f"SELECT COUNT(*) FROM ("
            f" SELECT match_id FROM {GOLD_MATCHES_TABLE}"
            f" GROUP BY match_id HAVING COUNT(*) != 2)"
        ).fetchone()
        assert bad_groups is not None
        if bad_groups[0]:
            raise SnapshotError(
                f"{GOLD_MATCHES_TABLE} has {bad_groups[0]} match_id groups with "
                "!= 2 rows; expected exactly 2 directional rows per match_id"
            )

        dup_keys = con.execute(
            f"SELECT COUNT(*) FROM ("
            f" SELECT match_id, player_id FROM {GOLD_MATCHES_TABLE}"
            f" GROUP BY match_id, player_id HAVING COUNT(*) > 1)"
        ).fetchone()
        assert dup_keys is not None
        if dup_keys[0]:
            raise SnapshotError(
                f'{GOLD_MATCHES_TABLE} has {dup_keys[0]} duplicate "(match_id, player_id)" rows'
            )

        bad_pairs = con.execute(
            f"SELECT COUNT(*) FROM ("
            f" SELECT a.match_id FROM {GOLD_MATCHES_TABLE} a"
            f" JOIN {GOLD_MATCHES_TABLE} b"
            f"   ON a.match_id = b.match_id AND a.player_id < b.player_id"
            f" WHERE NOT ("
            f"   a.opponent_id = b.player_id"
            f"   AND b.opponent_id = a.player_id"
            f"   AND a.match_won IN (0, 1) AND b.match_won IN (0, 1)"
            f"   AND a.match_won <> b.match_won))"
        ).fetchone()
        assert bad_pairs is not None
        if bad_pairs[0]:
            raise SnapshotError(
                f"{GOLD_MATCHES_TABLE} has {bad_pairs[0]} match_id groups whose two rows "
                "are not a valid reciprocal pair (mismatched opponent/player ids "
                "or non-complementary labels)"
            )

        # Recheck dbt's non-null, finite model-feature contract before training.
        col_checks = " OR ".join(
            f'"{c}" IS NULL OR isnan("{c}") OR isinf("{c}")' for c in FEATURE_COLS
        )
        bad_row = con.execute(
            f"SELECT COUNT(*) FROM {GOLD_MATCHES_TABLE} WHERE {col_checks}"
        ).fetchone()
        assert bad_row is not None
        if bad_row[0]:
            raise SnapshotError(
                f"{GOLD_MATCHES_TABLE} has {bad_row[0]} NULL or non-finite model feature cells"
            )

        # The model accepts only best_of values 1, 3, and 5.
        bad_best_of = con.execute(
            f"SELECT COUNT(*) FROM {GOLD_MATCHES_TABLE} WHERE best_of NOT IN (1, 3, 5)"
        ).fetchone()
        assert bad_best_of is not None
        if bad_best_of[0]:
            raise SnapshotError(
                f"{GOLD_MATCHES_TABLE} has {bad_best_of[0]} rows with best_of outside "
                "the 1/3/5 domain"
            )
    finally:
        con.close()


# Query and fragment keys whose values must be redacted in logs.
_SENSITIVE_PARAMS = frozenset(
    (
        "password",
        "passwd",
        "pwd",
        "secret",
        "token",
        "access_token",
        "api_key",
        "apikey",
        "api-key",
        "api_token",
        "apitoken",
        "api-token",
    )
)


def _mask_secret_values(raw: str) -> str:
    """Replace values of sensitive key=value parameters in a query string with ****."""
    masked = []
    for item in raw.split("&"):
        if not item:
            continue
        key, eq, _value = item.partition("=")
        if eq and unquote(key).lower() in _SENSITIVE_PARAMS:
            masked.append(f"{key}=****")
        else:
            masked.append(item)
    return "&".join(masked)


def _redact_pg_url(url: str) -> str:
    """Return *url* safe to log with passwords and sensitive values redacted."""
    parts = urlsplit(url)
    if "@" in parts.netloc:
        userinfo, host = parts.netloc.rsplit("@", 1)
        user, _, password = userinfo.partition(":")
        netloc = f"{user}:****@{host}" if password else f"{userinfo}@{host}"
    else:
        netloc = parts.netloc
    fragment = _mask_secret_values(parts.fragment) if "=" in parts.fragment else parts.fragment
    return urlunsplit(
        (parts.scheme, netloc, parts.path, _mask_secret_values(parts.query), fragment)
    )


def _row_counts(path: Path) -> dict[tuple[str, str], int]:
    """Rows per table in *path* (the snapshot tables once validation passes)."""
    con = duckdb.connect(str(path), read_only=True)
    try:
        counts: dict[tuple[str, str], int] = {}
        for schema, table in con.execute(
            "SELECT schema_name, table_name FROM duckdb_tables()"
        ).fetchall():
            count_row = con.execute(f'SELECT COUNT(*) FROM "{schema}"."{table}"').fetchone()
            assert count_row is not None
            counts[(str(schema), str(table))] = int(count_row[0])
        return counts
    finally:
        con.close()


def _database_size_summary(path: Path) -> str:
    """One-line PRAGMA database_size summary for the installed file."""
    con = duckdb.connect(str(path), read_only=True)
    try:
        pragma = con.execute("PRAGMA database_size")
        row = pragma.fetchone()
        assert row is not None
    finally:
        con.close()
    _, size, block_size, total_blocks, used_blocks, free_blocks, wal_size, _, _ = row
    return (
        f"database_size={size} block_size={block_size} total_blocks={total_blocks} "
        f"used_blocks={used_blocks} free_blocks={free_blocks} wal_size={wal_size}"
    )


def refresh_snapshot(path: Path = SNAPSHOT_PATH, pg_url: str | None = None) -> Path:
    """Build, validate, and atomically replace the training snapshot."""
    pg_url = pg_url or get_database_url()
    print(f"Snapshot source: {_redact_pg_url(pg_url)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        _copy_tables(tmp, pg_url)
        for (schema, table), count in _row_counts(tmp).items():
            print(f"  copied {schema}.{table}: {count} rows")
        validate_snapshot(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    size = path.stat().st_size
    print(f"Installed snapshot: {path} ({size / 1_000_000:.1f} MB on disk)")
    print(f"  {_database_size_summary(path)}")
    return path


if __name__ == "__main__":
    print(f"Training snapshot refreshed: {refresh_snapshot()}")
