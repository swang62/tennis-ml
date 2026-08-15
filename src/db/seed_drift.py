"""Dev-only: insert deterministic cloned real matches for local drift testing.

Runs the drift monitor's minimum-data path (DRIFT_MIN_N_FOR_CHECK = 10 physical
matches, which expand into 20 symmetric scored orientations) against real
matches cloned from the local seed corpus. Fully opt-in: normal `just seed` /
`just seed --all` never produce these rows — only this command writes them, and
it never touches profiles, rankings, or any other table.

Usage::

    just drift-seed                     # insert 12 matches (idempotent, skips existing)
    just drift-seed --force             # overwrite the fixture's own rows
    just drift-seed --after 2026-09-01  # first match date (must be after the champion watermark)
    just drift-seed --dry-run           # build + validate rows, write nothing

Rows are built deterministically from the local seed corpus: 12 distinct real
matches evenly spread across data/raw/2026.csv pass through the shared
raw->bronze transform, so each row keeps its real players, tournament, round,
surface, indoor flag, rankings, ages, rolling form, and match statistics. Only
match_id and match_date are overridden: match_id gets the `drift-` prefix,
which the real ingest never produces, so the fixture is uniquely identifiable
and re-runs converge; each row moves to a consecutive day starting at the
anchor date, so the fixture lands after the champion's training-data cutoff.

Prerequisites
-------------
Run the normal local setup first (`just seed`, then `just etl`) so bronze has
pre-cutoff matches for the drift reference window and silver/gold exist for the
champion Bento to score the fixture's real players. After inserting, run
`just etl` again if you want the fixture rows folded into silver/gold; the drift
check itself reads bronze.match_events directly.

Removing the rows (safe, scoped to the fixture's own match_ids; this command
never deletes anything)::

    psql "$DATABASE_URL" -c "DELETE FROM bronze.match_events WHERE match_id LIKE 'drift-%';"

Avoiding them: simply don't run `just drift-seed`; re-running it is idempotent
(skips existing match_ids) unless --force is passed.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from typing import Any

import pandas as pd

from src.constants import DRIFT_MIN_N_FOR_CHECK, ROOT
from src.db.ingest import (
    BRONZE_MATCHES_TABLE,
    atp_rows_to_bronze,
    insert_bronze_rows,
    load_raw_atp_rows,
)
from src.features.columns import BRONZE_COLUMNS
from src.features.validate import run_ingestion_checks

RAW_YEAR = ROOT / "data" / "raw" / "2026.csv"

# 12 physical matches clears drift's >=10 minimum with margin.
N_MATCHES = 12

# Deterministic default anchor: after the current seed horizon (max tourney_date
# in data/raw/2026.csv is 2026-08-01), hence after any champion trained on this
# data. Override with --after when a champion watermark is later.
DEFAULT_AFTER = date(2026, 8, 15)

# match_id prefix: never produced by atp_rows_to_bronze, so a scoped DELETE
# removes exactly the fixture rows and nothing else.
MATCH_ID_PREFIX = "drift"


def _source_match_id(m: dict[str, Any]) -> str:
    """The bronze match_id atp_rows_to_bronze assigns to a raw row."""
    return f"{int(m['tourney_date'])}-{m['tourney_id']}-{int(m['match_num']):03d}"


def _select_source_matches(rows: list[dict[str, Any]], n: int) -> set[str]:
    """n distinct real source match ids, evenly spread across the corpus.

    rows must be sorted with the same key atp_rows_to_bronze uses so the chosen
    indices line up with the transform's iteration order.
    """
    step = len(rows) - 1
    return {_source_match_id(rows[i * step // (n - 1)]) for i in range(n)}


def build_fixture_rows(after: date = DEFAULT_AFTER) -> pd.DataFrame:
    """Deterministic bronze rows cloned from real 2026 matches.

    Twelve real matches evenly spread across the season pass through the shared
    raw->bronze transform (with full-corpus history, so the rolling-form
    columns are real), then only match_id and match_date are overridden: each
    row moves to `after + i days` and match_id gets the drift- prefix.
    """
    rows = sorted(
        load_raw_atp_rows(RAW_YEAR),
        key=lambda m: (int(m["tourney_date"]), m["tourney_id"], m["match_num"]),
    )
    df = atp_rows_to_bronze(rows, selected_ids=_select_source_matches(rows, N_MATCHES))
    dates = [after + timedelta(days=i) for i in range(len(df))]
    df["match_date"] = [d.isoformat() for d in dates]
    df["match_id"] = [f"{MATCH_ID_PREFIX}-{d:%Y%m%d}-{i:03d}" for i, d in enumerate(dates)]
    return df[list(BRONZE_COLUMNS)]


def validate_fixture(df: pd.DataFrame) -> None:
    """Run the shared bronze ingestion checks; fail loudly on any dropped row.

    The fixture must never silently insert a partial set: 12 complete rows are
    required to clear drift's minimum-data guard.
    """
    report = run_ingestion_checks(df)
    for issue in report["results"]:
        print(f"  DROP: {issue}")
    if report["dropped_rows"]:
        raise ValueError(
            f"drift fixture failed bronze validation: {report['dropped_rows']} rows dropped"
        )


def _seed_horizon() -> date:
    """Latest match date in the local seed corpus (deterministic guard)."""
    rows = load_raw_atp_rows(RAW_YEAR)
    latest = max(int(r["tourney_date"]) for r in rows)
    return date(latest // 10000, latest % 10000 // 100, latest % 100)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--after",
        type=date.fromisoformat,
        default=DEFAULT_AFTER,
        help="first cloned match date (default %(default)s); must be after the "
        "seed horizon and the champion's training-data watermark",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite the fixture's own existing rows (default: idempotent skip)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build and validate the fixture rows, but write nothing to the database",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.after <= _seed_horizon():
        raise ValueError(
            f"--after {args.after} is not after the seed horizon {_seed_horizon()}: "
            "cloned matches must be newer than the champion's training data "
            "so drift sees them in the current window"
        )
    df = build_fixture_rows(after=args.after)
    validate_fixture(df)

    players = sorted(set(df["player1_id"]) | set(df["player2_id"]))
    print(
        f"Drift fixture: {len(df)} real cloned bronze matches from "
        f"{df['match_date'].min()} to {df['match_date'].max()} "
        f"({len(players)} real players: {', '.join(players)})"
    )
    if len(df) < DRIFT_MIN_N_FOR_CHECK:
        raise ValueError(
            f"fixture has {len(df)} matches; drift requires at least "
            f"{DRIFT_MIN_N_FOR_CHECK} physical matches"
        )
    print(
        f"Drift minimum-data guard: {len(df)} physical matches >= "
        f"{DRIFT_MIN_N_FOR_CHECK}, expanding to {len(df) * 2} scored orientations."
    )

    if args.dry_run:
        print("Dry run: validated above; no rows written.")
        return 0

    inserted = insert_bronze_rows(df, overwrite=args.force)
    if args.force:
        print(f"Inserted {inserted} rows into {BRONZE_MATCHES_TABLE} (overwrite)")
    else:
        print(
            f"Inserted {inserted} rows into {BRONZE_MATCHES_TABLE} ({len(df) - inserted} skipped existing)"
        )
    print(
        "Next: run `just etl` to fold the fixture into silver/gold, then "
        "`just drift` against your champion (drift reads bronze directly)."
    )
    print(
        "To remove the fixture rows (scoped to their drift- match_ids; this "
        "command never deletes):\n"
        '  psql "$DATABASE_URL" -c "DELETE FROM bronze.match_events '
        "WHERE match_id LIKE 'drift-%';\""
    )
    return 0


if __name__ == "__main__":
    main()
