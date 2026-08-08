"""Deterministically seed PostgreSQL from ATP CSVs without network enrichment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, LiteralString, cast

from src.constants import ROOT
from src.db.client import get_conn
from src.flows.ingest import (
    BRONZE_TABLE,
    atp_rows_to_bronze,
    insert_bronze_rows,
    load_profiles_for,
    load_raw_atp_rows,
    player_history,
)

RAW_DIR = ROOT / "data" / "raw"
RAW_YEAR = RAW_DIR / "2026.csv"

TOP_PLAYERS = 10
RECENT = 10


def discover_atp_csvs(raw_dir: Path = RAW_DIR) -> list[Path]:
    """Return sorted regular-tour ATP CSVs, excluding Challenger files."""
    return sorted(p for p in raw_dir.glob("*.csv") if p.is_file() and "_challenger" not in p.name)


def load_all_raw_atp_rows(csv_paths: list[Path]) -> list[dict[str, Any]]:
    """Load and chronologically sort ATP rows across CSV boundaries."""
    rows = [row for path in csv_paths for row in load_raw_atp_rows(path)]
    return sorted(rows, key=lambda m: (int(m["tourney_date"]), m["tourney_id"], m["match_num"]))


def select_matches(
    matches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select recent distinct matches for the best-ranked players."""
    ranks: dict[str, int] = {}
    for m in matches:
        ranks[m["winner_id"]] = m["winner_rank"]
        ranks[m["loser_id"]] = m["loser_rank"]
    top = sorted(ranks, key=lambda pid: (ranks[pid], pid))[:TOP_PLAYERS]

    history = player_history(matches)
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    for pid in top:
        for m in history[pid][-RECENT:]:
            selected[m["tourney_id"], m["match_num"]] = m
    return sorted(
        selected.values(),
        key=lambda m: (int(m["tourney_date"]), m["tourney_id"], m["match_num"]),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="seed every ATP match CSV under data/raw/ (not just the default miniset)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.all:
        main_all()
    else:
        main_default()


def main_default() -> None:
    matches = sorted(
        load_raw_atp_rows(RAW_YEAR),
        key=lambda m: (int(m["tourney_date"]), m["tourney_id"], m["match_num"]),
    )
    selected = select_matches(matches)
    selected_ids = {
        f"{int(m['tourney_date'])}-{m['tourney_id']}-{int(m['match_num']):03d}" for m in selected
    }

    bronze = atp_rows_to_bronze(matches, selected_ids=selected_ids)
    distinct_players = len(set(bronze["player1_id"]) | set(bronze["player2_id"]))
    print(
        f"Seeding {len(bronze)} bronze matches from {RAW_YEAR.name} "
        f"({bronze['match_date'].min()} .. {bronze['match_date'].max()}), "
        f"{distinct_players} players"
    )

    conn = get_conn()
    # Re-seeding replaces the seed's own rows so `just db-seed` stays idempotent.
    conn.execute(
        cast(
            LiteralString,
            f"DELETE FROM {BRONZE_TABLE} WHERE match_id IN ({', '.join(['%s'] * len(selected_ids))})",
        ),
        list(selected_ids),
    )
    insert_bronze_rows(bronze)
    print(f"Inserted {len(bronze)} rows into {BRONZE_TABLE}")

    player_ids = sorted(set(bronze["player1_id"]) | set(bronze["player2_id"]))
    load_profiles_for(player_ids, "seeded")


def main_all() -> None:
    """Seed every ATP CSV idempotently without rewriting sources or enriching bios."""
    csv_paths = discover_atp_csvs(RAW_DIR)
    if not csv_paths:
        print(f"No ATP CSVs found under {RAW_DIR}; nothing to seed")
        return
    matches = load_all_raw_atp_rows(csv_paths)
    bronze = atp_rows_to_bronze(matches)
    distinct_players = len(set(bronze["player1_id"]) | set(bronze["player2_id"]))
    print(
        f"Seeding ALL: {len(bronze)} bronze matches from {len(csv_paths)} CSVs "
        f"({bronze['match_date'].min()} .. {bronze['match_date'].max()}), "
        f"{distinct_players} players"
    )

    insert_bronze_rows(bronze)
    print(f"Inserted {len(bronze)} rows into {BRONZE_TABLE} (upsert, idempotent)")

    player_ids = sorted(set(bronze["player1_id"]) | set(bronze["player2_id"]))
    load_profiles_for(player_ids, "seeded")


if __name__ == "__main__":
    main()
