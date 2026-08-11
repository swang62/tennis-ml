"""Deterministically seed PostgreSQL from ATP CSVs.

Local match/profile data and official rank history always seed offline;
Wikipedia bios are fetched only with the explicit --enrich flag.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.constants import ROOT
from src.db.ingest import (
    BRONZE_TABLE,
    atp_rows_to_bronze,
    enrich_players,
    ingest_rankings,
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
    parser.add_argument(
        "--enrich",
        action="store_true",
        help="enrich the selected players' profiles with Wikipedia bios (default/--all stay offline)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.all:
        main_all(enrich=args.enrich)
    else:
        main_default(enrich=args.enrich)


def seed_rankings_and_enrichment(player_ids: list[str], enrich: bool) -> None:
    """Import local official rank history for seeded players; enrich when asked.

    Rankings come only from the local archive (offline); the import is scoped
    to the seeded set and is silent about players without rank coverage — never
    name-matched. ATP_player_database.csv stays the primary IOC source; the
    ranking-source atp_players.csv fallback fills only seeded profiles still
    missing an IOC (NULL/empty/UNK) and never overwrites a verified one.
    Wikipedia enrichment is gated on --enrich and forces a refresh: every
    selected profile's summary is rewritten even when one already exists.
    """
    if not player_ids:
        return
    ingest_rankings(player_ids=set(player_ids))
    if enrich:
        # enrich_players emits its own per-player lines and batch summary.
        enrich_players(player_ids, force=True)


def main_default(enrich: bool = False) -> None:
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

    # Re-seeding overwrites the seed's own rows (ON CONFLICT UPDATE) so
    # `just db-seed` converges on repeat runs with the same source.
    insert_bronze_rows(bronze, overwrite=True)
    print(f"Inserted {len(bronze)} rows into {BRONZE_TABLE} (overwrite)")

    player_ids = sorted(set(bronze["player1_id"]) | set(bronze["player2_id"]))
    load_profiles_for(player_ids, "seeded")
    seed_rankings_and_enrichment(player_ids, enrich)


def main_all(enrich: bool = False) -> None:
    """Seed every ATP CSV, overwriting selected match rows on re-runs."""
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

    insert_bronze_rows(bronze, overwrite=True)
    print(f"Inserted {len(bronze)} rows into {BRONZE_TABLE} (overwrite)")

    player_ids = sorted(set(bronze["player1_id"]) | set(bronze["player2_id"]))
    load_profiles_for(player_ids, "seeded")
    seed_rankings_and_enrichment(player_ids, enrich)


if __name__ == "__main__":
    main()
