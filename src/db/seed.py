"""Deterministically seed PostgreSQL from ATP CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.constants import ROOT
from src.db.client import clear_active_sessions
from src.db.ingest import (
    BRONZE_MATCHES_TABLE,
    atp_rows_to_bronze,
    canonical_match_id,
    canonical_players,
    clear_match_events,
    discover_ranking_csvs,
    enrich_players,
    ingest_rankings,
    insert_bronze_rows,
    load_profiles_for,
    load_ranking_player_map,
    load_ranking_rows,
    load_raw_atp_rows,
    player_history,
)

RAW_DIR = ROOT / "data" / "raw"
RAW_YEAR = RAW_DIR / "2026.csv"

TOP_PLAYERS = 10
RECENT = 10
DEFAULT_YEAR = int(RAW_YEAR.stem)


def discover_atp_csvs(raw_dir: Path = RAW_DIR) -> list[Path]:
    """Return sorted regular-tour ATP CSVs, excluding Challenger files."""
    return sorted(p for p in raw_dir.glob("*.csv") if p.is_file() and "_challenger" not in p.name)


def load_all_raw_atp_rows(csv_paths: list[Path]) -> list[dict[str, Any]]:
    """Load and chronologically sort ATP rows across CSV boundaries."""
    rows = [row for path in csv_paths for row in load_raw_atp_rows(path)]
    return sorted(rows, key=lambda m: (int(m["tourney_date"]), m["tourney_id"], m["match_num"]))


def select_matches(
    matches: list[dict[str, Any]],
    official_ranks: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Select each top player's full default-year history and recent prior matches."""
    history = {
        pid: sorted(hist, key=lambda m: (int(m["tourney_date"]), m["tourney_id"], m["match_num"]))
        for pid, hist in player_history(matches).items()
    }
    if official_ranks is None:
        official_ranks = {}
        for m in matches:
            for player_id, rank_key in (
                (m["winner_id"], "winner_rank"),
                (m["loser_id"], "loser_rank"),
            ):
                try:
                    rank = int(m[rank_key])
                except (TypeError, ValueError):
                    continue
                if rank > 0:
                    official_ranks[player_id] = rank
    top = sorted(
        (pid for pid in history if pid in official_ranks),
        key=lambda pid: (official_ranks[pid], pid),
    )[:TOP_PLAYERS]
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    for pid in top:
        player_matches = history[pid]
        for m in player_matches:
            if int(m["tourney_date"]) // 10000 == DEFAULT_YEAR:
                selected[m["tourney_id"], m["match_num"]] = m
        prior_years = [m for m in player_matches if int(m["tourney_date"]) // 10000 != DEFAULT_YEAR]
        for m in prior_years[-RECENT:]:
            selected[m["tourney_id"], m["match_num"]] = m
    return sorted(
        selected.values(),
        key=lambda m: (int(m["tourney_date"]), m["tourney_id"], m["match_num"]),
    )


def latest_official_ranks() -> dict[str, int]:
    """Return latest archived top-200 ranks mapped to canonical player ids.

    A ranking row resolves to a canonical id either directly, when its own
    player id is a canonical ATP id, or through the reviewed source-to-canonical
    map. Rows that resolve through neither route are excluded (preserving rank
    order and the existing top-10 selection downstream).
    """
    rows = load_ranking_rows(discover_ranking_csvs(), rank_limit=200)
    if rows.empty:
        raise ValueError("miniseed selection requires archived official rankings")
    latest = rows[rows["ranking_date"] == rows["ranking_date"].max()]
    canonical_ids = set(canonical_players())
    rank_map = load_ranking_player_map(canonical_ids=canonical_ids)
    resolved: dict[str, int] = {}
    for source, rank in zip(latest["player_id"], latest["rank"], strict=True):
        source = str(source)
        if source in canonical_ids:
            canonical = source
        elif (canonical := rank_map.get(source)) is not None:
            canonical = canonical
        else:
            continue
        resolved[canonical] = int(rank)
    return resolved


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
    parser.add_argument(
        "--reset",
        action="store_true",
        help="clear and rebuild bronze.match_events, and overwrite rankings, profiles, bios",
    )
    args, _ignored = parser.parse_known_args(argv)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cancelled, terminated = clear_active_sessions()
    if cancelled or terminated:
        print(f"Cleared {len(cancelled)} active queries and {len(terminated)} idle transactions")
    if args.all:
        main_all(enrich=args.enrich, reset=args.reset)
    else:
        main_default(enrich=args.enrich, reset=args.reset)


def seed_rankings_and_enrichment(
    player_ids: list[str],
    enrich: bool,
    reset: bool,
    match_rows: list[dict[str, Any]] | None = None,
) -> None:
    """Import local ranks for seeded players and optionally enrich their profiles."""
    if not player_ids:
        return
    summary = ingest_rankings(player_ids=set(player_ids), force=reset, match_rows=match_rows)
    coverage = (summary or {}).get("coverage")
    if coverage:
        print(
            f"Rankings coverage: {coverage['covered']}/{coverage['seeded']} seeded players "
            f"with official top-200 history; {coverage['auto_mapped']} auto-mapped source IDs; "
            f"{coverage['unresolved']} unresolved."
        )
    if enrich:
        enrich_players(player_ids, force=reset)


def main_default(enrich: bool = False, reset: bool = False) -> None:
    matches = sorted(
        load_raw_atp_rows(RAW_YEAR),
        key=lambda m: (int(m["tourney_date"]), m["tourney_id"], m["match_num"]),
    )
    selected = select_matches(matches, official_ranks=latest_official_ranks() if matches else {})
    selected_ids = {
        canonical_match_id(m["tourney_id"], int(m["match_num"]), int(m["tourney_date"]) // 10000)
        for m in selected
    }

    bronze = atp_rows_to_bronze(matches, selected_ids=selected_ids)
    distinct_players = len(set(bronze["player1_id"]) | set(bronze["player2_id"]))
    print(
        f"Seeding {len(bronze)} bronze matches from {RAW_YEAR.name} "
        f"({bronze['match_date'].min()} .. {bronze['match_date'].max()}), "
        f"{distinct_players} players"
    )

    if reset:
        clear_match_events()
        print(f"Cleared {BRONZE_MATCHES_TABLE} for a clean rebuild")
    inserted = insert_bronze_rows(bronze, overwrite=False)
    if reset:
        print(f"Inserted {inserted} rows into {BRONZE_MATCHES_TABLE} (clean rebuild)")
    else:
        print(
            f"Inserted {inserted} rows into {BRONZE_MATCHES_TABLE} "
            f"({len(bronze) - inserted} skipped existing)"
        )

    player_ids = sorted(set(bronze["player1_id"]) | set(bronze["player2_id"]))
    load_profiles_for(player_ids, "seeded", force=reset)
    seed_rankings_and_enrichment(player_ids, enrich, reset, match_rows=matches)


def main_all(enrich: bool = False, reset: bool = False) -> None:
    """Seed every ATP CSV; --reset clears bronze.match_events first."""
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

    if reset:
        clear_match_events()
        print(f"Cleared {BRONZE_MATCHES_TABLE} for a clean rebuild")
    inserted = insert_bronze_rows(bronze, overwrite=False)
    if reset:
        print(f"Inserted {inserted} rows into {BRONZE_MATCHES_TABLE} (clean rebuild)")
    else:
        print(
            f"Inserted {inserted} rows into {BRONZE_MATCHES_TABLE} "
            f"({len(bronze) - inserted} skipped existing)"
        )

    player_ids = sorted(set(bronze["player1_id"]) | set(bronze["player2_id"]))
    load_profiles_for(player_ids, "seeded", force=reset)
    seed_rankings_and_enrichment(player_ids, enrich, reset, match_rows=matches)


if __name__ == "__main__":
    main()
