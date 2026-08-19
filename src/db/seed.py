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
    BRONZE_MATCHES_TABLE,
    atp_rows_to_bronze,
    canonical_match_id,
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
# Calendar year of the default seed corpus (data/raw/2026.csv); kept in sync
# with the file name so the full-history rule never drifts from the data.
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
    """Select distinct matches for the best-ranked players: their full
    default-year (DEFAULT_YEAR) history plus their RECENT most recent
    other-year matches."""
    history = {
        pid: sorted(hist, key=lambda m: (int(m["tourney_date"]), m["tourney_id"], m["match_num"]))
        for pid, hist in player_history(matches).items()
    }
    if official_ranks is None:
        # Test/helper fallback only; production miniseed always supplies the
        # official archive ranks below.
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
    """Latest archived official ATP ranks, resolved into canonical player ids.

    Only the top-200 archive rows are needed — the global top-10 the miniseed
    selects are always within the top 200 — so the sub-200 archive is skipped.
    """
    rows = load_ranking_rows(discover_ranking_csvs(), rank_limit=200)
    if rows.empty:
        raise ValueError("miniseed selection requires archived official rankings")
    latest = rows[rows["ranking_date"] == rows["ranking_date"].max()]
    rank_map = load_ranking_player_map()
    return {
        canonical: int(rank)
        for source, rank in zip(latest["player_id"], latest["rank"], strict=True)
        if (canonical := rank_map.get(str(source))) is not None
    }


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
        "--force",
        action="store_true",
        help="overwrite existing rows and ignore idempotent skips (matches, rankings, profiles, bios)",
    )
    args, _ignored = parser.parse_known_args(argv)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.all:
        main_all(enrich=args.enrich, force=args.force)
    else:
        main_default(enrich=args.enrich, force=args.force)


def seed_rankings_and_enrichment(
    player_ids: list[str],
    enrich: bool,
    force: bool,
    match_rows: list[dict[str, Any]] | None = None,
) -> None:
    """Import local official rank history for seeded players; enrich when asked.

    Rankings come only from the local archive (offline); the import is scoped
    to the seeded set. Source ids absent from the reviewed map are auto-mapped
    by normalized name (with deterministic activity/rank/id tie-breaks using
    match_rows), and the returned summary prints seed coverage — covered /
    seeded players with official top-200 history, auto-mapped source ids, and
    unresolved identities. ATP_player_database.csv stays the primary IOC source;
    the ranking-source atp_players.csv fallback fills only seeded profiles still
    missing an IOC (NULL/empty/UNK) and never overwrites a verified one.
    Wikipedia enrichment is gated on --enrich and idempotent unless --force is
    given: profiles that already have a summary are skipped, never overwritten.
    """
    if not player_ids:
        return
    summary = ingest_rankings(player_ids=set(player_ids), force=force, match_rows=match_rows)
    coverage = (summary or {}).get("coverage")
    if coverage:
        print(
            f"Rankings coverage: {coverage['covered']}/{coverage['seeded']} seeded players "
            f"with official top-200 history; {coverage['auto_mapped']} auto-mapped source IDs; "
            f"{coverage['unresolved']} unresolved."
        )
    if enrich:
        # enrich_players emits its own per-player lines and batch summary.
        enrich_players(player_ids, force=force)


def main_default(enrich: bool = False, force: bool = False) -> None:
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

    # Idempotent by default (DO NOTHING on an existing match_id); --force
    # overwrites the seed's own rows (DO UPDATE) so repeat runs converge.
    inserted = insert_bronze_rows(bronze, overwrite=force)
    if force:
        print(f"Inserted {inserted} rows into {BRONZE_MATCHES_TABLE} (overwrite)")
    else:
        print(
            f"Inserted {inserted} rows into {BRONZE_MATCHES_TABLE} "
            f"({len(bronze) - inserted} skipped existing)"
        )

    player_ids = sorted(set(bronze["player1_id"]) | set(bronze["player2_id"]))
    load_profiles_for(player_ids, "seeded", force=force)
    seed_rankings_and_enrichment(player_ids, enrich, force, match_rows=matches)


def main_all(enrich: bool = False, force: bool = False) -> None:
    """Seed every ATP CSV, skipping existing rows unless --force."""
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

    inserted = insert_bronze_rows(bronze, overwrite=force)
    if force:
        print(f"Inserted {inserted} rows into {BRONZE_MATCHES_TABLE} (overwrite)")
    else:
        print(
            f"Inserted {inserted} rows into {BRONZE_MATCHES_TABLE} "
            f"({len(bronze) - inserted} skipped existing)"
        )

    player_ids = sorted(set(bronze["player1_id"]) | set(bronze["player2_id"]))
    load_profiles_for(player_ids, "seeded", force=force)
    seed_rankings_and_enrichment(player_ids, enrich, force, match_rows=matches)


if __name__ == "__main__":
    main()
