"""Deterministic dev seed: raw ATP matches -> bronze -> profiles.

Single seed entrypoint for the dev dataset. Mirrors the prod ingest path
(src/flows/ingest.py): the same raw ATP -> bronze transform and the same ATP
player-profile load (filtered to the seeded players only).

Selection is deterministic: the RECENT most recent matches of the TOP_PLAYERS
players with the best ranking at their latest match in data/raw/2026.csv,
deduped (~100 matches).

Permanently offline: the seed never performs live Wikipedia/bio enrichment.
Enrichment is owned exclusively by the ETL flow (src/flows/etl.py) as an
explicit operator opt-in; seeding stays offline by default and --all.

Usage:
    uv run python infra/duckdb/initialize_schemas.py init   # schemas first
    uv run python infra/duckdb/seed.py                      # or `just db-seed`
    uv run python infra/duckdb/seed.py --all                # seed every ATP CSV under data/raw/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.db.client import get_conn
from src.flows.ingest import (
    BRONZE_TABLE,
    atp_rows_to_bronze,
    insert_bronze_rows,
    load_profiles_for,
    load_raw_atp_rows,
    player_history,
)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
RAW_YEAR = RAW_DIR / "2026.csv"

TOP_PLAYERS = 10
RECENT = 10


def discover_atp_csvs(raw_dir: Path = RAW_DIR) -> list[Path]:
    """Every regular ATP match CSV under `raw_dir`, sorted.

    Only regular tour files are in scope: Challenger CSVs are excluded so a
    future `*_challenger.csv` file is never loaded. Excludes non-CSV files
    (.DS_Store, etc.). Deterministic order: the paths are sorted so matches
    seed chronologically regardless of on-disk order.
    """
    return sorted(p for p in raw_dir.glob("*.csv") if p.is_file() and "_challenger" not in p.name)


def load_all_raw_atp_rows(csv_paths: list[Path]) -> list[dict[str, Any]]:
    """Load every raw ATP CSV and sort all rows chronologically.

    The per-file rows are concatenated, then sorted by (tourney_date,
    tourney_id, match_num) so rolling form computed over the full history in
    atp_rows_to_bronze is correct across file boundaries.
    """
    rows = [row for path in csv_paths for row in load_raw_atp_rows(path)]
    return sorted(rows, key=lambda m: (int(m["tourney_date"]), m["tourney_id"], m["match_num"]))


def select_matches(
    matches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The RECENT most recent matches of the TOP_PLAYERS best-ranked players.

    Ranks every player by their ranking at their latest match in `matches`,
    takes the top TOP_PLAYERS by (latest_rank, player_id), and keeps their
    RECENT most recent matches, deduped to distinct matches.
    """
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
        f"DELETE FROM {BRONZE_TABLE} WHERE match_id IN ({', '.join('?' * len(selected_ids))})",
        list(selected_ids),
    )
    insert_bronze_rows(bronze)
    print(f"Inserted {len(bronze)} rows into {BRONZE_TABLE}")

    player_ids = sorted(set(bronze["player1_id"]) | set(bronze["player2_id"]))
    load_profiles_for(player_ids, "seeded")


def main_all() -> None:
    """Seed every ATP CSV under data/raw/ into bronze.

    Idempotent at the database level: rows are inserted via insert_bronze_rows
    (match_id PK, ON CONFLICT DO NOTHING), so re-running --all never duplicates
    or deletes existing data. No source CSV is rewritten and the database is not
    dropped. Permanently offline: no Wikipedia/bio enrichment is ever performed.
    """
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
