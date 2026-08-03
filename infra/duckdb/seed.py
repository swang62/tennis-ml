"""Deterministic dev seed: raw ATP matches -> bronze -> profiles -> Wikipedia.

Single seed entrypoint for the dev dataset. Mirrors the prod ingest path
(src/flows/ingest.py): the same raw ATP -> bronze transform, the same ATP
player-profile load (filtered to the seeded players only), and the same
best-effort Wikipedia enrichment.

Selection is deterministic: the RECENT most recent matches of the TOP_PLAYERS
players with the best ranking at their latest match in data/raw/2026.csv,
deduped (~100 matches).

Usage:
    uv run python infra/duckdb/initialize_schemas.py init   # schemas first
    uv run python infra/duckdb/seed.py                      # or `just db-seed`
    uv run python infra/duckdb/seed.py --offline            # skip live Wikipedia (tests/offline)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from src.db.client import get_conn
from src.flows.ingest import (
    BRONZE_TABLE,
    atp_rows_to_bronze,
    enrich_players,
    insert_bronze_rows,
    load_profiles_for,
    load_raw_atp_rows,
    player_history,
)

ROOT = Path(__file__).resolve().parents[2]
RAW_YEAR = ROOT / "data" / "raw" / "2026.csv"

TOP_PLAYERS = 10
RECENT = 10


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


def main() -> None:
    matches = sorted(
        load_raw_atp_rows(RAW_YEAR),
        key=lambda m: (int(m["tourney_date"]), m["tourney_id"], m["match_num"]),
    )
    selected = select_matches(matches)
    selected_ids = {f"{m['tourney_id']}-{int(m['match_num']):03d}" for m in selected}

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

    # Best-effort: enrichment is non-fatal (no live Wikipedia in CI/tests),
    # and skippable with --offline so the seed stays fast offline.
    if "--offline" in sys.argv[1:]:
        print("Wikipedia enrichment skipped (--offline)")
        return
    enriched = enrich_players(player_ids)
    print(f"Enriched {enriched} player profiles with non-empty summaries")


if __name__ == "__main__":
    main()
