"""CSV ingestion and player profile enrichment for tennis match data.

Usage:
    uv run python -m src.flows.ingest data/matches.csv

Shared enrichment logic also used by the Prefect ETL flow."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd
import requests

from src.db.client import get_client, to_dataframe
from src.features.validate import run_ingestion_checks

if TYPE_CHECKING:
    from clickhouse_connect.driver import Client

BRONZE_TABLE = "bronze.match_events"
GOLD_TABLE = "gold.match_features"
PROFILES_TABLE = "gold.player_profiles"

EXPECTED_COLUMNS = [
    "match_id",
    "match_date",
    "player_id",
    "opponent_id",
    "tournament",
    "round",
    "surface",
    "player_ranking",
    "opponent_ranking",
    "wins_last_10",
    "matches_last_10",
    "aces",
    "double_faults",
    "first_serves_made",
    "total_serve_points",
    "break_points_won",
    "break_points_total",
    "match_won",
]

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "TennisML/0.1 (research project; contact@tennis-ml.local)"


# ── CSV Loading ──────────────────────────────────────────────────


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return cast(pd.DataFrame, df[EXPECTED_COLUMNS])


# ── Wikipedia Profile Enrichment ────────────────────────────────


def get_players_without_profiles(client: Client) -> list[str]:
    sql = f"""
        SELECT DISTINCT gold.player_id
        FROM {GOLD_TABLE} gold
        LEFT JOIN {PROFILES_TABLE} prof ON gold.player_id = prof.player_id
        WHERE prof.player_id IS NULL
    """
    df = to_dataframe(sql, client)
    return df["player_id"].tolist()


def search_wikipedia(name: str) -> str | None:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": f"{name} tennis player",
        "format": "json",
        "srlimit": 1,
    }
    resp = requests.get(WIKI_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=10)
    data = resp.json()
    pages = data.get("query", {}).get("search", [])
    return pages[0]["title"] if pages else None


def fetch_summary(title: str) -> dict | None:
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts|pageprops",
        "exintro": True,
        "explaintext": True,
        "format": "json",
    }
    resp = requests.get(WIKI_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=10)
    data = resp.json()
    pages = data.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if page_id != "-1":
            return {
                "title": page.get("title", ""),
                "summary": page.get("extract", ""),
                "page_id": page_id,
            }
    return None


def extract_infobox_fields(summary: str) -> dict:
    fields = {}
    patterns = {
        "plays": r"Plays?\s*[:\-]\s*([A-Za-z\-]+)",
        "backhand": r"Backhand\s*[:\-]\s*([A-Za-z\-]+)",
        "height": r"Height\s*[:\-]\s*([\d.]+)\s*m",
        "turned_pro": r"Turned pro\s*[:\-]\s*(\d{4})",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, summary, re.IGNORECASE)
        if match:
            fields[key] = match.group(1).strip()
    return fields


def classify_style(extract: str) -> list[str]:
    extract_lower = extract.lower()
    keywords = {
        "aggressive baseliner": [
            "aggressive baseline",
            "powerful groundstroke",
            "big forehand",
            "heavy topspin",
        ],
        "serve-and-volleyer": ["serve and volley", "serve-and-volley", "net rusher", "net game"],
        "defensive counterpuncher": [
            "defensive",
            "counterpunch",
            "exceptional speed",
            "retrieves",
            "pusher",
        ],
        "all-court player": ["all-court", "complete game", "versatile", "all round"],
        "big server": ["big serve", "powerful serve", "ace machine"],
        "clay specialist": ["clay specialist", "king of clay", "dominant on clay"],
        "grinder": ["grinder", "grind", "relentless"],
    }
    found = []
    for style, signals in keywords.items():
        if any(signal in extract_lower for signal in signals):
            found.append(style)
    return found if found else ["unknown"]


def enrich_player(client: Client, player: str) -> bool:
    """Fetch Wikipedia bio for a single player and insert into profiles table.

    Returns True if profile was inserted, False if skipped.
    """
    title = search_wikipedia(player)
    if not title:
        print(f"  SKIP {player}: no Wikipedia match")
        return False

    page = fetch_summary(title)
    if not page:
        print(f"  SKIP {player}: no page data")
        return False

    infobox = extract_infobox_fields(page["summary"])
    styles = classify_style(page["summary"])

    safe_summary = page["summary"][:1000].replace("'", "\\'")
    safe_title = page["title"].replace("'", "\\'")

    client.command(f"""
        INSERT INTO {PROFILES_TABLE}
            (player_id, display_name, summary, handedness, backhand,
             play_style, height, turned_pro)
        VALUES (
            '{player}',
            '{safe_title}',
            '{safe_summary}',
            '{infobox.get("plays", "").lower().replace(" ", "_")}',
            '{infobox.get("backhand", "").lower().replace(" ", "_")}',
            '{", ".join(styles).lower().replace(" ", "_")}',
            '{infobox.get("height", "")}',
            {int(infobox.get("turned_pro", 0))}
        )
    """)
    print(f"  OK {player} → {page['title']}")
    return True


def enrich_missing(client: Client | None = None) -> int:
    """Find all players in gold missing from profiles, fetch from Wikipedia.

    Returns count of profiles inserted.
    """
    client = client or get_client()

    missing = get_players_without_profiles(client)
    if not missing:
        print("All players have profiles. Nothing to do.")
        return 0

    print(f"Found {len(missing)} players without profiles")
    inserted = 0
    for player in missing:
        try:
            if enrich_player(client, player):
                inserted += 1
        except Exception as e:
            print(f"  ERROR {player}: {e}")

    print(f"Done: {inserted}/{len(missing)} profiles inserted")
    return inserted


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python -m src.flows.ingest data/matches.csv", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = load_csv(str(csv_path))
    print(f"Loaded {len(df)} rows from {csv_path.name}")

    result = run_ingestion_checks(df)
    if not result["passed"]:
        print("Validation failed. Fix the data or adjust the checks.")
        sys.exit(1)

    client = get_client()
    client.insert_df(BRONZE_TABLE, df)
    print(f"Inserted {len(df)} rows into {BRONZE_TABLE}")

    enriched = enrich_missing(client)
    print(f"Enriched {enriched} player profiles")
