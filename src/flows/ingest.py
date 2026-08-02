"""CSV ingestion and player profile enrichment for tennis match data.

Usage:
    uv run python -m src.flows.ingest data/matches.csv

Shared enrichment logic also used by the Prefect ETL flow."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import cast

import pandas as pd
import requests

from src.constants import ROOT
from src.db.client import get_conn, to_dataframe
from src.features.validate import run_ingestion_checks

BRONZE_TABLE = "bronze.match_events"
GOLD_TABLE = "gold.match_features"
PROFILES_TABLE = "gold.player_profiles"

# ATP_Database.csv is the identity backbone for player_profiles; the Wikipedia
# flow below is the enrichment fallback for players it does not cover.
ATP_DATABASE_CSV = ROOT / "data" / "raw" / "ATP_Database.csv"

# Base metadata columns mirrored from ATP_Database.csv (id, player, atpname,
# birthdate, weight, height, turnedpro, birthplace, coaches, hand, backhand, ioc).
# Enrichment columns (summary, play_style, wiki_title, enriched_at) are not
# loaded here; they are filled by the Wikipedia fallback in enrich_player().
ATP_PROFILE_COLUMNS = [
    "player_id",
    "display_name",
    "atp_name",
    "birthdate",
    "weight",
    "height",
    "turned_pro",
    "birthplace",
    "coaches",
    "handedness",
    "backhand",
    "ioc",
]

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


# ── ATP Database Identity Load ─────────────────────────────────


def _parse_int(
    series: pd.Series, column: str, player_ids: pd.Series, low: int, high: int
) -> pd.Series:
    """Parse an ATP integer column once at load time (fail loudly on garbage).

    Empty cells become NULL; '0' is the ATP CSV's missing marker and also
    becomes NULL. Any other non-empty value that is not an integer within
    [low, high] raises, so malformed strings never silently land in the
    table as a real number.
    """
    s = series.fillna("").astype(str).str.strip()
    nonempty = s != ""
    nums = cast(pd.Series, pd.to_numeric(s.mask(~nonempty), errors="coerce"))
    is_int = nums.notna() & (nums == nums.astype("float64").round())
    valid = is_int & (nums >= low) & (nums <= high)
    bad = nonempty & ~valid & (nums != 0)
    if bad.any():
        offenders = ", ".join(
            f"{pid}={val!r}" for pid, val in zip(player_ids[bad], s[bad], strict=True)
        )
        raise ValueError(
            f"ATP {column} malformed (expected integer in [{low}, {high}], "
            f"or 0/empty for unknown): {offenders}"
        )
    result = pd.Series(pd.NA, index=series.index, dtype="Int64")
    result[valid] = nums[valid].astype("int64")
    return result


def _parse_birthdate(series: pd.Series, player_ids: pd.Series) -> pd.Series:
    """Parse ATP birthdate (YYYYMMDD) once at load time; NULL when empty.

    Raises on any non-empty value that is not a plausible date so malformed
    strings never land in the table.
    """
    s = series.fillna("").astype(str).str.strip()
    nonempty = s != ""
    parsed = pd.to_datetime(s.mask(~nonempty), format="%Y%m%d", errors="coerce")
    bad = nonempty & (parsed.isna() | (parsed.dt.year < 1900) | (parsed.dt.year > 2100))
    if bad.any():
        offenders = ", ".join(
            f"{pid}={val!r}" for pid, val in zip(player_ids[bad], s[bad], strict=True)
        )
        raise ValueError(
            f"ATP birthdate malformed (expected YYYYMMDD in 1900-2100, or empty): {offenders}"
        )
    return parsed


def load_atp_profiles(csv_path: str | Path = ATP_DATABASE_CSV) -> int:
    """Load ATP_Database.csv into gold.player_profiles (identity backbone).

    Maps the ATP header (id, player, atpname, birthdate, weight, height,
    turnedpro, birthplace, coaches, hand, backhand, ioc) onto the table's
    base metadata columns. Typed fields (birthdate, weight, height,
    turned_pro) are parsed exactly once here and land in the table with
    their real DB types; malformed non-empty values raise. Enrichment
    columns are left untouched, so existing Wikipedia enrichment survives
    re-loads. Returns rows loaded.
    """
    atp = pd.read_csv(csv_path, dtype=str)
    if not {"id", "player", "atpname", "hand", "backhand", "ioc"} <= set(atp.columns):
        raise ValueError(f"ATP database CSV missing expected columns: {csv_path}")

    player_ids = cast(pd.Series, atp["id"])
    birthdate = _parse_birthdate(cast(pd.Series, atp["birthdate"]), player_ids)
    weight = _parse_int(
        cast(pd.Series, atp["weight"]), "weight", player_ids, low=20, high=300
    ).astype("Int16")
    height = _parse_int(
        cast(pd.Series, atp["height"]), "height", player_ids, low=100, high=250
    ).astype("Int16")
    turned_pro = _parse_int(
        cast(pd.Series, atp["turnedpro"]), "turnedpro", player_ids, low=1900, high=2100
    ).astype("Int32")
    df = pd.DataFrame(
        {
            "player_id": player_ids,
            "display_name": atp["player"],
            "atp_name": atp["atpname"],
            "birthdate": birthdate,
            "weight": weight,
            "height": height,
            "turned_pro": turned_pro,
            "birthplace": atp["birthplace"],
            "coaches": atp["coaches"],
            "handedness": atp["hand"],
            "backhand": atp["backhand"],
            "ioc": atp["ioc"],
        }
    )

    base_updates = ", ".join(
        f"{col} = excluded.{col}" for col in ATP_PROFILE_COLUMNS if col != "player_id"
    )
    conn = get_conn()
    conn.sql(f"""
        INSERT INTO {PROFILES_TABLE} ({", ".join(ATP_PROFILE_COLUMNS)})
        SELECT * FROM df
        ON CONFLICT (player_id) DO UPDATE SET {base_updates}
    """)
    return len(df)


# ── Wikipedia Profile Enrichment ────────────────────────────────


def get_players_without_profiles() -> list[str]:
    sql = f"""
        SELECT DISTINCT gold.player_id
        FROM {GOLD_TABLE} gold
        LEFT JOIN {PROFILES_TABLE} prof ON gold.player_id = prof.player_id
        WHERE prof.player_id IS NULL
    """
    df = to_dataframe(sql)
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


def fetch_summary(title: str) -> dict[str, str] | None:
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


def extract_infobox_fields(summary: str) -> dict[str, str]:
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


def enrich_player(player: str) -> bool:
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

    # Typed fields: infobox height is meters ("1.85 m") but the column is cm;
    # both height and turned_pro become NULL when Wikipedia has no value.
    height_m = cast(str | None, infobox.get("height"))
    height_cm = round(float(height_m) * 100) if height_m else None
    turned_pro_raw = cast(str | None, infobox.get("turned_pro"))
    turned_pro = int(turned_pro_raw) if turned_pro_raw else None
    height_value = str(height_cm) if height_cm is not None else "NULL"
    turned_pro_value = str(turned_pro) if turned_pro is not None else "NULL"

    safe_summary = page["summary"][:1000].replace("'", "\\'")
    safe_title = page["title"].replace("'", "\\'")

    conn = get_conn()
    conn.sql(f"""
        INSERT INTO {PROFILES_TABLE}
            (player_id, display_name, summary, handedness, backhand,
             play_style, height, turned_pro, wiki_title, enriched_at)
        VALUES (
            '{player}',
            '{safe_title}',
            '{safe_summary}',
            '{infobox.get("plays", "").lower().replace(" ", "_")}',
            '{infobox.get("backhand", "").lower().replace(" ", "_")}',
            '{", ".join(styles).lower().replace(" ", "_")}',
            {height_value},
            {turned_pro_value},
            '{safe_title}',
            CURRENT_TIMESTAMP
        )
    """)
    print(f"  OK {player} -> {page['title']}")
    return True


def enrich_missing() -> int:
    """Find all players in gold missing from profiles, fetch from Wikipedia.

    Returns count of profiles inserted.
    """
    missing = get_players_without_profiles()
    if not missing:
        print("All players have profiles. Nothing to do.")
        return 0

    print(f"Found {len(missing)} players without profiles")
    inserted = 0
    for player in missing:
        try:
            if enrich_player(player):
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

    conn = get_conn()
    conn.sql(f"INSERT INTO {BRONZE_TABLE} SELECT * FROM df")
    print(f"Inserted {len(df)} rows into {BRONZE_TABLE}")

    if ATP_DATABASE_CSV.exists():
        loaded = load_atp_profiles(ATP_DATABASE_CSV)
        print(f"Loaded {loaded} player profiles from {ATP_DATABASE_CSV.name}")
    else:
        print(f"ATP database not found at {ATP_DATABASE_CSV}, skipping identity load")

    enriched = enrich_missing()
    print(f"Enriched {enriched} player profiles")
