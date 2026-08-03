"""CSV ingestion and player profile enrichment for tennis match data.

Usage:
    uv run python -m src.flows.ingest data/matches.csv

Takes a raw ATP-format CSV (winner/loser columns, see
data/column_features_glossary.md) and maps it to bronze.match_events rows
using the same transform as the dev seed flow (infra/duckdb/seed.py). It then
loads ATP player profiles for the ingested players and runs best-effort
Wikipedia enrichment.

Shared enrichment logic also used by the Prefect ETL flow."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd
import requests

from src.constants import ROOT
from src.db.client import get_conn, to_dataframe
from src.features.columns import BRONZE_COLUMNS
from src.features.validate import run_ingestion_checks
from src.utils import load_env

BRONZE_TABLE = "bronze.match_events"
GOLD_TABLE = "gold.match_features"
PROFILES_TABLE = "gold.player_profiles"

# The ATP player database is the identity backbone for player_profiles; the
# Wikipedia flow below is the enrichment fallback for players it does not cover.
ATP_DATABASE_CSV = ROOT / "data" / "ATP_player_database.csv"

# Base metadata columns mirrored from the ATP player database (id, player,
# atpname, birthdate, weight, height, turnedpro, birthplace, coaches, hand,
# backhand, ioc). Enrichment columns (summary, enriched_at) are
# not loaded here; they are filled by the Wikipedia fallback in
# enrich_player().
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

# Columns every raw ATP match row carries (see data/column_features_glossary.md).
RAW_ATP_COLUMNS = [
    "tourney_id",
    "tourney_date",
    "match_num",
    "winner_id",
    "loser_id",
    "winner_rank",
    "loser_rank",
    "tourney_level",
    "round",
    "surface",
    "w_ace",
    "w_df",
    "w_svpt",
    "w_1stIn",
    "w_bpSaved",
    "w_bpFaced",
    "l_ace",
    "l_df",
    "l_svpt",
    "l_1stIn",
    "l_bpSaved",
    "l_bpFaced",
]

# Raw ATP tourney_level -> canonical bronze.tournament.
# D (Davis Cup) and F (Masters/ATP Finals) appear in full-year raw files but
# never in the dev seed selection; they map to level 0 downstream (see
# dbt/models/gold/match_features.sql).
LEVEL_MAP = {
    "G": "grand_slam",
    "M": "masters",
    "A": "atp_500",
    "500": "atp_500",
    "250": "atp_250",
    "D": "davis_cup",
    "F": "finals",
}

# Rolling-form window for wins_last_10 / matches_last_10.
RECENT = 10

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "TennisML/0.1 (research project; contact@tennis-ml.local)"

# Wikipedia summaries are truncated to this many chars before they are stored.
SUMMARY_MAX_CHARS = 2000

load_env()

# ── Raw ATP → Bronze transform (shared with infra/duckdb/seed.py) ───────────


def _stat(row: dict[str, Any], key: str) -> int:
    """Int stat value; 0 for empty/NaN (raw ATP CSVs leave stats empty)."""
    try:
        return int(row[key])
    except (TypeError, ValueError):
        return 0


def player_history(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Each player's full match history, oldest -> newest."""
    history: dict[str, list[dict[str, Any]]] = {}
    for m in rows:
        history.setdefault(m["winner_id"], []).append(m)
        history.setdefault(m["loser_id"], []).append(m)
    return history


def _recent_form(m: dict[str, Any], hist: list[dict[str, Any]]) -> tuple[int, int]:
    """wins/matches in the RECENT matches strictly before m, per player side."""
    i = hist.index(m)
    prior = hist[max(0, i - RECENT) : i]
    return (sum(1 for x in prior if x["winner_id"] == m["winner_id"]), len(prior))


def atp_rows_to_bronze(
    rows: list[dict[str, Any]], selected_ids: set[str] | None = None
) -> pd.DataFrame:
    """Map raw ATP-format rows to bronze.match_events rows.

    Shared by the ingest CLI and the dev seed (infra/duckdb/seed.py) so both
    paths use identical semantics: winner on the player1 side, ISO match dates
    from tourney_date, canonical tournament names (LEVEL_MAP), lowercased
    round/surface, break_points_won = faced - saved, and per-player
    wins/matches_last_10 over the RECENT matches strictly before each match.

    `rows` is the full history (rolling form is computed over all of it); when
    `selected_ids` is given, only those match_ids (tourney_id-match_num) are
    emitted.
    """
    if not rows:
        return pd.DataFrame({col: [] for col in BRONZE_COLUMNS})
    rows = sorted(rows, key=lambda m: (int(m["tourney_date"]), m["tourney_id"], m["match_num"]))
    history = player_history(rows)

    out = []
    for m in rows:
        match_id = f"{m['tourney_id']}-{int(m['match_num']):03d}"
        if selected_ids is not None and match_id not in selected_ids:
            continue
        level = LEVEL_MAP.get(str(m["tourney_level"]))
        if level is None:
            raise ValueError(
                f"tourney_level {m['tourney_level']!r} has no canonical mapping "
                f"({m['tourney_id']} match {m['match_num']})"
            )
        winner_wins, winner_matches = _recent_form(m, history[m["winner_id"]])
        loser_wins, loser_matches = _recent_form(m, history[m["loser_id"]])
        ymd = int(m["tourney_date"])
        out.append(
            {
                "match_id": match_id,
                "match_date": f"{ymd // 10000:04d}-{ymd % 10000 // 100:02d}-{ymd % 100:02d}",
                "player1_id": m["winner_id"],
                "player2_id": m["loser_id"],
                "tournament": level,
                "round": str(m["round"]).lower(),
                "surface": str(m["surface"]).lower(),
                "player1_ranking": m["winner_rank"],
                "player2_ranking": m["loser_rank"],
                "player1_wins_last_10": winner_wins,
                "player1_matches_last_10": winner_matches,
                "player1_aces": _stat(m, "w_ace"),
                "player1_double_faults": _stat(m, "w_df"),
                "player1_first_serves_made": _stat(m, "w_1stIn"),
                "player1_total_serve_points": _stat(m, "w_svpt"),
                "player1_break_points_won": _stat(m, "w_bpFaced") - _stat(m, "w_bpSaved"),
                "player1_break_points_total": _stat(m, "w_bpFaced"),
                "player2_wins_last_10": loser_wins,
                "player2_matches_last_10": loser_matches,
                "player2_aces": _stat(m, "l_ace"),
                "player2_double_faults": _stat(m, "l_df"),
                "player2_first_serves_made": _stat(m, "l_1stIn"),
                "player2_total_serve_points": _stat(m, "l_svpt"),
                "player2_break_points_won": _stat(m, "l_bpFaced") - _stat(m, "l_bpSaved"),
                "player2_break_points_total": _stat(m, "l_bpFaced"),
                "winner_id": m["winner_id"],
            }
        )

    return pd.DataFrame({col: [r[col] for r in out] for col in BRONZE_COLUMNS})


# ── CSV Loading ──────────────────────────────────────────────────


def load_raw_atp_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a raw ATP-format CSV into eligible raw rows (shared seed/ingest).

    Rows with missing rankings or player ids (walkovers, Davis Cup ties) are
    dropped, and empty stat cells become 0 — the same eligibility filter the
    dev seed applies (infra/duckdb/seed.py).
    """
    df = pd.read_csv(path)
    missing = set(RAW_ATP_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing raw ATP columns: {missing}")
    df = cast(pd.DataFrame, df[RAW_ATP_COLUMNS])
    eligible = (
        df["winner_rank"].notna()
        & df["loser_rank"].notna()
        & df["winner_id"].notna()
        & df["loser_id"].notna()
    )
    df = cast(pd.DataFrame, df.loc[eligible])
    return cast(list[dict[str, Any]], df.fillna(0).to_dict(orient="records"))


def load_atp_csv(path: str | Path) -> pd.DataFrame:
    """Read a raw ATP-format CSV and return bronze.match_events rows."""
    return atp_rows_to_bronze(load_raw_atp_rows(path))


def insert_bronze_rows(df: pd.DataFrame) -> int:
    """Insert or upsert bronze.match_events rows from a DataFrame; returns row count.

    Shared by the ingest CLI and the dev seed so both paths use one INSERT.
    match_id is the PK, so re-ingesting the same file (or an updated version
    of it) refreshes rows in place instead of duplicating them.
    """
    report = run_ingestion_checks(df)
    valid_df = cast(pd.DataFrame, report["valid_df"])

    for issue in cast(list[str], report["results"]):
        print(f"  DROP: {issue}")
    print(
        "Ingestion report: "
        f"input_rows={report['input_rows']} "
        f"valid_rows={report['valid_rows']} "
        f"dropped_rows={report['dropped_rows']}"
    )

    if valid_df.empty:
        return 0

    conn = get_conn()
    updates = ", ".join(f"{col} = excluded.{col}" for col in BRONZE_COLUMNS if col != "match_id")
    conn.sql(
        f"INSERT INTO {BRONZE_TABLE} ({', '.join(BRONZE_COLUMNS)}) "
        f"SELECT {', '.join(BRONZE_COLUMNS)} FROM valid_df "
        f"ON CONFLICT (match_id) DO UPDATE SET {updates}"
    )
    return len(valid_df)


def load_profiles_for(player_ids: list[str], label: str) -> int:
    """Load ATP identities for player_ids and print status (shared with seed).

    `label` names the caller in the status line ("seeded"/"ingested"). Returns
    the number of profiles loaded (0 when the ATP database file is absent).
    """
    if not ATP_DATABASE_CSV.exists():
        print(f"ATP database not found at {ATP_DATABASE_CSV}, skipping identity load")
        return 0
    loaded = load_atp_profiles(ATP_DATABASE_CSV, player_ids=set(player_ids))
    print(f"Loaded {loaded} player profiles for {len(set(player_ids))} {label} players")
    return loaded


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


def load_atp_profiles(
    csv_path: str | Path = ATP_DATABASE_CSV, player_ids: set[str] | None = None
) -> int:
    """Load the ATP player database into gold.player_profiles (identity backbone).

    Maps the ATP header (id, player, atpname, birthdate, weight, height,
    turnedpro, birthplace, coaches, hand, backhand, ioc) onto the table's
    base metadata columns. Typed fields (birthdate, weight, height,
    turned_pro) are parsed exactly once here and land in the table with
    their real DB types; malformed non-empty values raise. Enrichment
    columns are left untouched, so existing Wikipedia enrichment survives
    re-loads. When `player_ids` is given, only those ATP ids are loaded
    (used by the seed flow for the seeded players only). Returns the count
    of distinct player ids loaded (duplicate CSV ids collapse under the PK
    upsert).
    """
    atp = pd.read_csv(csv_path, dtype=str)
    if not {"id", "player", "atpname", "hand", "backhand", "ioc"} <= set(atp.columns):
        raise ValueError(f"ATP database CSV missing expected columns: {csv_path}")
    if player_ids is not None:
        atp = atp[atp["id"].isin(list(player_ids))]
    # gold.player_profiles.player_id is the PK, so duplicate CSV ids collapse
    # under the upsert below. Dedupe first (last row wins, same as the upsert)
    # so the returned count reflects distinct player ids, not raw CSV rows.
    atp = atp[~cast(pd.Series, atp["id"]).duplicated(keep="last")]

    ids = cast(pd.Series, atp["id"])
    birthdate = _parse_birthdate(cast(pd.Series, atp["birthdate"]), ids)
    weight = _parse_int(cast(pd.Series, atp["weight"]), "weight", ids, low=20, high=300).astype(
        "Int16"
    )
    height = _parse_int(cast(pd.Series, atp["height"]), "height", ids, low=100, high=250).astype(
        "Int16"
    )
    turned_pro = _parse_int(
        cast(pd.Series, atp["turnedpro"]), "turnedpro", ids, low=1900, high=2100
    ).astype("Int32")
    df = pd.DataFrame(
        {
            "player_id": ids,
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


def enrich_player(name: str, player_id: str | None = None) -> bool:
    """Fetch a Wikipedia bio for NAME and write it into the profiles table.

    The row is keyed by `player_id` (defaults to `name`). Enrichment columns
    (summary, enriched_at) are UPSERTed, so existing ATP base
    metadata on the row survives; a fresh row additionally gets the Wikipedia
    title as display_name plus any infobox plays/backhand/height/turned-pro.
    Returns True only when a non-empty summary was written. Pages without a
    search match, without extract data, or with an empty extract are SKIPped
    and never counted as success.
    """
    pid = player_id or name
    title = search_wikipedia(name)
    if not title:
        print(f"  SKIP {pid}: no Wikipedia match for {name!r}")
        return False

    page = fetch_summary(title)
    if not page:
        print(f"  SKIP {pid}: no page data for {title!r}")
        return False

    if not page["summary"].strip():
        print(f"  SKIP {pid}: empty Wikipedia summary for {title!r}")
        return False

    infobox = extract_infobox_fields(page["summary"])

    # Typed fields: infobox height is meters ("1.85 m") but the column is cm;
    # both height and turned_pro become NULL when Wikipedia has no value.
    height_m = cast(str | None, infobox.get("height"))
    height_cm = round(float(height_m) * 100) if height_m else None
    turned_pro_raw = cast(str | None, infobox.get("turned_pro"))
    turned_pro = int(turned_pro_raw) if turned_pro_raw else None
    height_value = str(height_cm) if height_cm is not None else "NULL"
    turned_pro_value = str(turned_pro) if turned_pro is not None else "NULL"

    # DuckDB escapes single quotes by doubling, not backslash.
    summary_text = page["summary"][:SUMMARY_MAX_CHARS]
    safe_summary = summary_text.replace("'", "''")
    safe_title = page["title"].replace("'", "''")

    conn = get_conn()
    conn.sql(f"""
        INSERT INTO {PROFILES_TABLE}
            (player_id, display_name, summary, handedness, backhand,
             height, turned_pro, enriched_at)
        VALUES (
            '{player_id}',
            '{safe_title}',
            '{safe_summary}',
            '{infobox.get("plays", "").lower().replace(" ", "_")}',
            '{infobox.get("backhand", "").lower().replace(" ", "_")}',
            {height_value},
            {turned_pro_value},
            CURRENT_TIMESTAMP
        )
        ON CONFLICT (player_id) DO UPDATE SET
            summary = excluded.summary,
            enriched_at = excluded.enriched_at
    """)
    print(f"  OK {pid}: wrote {len(summary_text)}-char summary from {page['title']}")
    return True


def enrich_players(player_ids: list[str]) -> int:
    """Enrich existing profile rows for the given player ids, by name.

    Looks each id up in gold.player_profiles and refreshes its Wikipedia
    enrichment columns (summary/enriched_at) using the profile's
    display name. Profiles that already carry a non-empty summary are skipped
    so re-running ingest/seed does not re-fetch Wikipedia. Best-effort: ids
    without a profile row, or where Wikipedia has no page or returns an empty
    summary, are skipped. Returns the count of profiles with a non-empty
    summary written.
    """
    if not player_ids:
        return 0
    conn = get_conn()
    rows = conn.execute(
        f"SELECT player_id, COALESCE(display_name, atp_name) AS name, summary "
        f"FROM {PROFILES_TABLE} "
        f"WHERE player_id IN ({', '.join('?' * len(player_ids))})",
        player_ids,
    ).fetchall()
    enriched = 0
    for pid, name, summary in rows:
        if not name:
            print(f"  SKIP {pid}: no profile name for enrichment")
            continue
        if summary and summary.strip():
            print(f"  SKIP {pid}: already enriched")
            continue
        try:
            if enrich_player(name, pid):
                enriched += 1
        except Exception as e:
            print(f"  ERROR {pid} ({name}): {e}")
    return enriched


def enrich_missing() -> int:
    """Find all players in gold missing from profiles, fetch from Wikipedia.

    Players without a profile row have no stored name, so the raw player id is
    used as the search key (best effort). Returns count of profiles inserted.
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

    df = load_atp_csv(str(csv_path))
    print(f"Loaded {len(df)} bronze rows from {csv_path.name}")

    inserted = insert_bronze_rows(df)
    print(f"Inserted {inserted} rows into {BRONZE_TABLE}")

    player_ids = sorted(set(df["player1_id"]) | set(df["player2_id"]))
    load_profiles_for(player_ids, "ingested")

    enriched = enrich_players(player_ids)
    print(f"Enriched {enriched} player profiles with non-empty summaries")
