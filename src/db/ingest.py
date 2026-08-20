"""Ingest ATP CSVs into bronze and best-effort enrich player profiles."""

from __future__ import annotations

import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, LiteralString, cast

import pandas as pd
import requests

from src.constants import (
    BRONZE_MATCHES_TABLE,
    BRONZE_PROFILES_TABLE,
    BRONZE_RANKINGS_TABLE,
    ENRICH_WORKERS,
    GOLD_MATCHES_TABLE,
    ROOT,
)
from src.countries import UNK, valid_ioc
from src.db.client import connection, to_dataframe
from src.features.columns import BRONZE_COLUMNS, CANONICAL_SURFACES
from src.features.validate import run_ingestion_checks
from src.utils import load_env

# ATP data provides player identity; Wikipedia adds missing enrichment.
ATP_DATABASE_CSV = ROOT / "data" / "ATP_player_database.csv"

# Reviewed ranking identity map: explicit source-id assignments take precedence.
# Unmapped source ids are resolved automatically from source/canonical names.
RANKING_PLAYER_MAP_CSV = ROOT / "data" / "ranking_player_map.csv"
RANKING_MAP_COLUMNS = ["ranking_player_id", "ranking_name", "player_id"]

# ATP metadata columns; Wikipedia owns summary and enriched_at.
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

# Columns every raw ATP match row carries (see data/column_glossary.md).
RAW_ATP_COLUMNS = [
    "tourney_id",
    "tourney_name",
    "tourney_date",
    "match_num",
    "winner_id",
    "loser_id",
    "winner_rank",
    "winner_rank_points",
    "winner_age",
    "loser_rank",
    "loser_rank_points",
    "loser_age",
    "tourney_level",
    "round",
    "surface",
    "score",
    "indoor",
    "w_ace",
    "w_df",
    "w_svpt",
    "w_1stIn",
    "w_1stWon",
    "w_2ndWon",
    "w_SvGms",
    "w_bpSaved",
    "w_bpFaced",
    "l_ace",
    "l_df",
    "l_svpt",
    "l_1stIn",
    "l_1stWon",
    "l_2ndWon",
    "l_SvGms",
    "l_bpSaved",
    "l_bpFaced",
]

# Raw ATP level to bronze tournament; non-tier events encode to 0 later.
LEVEL_MAP = {
    "G": "grand_slam",
    "M": "masters",
    "A": "atp_500",
    "500": "atp_500",
    "250": "atp_250",
    "D": "davis_cup",
    "F": "atp_finals",
    "O": "olympics",
    "P": "professional",
}

# Rolling-form window for wins_last_10 / matches_last_10.
RECENT = 10

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "TennisML/0.1 (research project; contact@tennis-ml.local)"

# Wikipedia bios stay brief enough for the profile view.
SUMMARY_MAX_CHARS = 1000

# ── Official Rankings Ingest ───────────────────────────────────

RANKINGS_DIR = ROOT / "data" / "raw" / "rankings"
ATP_PLAYERS_CSV = RANKINGS_DIR / "atp_players.csv"

# Only atp_rankings_*.csv are discovered; atp_players.csv is metadata.
RANKINGS_GLOB = "atp_rankings_*.csv"

# Documented raw ranking shape. `player` is the ATP ranking source id; `points`
# is empty (NULL) in early eras.
RANKINGS_COLUMNS = ["ranking_date", "rank", "player", "points"]
RANKING_TARGET_COLUMNS = ["ranking_date", "player_id", "rank", "points"]
PLAYER_ID_RE = re.compile(r"^\d+$")

load_env()

# ── Raw ATP → Bronze transform (shared with seed.py) ──────────────

# Canonical match id rule: the year derived from the match date is prepended to
# the opaque tourney_id only when that same year is not already repeated at the
# id's start — "2026-418" + 2026 stays "2026-418" (never "2026-2026-418"),
# while "1987-foo" + 2026 -> "2026-1987-foo". Shared by bronze ingestion, the
# seed selection filter, and the match scrape so every path derives the same id.


def canonical_match_id(tourney_id: str, match_num: int, year: int | None = None) -> str:
    """Canonical Sackmann match id ``YYYY-TOURNAMENT_ID-NNN``, date-free.

    The date-derived ``year`` is prepended once, and only when the tourney_id
    does not already repeat that same year at its start (``2026-418`` + 2026
    stays ``2026-418``; ``1987`` + 2026 -> ``2026-1987``). Any other four-digit
    start is a different year and still gets the prefix — the check is against
    the derived year, never "starts with any four digits". A date-like year
    (``19670220``) is reduced to its four-digit edition year (``1967``), so the
    id never embeds a YYYYMMDD. The id is opaque: internal dashes/nonstandard
    Davis Cup ids pass through untouched, never parsed as numeric. ``match_num``
    is zero-padded to three digits.
    """
    tid = str(tourney_id).strip()
    if year is not None:
        year = int(year)
        if year > 9999:
            year //= 10000
    if year is None or tid.startswith(str(year)):
        return f"{tid}-{int(match_num):03d}"
    return f"{year}-{tid}-{int(match_num):03d}"


def _stat(row: dict[str, Any], key: str) -> int:
    """Int stat value; 0 for empty/NaN (raw ATP CSVs leave stats empty)."""
    try:
        return int(row[key])
    except (TypeError, ValueError):
        return 0


def _rank(row: dict[str, Any], key: str) -> int | None:
    """Official match-time rank; zero/blank means unknown, never rank #0."""
    try:
        value = int(row[key])
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _float_stat(row: dict[str, Any], key: str) -> float:
    """Float stat value; 0.0 for empty/NaN (raw ages like `24.41` preserved)."""
    try:
        return float(row[key])
    except (TypeError, ValueError):
        return 0.0


def _normalize_indoor(value: Any) -> int | None:
    """Normalize raw ATP indoor field to 1 (indoor), 0 (outdoor), or None (unknown).

    Raw CSVs use 'I' for indoor, 'O' for outdoor; empty/NaN/other = unknown.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, int):
        return value
    s = str(value).strip().upper()
    if s == "I":
        return 1
    if s == "O":
        return 0
    return None


def _canonical_surface(value: Any) -> str:
    """Map a raw source surface to one of the four canonicals; unknown -> 'hard'."""
    if isinstance(value, str):
        text = value.strip().lower()
    elif isinstance(value, int | float) and not pd.isna(value):
        text = str(int(value))
    else:
        text = ""
    return text if text in CANONICAL_SURFACES else "hard"


def _score(m: dict[str, Any]) -> str | None:
    """Winner-perspective set score with tiebreak digits stripped.

    Raw CSVs put the winner's games first in every set (``6-4 7-6(5)``); the
    tiebreak point total in parentheses is display noise, so it is removed at
    ingest (``6-4 7-6``). Empty/0 cells mean no score recorded and map to NULL.
    """
    value = m.get("score")
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if text in ("", "0", "0.0", "nan"):
        return None
    return re.sub(r"\(\d+\)", "", text).strip()


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
    """Map ATP rows to bronze using full-history, strictly-prior form."""
    if not rows:
        return pd.DataFrame({col: [] for col in BRONZE_COLUMNS})
    rows = sorted(rows, key=lambda m: (int(m["tourney_date"]), m["tourney_id"], m["match_num"]))
    history = player_history(rows)

    out = []
    for m in rows:
        # Canonical, date-free id from the shared rule: an already year-prefixed
        # tourney_id (e.g. "2026-418") is kept verbatim; a bare one gets the
        # edition year once, so the raw CSV and a match scrape always agree.
        match_id = canonical_match_id(
            m["tourney_id"], int(m["match_num"]), int(m["tourney_date"]) // 10000
        )
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
                "tournament_name": str(m["tourney_name"]),
                "round": str(m["round"]).lower(),
                "surface": _canonical_surface(m["surface"]),
                "score": _score(m),
                "is_indoor": _normalize_indoor(m.get("indoor")),
                "player1_ranking": _rank(m, "winner_rank"),
                "player2_ranking": _rank(m, "loser_rank"),
                "player1_wins_last_10": winner_wins,
                "player1_matches_last_10": winner_matches,
                "player1_aces": _stat(m, "w_ace"),
                "player1_double_faults": _stat(m, "w_df"),
                "player1_first_serves_made": _stat(m, "w_1stIn"),
                "player1_total_serve_points": _stat(m, "w_svpt"),
                "player1_first_serve_points_won": _stat(m, "w_1stWon"),
                "player1_second_serve_points_won": _stat(m, "w_2ndWon"),
                "player1_service_games": _stat(m, "w_SvGms"),
                "player1_break_points_saved": _stat(m, "w_bpSaved"),
                "player1_break_points_faced": _stat(m, "w_bpFaced"),
                "player2_wins_last_10": loser_wins,
                "player2_matches_last_10": loser_matches,
                "player2_aces": _stat(m, "l_ace"),
                "player2_double_faults": _stat(m, "l_df"),
                "player2_first_serves_made": _stat(m, "l_1stIn"),
                "player2_total_serve_points": _stat(m, "l_svpt"),
                "player2_first_serve_points_won": _stat(m, "l_1stWon"),
                "player2_second_serve_points_won": _stat(m, "l_2ndWon"),
                "player2_service_games": _stat(m, "l_SvGms"),
                "player2_break_points_saved": _stat(m, "l_bpSaved"),
                "player2_break_points_faced": _stat(m, "l_bpFaced"),
                "player1_rank_points": _stat(m, "winner_rank_points"),
                "player2_rank_points": _stat(m, "loser_rank_points"),
                "player1_age": _float_stat(m, "winner_age"),
                "player2_age": _float_stat(m, "loser_age"),
                "winner_id": m["winner_id"],
            }
        )

    return pd.DataFrame({col: [r[col] for r in out] for col in BRONZE_COLUMNS})


# ── CSV Loading ──────────────────────────────────────────────────


def load_raw_atp_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read eligible ATP rows; require indoor data and preserve zero rank markers."""
    df = pd.read_csv(path)
    missing = set(RAW_ATP_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing raw ATP columns: {missing}")
    df = cast(pd.DataFrame, df[RAW_ATP_COLUMNS])
    eligible = df["winner_id"].notna() & df["loser_id"].notna()
    df = cast(pd.DataFrame, df.loc[eligible])
    return cast(list[dict[str, Any]], df.fillna(0).to_dict(orient="records"))


def load_atp_csv(path: str | Path) -> pd.DataFrame:
    """Read a raw ATP-format CSV and return bronze.match_events rows."""
    return atp_rows_to_bronze(load_raw_atp_rows(path))


def _copy_row_value(value: Any) -> Any:
    """Convert pandas nulls, integral floats, and timestamps for binary COPY."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.date()
    return value


def _copy_df_into(
    table: str,
    df: pd.DataFrame,
    *,
    conflict_col: str,
    update_cols: list[str] | None = None,
) -> int:
    """COPY via a transactional stage table so INSERT can apply ON CONFLICT.

    Returns the number of rows actually inserted/updated, as reported by the
    database (DO NOTHING skips existing PKs; DO UPDATE counts overwrites).
    """
    columns = list(df.columns)
    columns_sql = ", ".join(columns)
    with connection() as conn, conn.transaction(), conn.cursor() as cur:
        cur.execute(cast(LiteralString, f"CREATE TEMP TABLE stage (LIKE {table}) ON COMMIT DROP"))
        # NaN is pandas' NULL marker; COPY binary adapts Python None as SQL NULL.
        records = df.where(pd.notnull(df), None)
        with cur.copy(cast(LiteralString, f"COPY stage ({columns_sql}) FROM STDIN")) as copy:
            for row in records.itertuples(index=False, name=None):
                copy.write_row([_copy_row_value(v) for v in row])
        if update_cols is None:
            cur.execute(
                cast(
                    LiteralString,
                    f"INSERT INTO {table} ({columns_sql}) SELECT {columns_sql} FROM stage "
                    f"ON CONFLICT ({conflict_col}) DO NOTHING",
                )
            )
        else:
            updates = ", ".join(f"{col} = excluded.{col}" for col in update_cols)
            cur.execute(
                cast(
                    LiteralString,
                    f"INSERT INTO {table} ({columns_sql}) SELECT {columns_sql} FROM stage "
                    f"ON CONFLICT ({conflict_col}) DO UPDATE SET {updates}",
                )
            )
        return int(cur.rowcount or 0)


def clear_match_events() -> None:
    """Delete every bronze.match_events row inside one transaction.

    Called by the seed's --force path so the corpus inserts into an empty
    table; the table/schema itself is never dropped.
    """
    with connection() as conn, conn.transaction(), conn.cursor() as cur:
        cur.execute(cast(LiteralString, f"DELETE FROM {BRONZE_MATCHES_TABLE}"))


def insert_bronze_rows(df: pd.DataFrame, *, overwrite: bool = False) -> int:
    """Insert bronze.match_events rows from a DataFrame; returns the number of
    rows actually inserted (0 when every row already exists and overwrite is
    False).

    Shared by the ingest CLI and the dev seed so both paths use one INSERT.
    match_id is the PK: ingestion skips an existing match_id (DO NOTHING).
    The seed's --force path clears bronze.match_events first (see
    clear_match_events) and inserts fresh rows; overwrite=True stays available
    for callers that want full-row replacement (DO UPDATE).
    """
    df = df.copy()
    # Canonical surface boundary: absent/0/unmapped source values become hard
    # before validation and insertion, so no non-canonical value reaches bronze.
    df["surface"] = df["surface"].map(_canonical_surface)
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

    return _copy_df_into(
        BRONZE_MATCHES_TABLE,
        cast(pd.DataFrame, valid_df[list(BRONZE_COLUMNS)]),
        conflict_col="match_id",
        update_cols=[c for c in BRONZE_COLUMNS if c != "match_id"] if overwrite else None,
    )


def load_profiles_for(player_ids: list[str], label: str, force: bool = False) -> int:
    """Load ATP identities for player_ids and print status (shared with seed).

    `label` names the caller in the status line ("seeded"/"ingested"). Returns
    the number of profiles actually written (0 when the ATP database file is
    absent). Idempotent by default (existing player_id rows are skipped);
    force=True overwrites them.
    """
    if not ATP_DATABASE_CSV.exists():
        print(f"ATP database not found at {ATP_DATABASE_CSV}, skipping identity load")
        return 0
    loaded = load_atp_profiles(ATP_DATABASE_CSV, player_ids=set(player_ids), force=force)
    requested = len(set(player_ids))
    if force:
        print(f"Loaded {loaded} player profiles for {requested} {label} players (overwrite)")
    else:
        skipped = requested - loaded
        print(
            f"Loaded {loaded} player profiles for {requested} {label} players ({skipped} skipped existing)"
        )
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
    csv_path: str | Path = ATP_DATABASE_CSV,
    player_ids: set[str] | None = None,
    force: bool = False,
) -> int:
    """Load typed ATP identity metadata while preserving Wikipedia enrichment.

    Idempotent by default: an existing player_id row is skipped (DO NOTHING).
    Pass force=True to overwrite ATP identity fields of existing rows (DO
    UPDATE) — enrichment fields (summary, enriched_at) are never touched.
    Returns the number of profiles actually inserted/updated.
    """
    atp = pd.read_csv(csv_path, dtype=str)
    if not {"id", "player", "atpname", "hand", "backhand", "ioc"} <= set(atp.columns):
        raise ValueError(f"ATP database CSV missing expected columns: {csv_path}")
    if player_ids is not None:
        atp = atp[atp["id"].isin(list(player_ids))]
    # Dedupe before upsert so the count reflects distinct player ids.
    atp = atp[~cast(pd.Series, atp["id"]).duplicated(keep="last")]

    ids = cast(pd.Series, atp["id"])
    # IOC codes are trimmed/uppercased; verified codes are preserved and
    # missing/invalid values fall back to the UNK sentinel (see src/countries).
    ioc = cast(pd.Series, atp["ioc"]).map(valid_ioc)
    unresolved = int((ioc == UNK).sum())
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
            "ioc": ioc,
        }
    )

    inserted = _copy_df_into(
        BRONZE_PROFILES_TABLE,
        cast(pd.DataFrame, df[ATP_PROFILE_COLUMNS]),
        conflict_col="player_id",
        update_cols=[c for c in ATP_PROFILE_COLUMNS if c != "player_id"] if force else None,
    )
    if unresolved:
        print(f"  IOC: {unresolved}/{len(df)} profiles unresolved (missing/invalid -> {UNK})")
    return inserted


# ── Ranking Identity Map ───────────────────────────────────────


def canonical_players(csv_path: str | Path = ATP_DATABASE_CSV) -> dict[str, str]:
    """{canonical player_id: display_name} from the canonical profile reference.

    The canonical id space is the profiles' ATP_Database.id space (the same ids
    match events and player profiles are keyed on). This is the review target
    and the validation reference for ranking map targets.
    """
    df = pd.read_csv(csv_path, dtype=str)
    if not {"id", "player"} <= set(df.columns):
        raise ValueError(f"canonical player reference CSV missing columns: {csv_path}")
    return {
        str(pid).strip(): (name or "").strip()
        for pid, name in zip(df["id"], df["player"], strict=True)
    }


def _metadata_csv_value(value: Any) -> str:
    """Non-empty cell from the player-reference CSV; '' for missing/NaN."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def load_player_metadata(
    csv_path: str | Path = ATP_DATABASE_CSV,
) -> dict[str, dict[str, str]]:
    """In-memory {uppercase player_id: {display_name, hand, height, ioc}}.

    Reads the same local ATP reference ``canonical_players`` does, loaded once
    per flow run — never a per-match query. Values pass through in Sackmann
    vocabulary when the CSV has them (hand R/L/A, height in cm, IOC code) and
    stay blank when absent; the CSV's 0 height marker is treated as missing.
    Players without a row are simply absent from the map.
    """
    df = pd.read_csv(csv_path, dtype=str)
    if not {"id", "player", "hand", "height", "ioc"} <= set(df.columns):
        raise ValueError(f"player metadata CSV missing expected columns: {csv_path}")
    profiles: dict[str, dict[str, str]] = {}
    for record in df.to_dict(orient="records"):
        player_id = _metadata_csv_value(record.get("id")).upper()
        if not player_id:
            continue
        height = _metadata_csv_value(record.get("height"))
        profiles[player_id] = {
            "display_name": _metadata_csv_value(record.get("player")),
            "hand": _metadata_csv_value(record.get("hand")),
            "height": "" if height == "0" else height,
            "ioc": _metadata_csv_value(record.get("ioc")).upper(),
        }
    return profiles


def _normalize_name(name: str) -> str:
    """Deterministic normalized name for review-only candidate matching."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _normalize_name_variants(name: str) -> list[str]:
    """Normalized-name variants: original and reversed token order.

    Surname-first sources ("Wu Yibing") normalize differently from the
    canonical given-first form ("Yibing Wu"), so identity matching tries both
    orientations. Single-token names yield only the original.
    """
    tokens = [t for t in name.split() if t]
    variants = [_normalize_name(" ".join(tokens))]
    if len(tokens) >= 2:
        variants.append(_normalize_name(" ".join(reversed(tokens))))
    return variants


def load_ranking_player_map(
    csv_path: str | Path = RANKING_PLAYER_MAP_CSV,
    canonical_ids: set[str] | None = None,
) -> dict[str, str]:
    """Load and validate the reviewed ranking identity map; returns
    {ranking_player_id: player_id}.

    The map is authoritative by source id; ranking_name is an audit/review field
    and is never used for a production write. Raises ValueError on an invalid map
    before any rows are written: a missing column, an empty required cell, a
    duplicated source id, a canonical player_id targeted by more than one row, or
    a target id absent from the canonical player reference.
    """
    df = pd.read_csv(csv_path, dtype=str)
    missing = set(RANKING_MAP_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"ranking player map missing columns: {sorted(missing)}")
    df = df[list(RANKING_MAP_COLUMNS)]
    for col in RANKING_MAP_COLUMNS:
        df[col] = df[col].fillna("").astype(str).str.strip()

    for col in RANKING_MAP_COLUMNS:
        empties = df[col].eq("")
        if empties.any():
            raise ValueError(f"ranking player map has empty {col} rows")

    dup_src = df.loc[df["ranking_player_id"].duplicated(keep=False), "ranking_player_id"]
    if not dup_src.empty:
        raise ValueError(f"duplicate ranking source ids (mapped by >1 row): {sorted(set(dup_src))}")

    # Multiple ranking source ids may legitimately map to the same canonical
    # player id (separate entries in different ranking files, name variants, etc).
    # Only reject the same source id mapping to different targets (caught above).
    dup_src_conflict = df.groupby("ranking_player_id")["player_id"].nunique().loc[lambda x: x > 1]
    if not dup_src_conflict.empty:
        raise ValueError(
            f"conflicting ranking map (same source -> different targets): "
            f"{sorted(dup_src_conflict.index)}"
        )

    if canonical_ids is None:
        canonical_ids = set(canonical_players())
    unknown = sorted(set(df["player_id"]) - canonical_ids)
    if unknown:
        raise ValueError(f"unknown canonical player ids in ranking map: {unknown}")

    return dict(zip(df["ranking_player_id"], df["player_id"], strict=True))


def ranking_name_candidates(
    ranking_rows: list[dict[str, Any]],
    canonical: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Deterministic normalized-name candidate report for maintainer review.

    For each distinct ranking source row this reports the canonical players whose
    exact name (case-insensitive) or normalized name matches the ranking name,
    flagging `ambiguous` when a normalized name matches more than one canonical
    player. Review-only: it never writes and is never used as a production mapping.
    """
    if canonical is None:
        canonical = canonical_players()
    exact: dict[str, list[str]] = {}
    norm: dict[str, list[str]] = {}
    for pid, name in canonical.items():
        exact.setdefault(name.lower(), []).append(pid)
        norm.setdefault(_normalize_name(name), []).append(pid)

    report: dict[str, dict[str, Any]] = {}
    for row in ranking_rows:
        src = str(row["ranking_player_id"]).strip()
        name = str(row.get("ranking_name") or "").strip()
        entry = report.setdefault(
            src,
            {
                "ranking_player_id": src,
                "ranking_name": name,
                "exact_candidates": [],
                "normalized_candidates": [],
                "ambiguous": False,
            },
        )
        entry["exact_candidates"] = sorted(exact.get(name.lower(), []))
        normalized = sorted(norm.get(_normalize_name(name), []))
        entry["normalized_candidates"] = normalized
        entry["ambiguous"] = len(normalized) > 1
    return [report[k] for k in sorted(report)]


def unmapped_ranking_rows(
    ranking_rows: list[dict[str, Any]], rank_map: dict[str, str]
) -> list[dict[str, Any]]:
    """Ranking source rows not covered by the approved map: id, audit name, count.

    Unmapped rows do not fail an import; this report surfaces them so a maintainer
    can review and extend the approved map.
    """
    counts: dict[str, dict[str, Any]] = {}
    for row in ranking_rows:
        src = str(row["ranking_player_id"]).strip()
        if src in rank_map:
            continue
        entry = counts.setdefault(
            src,
            {
                "ranking_player_id": src,
                "ranking_name": str(row.get("ranking_name") or "").strip(),
                "count": 0,
            },
        )
        entry["count"] += 1
    return [counts[k] for k in sorted(counts)]


def _canonical_corpus_stats(
    match_rows: list[dict[str, Any]] | None,
) -> dict[str, dict[str, int | None]]:
    """{canonical player_id: {matches, best_rank}} from raw ATP match rows.

    match activity = appearances as winner or loser; best_rank = lowest positive
    winner/loser rank. Used only to break name-matching ties for identity choice,
    never to synthesize rank history. None/empty rows yield empty stats.
    """
    stats: dict[str, dict[str, int | None]] = {}
    for row in match_rows or []:
        for side in ("winner", "loser"):
            pid = str(row.get(f"{side}_id") or "").strip()
            if not pid:
                continue
            entry = stats.setdefault(pid, {"matches": 0, "best_rank": None})
            entry["matches"] = (entry["matches"] or 0) + 1
            try:
                raw = row.get(f"{side}_rank")
                rank = int(raw) if isinstance(raw, (int, str, float)) else 0
            except (TypeError, ValueError):
                rank = 0
            if rank > 0:
                best = entry["best_rank"]
                entry["best_rank"] = rank if best is None else min(best, rank)
    return stats


def resolve_ranking_identities(
    source_ids: set[str] | list[str],
    source_names: dict[str, str],
    canonical: dict[str, str],
    corpus_stats: dict[str, dict[str, int | None]] | None = None,
) -> dict[str, str]:
    """Auto-map ranking source ids to canonical ids by normalized name.

    Deterministic resolution for source ids absent from the approved map: a
    Exact normalized names map directly. Otherwise the closest normalized name
    is accepted when it has at least 80% character similarity. Ties resolve by
    greatest match activity, lower best rank, then lexicographic player_id.
    Returns {ranking source id: canonical player id}; ids with no usable source
    name are left out. Never consults the map (explicit entries always win by
    construction — callers resolve gaps only).
    """
    norm_index: dict[str, list[str]] = {}
    for pid, name in canonical.items():
        norm = _normalize_name(name)
        if norm:
            norm_index.setdefault(norm, []).append(pid)

    stats = corpus_stats or {}
    resolved: dict[str, str] = {}
    for src_id in sorted({str(s).strip() for s in source_ids}):
        variants = _normalize_name_variants(source_names.get(src_id, ""))
        candidates: list[str] = []
        for norm in variants:
            candidates = norm_index.get(norm, [])
            if candidates:
                break
        if not candidates:
            if not any(variants):
                continue
            scores = {
                pid: max(SequenceMatcher(None, v, _normalize_name(name)).ratio() for v in variants)
                for pid, name in canonical.items()
            }
            best_score = max(scores.values(), default=0.0)
            if best_score < 0.8:
                continue
            candidates = [pid for pid, score in scores.items() if score == best_score]
        resolved[src_id] = min(
            candidates,
            key=lambda pid: (
                -(stats.get(pid, {}).get("matches") or 0),
                stats.get(pid, {}).get("best_rank") or 10**9,
                pid,
            ),
        )
    return resolved


def _name_matches_seed(name: str, player_ids: set[str], canonical: dict[str, str]) -> bool:
    """Whether a source name normalized-matches one of the seeded canonical players."""
    norm = _normalize_name(name)
    if not norm:
        return False
    return any(_normalize_name(canonical.get(pid, "")) == norm for pid in player_ids)


def discover_ranking_csvs(rankings_dir: Path = RANKINGS_DIR) -> list[Path]:
    """Discover data/raw/rankings/atp_rankings_*.csv files, sorted by name."""
    return sorted(p for p in rankings_dir.glob(RANKINGS_GLOB) if p.is_file())


def load_ranking_rows(
    csv_paths: list[Path],
    rank_limit: int | None = None,
) -> pd.DataFrame:
    """Read and combine ranking CSVs into one validated, typed frame.

    Raises ValueError on any malformed input before a row is returned: a file
    missing the exact four documented columns, an empty/unparseable
    ranking_date (YYYYMMDD), a non-integer or non-positive rank, a non-integer
    player id, or non-empty non-integer points. Empty points are allowed (NULL).

    rank_limit drops rows with rank > limit before validation; dropped rows are
    outside the requested scope and never validated.
    """
    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        df = pd.read_csv(path, dtype=str)
        if list(df.columns) != RANKINGS_COLUMNS:
            raise ValueError(
                f"{path.name}: expected columns {RANKINGS_COLUMNS}, got {list(df.columns)}"
            )
        for col in RANKINGS_COLUMNS:
            df[col] = df[col].fillna("").astype(str).str.strip()
        if rank_limit is not None:
            rank_num = pd.to_numeric(df["rank"].mask(df["rank"].eq("")), errors="coerce")
            df = df.loc[rank_num.notna() & rank_num.le(rank_limit)]
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    if raw.empty:
        return pd.DataFrame(columns=RANKING_TARGET_COLUMNS)

    date_s = raw["ranking_date"]
    parsed = pd.to_datetime(date_s.mask(date_s.eq("")), format="%Y%m%d", errors="coerce")
    bad_date = parsed.isna()
    if bad_date.any():
        offenders = ", ".join(
            f"{i}: {v!r}" for i, v in zip(raw.index[bad_date], date_s[bad_date], strict=True)
        )
        raise ValueError(f"ranking_date malformed (expected YYYYMMDD): {offenders}")

    rank_s = raw["rank"]
    rank_num = pd.to_numeric(rank_s.mask(rank_s.eq("")), errors="coerce")
    bad_rank = ~(rank_num.notna() & (rank_num >= 1) & (rank_num == rank_num.round()))
    if bad_rank.any():
        offenders = ", ".join(
            f"{i}: {v!r}" for i, v in zip(raw.index[bad_rank], rank_s[bad_rank], strict=True)
        )
        raise ValueError(f"rank malformed (expected integer >= 1): {offenders}")

    player_s = raw["player"]
    bad_player = ~player_s.str.fullmatch(PLAYER_ID_RE)
    if bad_player.any():
        offenders = ", ".join(
            f"{i}: {v!r}" for i, v in zip(raw.index[bad_player], player_s[bad_player], strict=True)
        )
        raise ValueError(f"player malformed (expected integer id): {offenders}")

    pts_s = raw["points"]
    pts_num = pd.to_numeric(pts_s.mask(pts_s.eq("")), errors="coerce")
    bad_pts = pts_s.ne("") & ~(pts_num.notna() & (pts_num == pts_num.round()))
    if bad_pts.any():
        offenders = ", ".join(
            f"{i}: {v!r}" for i, v in zip(raw.index[bad_pts], pts_s[bad_pts], strict=True)
        )
        raise ValueError(f"points malformed (expected integer or empty): {offenders}")

    return pd.DataFrame(
        {
            "ranking_date": parsed,
            "player_id": player_s,
            "rank": rank_num.astype("Int64"),
            "points": pts_num.astype("Int64"),
        }
    )


def _atp_players_names(players_csv: Path) -> dict[str, str]:
    """{ranking source player id: display name} from data/raw/rankings/atp_players.csv."""
    df = pd.read_csv(players_csv, dtype=str)
    if not {"player_id", "name_first", "name_last"} <= set(df.columns):
        raise ValueError(f"atp_players.csv missing expected columns: {players_csv}")

    def cell(value: Any) -> str:
        return "" if value is None or pd.isna(value) else str(value).strip()

    names: dict[str, str] = {}
    for pid, first, last in zip(df["player_id"], df["name_first"], df["name_last"], strict=True):
        key = cell(pid)
        if not key:
            continue
        names[key] = f"{cell(first)} {cell(last)}".strip()
    return names


def _unmapped_report(
    top200: pd.DataFrame, rank_map: dict[str, str], players_csv: Path
) -> list[dict[str, Any]]:
    """Top-200 source rows whose player id is absent from the approved map.

    Grouped by source id with the atp_players display name and skipped-row
    count, so the import report is actionable for maintainers extending the map.
    """
    names = _atp_players_names(players_csv)
    counts: dict[str, dict[str, Any]] = {}
    for pid in top200["player_id"]:
        pid = str(pid).strip()
        if pid in rank_map:
            continue
        entry = counts.setdefault(
            pid,
            {"ranking_player_id": pid, "ranking_name": names.get(pid, ""), "count": 0},
        )
        entry["count"] += 1
    return [counts[k] for k in sorted(counts)]


def _atp_players_iocs(players_csv: Path) -> dict[str, str]:
    """{ranking source player id: normalized ioc} from atp_players.csv."""
    df = pd.read_csv(players_csv, dtype=str)
    if not {"player_id", "ioc"} <= set(df.columns):
        raise ValueError(f"atp_players.csv missing expected columns: {players_csv}")
    iocs: dict[str, str] = {}
    for pid, ioc in zip(df["player_id"], df["ioc"], strict=True):
        key = "" if pid is None or pd.isna(pid) else str(pid).strip()
        if not key:
            continue
        iocs[key] = valid_ioc(ioc)
    return iocs


def backfill_profile_iocs(
    rank_map: dict[str, str],
    player_ids: set[str],
    players_csv: Path = ATP_PLAYERS_CSV,
) -> None:
    """Backfill bronze.player_profiles.ioc for the given canonical player ids.

    atp_players.csv is the higher-confidence IOC source: only players present in
    the approved map are touched, and only a valid code replaces the UNK
    sentinel/empty value — a verified IOC is never overwritten. The caller
    derives the concrete player set (the seed's match corpus, or every mapped
    canonical id for a full import).
    """
    iocs = _atp_players_iocs(players_csv)
    updates: list[tuple[str, str]] = []
    for src_id, canonical in rank_map.items():
        if canonical not in player_ids:
            continue
        ioc = iocs.get(src_id)
        if ioc is not None and ioc != UNK:
            updates.append((ioc, canonical))
    if not updates:
        return
    with connection() as conn, conn.transaction(), conn.cursor() as cur:
        cur.executemany(
            cast(
                LiteralString,
                f"UPDATE {BRONZE_PROFILES_TABLE} SET ioc = %s "
                f"WHERE player_id = %s AND (ioc IS NULL OR ioc = '' OR ioc = 'UNK')",
            ),
            updates,
        )


def ingest_rankings(
    rankings_dir: Path = RANKINGS_DIR,
    map_csv: Path = RANKING_PLAYER_MAP_CSV,
    players_csv: Path = ATP_PLAYERS_CSV,
    player_ids: set[str] | None = None,
    force: bool = False,
    match_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Official-rankings import; returns the import summary.

    Discover -> validate -> filter rank <= 200 -> map to canonical ids -> upsert.
    Only the archive's top-200 rows are read (ranks > 200 are ignored). Raw
    ranking source ids never reach the table: player_id is resolved through the
    approved identity map, and source ids absent from the map are auto-mapped by
    normalized name (deterministic: unique candidate, else greatest match
    activity, then lower best rank, then lexicographic player_id, using
    match_rows for the activity/rank tie-break). Explicit map entries always
    win; auto-mapping only chooses identity and never invents ranking values —
    only official rank <= 200 archive rows are ingested. Source ids that still
    cannot be resolved are reported and skipped. Raises ValueError on malformed
    input or an invalid map before any database write.

    Idempotent by default: an existing (ranking_date, player_id) row is skipped
    (DO NOTHING). Pass force=True to overwrite existing rows (DO UPDATE).

    player_ids restricts the import to those canonical player ids — the seed
    passes its exact match-corpus player set; None imports every mapped top-200
    row. The filtered seed path is silent about global archive gaps and instead
    returns a `coverage` summary — {seeded, covered, auto_mapped, unresolved} —
    for the seed's coverage report. Only the full import (player_ids=None)
    reports global unmapped/auto-mapped rows.

    The ranking-source IOC fallback (atp_players.csv) always runs against a
    concrete player set — the selected seed ids when supplied, otherwise every
    mapped canonical id — and only fills NULL/empty/UNK profile IOCs; a
    verified IOC is never overwritten.
    """
    csv_paths = discover_ranking_csvs(rankings_dir)
    if not csv_paths:
        print("No atp_rankings_*.csv files found under data/raw/rankings; nothing to import")
        return {
            "files": 0,
            "source_rows": 0,
            "top200": 0,
            "upserted": 0,
            "skipped_existing": 0,
            "unmapped": 0,
        }

    rows = load_ranking_rows(csv_paths, rank_limit=200)
    top200 = cast(pd.DataFrame, rows)
    source_rows = len(rows)
    top200_count = len(top200)

    rank_map = load_ranking_player_map(map_csv)
    source_names = _atp_players_names(players_csv)
    canonical_ref = canonical_players()
    # Auto-map source ids absent from the approved map. Explicit entries are
    # never re-resolved: auto-mapping fills gaps only, for identity choice.
    missing = sorted({str(x).strip() for x in top200["player_id"]} - set(rank_map))
    auto_map = (
        resolve_ranking_identities(
            missing,
            source_names,
            canonical_ref,
            _canonical_corpus_stats(match_rows),
        )
        if missing
        else {}
    )
    resolve = {**rank_map, **auto_map}

    # The filtered seed path is scoped to the seeded set: archive rows outside
    # it are not imports and are never reported. The global unmapped report is
    # a full-import (player_ids=None) concern.
    unmapped: list[dict[str, Any]] = []
    if player_ids is None:
        unmapped = _unmapped_report(top200, rank_map, players_csv)

    canonical = top200["player_id"].map(resolve)
    mapped = cast(pd.DataFrame, top200.loc[canonical.notna()]).copy()
    mapped["player_id"] = canonical[mapped.index]
    retained_top200 = top200_count
    coverage: dict[str, int] | None = None
    if player_ids is not None:
        mapped = cast(pd.DataFrame, mapped[mapped["player_id"].isin(player_ids)])
        retained_top200 = len(mapped)
        # Seed coverage: how many seeded players got official top-200 history,
        # how many source ids auto-mapped for them, and how many seed-relevant
        # source ids (name matching a seeded player) remain unresolved.
        auto_mapped_count = sum(1 for sid, cid in auto_map.items() if cid in player_ids)
        unresolved_ids = {str(s).strip() for s in top200["player_id"]} - set(resolve)
        unresolved_count = sum(
            1
            for sid in unresolved_ids
            if _name_matches_seed(source_names.get(sid, ""), player_ids, canonical_ref)
        )
        coverage = {
            "seeded": len(player_ids),
            "covered": len(set(mapped["player_id"])),
            "auto_mapped": auto_mapped_count,
            "unresolved": unresolved_count,
        }
    # Dedupe on the PK before the upsert: duplicate (date, player) rows would
    # trip PostgreSQL's "cannot affect row a second time" ON CONFLICT rule.
    mapped = cast(
        pd.DataFrame,
        mapped.drop_duplicates(subset=["ranking_date", "player_id"], keep="last"),
    )
    mapped = cast(pd.DataFrame, mapped[RANKING_TARGET_COLUMNS]).reset_index(drop=True)

    upserted = 0
    if not mapped.empty:
        upserted = _copy_df_into(
            BRONZE_RANKINGS_TABLE,
            mapped,
            conflict_col="ranking_date, player_id",
            update_cols=["rank", "points"] if force else None,
        )

    # The IOC fallback is scoped to one concrete player set: the selected seed
    # ids when supplied, otherwise every mapped canonical id (full import). It
    # only fills NULL/empty/UNK profile IOCs — a verified IOC is never
    # overwritten.
    ioc_player_ids = player_ids if player_ids is not None else set(resolve.values())
    backfill_profile_iocs(resolve, ioc_player_ids, players_csv)

    skipped_existing = 0 if force else len(mapped) - upserted
    summary: dict[str, object] = {
        "files": len(csv_paths),
        "source_rows": source_rows,
        "top200": retained_top200,
        "upserted": upserted,
        "skipped_existing": skipped_existing,
        "unmapped": int(sum(u["count"] for u in unmapped)),
        "auto_mapped": len(auto_map),
        "unresolved": len({str(s).strip() for s in top200["player_id"]} - set(resolve)),
    }
    if coverage is not None:
        summary["coverage"] = coverage
    if force:
        print(f"Rankings import: {summary['upserted']} inserted/updated rows (overwrite)")
    else:
        print(
            f"Rankings import: {summary['upserted']} inserted/updated rows "
            f"({summary['skipped_existing']} skipped existing)"
        )
    if player_ids is None:  # full import: report global gaps and auto-maps
        for u in unmapped:
            print(
                f"  unmapped: source_id={u['ranking_player_id']} name={u['ranking_name']!r} "
                f"rows={u['count']}"
            )
        for sid in sorted(auto_map):
            print(
                f"  auto-mapped: source_id={sid} name={source_names.get(sid, '')!r} "
                f"-> {auto_map[sid]}"
            )
    return summary


# ── Wikipedia Profile Enrichment ────────────────────────────────


def get_players_without_summary() -> list[str]:
    """Players in bronze.player_profiles who lack a Wikipedia summary."""
    sql = f"""
        SELECT player_id
        FROM {BRONZE_PROFILES_TABLE}
        WHERE summary IS NULL OR summary = ''
    """
    df = to_dataframe(sql)
    return df["player_id"].tolist()


def _normalized_wiki_name(value: str) -> str:
    value = re.sub(r"\s*\([^)]*\)\s*$", "", value)
    value = "".join(
        char for char in unicodedata.normalize("NFKD", value) if not unicodedata.combining(char)
    )
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _normalized_wiki_variants(value: str) -> list[str]:
    """Normalized wiki-name variants: original and reversed token order.

    Surname-first article titles ("Wu Yibing") normalize differently from the
    given-first player name ("Yibing Wu"), so enrichment tries both.
    """
    base = _normalized_wiki_name(value)
    tokens = base.split()
    variants = [base]
    if len(tokens) >= 2:
        variants.append(" ".join(reversed(tokens)))
    return variants


def search_wikipedia(name: str) -> str | None:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": f"{name} tennis player",
        "format": "json",
        "srlimit": 5,
    }
    resp = requests.get(WIKI_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=10)
    data = resp.json()
    pages = data.get("query", {}).get("search", [])
    expected_variants = _normalized_wiki_variants(name)
    for page in pages:
        title = str(page.get("title", ""))
        if _normalized_wiki_name(title) in expected_variants:
            return title
    return None


def fetch_summary(title: str) -> dict[str, str] | None:
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts|pageprops",
        # Full plaintext extract (not just the intro) so the `Playing style`
        # section is available for paragraph extraction.
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


def extract_playing_style_paragraph(summary: str) -> str | None:
    """First paragraph of the article's `Playing style` section, or None.

    The plaintext Wikipedia extract uses `\n\n`-separated paragraphs and
    `=== Playing style ===` section headers. Returns the first non-empty
    paragraph after the header, trimmed of the header marker; None when the
    section is absent or has no usable paragraph.
    """
    header = re.search(r"^==+\s*Playing style\s*==+\s*$", summary, re.MULTILINE)
    if not header:
        return None
    tail = summary[header.end() :]
    # Stop at the next section header (start of a sibling section).
    body = re.split(r"^==+ .*$", tail, maxsplit=1, flags=re.MULTILINE)[0]
    for paragraph in body.split("\n\n"):
        text = paragraph.strip()
        if text:
            return text
    return None


def extract_lead_paragraph(summary: str) -> str | None:
    """First non-empty paragraph of the article lead (before any header)."""
    lead = re.split(r"^==+ .*$", summary, maxsplit=1, flags=re.MULTILINE)[0]
    for paragraph in lead.split("\n\n"):
        text = paragraph.strip()
        if text:
            return text
    return None


def clean_bio_paragraph(text: str) -> str:
    """Collapse wiki whitespace and avoid cutting a bio mid-sentence."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= SUMMARY_MAX_CHARS:
        return cleaned
    last_period = cleaned.rfind(".", 0, SUMMARY_MAX_CHARS + 1)
    return cleaned[: last_period + 1] if last_period >= 0 else ""


def _fetch_wiki_bio(name: str, pid: str) -> tuple[str, str] | None:
    """Fetch and parse a Wikipedia bio for one player; no DB access.

    Returns ``(summary_text, page_title)`` when a usable bio is found, else
    None (printing the per-player SKIP line). Pure HTTP + string parsing, so it
    is safe to run from worker threads — the batch enricher parallelizes this
    and performs the DB write on the main thread only.
    """
    title = search_wikipedia(name)
    if not title:
        print(f"  SKIP {pid}: no Wikipedia match for {name!r}")
        return None

    page = fetch_summary(title)
    if not page:
        print(f"  SKIP {pid}: no page data for {title!r}")
        return None

    if not page["summary"].strip():
        print(f"  SKIP {pid}: empty Wikipedia summary for {title!r}")
        return None

    # Prefer Playing style; fall back to the lead paragraph.
    bio_paragraph = extract_playing_style_paragraph(page["summary"]) or extract_lead_paragraph(
        page["summary"]
    )
    if not bio_paragraph:
        print(f"  SKIP {pid}: no usable paragraph for {title!r}")
        return None

    # Prepared statement: None binds as NULL, apostrophes need no escaping.
    summary_text = clean_bio_paragraph(bio_paragraph)
    if not summary_text:
        print(f"  SKIP {pid}: no complete sentence within summary limit")
        return None

    return summary_text, page["title"]


def _write_summary(pid: str, summary_text: str, title: str) -> None:
    """Write a fetched bio to bronze.player_profiles (main thread only)."""
    with connection() as conn:
        conn.execute(
            cast(
                LiteralString,
                f"""UPDATE {BRONZE_PROFILES_TABLE}
                SET summary = %s, enriched_at = CURRENT_TIMESTAMP
                WHERE player_id = %s""",
            ),
            [summary_text, pid],
        )
    print(f"  OK {pid}: wrote {len(summary_text)}-char summary from {title}")


def enrich_player(name: str, player_id: str | None = None) -> bool:
    """Upsert a usable Wikipedia bio, preferring the Playing style paragraph."""
    pid = player_id or name
    fetched = _fetch_wiki_bio(name, pid)
    if fetched is None:
        return False
    summary_text, title = fetched
    _write_summary(pid, summary_text, title)
    return True


def enrich_players(player_ids: list[str], force: bool = False) -> int:
    """Best-effort enrich of bronze profiles with Wikipedia bios.

    Idempotent by default: profiles that already have a non-empty summary are
    counted as already enriched and silently skipped, never overwritten. Pass
    force=True to re-fetch and overwrite every summary. Profiles without a
    name are counted as no-name skips and never attempted.

    The slow HTTP fetch + parse runs in a thread pool (ENRICH_WORKERS workers);
    the DB write stays on the main thread on one pooled connection. Per-player
    lines print only for currently enriching (OK) and failed (SKIP/ERROR)
    players; pre-skip categories are summarized without per-player lines. The
    final batch summary distinguishes attempted, already enriched, no name,
    enriched, and failed (no usable bio or exception). Returns the number of
    profiles enriched.
    """
    if not player_ids:
        return 0
    with connection() as conn:
        rows = conn.execute(
            cast(
                LiteralString,
                f"SELECT player_id, COALESCE(display_name, atp_name) AS name, summary "
                f"FROM {BRONZE_PROFILES_TABLE} "
                f"WHERE player_id IN ({', '.join(['%s'] * len(player_ids))})",
            ),
            player_ids,
        ).fetchall()
    enriched = failed = already_enriched = no_name = 0
    to_enrich: list[tuple[str, str]] = []
    for pid, name, summary in rows:
        if not name:
            no_name += 1
            continue
        if not force and summary and summary.strip():
            already_enriched += 1
            continue
        to_enrich.append((pid, name))
    attempted = len(to_enrich)
    if to_enrich:
        with ThreadPoolExecutor(max_workers=min(ENRICH_WORKERS, len(to_enrich))) as pool:
            futures = {
                pool.submit(_fetch_wiki_bio, name, pid): (pid, name) for pid, name in to_enrich
            }
            for future in as_completed(futures):
                pid, name = futures[future]
                try:
                    fetched = future.result()
                except Exception as e:
                    failed += 1
                    print(f"  ERROR {pid} ({name}): {e}")
                    continue
                if fetched is None:
                    failed += 1
                    continue
                summary_text, title = fetched
                _write_summary(pid, summary_text, title)
                enriched += 1
    print(
        f"Enrichment summary: {attempted} attempted, {already_enriched} already enriched, "
        f"{no_name} no name, {enriched} enriched, {failed} failed"
    )
    return enriched


def enrich_missing() -> int:
    """Idempotent Wikipedia enrichment for bronze profiles missing a summary.

    Queries bronze.player_profiles for rows with null/empty summary and fetches
    bios from Wikipedia. Skips profiles that already have a non-empty summary.
    Returns count of profiles enriched.
    """
    missing_summary = get_players_without_summary()
    if not missing_summary:
        print("All profiles have summaries. Nothing to do.")
        return 0

    print(f"Found {len(missing_summary)} profiles without summaries")
    return enrich_players(missing_summary)
