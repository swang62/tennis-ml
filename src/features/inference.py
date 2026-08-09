"""Build canonical, as-of-dated inference rows from PostgreSQL.

Rolling values use snapshots strictly before the requested date. Missing values
use the materialized gold.feature_defaults row at or before the as-of date
(never on-demand AVG/PERCENTILE); SQL values are always passed as parameters.
"""

from __future__ import annotations

import builtins
from datetime import date, datetime
from decimal import Decimal
from numbers import Real
from time import perf_counter
from typing import Any, NamedTuple, cast

import pandas as pd

from src.constants import (
    FEATURE_DEFAULTS_TABLE,
    PROFILES_TABLE,
    SILVER_PLAYER_MATCHES,
    SILVER_ROLLING_FEATURES,
)
from src.db.client import execute_df, first_row_dict
from src.features.columns import FEATURE_COLS, FEATURE_DEFAULTS_COLS

# "0" is the explicit unknown surface marker used by gold; it maps to all-zero
# surface indicator columns and the fixed rate default, matching match_features.
VALID_SURFACES = {"clay", "grass", "hard", "carpet", "0"}
VALID_TOURNAMENT_LEVELS = {0, 1, 2, 3, 4}
VALID_ROUND_ENCODINGS = {0, 1, 2, 3, 4, 5, 6, 7}

# Upper bound for a single bulk inference request (Nginx chunks below this).
BULK_MAX_ROWS = 1000

# String aliases mirror the dbt context codebook; unknown values map to 0.
_TOURNAMENT_LEVELS = {
    "grand_slam": 4,
    "masters": 3,
    "atp_500": 2,
    "atp_250": 1,
    "davis_cup": 0,
    "atp_finals": 0,
    "olympics": 0,
    "professional": 0,
}
_ROUND_ENCODINGS = {
    "r128": 1,
    "r64": 2,
    "r32": 3,
    "r16": 4,
    "qf": 5,
    "sf": 6,
    "f": 7,
}

_SURFACE_TO_SNAPSHOT_COL = {
    "clay": "clay_win_rate_10",
    "grass": "grass_win_rate_10",
    "hard": "hard_win_rate_10",
}

_DEFAULTS_COLUMNS_SQL = ", ".join(FEATURE_DEFAULTS_COLS)

# Newest materialized defaults row at or before the as-of date (future dates
# resolve to the dbt run-date row); pre-history dates fall back to the oldest
# row, whose empty-pool constants are the correct prior state. This replaces
# on-demand AVG/PERCENTILE imputation queries.
_DEFAULTS_AT_OR_BEFORE_SQL = f"""
SELECT as_of_date, {_DEFAULTS_COLUMNS_SQL} FROM {FEATURE_DEFAULTS_TABLE}
WHERE as_of_date <= %s::date
ORDER BY as_of_date DESC
LIMIT 1
"""

_DEFAULTS_FIRST_SQL = f"""
SELECT as_of_date, {_DEFAULTS_COLUMNS_SQL} FROM {FEATURE_DEFAULTS_TABLE}
ORDER BY as_of_date ASC
LIMIT 1
"""

# Latest snapshot per player strictly before the as-of date. Point lookup.
_LATEST_SNAPSHOT_SQL = f"""
SELECT * FROM {SILVER_ROLLING_FEATURES}
WHERE player_id = %s
  AND snapshot_date < %s::date
ORDER BY player_match_number DESC
LIMIT 1
"""

# Pre-match activity count for [as_of - 30 days, as_of).
_MATCHES_30D_SQL = f"""
SELECT COUNT(*) AS n
FROM {SILVER_PLAYER_MATCHES}
WHERE player_id = %s
  AND match_date >= %s::date - INTERVAL '30 days'
  AND match_date < %s::date
"""

# Last five distinct, strictly-prior meetings for the canonical pair.
_H2H_PRIOR_SQL = f"""
SELECT match_id, a_won
FROM (
    SELECT
        CASE WHEN player_id < opponent_id THEN player_id ELSE opponent_id END AS a,
        CASE WHEN player_id < opponent_id THEN opponent_id ELSE player_id END AS b,
        match_id,
        match_date,
        MAX(CASE WHEN player_id < opponent_id THEN match_won
                 ELSE 1 - match_won END) AS a_won
    FROM {SILVER_PLAYER_MATCHES}
    WHERE ((%s = player_id AND %s = opponent_id)
        OR (%s = opponent_id AND %s = player_id))
      AND match_date < %s::date
    GROUP BY 1, 2, 3, 4
)
ORDER BY match_date DESC, match_id DESC
LIMIT 5
"""

# Per-player static identity lookup.
_PROFILE_SQL = f"""
SELECT player_id, height, handedness, turned_pro
FROM {PROFILES_TABLE}
WHERE player_id = %s
"""

# ── Set-oriented bulk queries ─────────────────────────────────────────────
# Every query resolves all requested rows in one round trip (unnest + LATERAL
# or ANY), so cost stays flat as the batch grows. Values are always bound via
# `%s`; the pairs below mirror the scalar query semantics exactly.

_DEFAULTS_BULK_SQL = f"""
SELECT req.as_of_iso, d.*
FROM unnest(%s::date[]) AS req(as_of_iso)
LEFT JOIN LATERAL (
    SELECT * FROM {FEATURE_DEFAULTS_TABLE}
    WHERE as_of_date <= req.as_of_iso
    ORDER BY as_of_date DESC
    LIMIT 1
) d ON true
"""

_SNAPSHOTS_BULK_SQL = f"""
SELECT req.player_id AS req_player_id, req.as_of_iso, s.*
FROM unnest(%s::text[], %s::date[]) AS req(player_id, as_of_iso)
LEFT JOIN LATERAL (
    SELECT * FROM {SILVER_ROLLING_FEATURES}
    WHERE player_id = req.player_id
      AND snapshot_date < req.as_of_iso
    ORDER BY player_match_number DESC
    LIMIT 1
) s ON true
"""

_MATCHES_30D_BULK_SQL = f"""
SELECT req.player_id AS req_player_id, req.as_of_iso, COUNT(pm.player_id) AS n
FROM unnest(%s::text[], %s::date[]) AS req(player_id, as_of_iso)
LEFT JOIN {SILVER_PLAYER_MATCHES} pm
  ON pm.player_id = req.player_id
 AND pm.match_date >= req.as_of_iso::date - INTERVAL '30 days'
 AND pm.match_date < req.as_of_iso::date
GROUP BY req.player_id, req.as_of_iso
"""

_PROFILES_BULK_SQL = f"""
SELECT player_id, height, handedness, turned_pro
FROM {PROFILES_TABLE}
WHERE player_id = ANY(%s::text[])
"""

_H2H_PRIOR_BULK_SQL = f"""
SELECT req.a, req.b, req.as_of_iso, h.match_id, h.a_won
FROM unnest(%s::text[], %s::text[], %s::date[]) AS req(a, b, as_of_iso)
LEFT JOIN LATERAL (
    SELECT match_id,
        MAX(CASE WHEN player_id < opponent_id THEN match_won
                 ELSE 1 - match_won END) AS a_won,
        MAX(match_date) AS max_date
    FROM {SILVER_PLAYER_MATCHES}
    WHERE ((req.a = player_id AND req.b = opponent_id)
        OR (req.b = player_id AND req.a = opponent_id))
      AND match_date < req.as_of_iso::date
    GROUP BY match_id
    ORDER BY max_date DESC, match_id DESC
    LIMIT 5
) h ON true
"""


# Keep the real type because tests monkeypatch the module's `date` name.
_DATE_TYPE = date


def _to_date(value: object) -> date:
    """Coerce a PostgreSQL DATE cell (datetime.date/Timestamp/datetime/str) to a
    plain date."""
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, _DATE_TYPE):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError(f"cannot coerce {value!r} to a date")


def _load_defaults_oldest() -> dict[str, float]:
    """Return the oldest materialized defaults row (pre-history fallback).

    Its empty-pool constants are the correct prior state for dates before the
    first defaults row.
    """
    df = execute_df(_DEFAULTS_FIRST_SQL)
    if df.empty:
        raise RuntimeError(f"{FEATURE_DEFAULTS_TABLE} is empty: run dbt build (just db-dbt) first")
    return cast(dict[str, float], first_row_dict(df))


def _load_defaults(as_of_iso: str) -> dict[str, float]:
    """Return the materialized defaults row for the as-of date.

    Uses the newest row at or before the date (the dbt run-date row covers
    future dates) and the oldest row — whose empty-pool constants are the
    correct prior state — for pre-history dates. One lookup, no aggregates.
    """
    df = execute_df(_DEFAULTS_AT_OR_BEFORE_SQL, [as_of_iso])
    if df.empty:
        return _load_defaults_oldest()
    return first_row_dict(df)


def _load_defaults_bulk(as_of_isos: list[str]) -> dict[str, dict[str, float]]:
    """Return date-keyed defaults rows for many as-of dates in one query.

    Mirrors `_load_defaults` per date (newest row at or before, oldest-row
    fallback for pre-history dates) using a LATERAL join over the distinct
    dates; the oldest-row fallback is fetched once, only when needed.
    """
    distinct = list(dict.fromkeys(as_of_isos))
    out: dict[str, dict[str, float]] = {}
    missing: list[str] = []
    for rec in execute_df(_DEFAULTS_BULK_SQL, [distinct]).to_dict("records"):
        iso = _to_date(rec["as_of_iso"]).isoformat()
        if rec["as_of_date"] is None:
            missing.append(iso)
        else:
            out[iso] = cast(dict[str, float], {k: v for k, v in rec.items() if k != "as_of_iso"})
    if missing:
        oldest = _load_defaults_oldest()
        for iso in missing:
            out[iso] = oldest
    return out


def _side_values(
    row: dict[str, object] | None,
    as_of_date: date,
    surface: str,
    defaults: dict[str, float],
    matches_30d: int | None = None,
) -> dict[str, int | float]:
    """Build one side's values from its latest snapshot or date-keyed defaults.

    `matches_30d` supplies the pre-match activity count when the caller already
    looked it up (bulk path); when omitted, the scalar path queries it on
    demand for snapshot-bearing sides.
    """

    def cell(snapshot_col: str, default_col: str) -> float:
        if row is not None:
            value = row.get(snapshot_col)
            # NaN is the only supported numeric value that is not self-equal.
            if value is not None and isinstance(value, (Real, Decimal)) and value == value:
                return float(value)
        return float(defaults[default_col])

    ranking = cell("latest_player_ranking", "latest_player_ranking")
    rank_points = cell("latest_player_rank_points", "latest_player_rank_points")
    age = cell("latest_player_age", "latest_player_age")
    avg_rank_10 = cell("avg_player_rank_10", "avg_player_rank_10")

    if row is not None:
        days_since = int((as_of_date - _to_date(row["snapshot_date"])).days)
        if matches_30d is None:
            matches_30d = int(
                execute_df(
                    _MATCHES_30D_SQL,
                    [row["player_id"], as_of_date.isoformat(), as_of_date.isoformat()],
                ).iloc[0]["n"]
            )
    else:
        # Defaults are pre-rounded whole values in the materialized table.
        days_since = int(defaults["days_since_default"])
        matches_30d = int(defaults["matches_30d_default"])

    surface_win_rate = (
        cell(_SURFACE_TO_SNAPSHOT_COL[surface], _SURFACE_TO_SNAPSHOT_COL[surface])
        if surface in _SURFACE_TO_SNAPSHOT_COL
        else float(defaults["rate_default"])
    )
    return {
        "ranking": ranking,
        "rank_points": rank_points,
        "age": age,
        "weighted_form_10": cell("weighted_form_10", "weighted_form_10"),
        "win_rate_10": cell("win_rate_10", "win_rate_10"),
        "ace_rate_10": cell("ace_rate_10", "ace_rate_10"),
        "first_serve_pct_10": cell("first_serve_pct_10", "first_serve_pct_10"),
        "break_points_saved_pct_10": cell("break_points_saved_pct_10", "break_points_saved_pct_10"),
        "first_serve_win_pct_10": cell("first_serve_win_pct_10", "first_serve_win_pct_10"),
        "second_serve_win_pct_10": cell("second_serve_win_pct_10", "second_serve_win_pct_10"),
        "serve_win_pct_10": cell("serve_win_pct_10", "serve_win_pct_10"),
        "df_rate_10": cell("df_rate_10", "df_rate_10"),
        "aces_per_svc_game_10": cell("aces_per_svc_game_10", "aces_per_svc_game_10"),
        "rank_trend_10": avg_rank_10 - ranking,
        "avg_rank_faced_10": cell("avg_rank_faced_10", "avg_rank_faced_10"),
        "streak": int(cell("streak", "streak")),
        "days_since_last_match": days_since,
        "matches_30d": matches_30d,
        "surface_win_rate_10": surface_win_rate,
    }


def _profile_values_from_row(
    row: dict[str, object] | None,
    as_of_date: date,
    defaults: dict[str, float],
) -> dict[str, float]:
    """Profile values from one profile row (None for a missing player)."""
    if row is None:
        return {
            "is_left_handed": float(defaults["left_handed_rate"]),
            "years_pro": float(defaults["avg_years_pro"]),
        }
    row_any = cast(dict[str, Any], row)
    turned_pro = row_any["turned_pro"]
    years_pro = (
        float(as_of_date.year - int(cast(Any, turned_pro)))
        if not pd.isna(turned_pro)
        else float(defaults["avg_years_pro"])
    )
    # Non-L/R handedness uses the pool rate, matching gold imputation.
    handedness = row_any["handedness"]
    is_left_handed = (
        float(handedness == "L")
        if isinstance(handedness, str) and handedness in ("L", "R")
        else float(defaults["left_handed_rate"])
    )
    return {
        "is_left_handed": is_left_handed,
        "years_pro": years_pro,
    }


def _profile_values(pid: str, as_of_date: date, defaults: dict[str, float]) -> dict[str, float]:
    """Return profile values, using date-keyed defaults for missing cells."""
    df = execute_df(_PROFILE_SQL, [pid])
    return _profile_values_from_row(
        first_row_dict(df) if not df.empty else None, as_of_date, defaults
    )


class _RowContext(NamedTuple):
    """Validated + canonicalized inputs shared by the scalar and bulk builders."""

    raw_player_id: str
    raw_opponent_id: str
    surface: str
    as_of_date: date
    as_of_iso: str
    tournament_level: int
    round_encoded: int
    is_indoor: int
    lower_id: str
    higher_id: str


def _normalize_inputs(
    player_id: str,
    opponent_id: str,
    surface: str,
    *,
    as_of_date: date | None = None,
    tournament_level: int = 0,
    round_encoded: int = 0,
    tournament: str | None = None,
    round: str | None = None,
    is_indoor: int | None = None,
) -> _RowContext:
    """Validate and canonicalize one request; shared by both builders."""

    # ── Boundary validation ──
    if surface not in VALID_SURFACES:
        raise ValueError(f"surface must be one of {sorted(VALID_SURFACES)}, got {surface!r}")
    if tournament is not None:
        if tournament_level != 0:
            raise ValueError("pass either tournament_level (int) or tournament (str), not both")
        if not isinstance(tournament, str):
            raise TypeError(f"tournament must be a string, got {tournament!r}")
        tournament_level = _TOURNAMENT_LEVELS.get(tournament, 0)
    if round is not None:
        if round_encoded != 0:
            raise ValueError("pass either round_encoded (int) or round (str), not both")
        if not isinstance(round, str):
            raise TypeError(f"round must be a string, got {round!r}")
        round_encoded = _ROUND_ENCODINGS.get(round, 0)
    if isinstance(tournament_level, bool) or (
        not isinstance(tournament_level, int) or tournament_level not in VALID_TOURNAMENT_LEVELS
    ):
        raise ValueError(
            f"tournament_level must be an int in {sorted(VALID_TOURNAMENT_LEVELS)}, "
            f"got {tournament_level!r}"
        )
    if isinstance(round_encoded, bool) or (
        not isinstance(round_encoded, int) or round_encoded not in VALID_ROUND_ENCODINGS
    ):
        raise ValueError(
            f"round_encoded must be an int in {sorted(VALID_ROUND_ENCODINGS)}, "
            f"got {round_encoded!r}"
        )
    if not isinstance(player_id, str) or not player_id.strip():
        raise ValueError(f"player_id must be a non-empty string, got {player_id!r}")
    if not isinstance(opponent_id, str) or not opponent_id.strip():
        raise ValueError(f"opponent_id must be a non-empty string, got {opponent_id!r}")
    if as_of_date is None:
        as_of_date = date.today()
    elif isinstance(as_of_date, datetime):
        # A datetime is accepted and truncated to its date component.
        as_of_date = as_of_date.date()
    elif not isinstance(as_of_date, date):
        raise TypeError(f"as_of_date must be a datetime.date (or datetime), got {as_of_date!r}")

    # ── Canonicalization: lower id is the player_* side ──
    lower_id, higher_id = sorted([player_id.strip(), opponent_id.strip()])
    return _RowContext(
        raw_player_id=player_id.strip(),
        raw_opponent_id=opponent_id.strip(),
        surface=surface,
        as_of_date=as_of_date,
        as_of_iso=as_of_date.isoformat(),
        tournament_level=tournament_level,
        round_encoded=round_encoded,
        is_indoor=is_indoor if is_indoor is not None else 0,
        lower_id=lower_id,
        higher_id=higher_id,
    )


def _assemble_row(
    ctx: _RowContext,
    player_side: dict[str, int | float],
    opponent_side: dict[str, int | float],
    h2h_matches: int,
    h2h_player_wins: int,
) -> dict[str, int | float | str]:
    """Assemble one canonical row in FEATURE_COLS order (shared by builders)."""
    row: dict[str, int | float | str] = {}

    def side(name: str, p: dict[str, int | float], o: dict[str, int | float]) -> int | float:
        """Return the player/opponent value for a per-side feature name."""
        if name in ("player_ranking", "opponent_ranking"):
            return p["ranking"] if name.startswith("player_") else o["ranking"]
        return (p if name.startswith("player_") else o)[
            name.removeprefix("player_").removeprefix("opponent_")
        ]

    # Matchup differences (canonical side minus opponent).
    row["rank_diff"] = player_side["ranking"] - opponent_side["ranking"]
    row["rank_points_diff"] = player_side["rank_points"] - opponent_side["rank_points"]
    row["age_diff"] = player_side["age"] - opponent_side["age"]
    row["win_rate_diff"] = player_side["win_rate_10"] - opponent_side["win_rate_10"]
    row["ace_rate_diff"] = player_side["ace_rate_10"] - opponent_side["ace_rate_10"]
    row["first_serve_pct_diff"] = (
        player_side["first_serve_pct_10"] - opponent_side["first_serve_pct_10"]
    )
    row["break_points_saved_pct_diff"] = (
        player_side["break_points_saved_pct_10"] - opponent_side["break_points_saved_pct_10"]
    )
    row["first_serve_win_pct_diff"] = (
        player_side["first_serve_win_pct_10"] - opponent_side["first_serve_win_pct_10"]
    )
    row["second_serve_win_pct_diff"] = (
        player_side["second_serve_win_pct_10"] - opponent_side["second_serve_win_pct_10"]
    )
    row["serve_win_pct_diff"] = player_side["serve_win_pct_10"] - opponent_side["serve_win_pct_10"]
    row["df_rate_diff"] = player_side["df_rate_10"] - opponent_side["df_rate_10"]
    row["aces_per_svc_game_diff"] = (
        player_side["aces_per_svc_game_10"] - opponent_side["aces_per_svc_game_10"]
    )
    row["rank_trend_diff"] = player_side["rank_trend_10"] - opponent_side["rank_trend_10"]
    row["avg_rank_faced_diff"] = (
        player_side["avg_rank_faced_10"] - opponent_side["avg_rank_faced_10"]
    )
    row["streak_diff"] = player_side["streak"] - opponent_side["streak"]

    # Absolute state values (both sides matter).
    for name in (
        "player_weighted_form_10",
        "opponent_weighted_form_10",
        "player_days_since_last_match",
        "opponent_days_since_last_match",
        "player_matches_30d",
        "opponent_matches_30d",
        "player_surface_win_rate_10",
        "opponent_surface_win_rate_10",
        "player_is_left_handed",
        "opponent_is_left_handed",
        "player_years_pro",
        "opponent_years_pro",
    ):
        row[name] = side(name, player_side, opponent_side)

    # Canonical-player head-to-head history.
    row["player_h2h_matches"] = h2h_matches
    row["player_h2h_wins"] = h2h_player_wins

    # Numeric match context (one-hot surface).
    row["is_clay"] = int(ctx.surface == "clay")
    row["is_grass"] = int(ctx.surface == "grass")
    row["is_hard"] = int(ctx.surface == "hard")
    row["is_carpet"] = int(ctx.surface == "carpet")
    row["is_indoor"] = ctx.is_indoor
    row["tournament_level"] = ctx.tournament_level
    row["round_encoded"] = ctx.round_encoded

    # Preserve the canonical ids, not the raw input order.
    row["player_id"] = ctx.lower_id
    row["opponent_id"] = ctx.higher_id
    return row


def _build_inference_features_with_meta(
    player_id: str,
    opponent_id: str,
    surface: str,
    *,
    as_of_date: date | None = None,
    tournament_level: int = 0,
    round_encoded: int = 0,
    tournament: str | None = None,
    round: str | None = None,
    is_indoor: int | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build one NaN-free canonical row in the FEATURE_COLS contract."""
    started_at = perf_counter()
    ctx = _normalize_inputs(
        player_id,
        opponent_id,
        surface,
        as_of_date=as_of_date,
        tournament_level=tournament_level,
        round_encoded=round_encoded,
        tournament=tournament,
        round=round,
        is_indoor=is_indoor,
    )

    # ── Materialized date-keyed defaults (no on-demand aggregates) ──
    defaults = _load_defaults(ctx.as_of_iso)

    def _latest_snapshot(pid: str) -> dict[str, object] | None:
        df = execute_df(_LATEST_SNAPSHOT_SQL, [pid, ctx.as_of_iso])
        if df.empty:
            return None
        return first_row_dict(df)

    player_snapshot = _latest_snapshot(ctx.lower_id)
    opponent_snapshot = _latest_snapshot(ctx.higher_id)

    player_side = _side_values(player_snapshot, ctx.as_of_date, ctx.surface, defaults)
    opponent_side = _side_values(opponent_snapshot, ctx.as_of_date, ctx.surface, defaults)
    player_side.update(_profile_values(ctx.lower_id, ctx.as_of_date, defaults))
    opponent_side.update(_profile_values(ctx.higher_id, ctx.as_of_date, defaults))

    # ── Head-to-head: last five prior meetings for the canonical pair ──
    h2h_df = execute_df(
        _H2H_PRIOR_SQL,
        [ctx.lower_id, ctx.higher_id, ctx.lower_id, ctx.higher_id, ctx.as_of_iso],
    )
    h2h_matches = len(h2h_df)
    h2h_player_wins = int(h2h_df["a_won"].sum()) if h2h_matches else 0

    # ── Assemble the canonical row in FEATURE_COLS order ──
    row = _assemble_row(ctx, player_side, opponent_side, h2h_matches, h2h_player_wins)

    final_cols = [*FEATURE_COLS, "player_id", "opponent_id"]
    out = pd.DataFrame({col: [row[col]] for col in final_cols})
    assert not out.isnull().to_numpy().any(), "inference row contains NaN"

    meta: dict[str, object] = {
        "raw_player_id": ctx.raw_player_id,
        "raw_opponent_id": ctx.raw_opponent_id,
        "canonical_player_id": ctx.lower_id,
        "canonical_opponent_id": ctx.higher_id,
        "surface": ctx.surface,
        "as_of_date": ctx.as_of_iso,
        "tournament_level": ctx.tournament_level,
        "round_encoded": ctx.round_encoded,
        "feature_count": len(FEATURE_COLS),
        "snapshot_pool_rows": int(float(defaults["snapshot_pool_rows"] or 0)),
        "snapshot_pool_players": int(float(defaults["snapshot_pool_players"] or 0)),
        "profile_rows": int(float(defaults["profile_rows"] or 0)),
        "median_days_since": float(defaults["days_since_default"]),
        "median_matches_30d": float(defaults["matches_30d_default"]),
        "player_snapshot_found": player_snapshot is not None,
        "opponent_snapshot_found": opponent_snapshot is not None,
        "player_snapshot_date": None
        if player_snapshot is None
        else _to_date(player_snapshot["snapshot_date"]).isoformat(),
        "opponent_snapshot_date": None
        if opponent_snapshot is None
        else _to_date(opponent_snapshot["snapshot_date"]).isoformat(),
        "player_rolling_match_number": None
        if player_snapshot is None
        else int(float(cast(Real, player_snapshot["player_match_number"]))),
        "opponent_rolling_match_number": None
        if opponent_snapshot is None
        else int(float(cast(Real, opponent_snapshot["player_match_number"]))),
        "player_matches_30d": int(player_side["matches_30d"]),
        "opponent_matches_30d": int(opponent_side["matches_30d"]),
        "player_days_since_last_match": int(player_side["days_since_last_match"]),
        "opponent_days_since_last_match": int(opponent_side["days_since_last_match"]),
        "h2h_prior_meetings": h2h_matches,
        "build_ms": builtins.round((perf_counter() - started_at) * 1000, 3),
    }
    return out, meta


def build_inference_features(
    player_id: str,
    opponent_id: str,
    surface: str,
    *,
    as_of_date: date | None = None,
    tournament_level: int = 0,
    round_encoded: int = 0,
    tournament: str | None = None,
    round: str | None = None,
    is_indoor: int | None = None,
) -> pd.DataFrame:
    out, _meta = _build_inference_features_with_meta(
        player_id,
        opponent_id,
        surface,
        as_of_date=as_of_date,
        tournament_level=tournament_level,
        round_encoded=round_encoded,
        tournament=tournament,
        round=round,
        is_indoor=is_indoor,
    )
    return out


def build_inference_features_bulk(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build canonical rows for many matches in one set-oriented pass.

    Each row accepts the same fields and validation as
    `build_inference_features` (including its own historical `as_of_date`).
    Snapshots, defaults, 30-day activity, profiles, and H2H history are
    resolved with unnest/LATERAL/ANY queries — a constant number of round
    trips regardless of the batch size — then every row is assembled with the
    exact same per-side value and row logic as the scalar builder, so output
    matches repeated scalar calls. Input order is preserved.
    """
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must be a non-empty list of match contexts")
    if len(rows) > BULK_MAX_ROWS:
        raise ValueError(f"bulk inference accepts at most {BULK_MAX_ROWS} rows, got {len(rows)}")

    ctxs = [_normalize_inputs(**cast(dict[str, Any], row)) for row in rows]
    defaults_by_date = _load_defaults_bulk([c.as_of_iso for c in ctxs])

    # Distinct (player, as-of) pairs drive the set-oriented snapshot queries.
    pairs = list({(pid, c.as_of_date) for c in ctxs for pid in (c.lower_id, c.higher_id)})
    pair_pids = [p for p, _ in pairs]
    pair_dates = [d for _, d in pairs]

    snapshots: dict[tuple[str, date], dict[str, object] | None] = {}
    for rec in execute_df(_SNAPSHOTS_BULK_SQL, [pair_pids, pair_dates]).to_dict("records"):
        key = (rec["req_player_id"], _to_date(rec["as_of_iso"]))
        if rec["player_id"] is None:
            snapshots[key] = None
        else:
            snapshots[key] = cast(
                dict[str, object],
                {k: v for k, v in rec.items() if k not in ("req_player_id", "as_of_iso")},
            )

    matches_30d: dict[tuple[str, date], int] = {}
    for rec in execute_df(_MATCHES_30D_BULK_SQL, [pair_pids, pair_dates]).to_dict("records"):
        matches_30d[(rec["req_player_id"], _to_date(rec["as_of_iso"]))] = int(rec["n"])

    profiles: dict[str, dict[str, object]] = {}
    players = sorted({p for c in ctxs for p in (c.lower_id, c.higher_id)})
    for rec in execute_df(_PROFILES_BULK_SQL, [players]).to_dict("records"):
        profiles[str(rec["player_id"])] = cast(dict[str, object], rec)

    h2h: dict[tuple[str, str, str], list[tuple[str, int]]] = {}
    h2h_triples = list({(c.lower_id, c.higher_id, c.as_of_iso) for c in ctxs})
    h2h_rows = execute_df(
        _H2H_PRIOR_BULK_SQL,
        [
            [t[0] for t in h2h_triples],
            [t[1] for t in h2h_triples],
            [_to_date(t[2]) for t in h2h_triples],
        ],
    )
    for rec in h2h_rows.to_dict("records"):
        if rec["match_id"] is None:  # pair with no prior meetings
            continue
        key = (rec["a"], rec["b"], _to_date(rec["as_of_iso"]).isoformat())
        h2h.setdefault(key, []).append((str(rec["match_id"]), int(rec["a_won"])))

    out_rows: list[dict[str, int | float | str]] = []
    for ctx in ctxs:
        defaults = defaults_by_date[ctx.as_of_iso]
        player_snapshot = snapshots.get((ctx.lower_id, ctx.as_of_date))
        opponent_snapshot = snapshots.get((ctx.higher_id, ctx.as_of_date))
        player_side = _side_values(
            player_snapshot,
            ctx.as_of_date,
            ctx.surface,
            defaults,
            matches_30d.get((ctx.lower_id, ctx.as_of_date)),
        )
        opponent_side = _side_values(
            opponent_snapshot,
            ctx.as_of_date,
            ctx.surface,
            defaults,
            matches_30d.get((ctx.higher_id, ctx.as_of_date)),
        )
        player_side.update(
            _profile_values_from_row(profiles.get(ctx.lower_id), ctx.as_of_date, defaults)
        )
        opponent_side.update(
            _profile_values_from_row(profiles.get(ctx.higher_id), ctx.as_of_date, defaults)
        )
        meetings = h2h.get((ctx.lower_id, ctx.higher_id, ctx.as_of_iso), [])
        out_rows.append(
            _assemble_row(
                ctx,
                player_side,
                opponent_side,
                len(meetings),
                sum(a_won for _, a_won in meetings),
            )
        )

    out = pd.DataFrame(out_rows, columns=[*FEATURE_COLS, "player_id", "opponent_id"])
    assert not out.isnull().to_numpy().any(), "bulk inference rows contain NaN"
    return out
