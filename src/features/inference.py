"""Build directional, as-of-dated inference rows from PostgreSQL."""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.features.nn_inference import GRUBatch, GRUPreprocessing
import math
from datetime import date, datetime
from numbers import Real
from time import perf_counter
from typing import Any, NamedTuple, cast

import pandas as pd

from src.constants import (
    BRONZE_MATCHES_TABLE,
    BRONZE_PROFILES_TABLE,
    BULK_MAX_ROWS,
    DAYS_SINCE_LAST_MATCH_MAX,
    ELO_DEFAULT_RATING,
    SILVER_ELO_SNAPSHOTS,
    SILVER_PLAYER_MATCHES,
    SILVER_ROLLING_FEATURES,
)
from src.db.client import execute_df, first_row_dict
from src.features.columns import CANONICAL_SURFACES, FEATURE_COLS
from src.features.elo_math import regress_rating
from src.features.tour_averages import load_tour_averages

VALID_SURFACES = CANONICAL_SURFACES
VALID_TOURNAMENT_LEVELS = {0, 1, 2, 3, 4}
VALID_ROUND_ENCODINGS = {0, 1, 2, 3, 4, 5, 6, 7}

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

# Latest post-match snapshot through as_of_date inclusively, ordered by the
# causal (snapshot_date, match_num) so the highest match_num on as_of_date wins.
_LATEST_SNAPSHOT_SQL = f"""
SELECT rf.* FROM {SILVER_ROLLING_FEATURES} rf
JOIN {SILVER_PLAYER_MATCHES} pm
  ON pm.player_id = rf.player_id AND pm.match_id = rf.match_id
WHERE rf.player_id = %s
  AND rf.snapshot_date <= %s::date
ORDER BY rf.snapshot_date DESC, pm.match_num DESC, rf.match_id DESC
LIMIT 1
"""

_H2H_PRIOR_SQL = f"""
SELECT match_id, winner_id, surface
FROM {BRONZE_MATCHES_TABLE}
WHERE ((player1_id = %s AND player2_id = %s)
    OR (player1_id = %s AND player2_id = %s))
    AND match_date < %s::date
    ORDER BY match_date DESC, match_num DESC, match_id DESC
    LIMIT 10
"""

_PROFILE_SQL = f"""
SELECT player_id, height, handedness, turned_pro
FROM {BRONZE_PROFILES_TABLE}
WHERE player_id = %s
"""

_SNAPSHOTS_BULK_SQL = f"""
SELECT req.player_id AS req_player_id, req.as_of_iso, s.*
FROM unnest(%s::text[], %s::date[]) AS req(player_id, as_of_iso)
LEFT JOIN LATERAL (
    SELECT rf.* FROM {SILVER_ROLLING_FEATURES} rf
    JOIN {SILVER_PLAYER_MATCHES} pm
      ON pm.player_id = rf.player_id AND pm.match_id = rf.match_id
    WHERE rf.player_id = req.player_id
      AND rf.snapshot_date <= req.as_of_iso
    ORDER BY rf.snapshot_date DESC, pm.match_num DESC, rf.match_id DESC
    LIMIT 1
) s ON true
"""

_PROFILES_BULK_SQL = f"""
SELECT player_id, height, handedness, turned_pro
FROM {BRONZE_PROFILES_TABLE}
WHERE player_id = ANY(%s::text[])
"""

_H2H_PRIOR_BULK_SQL = f"""
SELECT req.player_id AS req_player_id, req.opponent_id AS req_opponent_id,
       req.as_of_iso, h.match_id, h.winner_id, h.surface
FROM unnest(%s::text[], %s::text[], %s::date[]) AS req(player_id, opponent_id, as_of_iso)
LEFT JOIN LATERAL (
    SELECT match_id, winner_id, surface
    FROM {BRONZE_MATCHES_TABLE}
    WHERE ((req.player_id = player1_id AND req.opponent_id = player2_id)
        OR (req.opponent_id = player1_id AND req.player_id = player2_id))
       AND match_date < req.as_of_iso::date
    ORDER BY match_date DESC, match_num DESC, match_id DESC
    LIMIT 10
) h ON true
"""

# Latest completed overall Elo (post-match rating) through as_of_date, inclusive
# of a same-day completed match; inactivity-regressed in _elo_rating.
_LATEST_ELO_SQL = f"""
SELECT post_elo, match_date
FROM {SILVER_ELO_SNAPSHOTS}
WHERE player_id = %s
  AND match_date <= %s::date
ORDER BY match_date DESC, match_num DESC, match_id DESC
LIMIT 1
"""

_ELO_BULK_SQL = f"""
SELECT req.player_id AS req_player_id, req.as_of_iso, s.post_elo, s.match_date
FROM unnest(%s::text[], %s::date[]) AS req(player_id, as_of_iso)
LEFT JOIN LATERAL (
    SELECT post_elo, match_date FROM {SILVER_ELO_SNAPSHOTS}
    WHERE player_id = req.player_id
      AND match_date <= req.as_of_iso
    ORDER BY match_date DESC, match_num DESC, match_id DESC
    LIMIT 1
) s ON true
"""

# Up-to-ten completed post-match Elo ratings strictly before as_of_date, used to
# fit each side's Elo-gradient (OLS slope). Same-day/future matches are excluded
# to mirror the gold match_features.sql strictly-before gradient window.
_GRADIENT_BULK_SQL = f"""
SELECT req.player_id AS req_player_id, req.as_of_iso, s.post_elo, s.match_date, s.match_num, s.match_id
FROM unnest(%s::text[], %s::date[]) AS req(player_id, as_of_iso)
LEFT JOIN LATERAL (
    SELECT post_elo, match_date, match_num, match_id FROM {SILVER_ELO_SNAPSHOTS}
    WHERE player_id = req.player_id
      AND match_date < req.as_of_iso
    ORDER BY match_date DESC, match_num DESC, match_id DESC
    LIMIT 10
) s ON true
"""

# Keep the imported type stable when tests replace the module's `date` name.
_DATE_TYPE = date


def _to_date(value: object) -> date:
    """Coerce a PostgreSQL date cell to a plain date."""
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, _DATE_TYPE):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError(f"cannot coerce {value!r} to a date")


def _side_values(
    row: dict[str, object] | None,
    defaults: dict[str, float],
    *,
    surface: str,
    as_of_date: date,
) -> dict[str, int | float]:
    """Build one side from its latest snapshot, with tour-average fallbacks."""

    def cell_lit(snapshot_col: str, fallback: float) -> float:
        if row is not None:
            value = row.get(snapshot_col)
            if value is not None and isinstance(value, Real) and value == value:
                return float(value)
        return fallback

    def cell(snapshot_col: str, default_col: str) -> float:
        return cell_lit(snapshot_col, float(defaults[default_col]))

    ranking = cell("latest_player_ranking", "latest_player_ranking")
    rank_points = cell("latest_player_rank_points", "latest_player_rank_points")
    age = cell("latest_player_age", "latest_player_age")
    avg_rank_10 = cell("avg_player_rank_10", "avg_player_rank_10")

    surface_form = (
        cell_lit(f"{surface}_win_rate_10", float(defaults[f"{surface}_win_rate_10"]))
        if surface != "carpet"
        else float(defaults["rate_default"])
    )
    raw_days = (
        float((as_of_date - _to_date(row["snapshot_date"])).days)
        if row is not None
        else float(DAYS_SINCE_LAST_MATCH_MAX)
    )
    days_since_last_match = min(raw_days, float(DAYS_SINCE_LAST_MATCH_MAX))

    return {
        "ranking": ranking,
        "rank_points": rank_points,
        "age": age,
        "win_rate_10": cell("win_rate_10", "win_rate_10"),
        "ace_rate_10": cell("ace_rate_10", "ace_rate_10"),
        "first_serve_pct_10": cell("first_serve_pct_10", "first_serve_pct_10"),
        "break_points_saved_pct_10": cell("break_points_saved_pct_10", "break_points_saved_pct_10"),
        "first_serve_win_pct_10": cell("first_serve_win_pct_10", "first_serve_win_pct_10"),
        "second_serve_win_pct_10": cell("second_serve_win_pct_10", "second_serve_win_pct_10"),
        "serve_win_pct_10": cell("serve_win_pct_10", "serve_win_pct_10"),
        "return_points_won_pct_10": cell("return_points_won_pct_10", "return_points_won_pct_10"),
        "dominance": cell("dominance", "dominance"),
        "df_rate_10": cell("df_rate_10", "df_rate_10"),
        "aces_per_svc_game_10": cell("aces_per_svc_game_10", "aces_per_svc_game_10"),
        "rank_trend_10": avg_rank_10 - ranking,
        "avg_rank_faced_10": cell("avg_rank_faced_10", "avg_rank_faced_10"),
        "streak": int(cell("streak", "streak")),
        "matches_10": int(float(cast(Real, row["matches_10"]))) if row is not None else 0,
        "surface_form": surface_form,
        "days_since_last_match": days_since_last_match,
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
    """Return profile values, using singleton fallbacks for missing cells."""
    df = execute_df(_PROFILE_SQL, [pid])
    return _profile_values_from_row(
        first_row_dict(df) if not df.empty else None, as_of_date, defaults
    )


def _elo_rating(pid: str, as_of_date: date, as_of_iso: str) -> float:
    """Latest completed overall Elo through as_of_date.

    Each rating is the player's most recent post-match Elo at or before
    ``as_of_date`` (inclusive of a same-day completed match), inactive-regressed
    forward to ``as_of_date``; missing history defaults to 1500.
    """
    overall = execute_df(_LATEST_ELO_SQL, [pid, as_of_iso])
    elo = ELO_DEFAULT_RATING
    if not overall.empty:
        row = first_row_dict(overall)
        if not pd.isna(row["post_elo"]):
            elo = regress_rating(
                float(row["post_elo"]), (as_of_date - _to_date(row["match_date"])).days
            )
    return elo


def _elo_ratings_bulk(
    overall_keys: list[tuple[str, date]],
) -> dict[tuple[str, date], float]:
    """Set-oriented latest completed overall Elo, inactivity-regressed."""
    overall: dict[tuple[str, date], float] = {}
    for rec in execute_df(
        _ELO_BULK_SQL,
        [[k[0] for k in overall_keys], [k[1] for k in overall_keys]],
    ).to_dict("records"):
        key = (rec["req_player_id"], _to_date(rec["as_of_iso"]))
        if rec["post_elo"] is None or pd.isna(rec["post_elo"]):
            overall[key] = ELO_DEFAULT_RATING
        else:
            overall[key] = regress_rating(
                float(rec["post_elo"]), (key[1] - _to_date(rec["match_date"])).days
            )
    return overall


def _elo_gradient_from_records(records: list[dict[str, object]]) -> float:
    """OLS slope of post_elo over the chronological index of up to 10 snapshots.

    Snapshots are the most-recent completed ratings strictly before as_of; fewer
    than two yields exactly 0.0, matching the gold match_features.sql formula
    (COUNT(*) < 2 THEN 0; else the ordinary-least-squares slope on idx).
    """
    ordered: list[tuple[date, int, str, float]] = []
    for rec in records:
        pe = rec["post_elo"]
        if not isinstance(pe, Real) or pe != pe:  # skip None and NaN (mirrors _side_values)
            continue
        ordered.append(
            (
                _to_date(rec["match_date"]),
                int(float(cast(Real, rec["match_num"]))),
                str(rec["match_id"]),
                float(pe),
            )
        )
    ordered.sort(key=lambda t: (t[0], t[1], t[2]))
    ys = [t[3] for t in ordered]
    n = len(ys)
    if n < 2:
        return 0.0
    xs = list(range(1, n + 1))
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys, strict=True))
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0
    return (n * sxy - sx * sy) / denom


def _elo_gradients_bulk(
    keys: list[tuple[str, date]],
) -> dict[tuple[str, date], float]:
    """Set-oriented Elo-gradient (OLS slope) per (player, as_of_date).

    One query for all requested players; no per-request DB loops. Same-day and
    future Elo snapshots are excluded by the SQL window, so a player with fewer
    than two strictly-before ratings defaults to 0.0.
    """
    by_key: dict[tuple[str, date], list[dict[str, object]]] = {}
    for rec in execute_df(
        _GRADIENT_BULK_SQL,
        [[k[0] for k in keys], [k[1] for k in keys]],
    ).to_dict("records"):
        key = (rec["req_player_id"], _to_date(rec["as_of_iso"]))
        by_key.setdefault(key, [])
        if isinstance(rec["post_elo"], Real) and rec["post_elo"] == rec["post_elo"]:
            by_key[key].append(cast(dict[str, object], rec))
    grads = {key: _elo_gradient_from_records(recs) for key, recs in by_key.items()}
    for key in keys:
        grads.setdefault(key, 0.0)
    return grads


class _RowContext(NamedTuple):
    """Validated + directionally-oriented inputs shared by the builders."""

    raw_player_id: str
    raw_opponent_id: str
    surface: str
    as_of_date: date
    as_of_iso: str
    tournament_level: int
    round_encoded: int
    is_indoor: int
    best_of: int
    player_id: str
    opponent_id: str


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
    best_of: int = 3,
) -> _RowContext:
    """Validate one request while preserving player_id as the player side."""
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
    if isinstance(is_indoor, bool) or (
        is_indoor is not None and (not isinstance(is_indoor, int) or is_indoor not in (0, 1))
    ):
        raise ValueError(f"is_indoor must be exactly 0 or 1, got {is_indoor!r}")
    if isinstance(best_of, bool) or not isinstance(best_of, int) or best_of not in (1, 3, 5):
        raise ValueError(f"best_of must be exactly 1, 3, or 5, got {best_of!r}")
    if not isinstance(player_id, str) or not player_id.strip():
        raise ValueError(f"player_id must be a non-empty string, got {player_id!r}")
    if not isinstance(opponent_id, str) or not opponent_id.strip():
        raise ValueError(f"opponent_id must be a non-empty string, got {opponent_id!r}")
    if as_of_date is None:
        as_of_date = date.today()
    elif isinstance(as_of_date, datetime):
        as_of_date = as_of_date.date()
    elif not isinstance(as_of_date, date):
        raise TypeError(f"as_of_date must be a datetime.date (or datetime), got {as_of_date!r}")

    player_id = player_id.strip()
    opponent_id = opponent_id.strip()
    return _RowContext(
        raw_player_id=player_id,
        raw_opponent_id=opponent_id,
        surface=surface,
        as_of_date=as_of_date,
        as_of_iso=as_of_date.isoformat(),
        tournament_level=tournament_level,
        round_encoded=round_encoded,
        is_indoor=is_indoor if is_indoor is not None else 0,
        best_of=best_of,
        player_id=player_id,
        opponent_id=opponent_id,
    )


def _assemble_row(
    ctx: _RowContext,
    player_side: dict[str, int | float],
    opponent_side: dict[str, int | float],
    h2h_exposure: int,
    h2h_wins_for_requested_player: int,
    h2h_surface_meetings: int,
    h2h_surface_wins_for_requested_player: int,
    player_elo: float,
    opponent_elo: float,
    player_elo_gradient_10: float,
    opponent_elo_gradient_10: float,
) -> dict[str, int | float | str]:
    """Assemble one directional row in the FEATURE_COLS contract."""
    row: dict[str, int | float | str] = {}

    def side(name: str, p: dict[str, int | float], o: dict[str, int | float]) -> int | float:
        """Return the player/opponent value for a per-side feature name."""
        if name in ("player_ranking", "opponent_ranking"):
            return p["ranking"] if name.startswith("player_") else o["ranking"]
        return (p if name.startswith("player_") else o)[
            name.removeprefix("player_").removeprefix("opponent_")
        ]

    row["rank_diff"] = player_side["ranking"] - opponent_side["ranking"]
    row["rank_points_diff"] = player_side["rank_points"] - opponent_side["rank_points"]
    row["age_diff"] = player_side["age"] - opponent_side["age"]
    row["form_diff"] = player_side["win_rate_10"] - opponent_side["win_rate_10"]
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
    row["return_points_won_pct_diff"] = (
        player_side["return_points_won_pct_10"] - opponent_side["return_points_won_pct_10"]
    )
    row["dominance_diff"] = player_side["dominance"] - opponent_side["dominance"]
    row["df_rate_diff"] = player_side["df_rate_10"] - opponent_side["df_rate_10"]
    row["aces_per_svc_game_diff"] = (
        player_side["aces_per_svc_game_10"] - opponent_side["aces_per_svc_game_10"]
    )
    row["rank_trend_diff"] = player_side["rank_trend_10"] - opponent_side["rank_trend_10"]
    row["avg_rank_faced_diff"] = (
        player_side["avg_rank_faced_10"] - opponent_side["avg_rank_faced_10"]
    )
    row["streak_diff"] = player_side["streak"] - opponent_side["streak"]
    row["surface_form_diff"] = player_side["surface_form"] - opponent_side["surface_form"]
    row["days_since_last_match_diff"] = math.log(
        1.0 + player_side["days_since_last_match"]
    ) - math.log(1.0 + opponent_side["days_since_last_match"])
    row["elo_diff"] = player_elo - opponent_elo
    row["player_elo_gradient_10"] = player_elo_gradient_10
    row["opponent_elo_gradient_10"] = opponent_elo_gradient_10

    for name in (
        "player_matches_10",
        "opponent_matches_10",
        "player_is_left_handed",
        "opponent_is_left_handed",
        "player_years_pro",
        "opponent_years_pro",
    ):
        row[name] = side(name, player_side, opponent_side)

    row["h2h_exposure"] = h2h_exposure
    row["h2h_advantage"] = (h2h_wins_for_requested_player + 1.0) / (h2h_exposure + 2.0) - 0.5
    row["h2h_surface_advantage"] = (h2h_surface_wins_for_requested_player + 1.0) / (
        h2h_surface_meetings + 2.0
    ) - 0.5

    row["is_clay"] = int(ctx.surface == "clay")
    row["is_grass"] = int(ctx.surface == "grass")
    row["is_hard"] = int(ctx.surface == "hard")
    row["is_indoor"] = ctx.is_indoor
    row["best_of"] = ctx.best_of
    row["tournament_level"] = ctx.tournament_level
    row["round_encoded"] = ctx.round_encoded

    row["player_id"] = ctx.player_id
    row["opponent_id"] = ctx.opponent_id
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
    best_of: int = 3,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build one NaN-free directional row in the FEATURE_COLS contract."""
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
        best_of=best_of,
    )

    ta = load_tour_averages()
    defaults = cast(dict[str, float], ta)

    def _latest_snapshot(pid: str) -> dict[str, object] | None:
        df = execute_df(_LATEST_SNAPSHOT_SQL, [pid, ctx.as_of_iso])
        if df.empty:
            return None
        return first_row_dict(df)

    player_snapshot = _latest_snapshot(ctx.player_id)
    opponent_snapshot = _latest_snapshot(ctx.opponent_id)

    player_side = _side_values(
        player_snapshot,
        defaults,
        surface=ctx.surface,
        as_of_date=ctx.as_of_date,
    )
    opponent_side = _side_values(
        opponent_snapshot,
        defaults,
        surface=ctx.surface,
        as_of_date=ctx.as_of_date,
    )
    player_side.update(_profile_values(ctx.player_id, ctx.as_of_date, defaults))
    opponent_side.update(_profile_values(ctx.opponent_id, ctx.as_of_date, defaults))

    player_elo = _elo_rating(ctx.player_id, ctx.as_of_date, ctx.as_of_iso)
    opponent_elo = _elo_rating(ctx.opponent_id, ctx.as_of_date, ctx.as_of_iso)

    grad_pairs = [(ctx.player_id, ctx.as_of_date), (ctx.opponent_id, ctx.as_of_date)]
    grads = _elo_gradients_bulk(grad_pairs)
    player_grad = grads[(ctx.player_id, ctx.as_of_date)]
    opponent_grad = grads[(ctx.opponent_id, ctx.as_of_date)]

    h2h_df = execute_df(
        _H2H_PRIOR_SQL,
        [ctx.player_id, ctx.opponent_id, ctx.opponent_id, ctx.player_id, ctx.as_of_iso],
    )
    if h2h_df.empty:
        h2h_wins = h2h_surface_meetings = h2h_surface_wins = 0
        h2h_exposure = 0
    else:
        winner_id_values = h2h_df["winner_id"].tolist()
        surface_values = h2h_df["surface"].tolist()
        h2h_exposure = len(winner_id_values)
        h2h_wins = sum(1 for w in winner_id_values if str(w) == ctx.player_id)
        on_surface = [s == ctx.surface for s in surface_values]
        h2h_surface_meetings = int(sum(on_surface))
        h2h_surface_wins = sum(
            1
            for w, matches in zip(winner_id_values, on_surface, strict=True)
            if str(w) == ctx.player_id and matches
        )

    row = _assemble_row(
        ctx,
        player_side,
        opponent_side,
        h2h_exposure,
        h2h_wins,
        h2h_surface_meetings,
        h2h_surface_wins,
        player_elo,
        opponent_elo,
        player_grad,
        opponent_grad,
    )

    final_cols = [*FEATURE_COLS, "player_id", "opponent_id"]
    out = pd.DataFrame({col: [row[col]] for col in final_cols})
    assert not out.isnull().to_numpy().any(), "inference row contains NaN"

    meta: dict[str, object] = {
        "raw_player_id": ctx.raw_player_id,
        "raw_opponent_id": ctx.raw_opponent_id,
        "player_id": ctx.player_id,
        "opponent_id": ctx.opponent_id,
        "surface": ctx.surface,
        "as_of_date": ctx.as_of_iso,
        "tournament_level": ctx.tournament_level,
        "round_encoded": ctx.round_encoded,
        "feature_count": len(FEATURE_COLS),
        "pool_as_of_date": _to_date(ta["pool_as_of_date"]).isoformat(),
        "snapshot_pool_rows": int(float(cast(Real, ta["snapshot_pool_rows"] or 0))),
        "snapshot_pool_players": int(float(cast(Real, ta["snapshot_pool_players"] or 0))),
        "profile_rows": int(float(cast(Real, ta["profile_rows"] or 0))),
        "player_match_rows": int(float(cast(Real, ta["player_match_rows"] or 0))),
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
        "h2h_prior_meetings": h2h_exposure,
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
    best_of: int = 3,
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
        best_of=best_of,
    )
    return out


def build_inference_features_bulk(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build many directional rows with set-oriented queries, preserving input order."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must be a non-empty list of match contexts")
    if len(rows) > BULK_MAX_ROWS:
        raise ValueError(f"bulk inference accepts at most {BULK_MAX_ROWS} rows, got {len(rows)}")

    ctxs = [_normalize_inputs(**cast(dict[str, Any], row)) for row in rows]
    ta = load_tour_averages()
    defaults = cast(dict[str, float], ta)

    pairs = list({(pid, c.as_of_date) for c in ctxs for pid in (c.player_id, c.opponent_id)})
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

    profiles: dict[str, dict[str, object]] = {}
    players = sorted({p for c in ctxs for p in (c.player_id, c.opponent_id)})
    for rec in execute_df(_PROFILES_BULK_SQL, [players]).to_dict("records"):
        profiles[str(rec["player_id"])] = cast(dict[str, object], rec)

    h2h: dict[tuple[str, str, str], list[tuple[str, str, str]]] = {}
    h2h_triples = list({(c.player_id, c.opponent_id, c.as_of_iso) for c in ctxs})
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
        key = (
            rec["req_player_id"],
            rec["req_opponent_id"],
            _to_date(rec["as_of_iso"]).isoformat(),
        )
        h2h.setdefault(key, []).append(
            (str(rec["match_id"]), str(rec["winner_id"]), str(rec["surface"]))
        )

    elo_overall = _elo_ratings_bulk(
        list({(pid, c.as_of_date) for c in ctxs for pid in (c.player_id, c.opponent_id)})
    )
    grad_overall = _elo_gradients_bulk(
        list({(pid, c.as_of_date) for c in ctxs for pid in (c.player_id, c.opponent_id)})
    )

    out_rows: list[dict[str, int | float | str]] = []
    for ctx in ctxs:
        player_snapshot = snapshots.get((ctx.player_id, ctx.as_of_date))
        opponent_snapshot = snapshots.get((ctx.opponent_id, ctx.as_of_date))
        player_side = _side_values(
            player_snapshot,
            defaults,
            surface=ctx.surface,
            as_of_date=ctx.as_of_date,
        )
        opponent_side = _side_values(
            opponent_snapshot,
            defaults,
            surface=ctx.surface,
            as_of_date=ctx.as_of_date,
        )
        player_side.update(
            _profile_values_from_row(profiles.get(ctx.player_id), ctx.as_of_date, defaults)
        )
        opponent_side.update(
            _profile_values_from_row(profiles.get(ctx.opponent_id), ctx.as_of_date, defaults)
        )
        player_elo = elo_overall[(ctx.player_id, ctx.as_of_date)]
        opponent_elo = elo_overall[(ctx.opponent_id, ctx.as_of_date)]
        player_grad = grad_overall[(ctx.player_id, ctx.as_of_date)]
        opponent_grad = grad_overall[(ctx.opponent_id, ctx.as_of_date)]
        meetings = h2h.get((ctx.player_id, ctx.opponent_id, ctx.as_of_iso), [])
        winner_id_values = [w for _, w, _ in meetings]
        surface_values = [s for _, _, s in meetings]
        on_surface = [s == ctx.surface for s in surface_values]
        h2h_wins = sum(1 for w in winner_id_values if w == ctx.player_id)
        h2h_surface_wins = sum(
            1
            for w, matches in zip(winner_id_values, on_surface, strict=True)
            if w == ctx.player_id and matches
        )
        out_rows.append(
            _assemble_row(
                ctx,
                player_side,
                opponent_side,
                len(winner_id_values),
                h2h_wins,
                int(sum(on_surface)),
                h2h_surface_wins,
                player_elo,
                opponent_elo,
                player_grad,
                opponent_grad,
            )
        )

    out = pd.DataFrame(out_rows, columns=[*FEATURE_COLS, "player_id", "opponent_id"])
    assert not out.isnull().to_numpy().any(), "bulk inference rows contain NaN"
    return out


def build_gru_request_inputs(
    rows: list[dict[str, object]],
    preprocessing: GRUPreprocessing,
) -> GRUBatch:
    """Build runtime GRU (``nn``) tensors from raw request dicts.

    Delegates to :mod:`src.features.nn_inference`, which first builds the
    validated directional rows via :func:`build_inference_features_bulk`, then
    constructs set-wise GRU history/context tensors. Imported lazily to keep the
    heavy GRU transform path out of callers that never need it.
    """
    from src.features import nn_inference

    return nn_inference.build_gru_request_inputs(rows, preprocessing)
