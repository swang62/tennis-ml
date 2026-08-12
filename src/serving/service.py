"""BentoML service for the stacked ensemble and read-only dashboard data.

The NN uses the deploy-time ONNX artifact; finalized features are built from
ids in-service (scalar and bulk) against the live PostgreSQL gold tables.
"""

import builtins
import json
import logging
import math
import os
import pickle
from collections.abc import Callable
from datetime import date, datetime
from decimal import Decimal
from time import perf_counter
from typing import Any, cast

import bentoml
import numpy as np
import onnxruntime as ort
import pandas as pd
from bentoml.exceptions import InvalidArgument
from bentoml.images import Image
from starlette.applications import Starlette
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from src.constants import (
    BRONZE_PROFILES_TABLE,
    BRONZE_TABLE,
    PRODUCTION_MODEL,
    PROFILES_TABLE,
    RANKINGS_TABLE,
    ROOT,
    SILVER_PLAYER_MATCHES,
    TOUR_AVERAGES_TABLE,
)
from src.countries import resolve_ioc, valid_ioc
from src.db.client import execute_df, first_row_dict
from src.db.init_db import init as init_db
from src.features.columns import FEATURE_COLS
from src.features.inference import (
    _build_inference_features_with_meta,
    _to_date,
    build_inference_features_bulk,
)
from src.models.similarity import PlayerSimilarity
from src.utils import load_env

AUX_DIR = ROOT / "data" / "processed"

# Canonical champion manifest baked at deploy time (written by deploy.py from
# the champion's exact lineage tags; packaged via bentofile.yaml).
MODEL_INFO_FILE = AUX_DIR / "model_info.json"

load_env()

# Set LOG_LEVEL=DEBUG to enable per-request observability prints.
# Suppress BentoML's noisy service lifecycle INFO (initialized/cleanup spam).
logging.getLogger("bentoml").setLevel(logging.WARNING)
_log = logging.getLogger("tennis_ml.serving")
_log.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO))
if not _log.handlers:
    _log.addHandler(logging.StreamHandler())

# Serving dependencies stay here. Model packages are pinned because models are pickled.
# base_image avoids BentoML's default build-essential injection; ca-certificates
# and bash are the only system deps needed at runtime.
SERVING_IMAGE = Image(
    base_image="python:3.12-slim",
    distro="",
    lock_python_packages=False,
    commands=[
        "apt-get update && apt-get install -y --no-install-recommends ca-certificates bash libgomp1 && rm -rf /var/lib/apt/lists/*"
    ],
).python_packages(
    "bentoml==1.4.39",
    "scikit-learn==1.8.0",
    "xgboost-cpu==3.2.0",
    "lightgbm==4.6.0",
    "psycopg[binary]==3.3.4",
    "pandas==2.3.3",
    "numpy==2.4.6",
    "onnxruntime==1.27.0",
    "faiss-cpu==1.14.3",
    "python-dotenv==1.2.2",
)


# ── Read-only data endpoints (GET, no auth) ────────────────────────────────
# Bento APIs are POST-only, so dashboard GET routes use a mounted Starlette app.
# SQL values are always parameterized; response shapes are a dashboard contract.

import psycopg.errors as _pg_errors


def _safe_query(sql: str, params: list | None = None) -> pd.DataFrame:
    """Run *sql* and return the result; return empty DataFrame only when a
    dbt-created relation (silver/gold table) does not exist yet, so the web
    dashboard renders an empty state instead of a 500 before ETL. Other
    failures (bad columns, connection errors) propagate to the 500 handler.
    """
    try:
        return execute_df(sql, params)
    except _pg_errors.UndefinedTable:
        _log.warning("safe query returned empty: dbt tables may not exist yet")
        return pd.DataFrame()


# Response envelope used by every data endpoint.
def _ok(data: dict[str, object]) -> JSONResponse:
    return JSONResponse({"ok": True, "data": data})


def _err(status: int, message: str) -> JSONResponse:
    return JSONResponse({"ok": False, "error": message}, status_code=status)


def _iso(value: object) -> object:
    """Convert database and pandas scalars to JSON-safe values."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (float, Decimal, np.floating)):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, (np.integer, np.bool_)):
        return int(value)
    return value


def _records(df: pd.DataFrame) -> list[dict[str, object]]:
    return [{str(k): _iso(v) for k, v in row.items()} for row in df.to_dict("records")]


def _complement(value: object) -> object:
    """1 - benchmark for return-side tour comparisons; None when the benchmark is null."""
    number = _iso(value)
    if number is None:
        return None
    return 1.0 - cast(float, number)


def _delta(player_value: object, tour_value: object) -> object:
    """Player-minus-tour delta; None when either side is unavailable."""
    player, tour = _iso(player_value), _iso(tour_value)
    if player is None or tour is None:
        return None
    return cast(float, player) - cast(float, tour)


def _require_id(request: Request, name: str) -> str | None:
    """Return the non-blank query param value or None (caller turns None into a 400)."""
    raw = request.query_params.get(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip()


_NORMALIZE_KEYS = frozenset(
    {
        "player_id",
        "opponent_id",
        "surface",
        "as_of_date",
        "tournament_level",
        "round_encoded",
        "tournament",
        "round",
        "is_indoor",
        "indoor",
    }
)


def _predict_from_ids_bulk_impl(
    rows: list[dict[str, object]], predict_proba: object
) -> pd.DataFrame:
    """Build + predict a batch; `predict_proba` is the service's shared ensemble.

    Each row accepts the same fields as `predict_from_ids` (including its own
    historical `as_of_date`); the endpoint's `indoor` field maps to the
    builder's `is_indoor`. Returns a DataFrame with the finalized FEATURE_COLS
    plus ids and the four probability columns, in input order.
    """
    started_at = perf_counter()
    normalized: list[dict[str, object]] = []
    for row in rows:
        # Strip BentoML auto-generated Pydantic extras before normalize.
        r = {k: v for k, v in row.items() if k in _NORMALIZE_KEYS}
        if "indoor" in r:  # scalar endpoint field -> builder field
            r["is_indoor"] = r.pop("indoor")
        if r.get("as_of_date") is not None:
            r["as_of_date"] = _to_date(r["as_of_date"])
        normalized.append(r)
    feature_df = build_inference_features_bulk(normalized)
    proba_df = cast(Callable[[pd.DataFrame], pd.DataFrame], predict_proba)(feature_df)
    out = feature_df.copy()
    for col in ("p_linear", "p_gbdt", "p_nn", "p_win"):
        out[col] = proba_df[col].to_numpy()
    _log.debug(
        "predict_from_ids_bulk_observability"
        f" rows={len(out)}"
        f" feature_count={len(FEATURE_COLS)}"
        f" build_ms={(perf_counter() - started_at) * 1000:.3f}"
        f" mean_p_win={float(out['p_win'].mean()):.6f}"
    )
    return out


# ── SQL (table names interpolated from constants; values always via `%s`) ──

# Directory read: bronze metadata (name/IOC) joined to the dbt-derived gold
# aggregates. current_rank is the player's latest official weekly rank
# (bronze.rankings), falling back to match-time rank from the most recent
# match when no ranking row exists — both materialized by dbt in gold.
_PLAYERS_SQL = f"""
SELECT bp.player_id, bp.display_name, bp.ioc,
       gp.match_count AS matches_played,
       gp.latest_rank_points,
       gp.current_rank
FROM {BRONZE_PROFILES_TABLE} bp
LEFT JOIN {PROFILES_TABLE} gp ON gp.player_id = bp.player_id
ORDER BY gp.current_rank NULLS LAST, bp.display_name, bp.player_id
"""

# One point query: bronze metadata (bp.*) joined to the dbt-materialized gold
# aggregates (gp.*) and the one-row tour singleton (cross join). current_rank
# is already materialized in gold.player_profiles via dbt (official ranking
# with match-time fallback); no per-query CTE needed. Tour comparison deltas
# are computed in Python, not SQL.
_PROFILE_SQL = f"""
SELECT
    bp.*,
    gp.match_count, gp.latest_match_date,
    gp.latest_rank_points, gp.earliest_rank_points,
    gp.earliest_rank_points_date, gp.latest_rank_points_date,
    gp.rank_points_delta, gp.current_rank,
    gp.first_serve_in_pct, gp.aces_per_first_serve,
    gp.first_serve_points_won_pct, gp.second_serve_points_won_pct,
    gp.overall_serve_points_won_pct, gp.double_faults_per_serve_point,
    gp.aces_per_service_game, gp.break_points_saved_pct,
    gp.return_points_won_pct, gp.first_serve_return_points_won_pct,
    gp.second_serve_return_points_won_pct, gp.break_point_conversion_pct,
    gp.break_point_opportunities_per_return_game,
    gp.hard_matches, gp.clay_matches, gp.grass_matches,
    gp.hard_win_rate, gp.clay_win_rate, gp.grass_win_rate,
    gp.recent_snapshot_date, gp.win_rate_10,
    ta.tour_first_serve_win_pct,
    ta.tour_second_serve_win_pct,
    ta.tour_ace_rate,
    ta.tour_first_serve_pct,
    ta.tour_break_points_saved_pct,
    ta.tour_serve_win_pct,
    ta.tour_return_points_won_pct,
    ta.tour_df_rate,
    ta.tour_aces_per_svc_game,
    ta.tour_break_point_opportunities_per_return_game
FROM {BRONZE_PROFILES_TABLE} bp
LEFT JOIN {PROFILES_TABLE} gp ON gp.player_id = bp.player_id
CROSS JOIN {TOUR_AVERAGES_TABLE} ta
WHERE bp.player_id = %s
"""

# Official weekly history only: bronze.rankings, chronological. Never derived
# from match rows. The response envelope stays {rank_date, rank}; points is
# selected for parity with the source but not exposed.
_RANK_HISTORY_SQL = f"""
SELECT ranking_date, rank, points
FROM {RANKINGS_TABLE}
WHERE player_id = %s
ORDER BY ranking_date
"""

# Individual match rows, newest first: no tournament dedup, so every round of
# an occurrence appears (e.g. the Rome final and its earlier rounds) up to the
# visible limit. bronze.match_date is the tournament start date, so
# same-occurrence rows tie and are broken deterministically by match_id.
_MATCH_HISTORY_SQL = f"""
SELECT
    pm.match_id, pm.match_date, br.tournament, br.tournament_name, pm.surface, br.round,
    pm.opponent_id, pr.display_name AS opponent_name,
    pm.opponent_ranking, pm.match_won,
    pm.aces, pm.double_faults,
    pm.first_serve_points_won, pm.second_serve_points_won,
    pm.total_serve_points, pm.service_games,
    pm.break_points_saved, pm.break_points_faced
FROM {SILVER_PLAYER_MATCHES} pm
LEFT JOIN {BRONZE_TABLE} br ON br.match_id = pm.match_id
LEFT JOIN {BRONZE_PROFILES_TABLE} pr ON pr.player_id = pm.opponent_id
WHERE pm.player_id = %s
ORDER BY pm.match_date DESC, pm.match_id DESC
LIMIT %s
"""

# Direct bronze pair read: one row per meeting, no silver expansion or dedup.
_H2H_MEETINGS_SQL = f"""
SELECT match_id, match_date, tournament, round, surface,
       player1_id, player2_id, winner_id
FROM {BRONZE_TABLE}
WHERE (player1_id = %s AND player2_id = %s)
   OR (player1_id = %s AND player2_id = %s)
ORDER BY match_date DESC, match_id DESC
LIMIT %s
"""


# ── Route handlers ─────────────────────────────────────────────────────────


def _players(_request: Request) -> JSONResponse:
    try:
        df = _safe_query(_PLAYERS_SQL)
        players = []
        for r in _records(df):
            ioc = valid_ioc(r.get("ioc"))
            iso2, country_name = resolve_ioc(ioc)
            players.append({**r, "ioc": ioc, "iso2": iso2, "country_name": country_name})
        return _ok({"players": players})
    except Exception as exc:  # DB errors -> 500 with message
        return _err(500, f"players query failed: {exc}")


def _player_profile(request: Request) -> JSONResponse:
    player_id = _require_id(request, "player_id")
    if player_id is None:
        return _err(400, "missing required query parameter: player_id")
    try:
        df = _safe_query(_PROFILE_SQL, [player_id])
    except Exception as exc:
        return _err(500, f"profile query failed: {exc}")
    # One point query: the player's materialized profile row plus the tour
    # singleton (cross join). An empty result means the player is unknown.
    if df.empty:
        return _err(404, f"unknown player_id: {player_id}")
    row = first_row_dict(df)

    def delta(col: str, tour_value: object) -> object:
        return _delta(row[col], tour_value)

    serve = {
        "first_serve_in_pct": _iso(row["first_serve_in_pct"]),
        "aces_per_first_serve": _iso(row["aces_per_first_serve"]),
        "first_serve_points_won_pct": _iso(row["first_serve_points_won_pct"]),
        "second_serve_points_won_pct": _iso(row["second_serve_points_won_pct"]),
        "overall_serve_points_won_pct": _iso(row["overall_serve_points_won_pct"]),
        "double_faults_per_serve_point": _iso(row["double_faults_per_serve_point"]),
        "aces_per_service_game": _iso(row["aces_per_service_game"]),
        "break_points_saved_pct": _iso(row["break_points_saved_pct"]),
    }
    return_metrics = {
        "return_points_won_pct": _iso(row["return_points_won_pct"]),
        "first_serve_return_points_won_pct": _iso(row["first_serve_return_points_won_pct"]),
        "second_serve_return_points_won_pct": _iso(row["second_serve_return_points_won_pct"]),
        "break_point_conversion_pct": _iso(row["break_point_conversion_pct"]),
        "break_point_opportunities_per_return_game": _iso(
            row["break_point_opportunities_per_return_game"]
        ),
    }

    # Deltas are player minus tour, computed here — never in SQL. The three
    # return-side benchmarks derive from their serve complements; nulls flow
    # through unchanged.
    tour_comparisons = {
        "first_serve_in_pct": delta("first_serve_in_pct", row["tour_first_serve_pct"]),
        "aces_per_first_serve": delta("aces_per_first_serve", row["tour_ace_rate"]),
        "first_serve_points_won_pct": delta(
            "first_serve_points_won_pct", row["tour_first_serve_win_pct"]
        ),
        "second_serve_points_won_pct": delta(
            "second_serve_points_won_pct", row["tour_second_serve_win_pct"]
        ),
        "overall_serve_points_won_pct": delta(
            "overall_serve_points_won_pct", row["tour_serve_win_pct"]
        ),
        "double_faults_per_serve_point": delta(
            "double_faults_per_serve_point", row["tour_df_rate"]
        ),
        "aces_per_service_game": delta("aces_per_service_game", row["tour_aces_per_svc_game"]),
        "break_points_saved_pct": delta(
            "break_points_saved_pct", row["tour_break_points_saved_pct"]
        ),
        "return_points_won_pct": delta("return_points_won_pct", row["tour_return_points_won_pct"]),
        "first_serve_return_points_won_pct": delta(
            "first_serve_return_points_won_pct",
            _complement(row["tour_first_serve_win_pct"]),
        ),
        "second_serve_return_points_won_pct": delta(
            "second_serve_return_points_won_pct",
            _complement(row["tour_second_serve_win_pct"]),
        ),
        "break_point_conversion_pct": delta(
            "break_point_conversion_pct", _complement(row["tour_break_points_saved_pct"])
        ),
        "break_point_opportunities_per_return_game": delta(
            "break_point_opportunities_per_return_game",
            row["tour_break_point_opportunities_per_return_game"],
        ),
    }

    # Weighted tour benchmarks from the same row (the singleton side of the join).
    tour_averages_out = {
        "first_serve_win_pct": _iso(row["tour_first_serve_win_pct"]),
        "second_serve_win_pct": _iso(row["tour_second_serve_win_pct"]),
    }

    # Unplayed model surfaces are n/a (null win rate), not 0%.
    surface_rates = [
        {
            "surface": s,
            "matches": int(row[f"{s}_matches"]),
            "win_rate": _iso(row[f"{s}_win_rate"]),
        }
        for s in ("clay", "grass", "hard")
    ]

    # Recent form from the newest rolling snapshot (if the player has one).
    recent_form: dict[str, object] | None = None
    if row["recent_snapshot_date"] is not None:
        recent_form = {
            "snapshot_date": _iso(row["recent_snapshot_date"]),
            "last_10_win_rate": _iso(row["win_rate_10"]),
        }

    # Rank-points trend from the materialized latest vs earliest positive points.
    rank_points_trend: dict[str, object] | None = None
    if row["latest_rank_points"] is not None:
        rank_points_trend = {
            "earliest": _iso(row["earliest_rank_points"]),
            "latest": _iso(row["latest_rank_points"]),
            "delta": _iso(row["rank_points_delta"]),
        }

    # Country metadata resolved from the profile's stored IOC; missing/invalid
    # codes resolve to the UNK sentinel ("", "Country unknown").
    ioc = valid_ioc(row["ioc"])
    iso2, country_name = resolve_ioc(ioc)

    return _ok(
        {
            "player_id": row["player_id"],
            "display_name": row["display_name"],
            "handedness": row["handedness"],
            "backhand": row["backhand"],
            "height": _iso(row["height"]),
            "turned_pro": _iso(row["turned_pro"]),
            "birthplace": row["birthplace"],
            "summary": row["summary"],
            "atp_name": row["atp_name"],
            "birthdate": _iso(row["birthdate"]),
            "weight": _iso(row["weight"]),
            "coaches": row["coaches"],
            "ioc": ioc,
            "iso2": iso2,
            "country_name": country_name,
            "career": {
                "matches_played": int(row["match_count"]),
                "latest_match_date": _iso(row["latest_match_date"]),
            },
            "serve": serve,
            "return": return_metrics,
            "surface_rates": surface_rates,
            "recent_form": recent_form,
            "rank_points_trend": rank_points_trend,
            "rank": {
                "current_rank": _iso(row["current_rank"]),
                "latest_rank_points": _iso(row["latest_rank_points"]),
                "earliest_rank_points": _iso(row["earliest_rank_points"]),
                "earliest_rank_points_date": _iso(row["earliest_rank_points_date"]),
                "latest_rank_points_date": _iso(row["latest_rank_points_date"]),
                "rank_points_delta": _iso(row["rank_points_delta"]),
            },
            "tour_averages": tour_averages_out,
            "tour_comparisons": tour_comparisons,
        }
    )


def _rank_history(request: Request) -> JSONResponse:
    player_id = _require_id(request, "player_id")
    if player_id is None:
        return _err(400, "missing required query parameter: player_id")
    try:
        df = execute_df(_RANK_HISTORY_SQL, [player_id])
    except Exception as exc:
        return _err(500, f"rank history query failed: {exc}")
    history = [{"rank_date": _iso(r["ranking_date"]), "rank": r["rank"]} for r in _records(df)]
    return _ok({"player_id": player_id, "rank_history": history})


def _match_history(request: Request) -> JSONResponse:
    player_id = _require_id(request, "player_id")
    if player_id is None:
        return _err(400, "missing required query parameter: player_id")
    raw_limit = request.query_params.get("limit", "20")
    try:
        limit = int(raw_limit)
    except ValueError:
        return _err(400, "limit must be an integer")
    limit = max(1, min(limit, 100))  # clamp: dashboard never needs more
    try:
        df = _safe_query(_MATCH_HISTORY_SQL, [player_id, limit])
    except Exception as exc:
        return _err(500, f"match history query failed: {exc}")
    matches = []
    for r in _records(df):
        matches.append(
            {
                "match_id": r["match_id"],
                "match_date": r["match_date"],
                "tournament": r["tournament"],
                "tournament_name": r["tournament_name"] or None,
                "surface": r["surface"],
                "round": r["round"],
                "opponent_id": r["opponent_id"],
                "opponent_name": r["opponent_name"],
                "opponent_ranking": r["opponent_ranking"],
                "result": "won" if r["match_won"] == 1 else "lost",
                "aces": r["aces"],
                "double_faults": r["double_faults"],
                "first_serve_points_won": r["first_serve_points_won"],
                "second_serve_points_won": r["second_serve_points_won"],
                "total_serve_points": r["total_serve_points"],
                "service_games": r["service_games"],
                "break_points_saved": r["break_points_saved"],
                "break_points_faced": r["break_points_faced"],
            }
        )
    return _ok({"player_id": player_id, "matches": matches})


def _head_to_head(request: Request) -> JSONResponse:
    # Support both H2H and prediction parameter conventions.
    p1 = _require_id(request, "player1_id") or _require_id(request, "player_id")
    p2 = _require_id(request, "player2_id") or _require_id(request, "opponent_id")
    if p1 is None or p2 is None:
        return _err(
            400,
            "missing required query parameters: "
            "pass player1_id+player2_id (or player_id+opponent_id)",
        )
    # Canonical pair: lower id is the player_* side (matches model convention).
    lower, higher = sorted([p1, p2])
    raw_limit = request.query_params.get("limit", "100")
    try:
        limit = int(raw_limit)
    except ValueError:
        return _err(400, "limit must be an integer")
    limit = max(1, min(limit, 100))  # clamp: a pair never meets more than this
    try:
        df = execute_df(_H2H_MEETINGS_SQL, [lower, higher, higher, lower, limit])
    except Exception as exc:
        return _err(500, f"head-to-head query failed: {exc}")

    meetings = []
    for r in _records(df):
        winner_id = r["winner_id"]
        player1_won = winner_id == lower
        meetings.append(
            {
                "match_id": r["match_id"],
                "match_date": r["match_date"],
                "surface": r["surface"],
                "tournament": r["tournament"],
                "round": r["round"],
                "winner_id": winner_id,
                "loser_id": higher if player1_won else lower,
                "player1_won": bool(player1_won),
            }
        )

    n = len(meetings)
    p1_wins = sum(1 for m in meetings if m["player1_won"])
    # Last-5 win rate mirrors the model's H2H convention (most recent 5).
    last5 = meetings[:5]
    last5_p1_wins = sum(1 for m in last5 if m["player1_won"])
    summary = {
        "meetings": n,
        "player1_wins": p1_wins,
        "player2_wins": n - p1_wins,
        "player1_win_rate": round(p1_wins / n, 4) if n else 0.5,
        "last5_player1_win_rate": round(last5_p1_wins / len(last5), 4) if last5 else 0.5,
    }
    return _ok(
        {
            "player1_id": lower,
            "player2_id": higher,
            "meetings": meetings,
            "summary": summary,
        }
    )


# Lazy-load the packaged index so health checks do not pay its load cost.
_similarity_finder: PlayerSimilarity | None = None


def _get_similarity_finder() -> PlayerSimilarity | None:
    global _similarity_finder
    if _similarity_finder is None:
        finder = PlayerSimilarity()
        try:
            finder.load()
        except FileNotFoundError:
            _log.warning("similarity index not found — /similar_players returns empty")
            return None
        _similarity_finder = finder
    return _similarity_finder


def _similar_players(request: Request) -> JSONResponse:
    player_id = _require_id(request, "player_id")
    if player_id is None:
        return _err(400, "missing required query parameter: player_id")
    raw_limit = request.query_params.get("limit", "3")
    try:
        limit = int(raw_limit)
    except ValueError:
        return _err(400, "limit must be an integer")
    limit = max(1, min(limit, 3))  # bounded: a profile never shows more than 3
    try:
        finder = _get_similarity_finder()
        similar = finder.search(player_id, top_k=limit) if finder is not None else []
    except Exception as exc:
        return _err(500, f"similar players query failed: {exc}")
    # Entries carry player_id + display_name (score is a number, not an id);
    # the id is used only for the profile link, never rendered.
    return _ok({"player_id": player_id, "similar_players": similar})


def _non_secret_database_meta() -> dict[str, object]:
    """Non-secret connection metadata parsed from DATABASE_URL.

    Reports server address, port, and database name only — never credentials or
    the connection URL itself.
    """
    from urllib.parse import unquote, urlsplit

    raw = os.environ.get("DATABASE_URL") or ""
    if not raw:
        return {"server_address": None, "server_port": None, "database_name": None}
    parts = urlsplit(raw)
    db_name = unquote(parts.path.lstrip("/")) if parts.path else None
    return {
        "server_address": parts.hostname,
        "server_port": parts.port,
        "database_name": db_name or None,
    }


def _model_info(_request: Request) -> JSONResponse:
    """Baked champion manifest, deployment mode, and non-secret DB metadata.

    Production mode is claimed only when the image runs with SERVING_MODE=production
    AND the baked manifest is present; source-mode local serving always reports
    development.
    """
    manifest: dict[str, object] | None = None
    try:
        manifest = json.loads(MODEL_INFO_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        manifest = None
    production = os.environ.get("SERVING_MODE") == "production" and manifest is not None
    return _ok(
        {
            "mode": "production" if production else "development",
            "manifest": manifest,
            "database": _non_secret_database_meta(),
        }
    )


def _health(_request: Request) -> JSONResponse:
    """Liveness plus PostgreSQL reachability via an authenticated SELECT 1.

    A 200 implies the service finished initializing (models loaded, schema
    bootstrapped) — DATA_APP only answers once the service is serving — and
    that PostgreSQL is reachable right now. The error body is a static
    message so no connection details leak.
    """
    try:
        execute_df("SELECT 1")
    except Exception:
        _log.warning("health check failed: database unreachable")
        return _err(503, "database unavailable")
    return _ok({"status": "healthy"})


# Mounted at the service root; coexists with the POST-only @bentoml.api routes
# (the SDK's server checks its own routes first, then falls through to mounts).


async def _handle_starlette_error(_request: Request, exc: StarletteHTTPException):
    """Return every Starlette HTTPException as a json {ok, error} envelope."""
    return JSONResponse(
        {"ok": False, "error": exc.detail or "internal error"},
        status_code=exc.status_code,
        headers=getattr(exc, "headers", None) or {},
    )


async def _catch_all_error(_request: Request, _exc: Exception):
    """Catch-all for unexpected errors — log and return 500."""
    _log.exception("unhandled server error")
    return JSONResponse(
        {"ok": False, "error": "internal server error"},
        status_code=500,
    )


DATA_APP = Starlette(
    exception_handlers={  # type: ignore[arg-type]  # Starlette typing doesn't narrow per-key
        StarletteHTTPException: _handle_starlette_error,
        Exception: _catch_all_error,
    },
    routes=[
        Route("/players", _players, methods=["GET"]),
        Route("/player_profile", _player_profile, methods=["GET"]),
        Route("/rank_history", _rank_history, methods=["GET"]),
        Route("/match_history", _match_history, methods=["GET"]),
        Route("/head_to_head", _head_to_head, methods=["GET"]),
        Route("/similar_players", _similar_players, methods=["GET"]),
        Route("/model_info", _model_info, methods=["GET"]),
        Route("/health", _health, methods=["GET"]),
    ],
)


class _LGBMProbaAdapter:
    """Expose sklearn-style predict_proba over a native LightGBM Booster.

    bentoml.lightgbm.load_model returns the native Booster, which has no
    predict_proba; Booster.predict already returns P(class 1) for the binary
    objective, so predict_proba stacks [1 - p, p] to match the sklearn API.
    """

    def __init__(self, booster: Any) -> None:
        self._booster = booster

    def predict_proba(self, X: Any) -> np.ndarray:
        p = np.asarray(self._booster.predict(X), dtype=np.float64)
        return np.column_stack([1.0 - p, p])


@bentoml.service(
    image=SERVING_IMAGE,
    # Aligned with Nginx's 120s operational batch window so a large
    # (<=1,000 row) batch is not killed by the serving layer first.
    traffic={"timeout": 120},
    resources={"cpu": "500m"},
)
@bentoml.asgi_app(DATA_APP, path="/")
class TennisPredictor:
    bento_linear = bentoml.models.BentoModel("linear_best:latest")
    bento_gbdt = bentoml.models.BentoModel("gbdt_best:latest")
    # NN is not a BentoModel here — served from data/processed/nn_best.onnx
    # (materialized at deploy time from the pinned nn_best MLflow version).
    bento_production = bentoml.models.BentoModel(f"{PRODUCTION_MODEL}:latest")

    def __init__(self):
        # Idempotent schema bootstrap (applies init.sql). Must run before
        # any read endpoint queries the tables it creates.
        init_db()
        self.linear: Any = bentoml.sklearn.load_model(self.bento_linear)
        manifest = json.loads(MODEL_INFO_FILE.read_text())
        gbdt_framework = manifest["bases"]["gbdt"]["framework"]
        # xgboost/lightgbm use OpenMP, which can deadlock inside BentoML's
        # forked worker processes. Force single-threaded during model load.
        _old_omp = os.environ.get("OMP_NUM_THREADS")
        os.environ["OMP_NUM_THREADS"] = "1"
        try:
            if gbdt_framework == "xgboost":
                self.gbdt: Any = bentoml.xgboost.load_model(self.bento_gbdt)
            else:
                # bentoml.lightgbm.load_model returns a Booster (no predict_proba);
                # the adapter restores the sklearn-style interface.
                self.gbdt = _LGBMProbaAdapter(bentoml.lightgbm.load_model(self.bento_gbdt))
        finally:
            if _old_omp is not None:
                os.environ["OMP_NUM_THREADS"] = _old_omp
        self.production: Any = bentoml.sklearn.load_model(self.bento_production)
        self.nn_session = ort.InferenceSession(str(AUX_DIR / "nn_best.onnx"))

        with open(AUX_DIR / "linear_scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        with open(AUX_DIR / "bio_feature_cols.json") as f:
            self.bio_feature_cols = json.load(f)
        bio_data = np.load(str(AUX_DIR / "bio_embeddings.npz"), allow_pickle=True)
        # player_ids is a string array (object dtype, requires pickle); vectors is float32.
        self.bio_by_player = {pid: i for i, pid in enumerate(bio_data["player_ids"])}
        self.bio_array = bio_data["vectors"].astype(np.float32)

    def _predict_proba(self, input: pd.DataFrame) -> pd.DataFrame:
        """Run the stacked ensemble without recursively calling the HTTP endpoint."""
        started_at = perf_counter()
        features = input[FEATURE_COLS]

        # Linear + NN paths share the persisted train-fit scaler
        # (StandardScaler().fit(X_train), same contract the NN was trained on).
        scale_started_at = perf_counter()
        features_scaled = self.scaler.transform(features)
        scale_ms = (perf_counter() - scale_started_at) * 1000
        # Linear path: finalized row -> persisted scaler -> classifier.
        linear_started_at = perf_counter()
        p_linear = self.linear.predict_proba(features_scaled)[:, 1]
        linear_ms = (perf_counter() - linear_started_at) * 1000
        # GBDT path: raw finalized row.
        gbdt_started_at = perf_counter()
        p_gbdt = self.gbdt.predict_proba(features)[:, 1]
        gbdt_ms = (perf_counter() - gbdt_started_at) * 1000
        # ONNX inputs match the training forward signature.
        nn_inputs = {
            "tab": features_scaled.astype(np.float32),
            "bio_p": self._row_bio_np(input["player_id"].to_numpy()),
            "bio_o": self._row_bio_np(input["opponent_id"].to_numpy()),
        }
        nn_started_at = perf_counter()
        nn_logits = np.asarray(self.nn_session.run(None, nn_inputs)[0])
        p_nn = 1.0 / (1.0 + np.exp(-nn_logits.reshape(-1)))
        nn_ms = (perf_counter() - nn_started_at) * 1000

        # LR head: stack of base-model probabilities.
        stack = np.column_stack([p_linear, p_gbdt, p_nn])
        ensemble_started_at = perf_counter()
        p_win = self.production.predict_proba(stack)[:, 1]
        ensemble_ms = (perf_counter() - ensemble_started_at) * 1000

        # Aggregate-only observability: means (no per-row dumps). A single row
        # additionally logs its ids, preserving the scalar path's detail.
        first = input.iloc[0] if not input.empty else None
        first_ident = (
            f" player_id={first['player_id']} opponent_id={first['opponent_id']}"
            if first is not None and len(input) == 1
            else ""
        )
        _log.debug(
            "predict_observability"
            f" rows={len(input)}"
            f" feature_count={len(FEATURE_COLS)}"
            f" scale_ms={scale_ms:.3f}"
            f" linear_ms={linear_ms:.3f}"
            f" gbdt_ms={gbdt_ms:.3f}"
            f" nn_ms={nn_ms:.3f}"
            f" ensemble_ms={ensemble_ms:.3f}"
            f" total_ms={(perf_counter() - started_at) * 1000:.3f}"
            f" mean_p_win={float(p_win.mean()) if len(p_win) else float('nan'):.6f}"
            f" mean_p_linear={float(p_linear.mean()) if len(p_linear) else float('nan'):.6f}"
            f" mean_p_gbdt={float(p_gbdt.mean()) if len(p_gbdt) else float('nan'):.6f}"
            f" mean_p_nn={float(p_nn.mean()) if len(p_nn) else float('nan'):.6f}"
            f"{first_ident}"
        )

        return pd.DataFrame(
            {
                "player_id": input["player_id"].to_numpy(),
                "opponent_id": input["opponent_id"].to_numpy(),
                "p_win": p_win,
                "p_linear": p_linear,
                "p_gbdt": p_gbdt,
                "p_nn": p_nn,
            },
            index=input.index,
        )

    @bentoml.api
    def predict_from_ids(
        self,
        player_id: str,
        opponent_id: str,
        surface: str,
        *,
        tournament_level: int = 0,
        round_encoded: int = 0,
        tournament: str | None = None,
        round: str | None = None,
        as_of_date: date | None = None,
        indoor: int | None = None,
    ) -> dict[str, object]:
        """Build an as-of-dated feature row from player ids, then predict."""
        started_at = perf_counter()
        row, meta = _build_inference_features_with_meta(
            player_id,
            opponent_id,
            surface,
            tournament_level=tournament_level,
            round_encoded=round_encoded,
            tournament=tournament,
            round=round,
            as_of_date=as_of_date,
            is_indoor=indoor,
        )
        # Reuse the shared prediction path (no nested HTTP — see _predict_proba).
        out_df = self._predict_proba(row)
        # One row in, one row out — return the first record as a flat dict for
        # ergonomic JSON over HTTP.
        rec = first_row_dict(out_df)
        rec["p_win"] = builtins.round(float(rec["p_win"]), 4)
        rec["p_linear"] = builtins.round(float(rec["p_linear"]), 4)
        rec["p_gbdt"] = builtins.round(float(rec["p_gbdt"]), 4)
        rec["p_nn"] = builtins.round(float(rec["p_nn"]), 4)
        rec["predicted_winner"] = (
            rec["player_id"] if float(rec["p_win"]) >= 0.5 else rec["opponent_id"]
        )
        rec["response_ms"] = (perf_counter() - started_at) * 1000
        _log.debug(
            "predict_from_ids_observability"
            f" raw_player_id={meta['raw_player_id']}"
            f" raw_opponent_id={meta['raw_opponent_id']}"
            f" canonical_player_id={meta['canonical_player_id']}"
            f" canonical_opponent_id={meta['canonical_opponent_id']}"
            f" surface={meta['surface']}"
            f" as_of_date={meta['as_of_date']}"
            f" tournament_level={meta['tournament_level']}"
            f" round_encoded={meta['round_encoded']}"
            f" feature_count={meta['feature_count']}"
            f" snapshot_pool_rows={meta['snapshot_pool_rows']}"
            f" snapshot_pool_players={meta['snapshot_pool_players']}"
            f" profile_rows={meta['profile_rows']}"
            f" player_snapshot_found={meta['player_snapshot_found']}"
            f" opponent_snapshot_found={meta['opponent_snapshot_found']}"
            f" player_snapshot_date={meta['player_snapshot_date']}"
            f" opponent_snapshot_date={meta['opponent_snapshot_date']}"
            f" player_rolling_match_number={meta['player_rolling_match_number']}"
            f" opponent_rolling_match_number={meta['opponent_rolling_match_number']}"
            f" player_matches_30d={meta['player_matches_30d']}"
            f" opponent_matches_30d={meta['opponent_matches_30d']}"
            f" player_days_since_last_match={meta['player_days_since_last_match']}"
            f" opponent_days_since_last_match={meta['opponent_days_since_last_match']}"
            f" median_days_since={meta['median_days_since']}"
            f" median_matches_30d={meta['median_matches_30d']}"
            f" build_ms={meta['build_ms']}"
            f" response_ms={(perf_counter() - started_at) * 1000:.3f}"
            f" p_win={rec['p_win']:.6f}"
        )
        return rec

    def _predict_from_ids_bulk(self, rows: list[dict[str, object]]) -> pd.DataFrame:
        """Shared bulk path: normalize JSON rows, build features, predict once."""
        return _predict_from_ids_bulk_impl(rows, self._predict_proba)

    @bentoml.api
    def predict_from_ids_bulk(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        """Bulk predictions from minimal per-row contexts (internal endpoint).

        Each row accepts the same fields/defaults as `predict_from_ids`,
        including a per-row historical `as_of_date`. The batch is capped at
        1,000 rows and the ensemble runs once for the whole batch. This
        endpoint is internal: Nginx does not expose it publicly.
        """
        if not rows:
            raise InvalidArgument("rows must be a non-empty list")
        if len(rows) > 1000:
            raise InvalidArgument(f"max 1000 rows, got {len(rows)}")
        try:
            return _records(self._predict_from_ids_bulk(rows))
        except Exception:
            _log.exception("predict_from_ids_bulk failed")
            raise InvalidArgument("prediction failed — check input row format") from None

    def _row_bio_np(self, ids: np.ndarray) -> np.ndarray:
        """Map player ids to bio vectors (np.float32), zero-filled for unknown players."""
        out = np.zeros((len(ids), len(self.bio_feature_cols)), dtype=np.float32)
        for i, pid in enumerate(ids):
            j = self.bio_by_player.get(pid)
            if j is not None:
                out[i] = self.bio_array[j]
        return out
