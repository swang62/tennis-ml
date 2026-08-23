"""BentoML service for predictions and read-only dashboard data."""

import builtins
import json
import logging
import math
import os
import pickle
from collections.abc import Callable
from datetime import date, datetime
from decimal import Decimal
from enum import IntEnum, StrEnum
from time import perf_counter
from typing import Any, cast

import bentoml
import numpy as np
import onnxruntime as ort
import pandas as pd
import psycopg.errors as _pg_errors
from bentoml.exceptions import InvalidArgument
from bentoml.images import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.applications import Starlette
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from src.constants import (
    BRONZE_MATCHES_TABLE,
    BRONZE_PROFILES_TABLE,
    BRONZE_RANKINGS_TABLE,
    BULK_MAX_ROWS,
    DEPLOY_ARTIFACTS,
    FRAMEWORK_KEY,
    GOLD_PROFILES_TABLE,
    PRODUCTION_MODEL,
    SILVER_PLAYER_MATCHES,
    STACK_ORDER,
    TOUR_AVERAGES_TABLE,
    GBDTFramework,
    load_env,
    normalize_gbdt_framework,
)
from src.db.client import execute_df, first_row_dict
from src.evaluate.calibration import apply_temperature
from src.evaluate.symmetry import antisymmetric_evidence, evidence_to_probability
from src.features.columns import FEATURE_COLS
from src.features.inference import (
    _build_inference_features_with_meta,
    _to_date,
    build_inference_features_bulk,
)
from src.serving.directory import PLAYERS_SQL, directory_players
from src.training.similarity import PlayerSimilarity
from src.utils.countries import resolve_ioc, valid_ioc

# Deploy writes the champion manifest from its exact lineage tags.
AUX_DIR = DEPLOY_ARTIFACTS
MODEL_INFO_FILE = AUX_DIR / "model_info.json"


def _validate_feature_contract(estimator: Any, artifact_name: str) -> None:
    """Fail readiness when a fitted artifact disagrees with serving features."""
    fitted = getattr(estimator, "feature_names_in_", None)
    if fitted is None:
        return
    actual = [str(name) for name in fitted]
    expected = list(FEATURE_COLS)
    if actual != expected:
        missing = [name for name in actual if name not in expected]
        added = [name for name in expected if name not in actual]
        raise RuntimeError(
            f"{artifact_name} feature contract mismatch: fitted={len(actual)}, "
            f"serving={len(expected)}, missing_from_serving={missing}, "
            f"missing_from_model={added}; retrain before serving"
        )


# Pin model dependencies because the artifacts are pickled. Install only runtime OS deps.
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
    "psycopg-pool==3.3.1",
    "pandas==2.3.3",
    "numpy==2.4.6",
    "onnxruntime==1.27.0",
    "faiss-cpu==1.14.3",
    "python-dotenv==1.2.2",
)

_NORMALIZE_KEYS = frozenset(
    {
        "player_id",
        "opponent_id",
        "surface",
        "best_of",
        "as_of_date",
        "tournament_level",
        "round_encoded",
        "tournament",
        "round",
        "is_indoor",
        "indoor",
    }
)

# ── SQL (table names interpolated from constants; values always via `%s`) ──

# Gold supplies materialized profile aggregates and the tour singleton supplies benchmarks.
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
    gp.second_serve_return_points_won_pct, gp.return_games_won_pct,
    gp.break_point_conversion_pct,
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
    ta.tour_break_point_opportunities_per_return_game,
    ta.tour_return_games_won_pct
FROM {BRONZE_PROFILES_TABLE} bp
LEFT JOIN {GOLD_PROFILES_TABLE} gp ON gp.player_id = bp.player_id
CROSS JOIN {TOUR_AVERAGES_TABLE} ta
WHERE bp.player_id = %s
"""

# Use official weekly rankings only; match rows are not a fallback here.
_RANK_HISTORY_SQL = f"""
SELECT ranking_date, rank, points
FROM {BRONZE_RANKINGS_TABLE}
WHERE player_id = %s
ORDER BY ranking_date
"""

# Return every matching round, newest first, with deterministic match_id ties.
_MATCH_HISTORY_SQL = f"""
SELECT
    pm.match_id, pm.match_date, br.tournament, br.tournament_name, pm.surface, br.round,
    br.score,
    pm.opponent_id, pr.display_name AS opponent_name,
    COALESCE(pm.opponent_ranking, historical_rank.rank) AS opponent_ranking, pm.match_won,
    (pr.player_id IS NOT NULL) AS opponent_known,
    pm.aces, pm.double_faults,
    pm.first_serve_points_won, pm.second_serve_points_won,
    pm.total_serve_points, pm.service_games,
    pm.break_points_saved, pm.break_points_faced
FROM {SILVER_PLAYER_MATCHES} pm
LEFT JOIN {BRONZE_MATCHES_TABLE} br ON br.match_id = pm.match_id
LEFT JOIN {BRONZE_PROFILES_TABLE} pr ON pr.player_id = pm.opponent_id
LEFT JOIN LATERAL (
    SELECT rank
    FROM {BRONZE_RANKINGS_TABLE}
    WHERE player_id = pm.opponent_id
      AND ranking_date <= pm.match_date
    ORDER BY ranking_date DESC
    LIMIT 1
) historical_rank ON pm.opponent_ranking IS NULL
WHERE pm.player_id = %s
ORDER BY pm.match_date DESC, pm.match_id DESC
LIMIT %s
"""

# Read one row per meeting directly from bronze.
_H2H_MEETINGS_SQL = f"""
SELECT match_id, match_date, tournament, tournament_name, round, surface, score,
       player1_id, player2_id, winner_id
FROM {BRONZE_MATCHES_TABLE}
WHERE (player1_id = %s AND player2_id = %s)
   OR (player1_id = %s AND player2_id = %s)
ORDER BY match_date DESC, match_id DESC
LIMIT %s
"""

_DIRECTORY_SUMMARY_SQL = f"""
SELECT (
       SELECT source_watermark
           FROM bronze.etl_state
           WHERE pipeline = 'dbt'
       ) AS latest_match_date,
       (SELECT COUNT(match_id) FROM {BRONZE_MATCHES_TABLE}) AS total_matches
"""

# Set an explicit API description because mounted GET routes are not introspected.
SERVICE_DESCRIPTION = """\
# Tennis Match Prediction API

Symmetric player-vs-player predictions with feature snapshot construction.
Prediction context accepts optional tournament and round enums; encoded fields
are derived internally by the service. `is_indoor` defaults to 0 and `as_of_date` defaults
to today. The first-supplied player_id is the canonical player side and used for p_win.

## POST /predict_from_ids
Scalar ids-based prediction.
- `player_id`, `opponent_id` — required `str`
- `surface` — required `str` (`clay` / `grass` / `hard` / `carpet`)
- `best_of` — required `int` (`1` / `3` / `5`); best-of-N match format
- `tournament` — optional enum (`grand_slam` / `masters` / `atp_500` / `atp_250` / `davis_cup` / `atp_finals` / `olympics` / `professional`)
- `round` — optional enum (`r128` / `r64` / `r32` / `r16` / `qf` / `sf` / `f`)
- `as_of_date` — optional `date`, default today
- `is_indoor` — optional `int` (`0` / `1`), default 0

The service derives `tournament_level` and `round_encoded` from the enum values;
those numeric fields are not accepted request inputs.

## POST /predict_from_ids_bulk
Bulk ids-based prediction (API-key gated). Body envelope `{"rows": [ { ... } ]}` with the \
same per-row fields as `POST /predict_from_ids`; max 1000 rows; unknown fields are rejected.

## GET endpoints
Read-only dashboard data: `GET /directory`, `GET /player_profile`, `GET /rank_history`, `GET /match_history`, \
`GET /head_to_head`, `GET /similar_players`.
`GET /model_info` — API-key gated model metadata. `GET /health` — liveness.

Only the POST routes appear as OpenAPI paths; the GET routes are mounted Starlette
handlers that the OpenAPI generator does not introspect, so they are documented here only.
"""

ort.set_default_logger_severity(3)  # ERROR: suppress virtual-CPU warnings from ONNX Runtime.

load_env()


def _effective_log_level() -> int:
    """Map the LOG_LEVEL env var (case-insensitive) to a logging level; unknown -> INFO."""
    return getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)


# Keep lifecycle logs quiet while leaving request access logs at LOG_LEVEL.
_log_level = _effective_log_level()
logging.getLogger("bentoml").setLevel(logging.WARNING)
logging.getLogger("bentoml.access").setLevel(_log_level)
_log = logging.getLogger("tennis_ml.serving")
_log.setLevel(_log_level)
if not _log.handlers:
    _log.addHandler(logging.StreamHandler())


class _SuppressRequestValidationTraceback(logging.Filter):
    """Replace BentoML validation tracebacks with concise 400 warnings."""

    def filter(self, record: logging.LogRecord) -> bool:
        exc_info = record.exc_info
        if exc_info is None:
            return True
        exc_type, exc, _tb = exc_info
        if exc_type is None or not issubclass(exc_type, ValidationError):
            return True
        assert isinstance(exc, ValidationError)
        errors = exc.errors(include_input=False, include_context=False)
        first = errors[0]["msg"] if errors else "invalid request body"
        _log.warning(
            "request rejected with 400: %d validation error(s); first: %s",
            exc.error_count(),
            first,
        )
        return False


# Attach the filter to the emitting logger; ancestor filters do not see records.
logging.getLogger("bentoml._internal.server.http_app").addFilter(
    _SuppressRequestValidationTraceback()
)

# ── Read-only data endpoints (GET, no auth) ────────────────────────────────
# Dashboard GET routes use the mounted Starlette app; SQL values stay parameterized.


def _safe_query(sql: str, params: list[object] | None = None) -> pd.DataFrame:
    """Return an empty frame for missing dbt tables; propagate other failures."""
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


def _positive_class_probability(model: Any, features: Any) -> np.ndarray:
    """Return P(match_won=1), independent of the estimator's class order."""
    classes = np.asarray(model.classes_)
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise ValueError(f"model must expose binary classes including 1; got {classes.tolist()}")
    return np.asarray(model.predict_proba(features))[:, matches[0]]


def _stack_evidence(
    pairs: dict[str, tuple[float, float]],
    stacker: Any,
    stack_order: list[str] | None = None,
    *,
    temperature: float = 1.0,
) -> dict[str, float]:
    """Stack paired base probabilities into symmetric base scores and ``p_win``."""
    order = list(stack_order) if stack_order is not None else list(STACK_ORDER)
    if list(pairs.keys()) != order:
        raise ValueError(
            f"evidence pairs must be keyed in stack order {order}, got {list(pairs.keys())}"
        )
    evidence = {name: antisymmetric_evidence(ab, ba) for name, (ab, ba) in pairs.items()}
    stack = pd.DataFrame([[evidence[name] for name in order]], columns=order, dtype=float)
    p_win = float(_positive_class_probability(stacker, stack)[0])
    if temperature != 1.0:
        p_win = float(apply_temperature(p_win, temperature))
    probs = {name: float(evidence_to_probability(evidence[name])) for name in order}
    probs["p_win"] = p_win
    return probs


class TournamentLevel(StrEnum):
    GRAND_SLAM = "grand_slam"
    MASTERS = "masters"
    ATP_500 = "atp_500"
    ATP_250 = "atp_250"
    DAVIS_CUP = "davis_cup"
    ATP_FINALS = "atp_finals"
    OLYMPICS = "olympics"
    PROFESSIONAL = "professional"


class Round(StrEnum):
    R128 = "r128"
    R64 = "r64"
    R32 = "r32"
    R16 = "r16"
    QF = "qf"
    SF = "sf"
    F = "f"


class Surface(StrEnum):
    CLAY = "clay"
    GRASS = "grass"
    HARD = "hard"
    CARPET = "carpet"


class BestOf(IntEnum):
    """Best-of-N match format; exactly one of the three canonical lengths."""

    BO1 = 1
    BO3 = 3
    BO5 = 5


class PredictFromIdsRow(BaseModel):
    """One row of the bulk prediction envelope; mirrors `predict_from_ids` fields."""

    model_config = ConfigDict(extra="forbid")

    player_id: str
    opponent_id: str
    surface: Surface
    best_of: BestOf
    tournament: TournamentLevel | None = None
    round: Round | None = None
    as_of_date: date = Field(default_factory=date.today)
    is_indoor: int = 0

    @field_validator("best_of", mode="before")
    @classmethod
    def _require_best_of(cls, value: object) -> object:
        # bool is rejected explicitly: python treats True/False as 1/0 but a
        # booleann is never a best_of length.
        if isinstance(value, bool) or not isinstance(value, int) or value not in (1, 3, 5):
            raise ValueError(f"best_of must be exactly 1, 3, or 5, got {value!r}")
        return value


def _predict_from_ids_bulk_impl(
    rows: list[PredictFromIdsRow], predict_proba: object
) -> pd.DataFrame:
    """Build both orientations, run the shared ensemble, and preserve input order."""
    started_at = perf_counter()
    normalized: list[dict[str, object]] = []
    for row in rows:
        # BentoML passes validated PredictFromIdsRow instances; accept dicts too.
        d = row if isinstance(row, dict) else row.model_dump()
        # Strip BentoML auto-generated Pydantic extras before normalize.
        r = {k: v for k, v in d.items() if k in _NORMALIZE_KEYS}
        if r.get("as_of_date") is not None:
            r["as_of_date"] = _to_date(r["as_of_date"])
        normalized.append(r)
    # Reversed contexts swap the requested ids; surface/as_of_date/etc. stay put.
    reversed_ = [
        {**r, "player_id": r["opponent_id"], "opponent_id": r["player_id"]} for r in normalized
    ]
    feature_ab = build_inference_features_bulk(normalized)
    feature_ba = build_inference_features_bulk(reversed_)
    proba_df = cast(Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame], predict_proba)(
        feature_ab, feature_ba
    )
    out = feature_ab.copy()
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


# ── Route handlers ─────────────────────────────────────────────────────────


def _directory(_request: Request) -> JSONResponse:
    try:
        players_df = execute_df(PLAYERS_SQL)
        summary_df = execute_df(_DIRECTORY_SUMMARY_SQL)
    except Exception as exc:
        return _err(500, f"directory query failed: {exc}")
    players = directory_players(players_df)
    if summary_df.empty:
        latest_match_date: object = None
        total_matches = 0
    else:
        row = first_row_dict(summary_df)
        latest_match_date = _iso(row["latest_match_date"])
        total_matches = int(row["total_matches"])
    return _ok(
        {
            "players": players,
            "total_players": len(players),
            "latest_match_date": latest_match_date,
            "total_matches": total_matches,
        }
    )


def _player_profile(request: Request) -> JSONResponse:
    player_id = _require_id(request, "player_id")
    if player_id is None:
        return _err(400, "missing required query parameter: player_id")
    try:
        df = _safe_query(_PROFILE_SQL, [player_id])
    except Exception as exc:
        return _err(500, f"profile query failed: {exc}")
    # The empty result means the player is unknown.
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
        "return_games_won_pct": _iso(row["return_games_won_pct"]),
        "break_point_conversion_pct": _iso(row["break_point_conversion_pct"]),
        "break_point_opportunities_per_return_game": _iso(
            row["break_point_opportunities_per_return_game"]
        ),
    }

    # Compute player-minus-tour deltas here; return benchmarks use serve complements.
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
        "return_games_won_pct": delta("return_games_won_pct", row["tour_return_games_won_pct"]),
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

    # Resolve stored IOC values through the shared UNK fallback.
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
        # Prefer the source rank, then the latest official rank on or before the match.
        ranking = r["opponent_ranking"]
        if ranking is not None:
            rank_display = ranking
        elif r.get("opponent_known"):
            rank_display = "200+"
        else:
            rank_display = "N/A"
        matches.append(
            {
                "match_id": r["match_id"],
                "match_date": r["match_date"],
                "tournament": r["tournament"],
                "tournament_name": r["tournament_name"] or None,
                "surface": r["surface"],
                "round": r["round"],
                "score": r["score"],
                "opponent_id": r["opponent_id"],
                "opponent_name": r["opponent_name"],
                "opponent_ranking": ranking,
                "opponent_rank_display": rank_display,
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
    raw_limit = request.query_params.get("limit", "100")
    try:
        limit = int(raw_limit)
    except ValueError:
        return _err(400, "limit must be an integer")
    limit = max(1, min(limit, 100))  # clamp: a pair never meets more than this
    try:
        df = execute_df(_H2H_MEETINGS_SQL, [p1, p2, p2, p1, limit])
    except Exception as exc:
        return _err(500, f"head-to-head query failed: {exc}")

    meetings = []
    for r in _records(df):
        winner_id = r["winner_id"]
        player1_won = winner_id == p1
        meetings.append(
            {
                "match_id": r["match_id"],
                "match_date": r["match_date"],
                "surface": r["surface"],
                "tournament": r["tournament"],
                "tournament_name": r["tournament_name"] or None,
                "round": r["round"],
                "score": r["score"],
                "winner_id": winner_id,
                "loser_id": p2 if player1_won else p1,
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
            "player1_id": p1,
            "player2_id": p2,
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
    """Return server, port, and database name without credentials."""
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
    """Return the champion manifest, serving mode, and non-secret DB metadata."""
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


def _load_serving_temperature() -> float:
    """Load a positive finite calibration temperature, or return the 1.0 no-op."""
    try:
        manifest = json.loads(MODEL_INFO_FILE.read_text())
        value = manifest["calibration"]["temperature"]
    except (FileNotFoundError, OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        _log.warning("calibration missing from model_info (%s); serving temperature=1.0", exc)
        return 1.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _log.warning("calibration temperature is not numeric; serving temperature=1.0")
        return 1.0
    temperature = float(value)
    if not temperature > 0 or not math.isfinite(temperature):
        _log.warning("calibration temperature %r is invalid; serving temperature=1.0", temperature)
        return 1.0
    return temperature


def _health(_request: Request) -> JSONResponse:
    """Report liveness and PostgreSQL reachability without leaking details."""
    try:
        execute_df("SELECT 1")
    except Exception:
        _log.warning("health check failed: database unreachable")
        return _err(503, "database unavailable")
    return _ok({"status": "healthy"})


# Mount at the service root alongside the POST-only Bento routes.


async def _handle_starlette_error(_request: Request, exc: Exception):
    """Return every Starlette HTTPException as a json {ok, error} envelope."""
    http_exc = cast(StarletteHTTPException, exc)
    return JSONResponse(
        {"ok": False, "error": http_exc.detail or "internal error"},
        status_code=http_exc.status_code,
        headers=getattr(http_exc, "headers", None) or {},
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
        Route("/directory", _directory, methods=["GET"]),
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
    """Adapt a native LightGBM Booster to sklearn's ``predict_proba`` interface."""

    def __init__(self, booster: Any) -> None:
        self._booster = booster
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: Any) -> np.ndarray:
        p = np.asarray(self._booster.predict(X), dtype=np.float64)
        return np.column_stack([1.0 - p, p])


@bentoml.service(
    image=SERVING_IMAGE,
    description=SERVICE_DESCRIPTION,
    traffic={"timeout": 120},
    resources={"cpu": "500m"},
    # Two workers x four max pooled DB connections each: at most eight app
    # connections, none held while idle.
    workers=2,
)
@bentoml.asgi_app(DATA_APP, path="/")
class TennisPredictor:
    bento_linear = bentoml.models.BentoModel("linear_best:latest")
    bento_gbdt = bentoml.models.BentoModel("gbdt_best:latest")
    # Serve the NN from the deploy-time ONNX artifact, not a BentoModel.
    bento_production = bentoml.models.BentoModel(f"{PRODUCTION_MODEL}:latest")
    # Calibration temperature; __init__ overrides from the packaged artifact.
    temperature: float = 1.0

    def __init__(self):
        self.linear: Any = bentoml.sklearn.load_model(self.bento_linear)
        manifest = json.loads(MODEL_INFO_FILE.read_text())
        contract = manifest.get("feature_contract", {})
        if contract.get("columns") != list(FEATURE_COLS):
            raise RuntimeError(
                "model_info feature contract does not match serving FEATURE_COLS; "
                "rebuild from the promoted model"
            )
        # Fixed evidence stack order shared by training and serving.
        self._stack_order: list[str] = list(STACK_ORDER)
        gbdt_framework = normalize_gbdt_framework(manifest["bases"]["gbdt"][FRAMEWORK_KEY])
        # xgboost/lightgbm use OpenMP, which can deadlock inside BentoML's
        # forked worker processes. Force single-threaded during model load.
        _old_omp = os.environ.get("OMP_NUM_THREADS")
        os.environ["OMP_NUM_THREADS"] = "1"
        try:
            if gbdt_framework == GBDTFramework.XGBOOST:
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
        _validate_feature_contract(self.scaler, "linear_scaler.pkl")
        self.temperature = _load_serving_temperature()

    def _predict_proba(self, row_ab: pd.DataFrame, row_ba: pd.DataFrame) -> pd.DataFrame:
        """Score paired orientations through the stacker without an HTTP call."""
        started_at = perf_counter()
        if len(row_ab) != len(row_ba):
            raise ValueError("paired orientations must have equal length")
        features_ab = row_ab[FEATURE_COLS]
        features_ba = row_ba[FEATURE_COLS]

        # Linear and NN paths share the persisted train-fit scaler.
        scale_started_at = perf_counter()
        scaled_ab = self.scaler.transform(features_ab)
        scaled_ba = self.scaler.transform(features_ba)
        scale_ms = (perf_counter() - scale_started_at) * 1000
        # Linear path: finalized row -> persisted scaler -> classifier.
        linear_started_at = perf_counter()
        p_linear_ab = _positive_class_probability(self.linear, scaled_ab)
        p_linear_ba = _positive_class_probability(self.linear, scaled_ba)
        linear_ms = (perf_counter() - linear_started_at) * 1000
        # GBDT path: raw finalized row.
        gbdt_started_at = perf_counter()
        p_gbdt_ab = _positive_class_probability(self.gbdt, features_ab)
        p_gbdt_ba = _positive_class_probability(self.gbdt, features_ba)
        gbdt_ms = (perf_counter() - gbdt_started_at) * 1000
        # ONNX input matches the training forward signature (tab-only).
        nn_inputs_ab = {"tab": scaled_ab.astype(np.float32)}
        nn_inputs_ba = {"tab": scaled_ba.astype(np.float32)}
        nn_started_at = perf_counter()
        nn_ab = np.asarray(self.nn_session.run(None, nn_inputs_ab)[0])
        nn_ba = np.asarray(self.nn_session.run(None, nn_inputs_ba)[0])
        p_nn_ab = 1.0 / (1.0 + np.exp(-nn_ab.reshape(-1)))
        p_nn_ba = 1.0 / (1.0 + np.exp(-nn_ba.reshape(-1)))
        nn_ms = (perf_counter() - nn_started_at) * 1000

        # Stack antisymmetric evidence per row through the no-intercept stacker.
        n = len(row_ab)
        p_linear = np.empty(n)
        p_gbdt = np.empty(n)
        p_nn = np.empty(n)
        p_win = np.empty(n)
        ensemble_started_at = perf_counter()
        for i in range(n):
            probs = _stack_evidence(
                {
                    "linear": (p_linear_ab[i], p_linear_ba[i]),
                    "gbdt": (p_gbdt_ab[i], p_gbdt_ba[i]),
                    "nn": (p_nn_ab[i], p_nn_ba[i]),
                },
                self.production,
                self._stack_order,
                temperature=self.temperature,
            )
            p_linear[i] = probs["linear"]
            p_gbdt[i] = probs["gbdt"]
            p_nn[i] = probs["nn"]
            p_win[i] = probs["p_win"]
        ensemble_ms = (perf_counter() - ensemble_started_at) * 1000

        # Log aggregate metrics only, plus ids for single-row requests.
        first = row_ab.iloc[0] if not row_ab.empty else None
        first_ident = (
            f" player_id={first['player_id']} opponent_id={first['opponent_id']}"
            if first is not None and len(row_ab) == 1
            else ""
        )
        _log.debug(
            "predict_observability"
            f" rows={len(row_ab)}"
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
                "player_id": row_ab["player_id"].to_numpy(),
                "opponent_id": row_ab["opponent_id"].to_numpy(),
                "p_win": p_win,
                "p_linear": p_linear,
                "p_gbdt": p_gbdt,
                "p_nn": p_nn,
            },
            index=row_ab.index,
        )

    @bentoml.api
    def predict_from_ids(
        self,
        row: PredictFromIdsRow,
    ) -> dict[str, object]:
        """Build an as-of-dated feature row from player ids, then predict."""
        started_at = perf_counter()
        try:
            row_ab, meta = _build_inference_features_with_meta(
                row.player_id,
                row.opponent_id,
                row.surface,
                tournament=row.tournament,
                round=row.round,
                as_of_date=row.as_of_date,
                is_indoor=row.is_indoor,
                best_of=row.best_of.value,
            )
            row_ba, _meta_ba = _build_inference_features_with_meta(
                row.opponent_id,
                row.player_id,
                row.surface,
                tournament=row.tournament,
                round=row.round,
                as_of_date=row.as_of_date,
                is_indoor=row.is_indoor,
                best_of=row.best_of.value,
            )
            # Reuse the shared prediction path (no nested HTTP - see _predict_proba).
            out_df = self._predict_proba(row_ab, row_ba)
        except Exception:
            _log.exception("predict_from_ids failed")
            raise InvalidArgument("prediction failed - check input parameters") from None
        # Return the requested orientation as a flat JSON record.
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
            f" requested_player_id={meta['player_id']}"
            f" requested_opponent_id={meta['opponent_id']}"
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
            f" median_days_since={meta['median_days_since']}"
            f" build_ms={meta['build_ms']}"
            f" response_ms={(perf_counter() - started_at) * 1000:.3f}"
            f" p_win={rec['p_win']:.6f}"
        )
        return rec

    def _predict_from_ids_bulk(self, rows: list[PredictFromIdsRow]) -> pd.DataFrame:
        """Shared bulk path: normalize JSON rows, build features, predict once."""
        return _predict_from_ids_bulk_impl(rows, self._predict_proba)

    @bentoml.api
    def predict_from_ids_bulk(self, rows: list[PredictFromIdsRow]) -> list[dict[str, object]]:
        """Return predictions for a batch of per-row contexts."""
        if not rows:
            raise InvalidArgument("rows must be a non-empty list")
        if len(rows) > BULK_MAX_ROWS:
            raise InvalidArgument(f"max {BULK_MAX_ROWS} rows, got {len(rows)}")
        try:
            return _records(self._predict_from_ids_bulk(rows))
        except Exception:
            _log.exception("predict_from_ids_bulk failed")
            raise InvalidArgument("prediction failed — check input row format") from None
