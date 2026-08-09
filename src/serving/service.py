"""BentoML service for the stacked ensemble and read-only dashboard data.

The NN uses the deploy-time ONNX artifact; finalized features are built from
ids in-service (scalar and bulk) against the live PostgreSQL gold tables.
"""

import builtins
import json
import math
import os
import pickle
from datetime import date, datetime
from decimal import Decimal
from time import perf_counter

import bentoml
import numpy as np
import onnxruntime as ort
import pandas as pd
from bentoml.images import Image
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from typing import Callable, cast

from src.constants import (
    BRONZE_TABLE,
    PRODUCTION_MODEL,
    PROFILES_TABLE,
    ROOT,
    SILVER_PLAYER_MATCHES,
    SILVER_ROLLING_FEATURES,
)
from src.db.client import execute_df, first_row_dict
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

# Serving dependencies stay here. Model packages are pinned because models are pickled.
SERVING_IMAGE = Image(
    python_version="3.12", distro="debian", lock_python_packages=False
).python_packages(
    "bentoml==1.4.39",
    "mlflow==3.13.0",
    "scikit-learn==1.8.0",
    "xgboost-cpu==3.2.0",
    "lightgbm==4.6.0",
    "catboost==1.2.10",  # 02_tune_gbdt tries xgb/lgbm/catboost; image must support whichever wins
    "psycopg[binary]==3.3.4",
    "psycopg-pool==3.3.1",
    "pandas==2.3.3",
    "pyarrow==24.0.0",
    "numpy==2.4.6",
    "scipy==1.17.1",
    "onnxruntime==1.27.0",
    "faiss-cpu==1.14.3",  # loads the packaged player-similarity index for /similar_players
)


# ── Read-only data endpoints (GET, no auth) ────────────────────────────────
# Bento APIs are POST-only, so dashboard GET routes use a mounted Starlette app.
# SQL values are always parameterized; response shapes are a dashboard contract.


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


def _require_id(request: Request, name: str) -> str | None:
    """Return the non-blank query param value or None (caller turns None into a 400)."""
    raw = request.query_params.get(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip()


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
        r = dict(row)
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
    print(
        "predict_from_ids_bulk_observability"
        f" rows={len(out)}"
        f" feature_count={len(FEATURE_COLS)}"
        f" build_ms={(perf_counter() - started_at) * 1000:.3f}"
        f" mean_p_win={float(out['p_win'].mean()):.6f}"
    )
    return out


# ── SQL (table names interpolated from constants; values always via `%s`) ──

_PLAYERS_SQL = f"""
SELECT
    p.player_id,
    p.display_name,
    COUNT(pm.player_id) AS matches_played
FROM {PROFILES_TABLE} p
LEFT JOIN {SILVER_PLAYER_MATCHES} pm ON pm.player_id = p.player_id
GROUP BY p.player_id, p.display_name
ORDER BY p.display_name
"""

# Bio block: identity columns the dashboard shows on the profile header.
_PROFILE_BIO_SQL = f"""
SELECT
    player_id, display_name, handedness, backhand, height,
    turned_pro, birthplace, summary
FROM {PROFILES_TABLE}
WHERE player_id = %s
"""

# Career rates use NULLIF and double precision to avoid divide-by-zero and truncation.
_PROFILE_CAREER_SQL = f"""
SELECT
    COUNT(*) AS matches_played,
    AVG(match_won) AS win_rate,
    SUM(first_serve_points_won)::double precision / NULLIF(SUM(first_serves_made), 0)
        AS first_serve_win_pct,
    SUM(second_serve_points_won)::double precision
        / NULLIF(SUM(total_serve_points - first_serves_made), 0)
        AS second_serve_win_pct,
    SUM(first_serve_points_won + second_serve_points_won)::double precision
        / NULLIF(SUM(total_serve_points), 0)
        AS serve_win_pct,
    SUM(break_points_saved)::double precision / NULLIF(SUM(break_points_faced), 0)
        AS break_points_saved_pct
FROM {SILVER_PLAYER_MATCHES}
WHERE player_id = %s
"""

# Unplayed surfaces have no row; the handler renders them as n/a, not 0%.
_PROFILE_SURFACE_SQL = f"""
SELECT
    surface,
    COUNT(*) AS matches,
    AVG(match_won) AS win_rate
FROM {SILVER_PLAYER_MATCHES}
WHERE player_id = %s
GROUP BY surface
"""

# Newest as-of rolling snapshot per player (recent form).
_PROFILE_FORM_SQL = f"""
SELECT snapshot_date, win_rate_10
FROM {SILVER_ROLLING_FEATURES}
WHERE player_id = %s
ORDER BY player_match_number DESC
LIMIT 1
"""

# Rank-points series ordered by date, for latest-vs-earliest trend.
_PROFILE_RANK_POINTS_SQL = f"""
SELECT match_date, player_rank_points
FROM {SILVER_PLAYER_MATCHES}
WHERE player_id = %s AND player_rank_points IS NOT NULL
ORDER BY match_date, match_id
"""

# Expand both bronze sides into a player ranking time series.
_RANK_HISTORY_SQL = f"""
SELECT match_date, ranking
FROM (
    SELECT match_date, player1_id AS player_id, player1_ranking AS ranking
    FROM {BRONZE_TABLE}
    UNION ALL
    SELECT match_date, player2_id AS player_id, player2_ranking AS ranking
    FROM {BRONZE_TABLE}
)
WHERE player_id = %s AND ranking IS NOT NULL AND ranking > 0
ORDER BY match_date
"""

# Keep the deepest round per tournament before applying the visible limit.
_MATCH_HISTORY_SQL = f"""
WITH per_match AS (
    SELECT
        pm.match_id, pm.match_date, br.tournament, pm.surface, br.round,
        pm.opponent_id, pr.display_name AS opponent_name,
        pm.player_ranking, pm.match_won,
        pm.aces, pm.double_faults,
        pm.first_serve_points_won, pm.second_serve_points_won,
        pm.total_serve_points, pm.service_games,
        pm.break_points_saved, pm.break_points_faced,
        regexp_replace(pm.match_id, '-[0-9]{{3}}$', '') AS tourney_key,
        CASE br.round
            WHEN 'R128' THEN 1
            WHEN 'R64' THEN 2
            WHEN 'R32' THEN 3
            WHEN 'R16' THEN 4
            WHEN 'QF' THEN 5
            WHEN 'SF' THEN 6
            WHEN 'F' THEN 7
            ELSE 0
        END AS round_encoded
    FROM {SILVER_PLAYER_MATCHES} pm
    LEFT JOIN {BRONZE_TABLE} br ON br.match_id = pm.match_id
    LEFT JOIN {PROFILES_TABLE} pr ON pr.player_id = pm.opponent_id
    WHERE pm.player_id = %s
),
ranked AS (
    SELECT per_match.*,
        ROW_NUMBER() OVER (
            PARTITION BY tourney_key
            ORDER BY round_encoded DESC, match_date DESC, match_id DESC
        ) AS rn
    FROM per_match
)
SELECT match_id, match_date, tournament, surface, round,
       opponent_id, opponent_name, player_ranking, match_won,
       aces, double_faults, first_serve_points_won, second_serve_points_won,
       total_serve_points, service_games, break_points_saved, break_points_faced
FROM ranked
WHERE rn = 1
ORDER BY match_date DESC, match_id DESC
LIMIT %s
"""

# Dedupe player perspectives; a_won is the canonical lower-id side's result.
_H2H_MEETINGS_SQL = f"""
SELECT
    pm.match_id, pm.match_date, br.surface, br.tournament, br.round,
    MAX(br.winner_id) AS winner_id,
    MAX(CASE WHEN pm.player_id < pm.opponent_id THEN pm.match_won
             ELSE 1 - pm.match_won END) AS a_won
FROM {SILVER_PLAYER_MATCHES} pm
LEFT JOIN {BRONZE_TABLE} br ON br.match_id = pm.match_id
WHERE ((%s = pm.player_id AND %s = pm.opponent_id)
    OR (%s = pm.opponent_id AND %s = pm.player_id))
GROUP BY pm.match_id, pm.match_date, br.surface, br.tournament, br.round
ORDER BY pm.match_date DESC, pm.match_id DESC
"""


# ── Route handlers ─────────────────────────────────────────────────────────


def _players(_request: Request) -> JSONResponse:
    try:
        df = execute_df(_PLAYERS_SQL)
        return _ok({"players": _records(df)})
    except Exception as exc:  # DB errors -> 500 with message
        return _err(500, f"players query failed: {exc}")


def _player_profile(request: Request) -> JSONResponse:
    player_id = _require_id(request, "player_id")
    if player_id is None:
        return _err(400, "missing required query parameter: player_id")
    try:
        bio_df = execute_df(_PROFILE_BIO_SQL, [player_id])
        if bio_df.empty:
            return _err(404, f"unknown player_id: {player_id}")
        career_df = execute_df(_PROFILE_CAREER_SQL, [player_id])
        surf_df = execute_df(_PROFILE_SURFACE_SQL, [player_id])
        form_df = execute_df(_PROFILE_FORM_SQL, [player_id])
        rp_df = execute_df(_PROFILE_RANK_POINTS_SQL, [player_id])
    except Exception as exc:
        return _err(500, f"profile query failed: {exc}")

    bio = first_row_dict(bio_df)
    career = first_row_dict(career_df)
    career_out: dict[str, object] = {
        "matches_played": _iso(career["matches_played"]),
        "win_rate": _iso(career["win_rate"]),
        "first_serve_win_pct": _iso(career["first_serve_win_pct"]),
        "second_serve_win_pct": _iso(career["second_serve_win_pct"]),
        "serve_win_pct": _iso(career["serve_win_pct"]),
        "break_points_saved_pct": _iso(career["break_points_saved_pct"]),
    }

    # Unplayed model surfaces are n/a, not 0%.
    surf_rates = {r["surface"]: r for r in surf_df.to_dict("records")}
    surface_rates = [
        {
            "surface": s,
            "matches": int(surf_rates[s]["matches"]) if s in surf_rates else 0,
            "win_rate": _iso(surf_rates[s]["win_rate"]) if s in surf_rates else None,
        }
        for s in ("clay", "grass", "hard")
    ]

    # Recent form: last-10 win rate from the newest rolling snapshot (if any).
    recent_form: dict[str, object] | None = None
    if not form_df.empty:
        form = first_row_dict(form_df)
        recent_form = {
            "snapshot_date": _iso(form["snapshot_date"]),
            "last_10_win_rate": _iso(form["win_rate_10"]),
        }

    # Rank-points trend from earliest to latest observation.
    rank_points_trend: dict[str, object] | None = None
    if not rp_df.empty:
        points = rp_df["player_rank_points"].tolist()
        earliest = float(points[0])
        latest = float(points[-1])
        # single point: no trend to show
        delta = latest - earliest if len(points) >= 2 else 0.0
        rank_points_trend = {
            "earliest": earliest,
            "latest": latest,
            "delta": delta,
        }

    return _ok(
        {
            "player_id": bio["player_id"],
            "display_name": bio["display_name"],
            "handedness": bio["handedness"],
            "backhand": bio["backhand"],
            "height": bio["height"],
            "turned_pro": bio["turned_pro"],
            "birthplace": bio["birthplace"],
            "summary": bio["summary"],
            "career": career_out,
            "surface_rates": surface_rates,
            "recent_form": recent_form,
            "rank_points_trend": rank_points_trend,
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
    history = [{"rank_date": _iso(r["match_date"]), "rank": r["ranking"]} for r in _records(df)]
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
        df = execute_df(_MATCH_HISTORY_SQL, [player_id, limit])
    except Exception as exc:
        return _err(500, f"match history query failed: {exc}")
    matches = []
    for r in _records(df):
        matches.append(
            {
                "match_id": r["match_id"],
                "match_date": r["match_date"],
                "tournament": r["tournament"],
                "surface": r["surface"],
                "round": r["round"],
                "opponent_id": r["opponent_id"],
                "opponent_name": r["opponent_name"],
                "ranking": r["player_ranking"],
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
    try:
        df = execute_df(_H2H_MEETINGS_SQL, [lower, higher, lower, higher])
    except Exception as exc:
        return _err(500, f"head-to-head query failed: {exc}")

    meetings = []
    for r in _records(df):
        winner_id = r["winner_id"]
        loser_id = higher if winner_id == lower else lower
        meetings.append(
            {
                "match_date": r["match_date"],
                "surface": r["surface"],
                "tournament": r["tournament"],
                "round": r["round"],
                "winner_id": winner_id,
                "loser_id": loser_id,
                "player1_won": bool(r["a_won"] == 1),
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


def _get_similarity_finder() -> PlayerSimilarity:
    global _similarity_finder
    if _similarity_finder is None:
        finder = PlayerSimilarity()
        finder.load()
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
        similar = _get_similarity_finder().search(player_id, top_k=limit)
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


# Mounted at the service root; coexists with the POST-only @bentoml.api routes
# (the SDK's server checks its own routes first, then falls through to mounts).
DATA_APP = Starlette(
    routes=[
        Route("/players", _players, methods=["GET"]),
        Route("/player_profile", _player_profile, methods=["GET"]),
        Route("/rank_history", _rank_history, methods=["GET"]),
        Route("/match_history", _match_history, methods=["GET"]),
        Route("/head_to_head", _head_to_head, methods=["GET"]),
        Route("/similar_players", _similar_players, methods=["GET"]),
        Route("/model_info", _model_info, methods=["GET"]),
    ]
)


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
        self.linear = bentoml.mlflow.load_model(self.bento_linear).get_raw_model()
        self.gbdt = bentoml.mlflow.load_model(self.bento_gbdt).get_raw_model()
        self.production = bentoml.mlflow.load_model(self.bento_production).get_raw_model()
        self.nn_session = ort.InferenceSession(str(AUX_DIR / "nn_best.onnx"))

        with open(AUX_DIR / "linear_scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        bio_df = pd.read_parquet(AUX_DIR / "bio_embeddings.parquet")
        with open(AUX_DIR / "bio_feature_cols.json") as f:
            self.bio_feature_cols = json.load(f)
        self.bio_by_player = {pid: i for i, pid in enumerate(bio_df["player_id"])}
        self.bio_array = bio_df[self.bio_feature_cols].to_numpy(np.float32)

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
        print(
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
        print(
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
        return _records(self._predict_from_ids_bulk(rows))

    def _row_bio_np(self, ids: np.ndarray) -> np.ndarray:
        """Map player ids to bio vectors (np.float32), zero-filled for unknown players."""
        out = np.zeros((len(ids), len(self.bio_feature_cols)), dtype=np.float32)
        for i, pid in enumerate(ids):
            j = self.bio_by_player.get(pid)
            if j is not None:
                out[i] = self.bio_array[j]
        return out
