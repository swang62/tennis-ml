"""BentoML service composing the 4 stacked-ensemble artifacts.

`linear_best` and `gbdt_best` are sklearn classifiers, `nn_best` is the
PyTorch MLP with the bio-embedding pathway, and `ensemble_lr_model` is the
logistic-regression stack head over `[p_linear, p_gbdt, p_nn]`.

The NN path runs through ONNX Runtime, not torch: the deploy flow exports the
pinned `nn_best` MLflow version to `data/processed/nn_best.onnx` at deploy
time, and this service loads that artifact at init. torch is not a serving
dependency.

The request carries a finalized `FEATURE_COLS` row plus `player_id` and
`opponent_id` (needed for the NN bio lookup); no rolling/diff/context
features are derived here. See `src/serving/README.md` for the full
payload contract (what upstream must precompute vs. what Bento does).
Decoupled aux artifacts (`linear_scaler.pkl`, `bio_embeddings.parquet`,
`bio_feature_cols.json`, `nn_best.onnx`) are packaged by the build step
and loaded from disk at init.
"""

import builtins
import json
import math
import pickle
from datetime import date, datetime
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

from src.constants import (
    GOLD_ROLLING_FEATURES,
    PRODUCTION_MODEL,
    PROFILES_TABLE,
    ROOT,
    SILVER_PLAYER_MATCHES,
    SILVER_PLAYER_RANKINGS,
)
from src.db.client import execute_df, first_row_dict
from src.features.columns import FEATURE_COLS
from src.features.inference import _build_inference_features_with_meta
from src.utils import load_env

AUX_DIR = ROOT / "data" / "processed"

load_env()

# v2 image spec: deps declared here, NOT in bentofile.yaml. Keeps the install
# list next to the service that consumes it. torch + lightning are dropped
# (the NN is served via ONNX Runtime); everything else is pinned to the exact
# versions the training venv used — the sklearn/lightgbm/xgboost models are
# pickled, so version drift breaks loading.
SERVING_IMAGE = Image(
    python_version="3.12", distro="debian", lock_python_packages=False
).python_packages(
    "bentoml==1.4.39",
    "mlflow==3.13.0",
    "scikit-learn==1.8.0",
    "xgboost-cpu==3.2.0",
    "lightgbm==4.6.0",
    "catboost==1.2.10",  # 02_tune_gbdt tries xgb/lgbm/catboost; image must support whichever wins
    "duckdb==1.5.4",
    "pandas==2.3.3",
    "pyarrow==24.0.0",
    "numpy==2.4.6",
    "scipy==1.17.1",
    "onnxruntime==1.27.0",
)


# ── Read-only data endpoints (GET, no auth) ────────────────────────────────
# Served from a mounted Starlette app: BentoML's `@bentoml.api` only registers
# POST routes (the SDK's HTTP server hardcodes methods=["POST"]), while the
# dashboard reads player/career data over plain GET with query params. The
# mounted app runs in the same process/worker as the service, so it shares the
# lazy DuckDB connection from src.db.client (same env-configurable path the
# inference builder uses).
#
# All SQL is parameterized (prepared statements via src.db.client.execute_df);
# ids/dates never appear in SQL text. JSON shapes below are the dashboard's
# contract — keep them stable.


# Response envelope used by every data endpoint.
def _ok(data: dict[str, object]) -> JSONResponse:
    return JSONResponse({"ok": True, "data": data})


def _err(status: int, message: str) -> JSONResponse:
    return JSONResponse({"ok": False, "error": message}, status_code=status)


def _iso(value: object) -> object:
    """JSON-safe scalar: dates to ISO strings, NaN/None stay null."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float):
        return None if math.isnan(value) else value
    return value


def _records(df: pd.DataFrame) -> list[dict[str, object]]:
    return [{str(k): _iso(v) for k, v in row.items()} for row in df.to_dict("records")]


def _require_id(request: Request, name: str) -> str | None:
    """Return the non-blank query param value or None (caller turns None into a 400)."""
    raw = request.query_params.get(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip()


# ── SQL (table names interpolated from constants; values always via `?`) ──

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
WHERE player_id = ?
"""

# Career stats from the player's oriented history in silver.player_matches.
# Serve/breakpoint rates are aggregate sums (NULL-safe: NULLIF guards zero
# denominators), mirroring the rolling_features rate conventions.
_PROFILE_CAREER_SQL = f"""
SELECT
    COUNT(*) AS matches_played,
    AVG(match_won) AS win_rate,
    SUM(first_serve_points_won)::DOUBLE / NULLIF(SUM(first_serves_made), 0)
        AS first_serve_win_pct,
    SUM(second_serve_points_won)::DOUBLE
        / NULLIF(SUM(total_serve_points - first_serves_made), 0)
        AS second_serve_win_pct,
    SUM(first_serve_points_won + second_serve_points_won)::DOUBLE
        / NULLIF(SUM(total_serve_points), 0)
        AS serve_win_pct,
    SUM(break_points_saved)::DOUBLE / NULLIF(SUM(break_points_faced), 0)
        AS break_points_saved_pct
FROM {SILVER_PLAYER_MATCHES}
WHERE player_id = ?
"""

# All-time per-surface win rates from the player's oriented history. One row
# per surface actually played; unplayed surfaces have no row at all. The
# handler fills those in with matches=0 and a NULL win_rate — the dashboard
# renders them as "n/a (n=0)", never 0%.
_PROFILE_SURFACE_SQL = f"""
SELECT
    surface,
    COUNT(*) AS matches,
    AVG(match_won) AS win_rate
FROM {SILVER_PLAYER_MATCHES}
WHERE player_id = ?
GROUP BY surface
"""

# Newest as-of rolling snapshot per player (recent form).
_PROFILE_FORM_SQL = f"""
SELECT snapshot_date, win_rate_10
FROM {GOLD_ROLLING_FEATURES}
WHERE player_id = ?
ORDER BY player_match_number DESC
LIMIT 1
"""

# Rank-points series ordered by date, for latest-vs-earliest trend.
_PROFILE_RANK_POINTS_SQL = f"""
SELECT match_date, player_rank_points
FROM {SILVER_PLAYER_MATCHES}
WHERE player_id = ? AND player_rank_points IS NOT NULL
ORDER BY match_date, match_id
"""

_RANK_HISTORY_SQL = f"""
SELECT match_date, ranking
FROM {SILVER_PLAYER_RANKINGS}
WHERE player_id = ?
ORDER BY match_date, match_id
"""

_MATCH_HISTORY_SQL = f"""
SELECT
    pm.match_id, pm.match_date, pm.tournament, pm.surface, pm.round,
    pm.opponent_id, pr.display_name AS opponent_name,
    pm.player_ranking, pm.match_won,
    pm.aces, pm.double_faults,
    pm.first_serve_points_won, pm.second_serve_points_won,
    pm.total_serve_points, pm.service_games,
    pm.break_points_saved, pm.break_points_faced
FROM {SILVER_PLAYER_MATCHES} pm
LEFT JOIN {PROFILES_TABLE} pr ON pr.player_id = pm.opponent_id
WHERE pm.player_id = ?
ORDER BY pm.match_date DESC, pm.match_id DESC
LIMIT ?
"""

# Prior meetings between a pair. silver.player_matches has TWO rows per match
# (one per player perspective); GROUP BY collapses them to distinct match_ids
# (same dedupe rule as the model's H2H lookup). a_won = 1 iff the canonical
# a-side (the lower id, always the player_* side) won the meeting; both
# perspective rows of a meeting agree on it, so MAX is safe. Params:
# lower_id, higher_id, lower_id, higher_id.
_H2H_MEETINGS_SQL = f"""
SELECT
    match_id, match_date, surface, tournament, round,
    MAX(winner_id) AS winner_id,
    MAX(CASE WHEN player_id < opponent_id THEN match_won
             ELSE 1 - match_won END) AS a_won
FROM {SILVER_PLAYER_MATCHES}
WHERE ((? = player_id AND ? = opponent_id)
    OR (? = opponent_id AND ? = player_id))
GROUP BY match_id, match_date, surface, tournament, round
ORDER BY match_date DESC, match_id DESC
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
        "matches_played": int(career["matches_played"]),
        "win_rate": career["win_rate"],
        "first_serve_win_pct": career["first_serve_win_pct"],
        "second_serve_win_pct": career["second_serve_win_pct"],
        "serve_win_pct": career["serve_win_pct"],
        "break_points_saved_pct": career["break_points_saved_pct"],
    }

    # All-time per-surface win rates; locked to clay/grass/hard (the model's
    # surface set). Unplayed surfaces: matches 0, NULL win_rate (n/a, not 0%).
    surf_rates = {r["surface"]: r for r in surf_df.to_dict("records")}
    surface_rates = [
        {
            "surface": s,
            "matches": int(surf_rates[s]["matches"]) if s in surf_rates else 0,
            "win_rate": surf_rates[s]["win_rate"] if s in surf_rates else None,
        }
        for s in ("clay", "grass", "hard")
    ]

    # Recent form: last-10 win rate from the newest rolling snapshot (if any).
    recent_form: dict[str, object] | None = None
    if not form_df.empty:
        form = first_row_dict(form_df)
        recent_form = {
            "snapshot_date": _iso(form["snapshot_date"]),
            "last_10_win_rate": form["win_rate_10"],
        }

    # Rank-points trend: latest vs earliest rank points across the player's
    # matches (fallback to last-5 delta when only one point exists).
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
    # Accept both parameter conventions: player1_id/player2_id (MUST DO) and
    # player_id/opponent_id (the /predict_from_ids style, used by the dashboard).
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


# Mounted at the service root; coexists with the POST-only @bentoml.api routes
# (the SDK's server checks its own routes first, then falls through to mounts).
DATA_APP = Starlette(
    routes=[
        Route("/players", _players, methods=["GET"]),
        Route("/player_profile", _player_profile, methods=["GET"]),
        Route("/rank_history", _rank_history, methods=["GET"]),
        Route("/match_history", _match_history, methods=["GET"]),
        Route("/head_to_head", _head_to_head, methods=["GET"]),
    ]
)


@bentoml.service(
    image=SERVING_IMAGE,
    traffic={"timeout": 10},
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

    # Non-batchable: BentoML 1.4.39's batch dispatcher has a bug sizing pandas
    # DataFrame inputs (`get_batch_size = lambda x: x.sample.batch_size` hits
    # `DataFrame.sample` the method and returns a tuple, not an int), raising
    # `TypeError: int + tuple` and surfacing as `ServiceUnavailable: process is
    # overloaded`. Single-match predictions don't need batch concat anyway.
    @bentoml.api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        required = [*FEATURE_COLS, "player_id", "opponent_id"]
        missing = [c for c in required if c not in input.columns]
        if missing:
            raise MissingColumnsError(missing)
        return self._predict_proba(input)

    def _predict_proba(self, input: pd.DataFrame) -> pd.DataFrame:
        """Run the stacked ensemble on a finalized row. Shared by the
        model-only `/predict` endpoint and `/predict-from-ids`.

        Called DIRECTLY (not via `self.predict`) so `/predict-from-ids` doesn't
        route a nested HTTP call back into the same single-worker service —
        that would self-deadlock (the worker is busy handling the outer call
        and can't service the inner one), surfacing as a timeout.
        """
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
        # NN path: scaled row (as in training) + player/opponent bio lookup,
        # run through ONNX Runtime. The ONNX graph was exported with the three
        # inputs the training forward() takes: tab, bio_p, bio_o.
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

        print(
            "predict_observability"
            f" player_id={input.iloc[0]['player_id'] if not input.empty else None}"
            f" opponent_id={input.iloc[0]['opponent_id'] if not input.empty else None}"
            f" rows={len(input)}"
            f" feature_count={len(FEATURE_COLS)}"
            f" scale_ms={scale_ms:.3f}"
            f" linear_ms={linear_ms:.3f}"
            f" gbdt_ms={gbdt_ms:.3f}"
            f" nn_ms={nn_ms:.3f}"
            f" ensemble_ms={ensemble_ms:.3f}"
            f" total_ms={(perf_counter() - started_at) * 1000:.3f}"
            f" p_win={float(p_win[0]) if len(p_win) else float('nan'):.6f}"
            f" p_linear={float(p_linear[0]) if len(p_linear) else float('nan'):.6f}"
            f" p_gbdt={float(p_gbdt[0]) if len(p_gbdt) else float('nan'):.6f}"
            f" p_nn={float(p_nn[0]) if len(p_nn) else float('nan'):.6f}"
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
    ) -> dict[str, object]:
        """Build the feature row on demand from the baked-in DuckDB and predict.

        Minimal human inputs: two player ids + surface, optional integer
        tournament_level/round_encoded (or their string aliases tournament/
        round, e.g. "grand_slam" / "f") and as_of_date. Queries the bundled
        gold tables (snapshot from deploy time).
        """
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

    def _row_bio_np(self, ids: np.ndarray) -> np.ndarray:
        """Map player ids to bio vectors (np.float32), zero-filled for unknown players."""
        out = np.zeros((len(ids), len(self.bio_feature_cols)), dtype=np.float32)
        for i, pid in enumerate(ids):
            j = self.bio_by_player.get(pid)
            if j is not None:
                out[i] = self.bio_array[j]
        return out


class MissingColumnsError(ValueError):
    def __init__(self, missing: list[str]) -> None:
        super().__init__(f"Missing columns: {missing}")
