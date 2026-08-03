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

import json
import pickle
from datetime import date
from time import perf_counter

import bentoml
import numpy as np
import onnxruntime as ort
import pandas as pd
from bentoml.images import Image

from src.constants import PRODUCTION_MODEL, ROOT
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


@bentoml.service(
    image=SERVING_IMAGE,
    traffic={"timeout": 10},
    resources={"cpu": "500m"},
)
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
        rec = out_df.iloc[0].to_dict()
        rec["p_win"] = float(rec["p_win"])
        rec["p_linear"] = float(rec["p_linear"])
        rec["p_gbdt"] = float(rec["p_gbdt"])
        rec["p_nn"] = float(rec["p_nn"])
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
            f" total_ms={(perf_counter() - started_at) * 1000:.3f}"
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
