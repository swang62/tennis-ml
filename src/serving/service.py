"""BentoML service composing the 4 stacked-ensemble artifacts.

`linear_best` and `gbdt_best` are sklearn classifiers, `nn_best` is the
PyTorch MLP with the bio-embedding pathway, and `production_model` is the
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

import bentoml
import numpy as np
import onnxruntime as ort
import pandas as pd
from bentoml.images import Image

from src.constants import ROOT
from src.features.rolling import FEATURE_COLS

AUX_DIR = ROOT / "data" / "processed"

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
    bento_production = bentoml.models.BentoModel("production_model:latest")

    def __init__(self):
        self.linear = bentoml.mlflow.load_model(self.bento_linear).unwrap_python_model()
        self.gbdt = bentoml.mlflow.load_model(self.bento_gbdt).unwrap_python_model()
        self.production = bentoml.mlflow.load_model(self.bento_production).unwrap_python_model()
        self.nn_session = ort.InferenceSession(str(AUX_DIR / "nn_best.onnx"))

        with open(AUX_DIR / "linear_scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        bio_df = pd.read_parquet(AUX_DIR / "bio_embeddings.parquet")
        with open(AUX_DIR / "bio_feature_cols.json") as f:
            self.bio_feature_cols = json.load(f)
        self.bio_by_player = {pid: i for i, pid in enumerate(bio_df["player_id"])}
        self.bio_array = bio_df[self.bio_feature_cols].to_numpy(np.float32)

    @bentoml.api(batchable=True, batch_dim=0)
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        required = [*FEATURE_COLS, "player_id", "opponent_id"]
        missing = [c for c in required if c not in input.columns]
        if missing:
            raise MissingColumnsError(missing)

        features = input[FEATURE_COLS]

        # Linear + NN paths share the persisted train-fit scaler
        # (StandardScaler().fit(X_train), same contract the NN was trained on).
        features_scaled = self.scaler.transform(features)
        # Linear path: finalized row -> persisted scaler -> classifier.
        p_linear = self.linear.predict_proba(features_scaled)[:, 1]
        # GBDT path: raw finalized row.
        p_gbdt = self.gbdt.predict_proba(features)[:, 1]
        # NN path: scaled row (as in training) + player/opponent bio lookup,
        # run through ONNX Runtime. The ONNX graph was exported with the three
        # inputs the training forward() takes: tab, bio_p, bio_o.
        nn_inputs = {
            "tab": features_scaled.astype(np.float32),
            "bio_p": self._row_bio_np(input["player_id"].to_numpy()),
            "bio_o": self._row_bio_np(input["opponent_id"].to_numpy()),
        }
        nn_logits = np.asarray(self.nn_session.run(None, nn_inputs)[0])
        p_nn = 1.0 / (1.0 + np.exp(-nn_logits.reshape(-1)))

        # LR head: stack of base-model probabilities.
        stack = np.column_stack([p_linear, p_gbdt, p_nn])
        p_win = self.production.predict_proba(stack)[:, 1]

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

    def _row_bio_np(self, ids: np.ndarray) -> np.ndarray:
        """Map player ids to bio vectors (np.float32), zero-filled for unknown players."""
        out = np.zeros((len(ids), len(self.bio_feature_cols)), dtype=np.float32)
        for i, pid in enumerate(ids):
            j = self.bio_by_player.get(pid)
            if j is not None:
                out[i] = self.bio_array[j]
        return out

    @bentoml.api
    def health(self) -> dict[str, str]:
        return {"status": "ok"}


class MissingColumnsError(ValueError):
    def __init__(self, missing: list[str]) -> None:
        super().__init__(f"Missing columns: {missing}")
