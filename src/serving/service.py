import bentoml
import numpy as np
import pandas as pd

from src.features.build import FEATURE_COLS


@bentoml.service(
    traffic={"timeout": 10},
    resources={"cpu": "500m"},
)
class TennisPredictor:
    bento_model = bentoml.models.BentoModel("tennis_prediction:latest")

    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.bento_model)

    @bentoml.api(batchable=True, batch_dim=0)
    def predict(self, input: pd.DataFrame) -> np.ndarray:
        missing = [c for c in FEATURE_COLS if c not in input.columns]
        if missing:
            raise MissingColumnsError(missing)
        return self.model.predict(input[FEATURE_COLS])

    @bentoml.api
    def health(self) -> dict[str, str]:
        return {"status": "ok"}


class MissingColumnsError(ValueError):
    def __init__(self, missing: list[str]) -> None:
        super().__init__(f"Missing columns: {missing}")
