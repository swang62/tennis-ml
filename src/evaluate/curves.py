"""Serialize and resolve the champion's pinned ROC, PR, and calibration curves."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.constants import CHAMPION_CURVE_URI_TAG, PRODUCTION_MODEL


@dataclass(frozen=True)
class CurveData:
    """Frozen ROC/PR/calibration reference points for one probability vector."""

    roc: dict[str, Any]
    pr: dict[str, Any]
    calibration: dict[str, Any]


def compute_curve_data(y_true, proba) -> CurveData:
    """Curve points (lists) + summary scalars for one (y, proba) pair."""
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    y_true = np.asarray(y_true, dtype=float)
    proba = np.asarray(proba, dtype=float)
    fpr, tpr, _ = roc_curve(y_true, proba)
    precision, recall, _ = precision_recall_curve(y_true, proba)
    prob_true, prob_pred = calibration_curve(y_true, proba, n_bins=10)
    return CurveData(
        roc={
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(roc_auc_score(y_true, proba)),
        },
        pr={
            "recall": recall.tolist(),
            "precision": precision.tolist(),
            "ap": float(average_precision_score(y_true, proba)),
        },
        calibration={
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
            "brier": float(brier_score_loss(y_true, proba)),
        },
    )


def curve_data_to_dict(data: CurveData) -> dict[str, Any]:
    return {"roc": data.roc, "pr": data.pr, "calibration": data.calibration}


def dict_to_curve_data(raw: dict[str, Any]) -> CurveData:
    return CurveData(roc=raw["roc"], pr=raw["pr"], calibration=raw["calibration"])


def serialize_curve_data(data: CurveData) -> tuple[bytes, str]:
    """JSON bytes + sha256, so promotion can pin exact champion curve bytes."""
    raw = json.dumps(curve_data_to_dict(data), separators=(",", ":")).encode("utf-8")
    return raw, hashlib.sha256(raw).hexdigest()


def resolve_champion_curves(client: Any, champion: Any) -> CurveData | None:
    """Download the pinned curve artifact, returning None when unavailable."""
    if champion is None:
        return None
    from mlflow.artifacts import download_artifacts

    version = client.get_model_version(PRODUCTION_MODEL, champion.version)
    uri = (getattr(version, "tags", None) or {}).get(CHAMPION_CURVE_URI_TAG)
    if not uri:
        return None
    try:
        path = download_artifacts(artifact_uri=uri)
        with open(path) as fh:
            return dict_to_curve_data(json.load(fh))
    except Exception:
        # Missing or broken incumbent curves must not break evaluation.
        return None
