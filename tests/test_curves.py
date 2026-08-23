"""Tests for champion reference-curve computation and serialization."""

from __future__ import annotations

import numpy as np

from src.evaluate.curves import (
    compute_curve_data,
    dict_to_curve_data,
    serialize_curve_data,
)


def test_compute_curve_data_keys_and_auc_range():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=200) < 0.5).astype(float)
    p = np.clip(rng.normal(0.5, 0.2, 200), 0.01, 0.99)
    data = compute_curve_data(y, p)
    assert set(data.roc) == {"fpr", "tpr", "auc"}
    assert set(data.pr) == {"recall", "precision", "ap"}
    assert set(data.calibration) == {"prob_true", "prob_pred", "brier"}
    assert 0.0 <= data.roc["auc"] <= 1.0
    assert len(data.roc["fpr"]) == len(data.roc["tpr"])


def test_serialize_round_trip():
    rng = np.random.default_rng(1)
    y = (rng.uniform(size=100) < 0.5).astype(float)
    p = np.clip(rng.uniform(0.1, 0.9, 100), 0.01, 0.99)
    data = compute_curve_data(y, p)
    raw, digest = serialize_curve_data(data)
    assert isinstance(digest, str) and len(digest) == 64
    restored = dict_to_curve_data(__import__("json").loads(raw.decode("utf-8")))
    assert restored.roc["auc"] == data.roc["auc"]
    assert restored.calibration["prob_pred"] == data.calibration["prob_pred"]
