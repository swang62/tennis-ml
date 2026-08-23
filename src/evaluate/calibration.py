"""Symmetry-preserving temperature scaling for ensemble win probabilities."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from src.evaluate.symmetry import clip_probability, logit, sigmoid


def apply_temperature(p, t) -> float | np.ndarray:
    """Temperature-scaled probability: sigmoid(t * logit(clip(p)))."""
    t = float(t)
    if not t > 0:
        raise ValueError(f"temperature must be positive, got {t!r}")
    return sigmoid(t * logit(clip_probability(p)))


def fit_temperature(proba, y, bounds=(0.05, 20.0)) -> float:
    """Fit the temperature minimizing log loss over bounds; fall back to 1.0."""
    from scipy.optimize import minimize_scalar

    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y, dtype=float)
    if proba.ndim != 1 or y.ndim != 1 or proba.shape != y.shape:
        raise ValueError(
            f"proba and y must be equal-length 1-D arrays, got {proba.shape} and {y.shape}"
        )
    if len(proba) == 0:
        return 1.0
    eps = 1e-12

    def loss(t: float) -> float:
        p = np.clip(apply_temperature(proba, t), eps, 1.0 - eps)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    result = minimize_scalar(loss, bounds=bounds, method="bounded")
    t_fit = float(result.x)  # type: ignore[reportAttributeAccessIssue]  # scipy stubs unavailable
    if not np.isfinite(t_fit) or not bounds[0] <= t_fit <= bounds[1]:
        return 1.0
    if loss(t_fit) < loss(1.0):
        return t_fit
    return 1.0


@dataclass(frozen=True)
class CalibrationSelection:
    """A temperature accepted only when it beats raw probabilities out of sample."""

    temperature: float
    fitted_temperature: float
    accepted: bool
    raw_log_loss: float
    calibrated_log_loss: float
    raw_brier: float
    calibrated_brier: float


def select_temperature(
    proba,
    y,
    folds,
    bounds=(0.05, 20.0),
    brier_guard_tolerance=1e-3,
) -> CalibrationSelection:
    """Select a temperature with chronological folds and bounded Brier degradation."""
    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y, dtype=float)
    folds = np.asarray(folds)
    if proba.ndim != 1 or y.ndim != 1 or proba.shape != y.shape or proba.shape != folds.shape:
        raise ValueError("proba, y, and folds must be equal-length 1-D arrays")
    if not np.all(np.isfinite(proba)) or not np.all(np.isfinite(y)):
        raise ValueError("proba and y must be finite")
    if folds.dtype.kind not in "iuf":
        raise ValueError("fold labels must be integer ordinals in chronological order")
    try:
        fold_ids = folds.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError("fold labels must be integer ordinals in chronological order") from exc
    if not np.array_equal(folds, fold_ids):
        raise ValueError("fold labels must be integer ordinals in chronological order")
    fold_values = np.unique(folds)
    if len(fold_values) < 2:
        raise ValueError("walk-forward calibration requires at least two folds")

    raw_parts: list[np.ndarray] = []
    calibrated_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for index, fold in enumerate(fold_values[1:], start=1):
        train = np.isin(folds, fold_values[:index])
        validation = folds == fold
        val_labels = y[validation]
        if len(val_labels) < 2 or np.unique(val_labels).size < 2:
            raise ValueError(
                f"validation fold {int(fold)} must hold both label classes, "
                f"got {np.unique(val_labels)}"
            )
        temperature = fit_temperature(proba[train], y[train], bounds=bounds)
        raw_parts.append(proba[validation])
        calibrated_parts.append(np.asarray(apply_temperature(proba[validation], temperature)))
        label_parts.append(y[validation])

    raw = np.clip(np.concatenate(raw_parts), 1e-12, 1.0 - 1e-12)
    calibrated = np.clip(np.concatenate(calibrated_parts), 1e-12, 1.0 - 1e-12)
    labels = np.concatenate(label_parts)
    raw_log_loss = float(-np.mean(labels * np.log(raw) + (1.0 - labels) * np.log(1.0 - raw)))
    calibrated_log_loss = float(
        -np.mean(labels * np.log(calibrated) + (1.0 - labels) * np.log(1.0 - calibrated))
    )
    raw_brier = float(np.mean((labels - raw) ** 2))
    calibrated_brier = float(np.mean((labels - calibrated) ** 2))
    accepted = (
        calibrated_log_loss < raw_log_loss and calibrated_brier < raw_brier + brier_guard_tolerance
    )
    fitted_temperature = fit_temperature(proba, y, bounds=bounds)
    return CalibrationSelection(
        temperature=fitted_temperature if accepted else 1.0,
        fitted_temperature=fitted_temperature,
        accepted=accepted,
        raw_log_loss=raw_log_loss,
        calibrated_log_loss=calibrated_log_loss,
        raw_brier=raw_brier,
        calibrated_brier=calibrated_brier,
    )


def expected_calibration_error(y_true, proba, n_bins=10) -> float:
    """Mean |mean_pred - positive_fraction| over bins of proba in [0, 1]."""
    y_true = np.asarray(y_true, dtype=float)
    proba = np.asarray(proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(proba)
    error = 0.0
    for lo, hi in itertools.pairwise(edges):
        mask = (proba >= lo) & (proba <= hi) if hi == 1.0 else (proba >= lo) & (proba < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        error += (count / total) * abs(float(proba[mask].mean()) - float(y_true[mask].mean()))
    return float(error)


def demo() -> None:
    """Self-check: t=1 identity, swap complementarity, 0.5 fixed point, fit."""
    for p in (0.05, 0.3, 0.5, 0.7, 0.95):
        assert np.isclose(apply_temperature(p, 1.0), clip_probability(p), atol=1e-12)
    for p in (0.2, 0.4, 0.6, 0.8):
        assert np.isclose(
            apply_temperature(p, 2.0) + apply_temperature(1.0 - p, 2.0), 1.0, atol=1e-9
        )
    assert np.isclose(apply_temperature(0.5, 3.0), 0.5, atol=1e-12)
    rng = np.random.default_rng(0)
    proba = rng.uniform(0.1, 0.9, 200)
    y = (rng.uniform(size=200) < proba).astype(float)
    t = fit_temperature(proba, y)
    assert isinstance(t, float) and t > 0
    ece = expected_calibration_error(y, proba)
    assert 0.0 <= ece <= 1.0
    print("calibration demo(): all self-checks passed")


if __name__ == "__main__":
    demo()
