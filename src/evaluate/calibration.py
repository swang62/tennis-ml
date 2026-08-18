"""Temperature scaling for the ensemble p_win (symmetry-preserving zero-intercept Platt).

p' = sigmoid(t * logit(clip_probability(p))). Preserves swap symmetry and the
0.5 fixed point. Pure numpy on the module surface so serving (no scipy in its
Bento image) and hermetic tests can import it; scipy is imported lazily inside
fit_temperature, which only training calls.
"""

from __future__ import annotations

import itertools

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
