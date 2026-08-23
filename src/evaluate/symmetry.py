"""Pure NumPy helpers for symmetric clipping and antisymmetric evidence."""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def _as_scalar(x: float | np.ndarray) -> float | np.ndarray:
    """Return a Python float for scalar input, the array otherwise."""
    arr = np.asarray(x)
    return arr.item() if np.ndim(arr) == 0 else arr


def clip_probability(p, eps: float = EPS) -> float | np.ndarray:
    """Clip probabilities to [eps, 1-eps]; raise on NaN/inf input."""
    arr = np.asarray(p, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"probabilities must be finite, got non-finite value: {p!r}")
    return _as_scalar(np.clip(arr, eps, 1.0 - eps))


def logit(p) -> float | np.ndarray:
    """Log-odds of a probability: log(p / (1 - p))."""
    arr = np.asarray(p, dtype=float)
    return _as_scalar(np.log(arr / (1.0 - arr)))


def sigmoid(x) -> float | np.ndarray:
    """Return a numerically stable logistic value, including for infinities."""
    arr = np.asarray(x, dtype=float)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        result = np.where(arr >= 0, 1.0 / (1.0 + np.exp(-arr)), np.exp(arr) / (1.0 + np.exp(arr)))
    return _as_scalar(result)


def antisymmetric_evidence(p_ab, p_ba, eps: float = EPS) -> float | np.ndarray:
    """(logit(clip(p_ab)) - logit(clip(p_ba))) / 2 — negates on swap."""
    return _as_scalar(
        (logit(clip_probability(p_ab, eps)) - logit(clip_probability(p_ba, eps))) / 2.0
    )


def evidence_to_probability(e) -> float | np.ndarray:
    """sigmoid of antisymmetric evidence: P(chosen side wins)."""
    return sigmoid(e)


def demo() -> None:
    """Lightweight self-check: swap negation, boundary clipping, complementarity."""
    a, b = 0.52, 0.48
    e = antisymmetric_evidence(a, b)
    assert np.isclose(e, -antisymmetric_evidence(b, a), atol=1e-12)
    assert np.isfinite(e)
    assert clip_probability(0.0) == EPS
    assert clip_probability(1.0) == 1.0 - EPS
    assert clip_probability(0.5) == 0.5
    assert sigmoid(0.0) == 0.5
    assert sigmoid(np.inf) == 1.0
    assert sigmoid(-np.inf) == 0.0
    assert np.isclose(evidence_to_probability(e) + evidence_to_probability(-e), 1.0, atol=1e-9)
    for bad in (np.nan, np.inf, -np.inf):
        try:
            clip_probability(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"clip_probability must reject {bad}")
    print("symmetry demo(): all self-checks passed")


if __name__ == "__main__":
    demo()
