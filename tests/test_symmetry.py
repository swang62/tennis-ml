"""Hermetic tests for the shared antisymmetry numerical helpers.

Pure numpy — no database, MLflow, or model imports, so this runs anywhere.
"""

import math

import numpy as np
import pytest

from src.evaluate.symmetry import (
    EPS,
    antisymmetric_evidence,
    clip_probability,
    evidence_to_probability,
    logit,
    sigmoid,
)


def test_antisymmetric_evidence_negates_on_swap():
    e = antisymmetric_evidence(0.52, 0.48)
    assert math.isfinite(e)
    assert math.isclose(antisymmetric_evidence(0.48, 0.52), -e, abs_tol=1e-12)
    assert antisymmetric_evidence(0.5, 0.5) == 0.0


def test_clip_probability_bounds_and_center():
    assert clip_probability(0.0) == EPS
    assert clip_probability(1.0) == 1.0 - EPS
    assert clip_probability(0.5) == 0.5
    assert clip_probability(-1.0) == EPS
    assert clip_probability(2.0) == 1.0 - EPS
    # At the clipping boundaries the evidence is finite and still antisymmetric.
    e = antisymmetric_evidence(0.0, 1.0)
    assert math.isfinite(e)
    assert math.isclose(e, logit(EPS), abs_tol=1e-12)
    assert math.isclose(antisymmetric_evidence(1.0, 0.0), -e, abs_tol=1e-12)


def test_sigmoid_complementarity():
    for x in (-100.0, -1.0, 0.0, 1.0, 100.0):
        assert abs(sigmoid(x) + sigmoid(-x) - 1.0) < 1e-9
    assert sigmoid(0.0) == 0.5
    assert sigmoid(np.inf) == 1.0
    assert sigmoid(-np.inf) == 0.0


def test_evidence_to_probability_roundtrip():
    e = antisymmetric_evidence(0.7, 0.3)
    p = evidence_to_probability(e)
    assert 0.0 < p < 1.0
    assert math.isclose(p + evidence_to_probability(-e), 1.0, abs_tol=1e-12)


def test_nan_inf_rejected():
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            clip_probability(bad)
        with pytest.raises(ValueError):
            antisymmetric_evidence(bad, 0.5)
        with pytest.raises(ValueError):
            antisymmetric_evidence(0.5, bad)


def test_vectorized_inputs():
    p_ab = np.array([0.7, 0.5, 0.9])
    p_ba = np.array([0.3, 0.5, 0.1])
    ev = np.asarray(antisymmetric_evidence(p_ab, p_ba))
    assert ev.shape == (3,)
    assert np.allclose(ev, -np.asarray(antisymmetric_evidence(p_ba, p_ab)), atol=1e-12)
    p = np.asarray(evidence_to_probability(ev))
    assert np.allclose(p + evidence_to_probability(-ev), 1.0, atol=1e-12)
