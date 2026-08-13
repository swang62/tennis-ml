"""Hermetic tests for the serving evidence-stacking helper.

Pure numpy/sklearn — no Bento, database, or MLflow. Verifies that the pure
`_stack_evidence` helper returns complementary p_win and base probabilities
when the paired orientation is reversed.
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression

from src.serving.service import _stack_evidence


def _toy_stacker() -> LogisticRegression:
    # Tiny no-intercept evidence matrix: a reversed request negates every
    # input, so logits negate and probabilities complement.
    X = pd.DataFrame(
        [
            [1.0, 0.5, 0.2],
            [0.1, 0.8, 0.4],
            [0.6, 0.3, 0.9],
        ],
        columns=["linear", "gbdt", "nn"],
    )
    y = np.array([1, 0, 1])
    return LogisticRegression(fit_intercept=False).fit(X, y)


def test_stack_evidence_reverse_complements():
    stacker = _toy_stacker()
    pairs = {"linear": (0.7, 0.3), "gbdt": (0.6, 0.4), "nn": (0.55, 0.45)}
    rev_pairs = {"linear": (0.3, 0.7), "gbdt": (0.4, 0.6), "nn": (0.45, 0.55)}

    out = _stack_evidence(pairs, stacker)
    rev = _stack_evidence(rev_pairs, stacker)

    # Each symmetric base probability complements its reverse.
    for name in ("linear", "gbdt", "nn"):
        assert abs(out[name] + rev[name] - 1.0) < 1e-6, name
    # The stacked p_win complements its reverse.
    assert abs(out["p_win"] + rev["p_win"] - 1.0) < 1e-6


def test_stack_evidence_explicit_order_matches_default():
    stacker = _toy_stacker()
    pairs = {"linear": (0.7, 0.3), "gbdt": (0.6, 0.4), "nn": (0.55, 0.45)}
    assert _stack_evidence(pairs, stacker) == _stack_evidence(
        pairs, stacker, stack_order=["linear", "gbdt", "nn"]
    )


def test_stack_evidence_preserves_stacker_feature_names():
    stacker = LogisticRegression(fit_intercept=False).fit(
        pd.DataFrame(
            [[1.0, 0.5, 0.2], [0.1, 0.8, 0.4], [0.6, 0.3, 0.9]],
            columns=["linear", "gbdt", "nn"],
        ),
        np.array([1, 0, 1]),
    )
    pairs = {"linear": (0.7, 0.3), "gbdt": (0.6, 0.4), "nn": (0.55, 0.45)}

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _stack_evidence(pairs, stacker)


def test_stack_evidence_rejects_wrong_key_order():
    stacker = _toy_stacker()
    pairs = {"gbdt": (0.6, 0.4), "linear": (0.7, 0.3), "nn": (0.55, 0.45)}
    try:
        _stack_evidence(pairs, stacker)
    except ValueError as exc:
        assert "stack order" in str(exc)
    else:
        raise AssertionError("wrong key order must be rejected")
