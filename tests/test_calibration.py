"""Hermetic tests for temperature scaling and ECE diagnostics."""

import math

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluate.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    select_temperature,
)
from src.evaluate.symmetry import clip_probability


def test_apply_temperature_unit_is_identity():
    for p in (0.1, 0.3, 0.5, 0.7, 0.9):
        assert math.isclose(apply_temperature(p, 1.0), clip_probability(p), abs_tol=1e-12)


def test_apply_temperature_preserves_complement_symmetry():
    for p in (0.1, 0.3, 0.7, 0.95):
        for t in (0.5, 2.0, 5.0):
            assert abs(apply_temperature(p, t) + apply_temperature(1 - p, t) - 1.0) < 1e-9


def test_apply_temperature_fixes_0_5():
    for t in (0.5, 1.0, 2.0, 20.0):
        assert math.isclose(apply_temperature(0.5, t), 0.5, abs_tol=1e-12)


def test_apply_temperature_preserves_auc():
    rng = np.random.default_rng(0)
    proba = rng.uniform(0.05, 0.95, size=500)
    y = (rng.uniform(size=500) < proba).astype(int)
    for t in (0.5, 2.0, 3.0):
        auc_raw = roc_auc_score(y, proba)
        auc_cal = roc_auc_score(y, apply_temperature(proba, t))
        assert math.isclose(auc_cal, auc_raw, abs_tol=1e-12)


def test_apply_temperature_preserves_threshold_classifier():
    rng = np.random.default_rng(1)
    proba = rng.uniform(size=200)
    for t in (0.5, 2.0, 4.0):
        cal = np.asarray(apply_temperature(proba, t))
        assert np.array_equal(cal >= 0.5, proba >= 0.5)


def test_apply_temperature_rejects_non_positive_temperature():
    for t in (0.0, -1.0):
        with pytest.raises(ValueError):
            apply_temperature(0.6, t)


def test_fit_temperature_returns_positive_finite():
    # Overconfident model: predictions too close to 0/1 for a noisier label.
    proba = np.array([0.95, 0.97, 0.02, 0.9, 0.1, 0.98, 0.05, 0.93, 0.08, 0.96])
    y = np.array([1, 1, 0, 1, 0, 1, 0, 1, 0, 0])
    t = fit_temperature(proba, y)
    assert isinstance(t, float)
    assert math.isfinite(t) and t > 0


def test_fit_temperature_no_improvement_returns_1():
    # All predictions exactly 0.5 (no class separation): every t on logit 0 is
    # identical, so the solver cannot improve on t=1.0 and falls back to it.
    proba = np.array([0.5, 0.5, 0.5, 0.5])
    y = np.array([1, 0, 1, 0])
    assert fit_temperature(proba, y) == 1.0


def test_fit_temperature_handles_tiny_input():
    proba = np.array([0.6, 0.4])
    y = np.array([1, 0])
    t = fit_temperature(proba, y)
    assert isinstance(t, float) and t > 0


def test_select_temperature_rejects_when_later_rows_are_worse(monkeypatch):
    monkeypatch.setattr("src.evaluate.calibration.fit_temperature", lambda *_args, **_kwargs: 0.5)
    result = select_temperature([0.9] * 8 + [0.9, 0.1], [1] * 8 + [1, 0], [0] * 8 + [1] * 2)

    assert result.fitted_temperature == 0.5
    assert result.temperature == 1.0
    assert not result.accepted
    assert result.calibrated_log_loss > result.raw_log_loss
    assert result.calibrated_brier > result.raw_brier


def test_select_temperature_accepts_only_when_both_later_metrics_improve(monkeypatch):
    monkeypatch.setattr("src.evaluate.calibration.fit_temperature", lambda *_args, **_kwargs: 0.5)
    result = select_temperature([0.9] * 8 + [0.9, 0.1], [1] * 8 + [0, 1], [0] * 8 + [1] * 2)

    assert result.temperature == 0.5
    assert result.accepted
    assert result.calibrated_log_loss < result.raw_log_loss
    assert result.calibrated_brier < result.raw_brier


def test_select_temperature_fits_each_fold_from_prior_folds_only(monkeypatch):
    seen: list[tuple[float, ...]] = []

    def fake_fit(proba, *_args, **_kwargs):
        seen.append(tuple(proba))
        return 1.0

    monkeypatch.setattr("src.evaluate.calibration.fit_temperature", fake_fit)
    # y must hold both classes in every validation fold (fold 1: rows 2-3, fold 2: rows 4-5).
    select_temperature([0.2, 0.3, 0.4, 0.5, 0.6, 0.7], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 2, 2])

    assert seen == [(0.2, 0.3), (0.2, 0.3, 0.4, 0.5), (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)]


def test_select_temperature_rejects_fractional_folds():
    # Walk-forward ordering is meaningful only for integer-encoded ordinals;
    # a fractional encoding silently misorders chronology and must be refused.
    with pytest.raises(ValueError, match="integer ordinals"):
        select_temperature([0.4, 0.6, 0.3], [1, 0, 1], [0.0, 0.5, 1.0])


def test_select_temperature_rejects_single_class_validation_fold():
    # A validation fold that cannot move between classes would drive a
    # degenerate pooled comparison; refuse loudly instead of accepting blindly.
    with pytest.raises(ValueError, match=r"validation fold 1.*both label classes"):
        select_temperature([0.5] * 8 + [0.6, 0.7], [1] * 8 + [1, 1], [0] * 8 + [1] * 2)


def test_select_temperature_brier_guardrail_accepts_within_tolerance(monkeypatch):
    # The small Brier regression is within tolerance, so the log-loss gain wins.
    monkeypatch.setattr("src.evaluate.calibration.fit_temperature", lambda *_args, **_kwargs: 0.7)
    p_val = [
        0.1663305428891164,
        0.2538198406045636,
        0.2857815239344147,
        0.34140834903838135,
        0.36919782351885067,
        0.555474259176121,
        0.7589521960679259,
        0.9517364361221391,
    ]
    y_val = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    result = select_temperature([0.5] * 8 + p_val, [1, 0] * 4 + y_val, [0] * 8 + [1] * 8)

    assert result.accepted
    assert result.calibrated_log_loss < result.raw_log_loss
    assert 0.0 < result.calibrated_brier - result.raw_brier < 1e-3


def test_select_temperature_brier_guardrail_rejects_beyond_tolerance(monkeypatch):
    # Zero tolerance rejects the Brier regression despite improved log loss.
    monkeypatch.setattr("src.evaluate.calibration.fit_temperature", lambda *_args, **_kwargs: 0.7)
    p_val = [
        0.1663305428891164,
        0.2538198406045636,
        0.2857815239344147,
        0.34140834903838135,
        0.36919782351885067,
        0.555474259176121,
        0.7589521960679259,
        0.9517364361221391,
    ]
    y_val = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    result = select_temperature(
        [0.5] * 8 + p_val, [1, 0] * 4 + y_val, [0] * 8 + [1] * 8, brier_guard_tolerance=0.0
    )

    assert not result.accepted
    assert result.temperature == 1.0
    assert result.calibrated_log_loss < result.raw_log_loss
    assert result.calibrated_brier > result.raw_brier


def test_expected_calibration_error_perfect():
    proba = np.array([0.0, 0.0, 1.0, 1.0])
    y = np.array([0, 0, 1, 1])
    assert abs(expected_calibration_error(y, proba, n_bins=2)) < 1e-12


def test_expected_calibration_error_miscalibrated():
    proba = np.array([0.9, 0.9, 0.1, 0.1])
    y = np.array([0, 1, 0, 1])
    assert expected_calibration_error(y, proba, n_bins=2) > 0.0
