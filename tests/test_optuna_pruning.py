"""Hermetic regression coverage for the GBDT Optuna reporting/pruning bridge.

Exercises with fake trials and fake callback
environments, plus a numeric check that probability-space binary cross-entropy
and BCE-with-logits agree. No optuna study, model, database, or network is
spawned.
"""

from __future__ import annotations

import numpy as np
import optuna
import pytest

from src.training.optuna_pruning import (
    make_lightgbm_pruning_callback,
    make_xgboost_pruning_callback,
    report_and_maybe_prune,
)


class _FakeTrial:
    """Records reported (value, step) pairs; prunes when asked."""

    def __init__(self, should_prune_at: set[int] | None = None) -> None:
        self.reports: list[tuple[float, int]] = []
        self.should_prune_at: set[int] = should_prune_at or set()

    def report(self, value: float, step: int) -> None:
        self.reports.append((float(value), step))

    def should_prune(self) -> bool:
        return self.reports[-1][1] in self.should_prune_at


class _FakeLGBMEnv:
    """Minimal LightGBM callback environment holding one metric row."""

    def __init__(self, iteration: int, evaluation_result_list: list[list]) -> None:
        self.iteration = iteration
        self.evaluation_result_list = evaluation_result_list


def test_report_and_maybe_prune_records_value_and_step() -> None:
    trial = _FakeTrial()
    report_and_maybe_prune(trial, 0.69, 0)  # type: ignore[arg-type]
    report_and_maybe_prune(trial, 0.68, 1)  # type: ignore[arg-type]
    report_and_maybe_prune(trial, 0.66, 2)  # type: ignore[arg-type]
    assert trial.reports == [(0.69, 0), (0.68, 1), (0.66, 2)]


def test_report_and_maybe_prune_raises_trial_pruned_when_requested() -> None:
    trial = _FakeTrial(should_prune_at={2})
    report_and_maybe_prune(trial, 0.9, 0)  # type: ignore[arg-type]
    report_and_maybe_prune(trial, 0.9, 1)  # type: ignore[arg-type]
    with pytest.raises(optuna.TrialPruned):
        report_and_maybe_prune(trial, 0.9, 2)  # type: ignore[arg-type]


def test_pruning_propagates_through_lightgbm_callback() -> None:
    trial = _FakeTrial(should_prune_at={1})
    callback = make_lightgbm_pruning_callback(trial)  # type: ignore[arg-type]
    callback(
        _FakeLGBMEnv(
            0,
            [["valid", "binary_logloss", 0.5, 1.0]],
        )
    )
    assert trial.reports == [(0.5, 0)]
    with pytest.raises(optuna.TrialPruned):
        callback(
            _FakeLGBMEnv(
                1,
                [["valid", "binary_logloss", 0.51, 1.0]],
            )
        )


def test_lightgbm_callback_reports_consistently_across_rounds() -> None:
    trial = _FakeTrial()
    callback = make_lightgbm_pruning_callback(trial)  # type: ignore[arg-type]
    losses = []
    for epoch in range(5):
        callback(
            _FakeLGBMEnv(
                epoch,
                [["valid", "binary_logloss", 0.60 - 0.01 * epoch, 1.0]],
            )
        )
        losses.append(0.60 - 0.01 * epoch)
    assert trial.reports == [(v, i) for i, v in enumerate(losses)]


def test_pruning_propagates_through_xgboost_callback() -> None:
    trial = _FakeTrial(should_prune_at={1})
    callback = make_xgboost_pruning_callback(trial)  # type: ignore[arg-type]
    evals_log = {"validation": {"logloss": [0.55, 0.54]}}
    assert callback.after_iteration(None, 0, evals_log) is False
    assert trial.reports == [(0.54, 0)]
    with pytest.raises(optuna.TrialPruned):
        callback.after_iteration(None, 1, evals_log)


def test_xgboost_callback_reports_last_logloss_per_round_and_never_stops() -> None:
    trial = _FakeTrial()
    callback = make_xgboost_pruning_callback(trial)  # type: ignore[arg-type]
    for epoch in range(3):
        evals_log = {"validation": {"logloss": [0.5, 0.49, 0.48 + 0.01 * epoch]}}
        # callback contract: never stop training itself (always returns False)
        assert callback.after_iteration(None, epoch, evals_log) is False
    assert trial.reports == [(0.48, 0), (0.49, 1), (0.50, 2)]


def test_lightgbm_callback_ignores_unrelated_metrics_and_empty_rows() -> None:
    trial = _FakeTrial()
    callback = make_lightgbm_pruning_callback(trial)  # type: ignore[arg-type]
    callback(_FakeLGBMEnv(0, [["valid", "auc", 0.7, 1.0]]))
    callback(_FakeLGBMEnv(1, []))
    assert trial.reports == []


def _log_loss(p: float, y: float) -> float:
    """Probability-space binary log loss for a single (p, y)."""
    eps = 1e-12
    p = min(max(p, eps), 1.0 - eps)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def test_probability_bce_matches_bce_with_logits() -> None:
    """Regression guard: never feed ``log(probability)`` into BCE-with-logits."""
    rng = np.random.default_rng(0)
    # Include extreme probabilities near 0 and 1 so naive use would diverge.
    probs = np.concatenate(
        [rng.uniform(0.0, 1.0, 64), np.array([1e-7, 0.9999995, 1e-9, 1.0 - 1e-9])]
    )
    labels = (rng.uniform(size=probs.size) < probs).astype(np.float32)

    # BCE-with-logits closed form: max(x,0) - x*y + log(1 + exp(-|x|)).
    # Corrected NN objective computes logits = logit(p), then applies this.
    logits = np.log(probs / (1.0 - probs))
    from_logits = np.mean(
        np.maximum(logits, 0.0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))
    )
    probability_space = np.mean([_log_loss(p, y) for p, y in zip(probs, labels, strict=True)])

    assert np.isclose(from_logits, probability_space, atol=1e-6)
    assert np.isfinite(from_logits) and np.isfinite(probability_space)

    # Regresses `log(p)` used directly as a logits input: that diverges.
    log_probs = np.log(np.clip(probs, 1e-12, 1.0))
    broken = np.mean(
        np.maximum(log_probs, 0.0) - log_probs * labels + np.log1p(np.exp(-np.abs(log_probs)))
    )
    assert not np.isfinite(broken) or not np.isclose(broken, probability_space, rtol=1e-2)
