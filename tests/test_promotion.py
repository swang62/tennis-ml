"""Focused promotion-gate tests: the probability-first gate in decide_promotion
and the exact 4-metric contract of compute_metrics. Pure functions only — no
MLflow, no DB, no HTTP."""

from __future__ import annotations

import pytest

from src.evaluate import promotion


def _metrics(**overrides: float) -> dict[str, float]:
    """Full 4-metric dict, defaulting to 0.5 for every metric."""
    metrics = dict.fromkeys(promotion.METRIC_NAMES, 0.5)
    metrics.update(overrides)
    return metrics


def _decide(
    cand: dict[str, float],
    prod: dict[str, float] | None = None,
    champion_run_id: object | None = None,
    candidate_run_id: object = "cand",
    force: bool = False,
) -> int:
    return promotion.decide_promotion(
        cand_metrics=cand,
        prod_metrics=prod,
        champion_run_id=champion_run_id,
        candidate_run_id=candidate_run_id,
        force=force,
    )


# ── compute_metrics returns exactly the 4 gate metrics ──

Y_TRUE = [0, 0, 1, 1]
PROBA = [0.1, 0.2, 0.8, 0.9]
PRED = [0, 0, 1, 1]


def test_compute_metrics_returns_exact_metric_contract():
    metrics = promotion.compute_metrics(Y_TRUE, PROBA, PRED)
    assert set(metrics) == {"log_loss", "roc_auc", "accuracy", "brier"}


def test_compute_metrics_values_match_sklearn():
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    metrics = promotion.compute_metrics(Y_TRUE, PROBA, PRED)
    assert metrics["log_loss"] == pytest.approx(log_loss(Y_TRUE, PROBA))
    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(Y_TRUE, PROBA))
    assert metrics["accuracy"] == pytest.approx(accuracy_score(Y_TRUE, PRED))
    assert metrics["brier"] == pytest.approx(brier_score_loss(Y_TRUE, PROBA))


# ── first promotion, idempotency, force ──


def test_first_promotion_when_no_incumbent():
    cand = _metrics(log_loss=0.9)  # losing candidate still promoted: no incumbent
    assert _decide(cand, prod=None) == 1


def test_already_promoted_skips_even_with_improvement():
    cand = _metrics(log_loss=0.4, roc_auc=0.9)
    prod = _metrics()
    assert _decide(cand, prod, champion_run_id="run1", candidate_run_id="run1") == 0


def test_force_overrides_gate():
    cand = _metrics(log_loss=0.9, roc_auc=0.1)
    prod = _metrics()
    assert _decide(cand, prod, force=True) == 1


def test_force_overrides_idempotency():
    cand = _metrics(log_loss=0.9)
    prod = _metrics()
    assert _decide(cand, prod, champion_run_id="run1", candidate_run_id="run1", force=True) == 1


# ── probability-first gate: strict log loss, bounded AUC decline ──


def test_promotes_when_log_loss_improves_and_auc_holds():
    cand = _metrics(log_loss=0.4)
    prod = _metrics()
    assert _decide(cand, prod) == 1


def test_requires_strict_log_loss_improvement():
    # Equal log loss is not an improvement; a higher AUC cannot compensate.
    cand = _metrics(log_loss=0.5, roc_auc=0.9)
    prod = _metrics()
    assert _decide(cand, prod) == 0


def test_rejects_worse_log_loss_even_with_much_better_auc():
    cand = _metrics(log_loss=0.6, roc_auc=0.9)
    prod = _metrics()
    assert _decide(cand, prod) == 0


def test_auc_can_decline_exactly_at_tolerance():
    # ROC-AUC may trail by exactly PROMOTION_AUC_TOLERANCE when log loss improves.
    cand = _metrics(log_loss=0.4, roc_auc=0.5 - promotion.PROMOTION_AUC_TOLERANCE)
    prod = _metrics()
    assert _decide(cand, prod) == 1


def test_rejects_auc_decline_beyond_tolerance():
    cand = _metrics(
        log_loss=0.4,
        roc_auc=0.5 - promotion.PROMOTION_AUC_TOLERANCE - 1e-9,
    )
    prod = _metrics()
    assert _decide(cand, prod) == 0
