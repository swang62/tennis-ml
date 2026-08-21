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
    champion_feature_hash: object = "contract",
    candidate_feature_hash: object = "contract",
) -> int:
    return promotion.decide_promotion(
        cand_metrics=cand,
        prod_metrics=prod,
        champion_run_id=champion_run_id,
        candidate_run_id=candidate_run_id,
        force=force,
        champion_feature_hash=champion_feature_hash,
        candidate_feature_hash=candidate_feature_hash,
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


# ── primary-metric gate: either metric may improve within the tolerance ──


def test_promotes_when_log_loss_improves_and_auc_holds():
    cand = _metrics(log_loss=0.4)
    prod = _metrics()
    assert _decide(cand, prod) == 1


def test_rejects_equal_metrics():
    # Neither metric improves, so the candidate is not promoted.
    cand = _metrics(log_loss=0.5, roc_auc=0.5)
    prod = _metrics()
    assert _decide(cand, prod) == 0


def test_rejects_worse_log_loss_even_with_much_better_auc():
    cand = _metrics(log_loss=0.6, roc_auc=0.9)
    prod = _metrics()
    assert _decide(cand, prod) == 0


def test_auc_can_decline_exactly_at_tolerance():
    # ROC-AUC may trail by exactly PROMOTION_TOLERANCE when log loss improves.
    cand = _metrics(log_loss=0.4, roc_auc=0.5 - promotion.PROMOTION_TOLERANCE)
    prod = _metrics()
    assert _decide(cand, prod) == 1


def test_rejects_auc_decline_beyond_tolerance():
    cand = _metrics(
        log_loss=0.4,
        roc_auc=0.5 - promotion.PROMOTION_TOLERANCE - 1e-9,
    )
    prod = _metrics()
    assert _decide(cand, prod) == 0


def test_promotes_when_auc_improves_and_log_loss_is_within_tolerance():
    cand = _metrics(log_loss=0.5 + promotion.PROMOTION_TOLERANCE, roc_auc=0.6)
    prod = _metrics()
    assert _decide(cand, prod) == 1


def test_rejects_when_auc_is_not_strictly_better_and_log_loss_is_worse():
    cand = _metrics(log_loss=0.5 + promotion.PROMOTION_TOLERANCE, roc_auc=0.5)
    prod = _metrics()
    assert _decide(cand, prod) == 0


# ── the gate is metric-agnostic: it consumes whatever (calibrated) dict is fed ──


def test_promotes_on_calibrated_log_loss_improvement():
    # The 04 notebook feeds CALIBRATED test metrics in; a lower calibrated log
    # loss with AUC within tolerance promotes, exactly as for raw metrics.
    cand = _metrics(log_loss=0.37)
    prod = _metrics(log_loss=0.42)
    assert _decide(cand, prod) == 1


def test_rejects_plateau_calibrated_log_loss():
    # Equal calibrated metrics cannot force a tie into a win.
    cand = _metrics(log_loss=0.42, roc_auc=0.5)
    prod = _metrics(log_loss=0.42)
    assert _decide(cand, prod) == 0


# ── FEATURE_COLS is the sole compatibility contract ──


def test_changed_feature_contract_promotes_despite_bad_metrics():
    # A candidate trained on a different FEATURE_COLS is incompatible with the
    # incumbent: promote to refresh the contract even when metrics are worse.
    cand = _metrics(log_loss=0.9, roc_auc=0.1)
    prod = _metrics()
    assert (
        _decide(
            cand,
            prod,
            champion_feature_hash="old-contract",
            candidate_feature_hash="new-contract",
        )
        == 1
    )


def test_matching_contract_still_rejects_bad_metrics():
    # Same FEATURE_COLS: the metric gate is unchanged, so worse log loss skips
    # even when every lineage pin (base models, scaler, embeddings) changed.
    cand = _metrics(log_loss=0.6, roc_auc=0.9)
    prod = _metrics()
    assert _decide(cand, prod) == 0


def test_lineage_only_changes_do_not_force_promotion():
    # Base-model/scaler/embeddings pins are lineage only; with an unchanged
    # contract the candidate must still beat the metric gate.
    cand = _metrics(log_loss=0.9)
    prod = _metrics()
    assert _decide(cand, prod) == 0


def test_missing_champion_contract_promotes():
    # A legacy champion without contract tags is incompatible; the next
    # candidate refreshes it regardless of metrics and even of run-id match.
    cand = _metrics(log_loss=0.9)
    prod = _metrics()
    assert (
        _decide(
            cand,
            prod,
            champion_run_id="run1",
            candidate_run_id="run1",
            champion_feature_hash=None,
            candidate_feature_hash="new-contract",
        )
        == 1
    )


# ── read_champion_metrics: incumbent metrics come from champion version tags ──


class _Champion:
    def __init__(self, version="1", run_id="run1", tags=None):
        self.version = version
        self.run_id = run_id
        self._tags = tags or {}


class _Version:
    def __init__(self, tags):
        self.tags = tags


class _FakeClient:
    def __init__(self, tags):
        self._tags = tags

    def get_model_version(self, name, version):
        assert (name, version) == (promotion.PRODUCTION_MODEL, "1")
        return _Version(self._tags)


def _full_tags(**overrides):
    tags = {f"{promotion.METRIC_PREFIX}{m}": str(0.5) for m in promotion.METRIC_NAMES}
    tags.update(overrides)
    return tags


def test_read_champion_metrics_returns_all_four_gate_metrics():
    tags = _full_tags(metric_roc_auc="0.72")
    metrics = promotion.read_champion_metrics(_FakeClient(tags), _Champion())
    assert set(metrics) == set(promotion.METRIC_NAMES)
    assert metrics["roc_auc"] == pytest.approx(0.72)


def test_read_champion_metrics_requires_every_metric_tag():
    tags = _full_tags()
    del tags["metric_brier"]
    with pytest.raises(RuntimeError, match=r"missing metric tag.*metric_brier"):
        promotion.read_champion_metrics(_FakeClient(tags), _Champion())


def test_read_champion_metrics_rejects_malformed_tag():
    tags = _full_tags(metric_log_loss="not-a-number")
    with pytest.raises(RuntimeError, match=r"metric_log_loss.*malformed"):
        promotion.read_champion_metrics(_FakeClient(tags), _Champion())


def test_read_champion_metrics_rejects_non_finite_tag():
    tags = _full_tags(metric_log_loss="inf")
    with pytest.raises(RuntimeError, match=r"metric_log_loss.*not finite"):
        promotion.read_champion_metrics(_FakeClient(tags), _Champion())


def test_read_champion_metrics_fails_without_champion():
    with pytest.raises(RuntimeError, match="without a champion"):
        promotion.read_champion_metrics(_FakeClient({}), None)
