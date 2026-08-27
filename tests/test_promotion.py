"""Focused promotion-gate tests: the probability-first gate in decide_promotion
and the exact 4-metric contract of compute_metrics. Pure functions only — no
MLflow, no DB, no HTTP."""

from __future__ import annotations

import json
from types import SimpleNamespace

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


# ── alias failure safety: a broken/absent @champion never raises here ──


class _RaisingMlflowClient:
    """Simulates an MLflow store where the @champion alias is missing or broken."""

    def get_model_version_by_alias(self, *_args, **_kwargs):
        raise RuntimeError("alias not found")


def test_resolve_champion_returns_none_when_alias_absent():
    # A missing/broken alias must not raise; the caller treats None as
    # "no standing champion" and never moves @champion on a failed promotion.
    assert promotion.resolve_champion(_RaisingMlflowClient()) is None


def test_resolve_champion_feature_contract_accepts_exact_version_tags():
    columns = ["rank_diff", "is_hard"]
    feature_hash = "contract-hash"
    version = SimpleNamespace(
        tags={
            "feature_cols": json.dumps(columns),
            "feature_cols_hash": feature_hash,
        }
    )

    class Client:
        def get_model_version(self, name, number):
            assert (name, number) == (promotion.PRODUCTION_MODEL, "4")
            return version

    changed, recorded_hash = promotion.resolve_champion_feature_contract(
        Client(), SimpleNamespace(version="4"), columns, feature_hash
    )

    assert (changed, recorded_hash) == (False, feature_hash)


def test_resolve_champion_feature_contract_rejects_changed_columns():
    version = SimpleNamespace(
        tags={
            "feature_cols": json.dumps(["rank_diff", "legacy_feature"]),
            "feature_cols_hash": "contract-hash",
        }
    )

    class Client:
        def get_model_version(self, _name, _number):
            return version

    assert promotion.resolve_champion_feature_contract(
        Client(), SimpleNamespace(version="4"), ["rank_diff", "is_hard"], "contract-hash"
    ) == (True, None)
