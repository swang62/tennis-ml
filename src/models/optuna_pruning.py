"""Plain-Python bridge for reporting per-iteration GBDT validation log loss to Optuna.

Both GBDT families train through ``eval_set=[(X_val, y_val)]`` with a single
chronological validation band. The shared core here turns each boosting round's
binary log loss into ``trial.report(value, step)`` and raises
``optuna.TrialPruned`` when ``trial.should_prune()`` is true, so the study
pruner can drop poor trials mid-training. The framework-specific adapters stay
thin: they only read the round's validation score out of xgboost's
``evals_log`` / lightgbm's ``evaluation_result_list``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import optuna


def report_and_maybe_prune(trial: optuna.trial.Trial, value: float, step: int) -> None:
    """Report ``value`` at ``step`` and prune the trial when Optuna asks.

    ``step`` is the zero-based boosting round, so reported steps are
    monotonically increasing across a trial. Raises ``optuna.TrialPruned`` (a
    subclass of ``TrialException``) when ``trial.should_prune()`` returns true;
    the manager catches it, so training stops and the trial is marked pruned.
    """
    trial.report(value, step)
    if trial.should_prune():
        raise optuna.TrialPruned(f"Trial pruned at boosting round {step}")


# ── xgboost adapter ──────────────────────────────────────────────────────────


def make_xgboost_pruning_callback(trial: optuna.trial.Trial) -> Any:
    """Build an ``xgboost.TrainingCallback`` that reports ``binary logloss``.

    The single ``eval_set`` is the chronological validation band, so the
    ``logloss`` history is exactly the per-iteration value Optuna should watch.
    Training never stops here; pruning surfaces as ``optuna.TrialPruned``,
    matching the callback contract (always return False).
    """
    from xgboost.callback import TrainingCallback

    class _XGBoostPruningCallback(TrainingCallback):
        def after_iteration(
            self,
            model: Any,  # noqa: ARG002 (unnamed base arg renamed in override breaks pyright)
            epoch: int,
            evals_log: dict[str, dict[str, list[Any]]],
        ) -> bool:
            for metrics in evals_log.values():
                history = metrics.get("logloss")
                if history:
                    report_and_maybe_prune(trial, float(history[-1]), epoch)
                    return False  # never stop training ourselves; pruner raises instead
            return False

    return _XGBoostPruningCallback()


# ── lightgbm adapter ─────────────────────────────────────────────────────────


def make_lightgbm_pruning_callback(trial: optuna.trial.Trial) -> Callable[..., None]:
    """Build a LightGBM callback that reports ``binary_logloss`` per round.

    ``env.iteration`` is the zero-based boosting round; the ``eval_set`` row
    holds the chronological validation band's binary log loss.
    """

    class _LightGBMPruningCallback:
        order = 20  # run every iteration, before/independent of early stopping

        def __call__(self, env: Any) -> None:
            if not env.evaluation_result_list:
                return
            for _, metric_name, value, *_ in env.evaluation_result_list:
                if metric_name == "binary_logloss":
                    report_and_maybe_prune(trial, float(value), env.iteration)
                    break

    return _LightGBMPruningCallback()
