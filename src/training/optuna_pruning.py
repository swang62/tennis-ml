"""Report GBDT validation loss to Optuna and prune poor trials."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import optuna


def report_and_maybe_prune(trial: optuna.trial.Trial, value: float, step: int) -> None:
    """Report a boosting-round value and raise when the trial should be pruned."""
    trial.report(value, step)
    if trial.should_prune():
        raise optuna.TrialPruned(f"Trial pruned at boosting round {step}")


# ── xgboost adapter ──────────────────────────────────────────────────────────


def make_xgboost_pruning_callback(trial: optuna.trial.Trial, step_offset: int = 0) -> Any:
    """Build an XGBoost callback that reports validation ``logloss``."""
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
                    report_and_maybe_prune(trial, float(history[-1]), step_offset + epoch)
                    return False  # never stop training ourselves; pruner raises instead
            return False

    return _XGBoostPruningCallback()


# ── lightgbm adapter ─────────────────────────────────────────────────────────


def make_lightgbm_pruning_callback(
    trial: optuna.trial.Trial, step_offset: int = 0
) -> Callable[..., None]:
    """Build a LightGBM callback that reports validation ``binary_logloss``."""

    class _LightGBMPruningCallback:
        order = 20  # run every iteration, before/independent of early stopping

        def __call__(self, env: Any) -> None:
            if not env.evaluation_result_list:
                return
            for _, metric_name, value, *_ in env.evaluation_result_list:
                if metric_name == "binary_logloss":
                    report_and_maybe_prune(trial, float(value), step_offset + env.iteration)
                    break

    return _LightGBMPruningCallback()
