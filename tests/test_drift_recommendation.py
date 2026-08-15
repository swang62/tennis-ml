"""Pure-function drift tests: recommendation mapping, pinned-metric parsing, and
the size-matched reference window clamp. No Evidently, no DB, no HTTP."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from src.evaluate.promotion import METRIC_NAMES
from src.flows import drift

# ── recommendation mapping (plan threshold table) ──


def _recommend(**kwargs) -> str:
    defaults = {
        "drift_share": 0.1,
        "prediction_psi": 0.1,
        "calibration_delta": 0.01,
        "n_current": 60,
        "auc_drop": 0.01,
        "per_feature_drift": {},
    }
    defaults.update(kwargs)
    return drift._recommendation(**defaults)


def test_recommendation_healthy_when_no_triggers():
    assert _recommend() == "healthy"


def test_recommendation_investigate_on_moderate_feature_psi():
    assert _recommend(per_feature_drift={"player1_first_serve_pct": 0.15}) == "investigate"


def test_recommendation_retrain_on_drift_share():
    assert _recommend(drift_share=drift.DRIFT_SHARE_THRESHOLD) == "retrain"


def test_recommendation_retrain_on_prediction_psi():
    assert _recommend(prediction_psi=drift.DRIFT_PRED_PSI_THRESHOLD) == "retrain"


def test_recommendation_retrain_on_calibration_delta():
    assert _recommend(calibration_delta=drift.DRIFT_CALIBRATION_DELTA + 0.01) == "retrain"


def test_recommendation_retrain_on_auc_drop_with_enough_matches():
    assert (
        _recommend(n_current=drift.DRIFT_MIN_N_FOR_AUC, auc_drop=drift.DRIFT_AUC_DROP + 0.01)
        == "retrain"
    )


def test_recommendation_ignores_auc_drop_when_sample_too_small():
    assert _recommend(n_current=drift.DRIFT_MIN_N_FOR_AUC - 1, auc_drop=0.9) == "healthy"


def test_recommendation_retrain_takes_precedence_over_investigate():
    assert (
        _recommend(
            drift_share=drift.DRIFT_SHARE_THRESHOLD,
            per_feature_drift={"player1_serve_win_pct": 0.15},
        )
        == "retrain"
    )


def test_recommendation_thresholds_are_strict_on_delta_triggers():
    # Delta triggers are strict >; share/PSI triggers are >=.
    assert _recommend(calibration_delta=drift.DRIFT_CALIBRATION_DELTA) == "healthy"
    assert (
        _recommend(n_current=drift.DRIFT_MIN_N_FOR_AUC, auc_drop=drift.DRIFT_AUC_DROP) == "healthy"
    )
    assert _recommend(prediction_psi=drift.DRIFT_PRED_PSI_THRESHOLD) == "retrain"
    assert _recommend(drift_share=drift.DRIFT_SHARE_THRESHOLD) == "retrain"


# ── pinned champion metric tags → metric dict ──


class _FakeClient:
    def __init__(self, tags: dict[str, str]):
        self._tags = tags

    def get_model_version(self, _name, _version):
        return SimpleNamespace(tags=self._tags)


def _champion():
    return SimpleNamespace(version="3")


def test_pinned_metrics_parses_champion_tags():
    tags: dict[str, str] = {}
    for name in METRIC_NAMES:
        tags[f"{drift.METRIC_PREFIX}{name}"] = "0.61"
    tags[drift.EVAL_SPLIT_SIZE_KEY] = "125"
    tags[drift.EVAL_MAX_DATE_KEY] = "2025-01-10"

    pinned = drift._pinned_metrics(_FakeClient(tags), _champion())  # type: ignore[arg-type]

    assert set(pinned) == set(METRIC_NAMES) | {
        "eval_split_size",
        "eval_max_date",
    }
    assert pinned["roc_auc"] == 0.61
    assert pinned["eval_split_size"] == 125
    assert pinned["eval_max_date"] == "2025-01-10"


def test_pinned_metrics_skips_missing_tags():
    client = _FakeClient({f"{drift.METRIC_PREFIX}roc_auc": "0.72"})
    pinned = drift._pinned_metrics(client, _champion())  # type: ignore[arg-type]
    assert pinned == {"roc_auc": 0.72}


# ── size-matched reference window (DRIFT_REF_MIN/MAX clamp) ──


def _window_frame(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(n)],
            "match_date": pd.Timestamp("2025-01-15"),
            "match_won": [1] * n,
        }
    )


def _reference_limit_for(n_current: int, monkeypatch: pytest.MonkeyPatch) -> int:
    captured: list[list[object]] = []

    def fake_execute_df(sql, params=None):
        if "match_date > %s" in sql:
            return _window_frame(n_current)
        if "match_date < %s" in sql:
            captured.append(list(params or []))
            return pd.DataFrame()
        raise AssertionError(f"unexpected drift SQL: {sql}")

    monkeypatch.setattr(drift, "execute_df", fake_execute_df)
    drift._pull_windows(date(2025, 1, 10))
    assert len(captured) == 1
    assert len(captured[0]) == 2
    return int(captured[0][1])  # type: ignore[arg-type]


def test_reference_window_floor_at_drift_ref_min(monkeypatch):
    assert _reference_limit_for(10, monkeypatch) == drift.DRIFT_REF_MIN


def test_reference_window_matches_current_size_in_band(monkeypatch):
    assert _reference_limit_for(500, monkeypatch) == 500


def test_reference_window_capped_at_drift_ref_max(monkeypatch):
    assert _reference_limit_for(5000, monkeypatch) == drift.DRIFT_REF_MAX
