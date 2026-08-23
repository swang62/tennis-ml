"""Pure-function drift tests: recommendation mapping, pinned-metric parsing, and
the size-matched reference window clamp. No Evidently, no DB, no HTTP."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

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
    assert _reference_limit_for(drift.DRIFT_REF_MIN - 1, monkeypatch) == drift.DRIFT_REF_MIN


def test_reference_window_matches_current_size_in_band(monkeypatch):
    in_band = (drift.DRIFT_REF_MIN + drift.DRIFT_REF_MAX) // 2
    assert _reference_limit_for(in_band, monkeypatch) == in_band


def test_reference_window_capped_at_drift_ref_max(monkeypatch):
    assert _reference_limit_for(drift.DRIFT_REF_MAX + 1, monkeypatch) == drift.DRIFT_REF_MAX
