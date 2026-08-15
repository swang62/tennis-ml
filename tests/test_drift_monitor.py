"""Offline drift-monitoring tests with mocked MLflow client and HTTP."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlflow
import numpy as np
import pandas as pd
import pytest

from src.flows import drift


class _FakeModelVersion:
    def __init__(self, version="3", run_id="champ-run-id", creation_timestamp=1710000000000):
        self.version = version
        self.run_id = run_id
        self.creation_timestamp = creation_timestamp


class _FakeExperiment:
    def __init__(self, experiment_id="exp-1"):
        self.experiment_id = experiment_id


class _FakeMlflowClient:
    def __init__(self, champion=None, tags=None):
        self._champion = champion
        self._tags = tags or {}
        self.logged_params: dict[str, dict[str, object]] = {}
        self.logged_metrics: dict[str, dict[str, float]] = {}
        self.logged_texts: list[tuple[str, str, str]] = []
        self.logged_artifacts: list[tuple[str, str]] = []

    def get_model_version_by_alias(self, name, alias):
        assert name == "ensemble_lr_model"
        assert alias == "champion"
        if self._champion is None:
            from mlflow.exceptions import MlflowException

            raise MlflowException("Alias 'champion' not found")
        return self._champion

    def get_model_version(self, name, version):
        del name, version
        return SimpleNamespace(tags=self._tags)

    def get_experiment_by_name(self, _name):
        return _FakeExperiment()

    def log_param(self, run_id, key, value):
        self.logged_params.setdefault(run_id, {})[key] = value

    def log_metric(self, run_id, key, value):
        self.logged_metrics.setdefault(run_id, {})[key] = value

    def log_text(self, run_id, text, artifact_file):
        self.logged_texts.append((run_id, text, artifact_file))

    def log_artifact(self, run_id, local_path):
        self.logged_artifacts.append((run_id, local_path))


def _stub_batch_response(ctxs, probs=None):
    if probs is None:
        probs = [0.75 if i % 2 == 0 else 0.35 for i in range(len(ctxs))]
    return [
        {"player_id": c["player_id"], "opponent_id": c["opponent_id"], "p_win": probs[i]}
        for i, c in enumerate(ctxs)
    ]


def _setup_model_info_stub(monkeypatch, mode="production", version="3", run_id="champ-run-id"):
    monkeypatch.setattr(drift, "BENTO_API_KEY", "")
    monkeypatch.setattr(drift, "PRODUCTION_BENTO_URL", "http://127.0.0.1:8187")
    monkeypatch.setattr(drift, "MODEL_INFO_ROUTE", "/api/model_info")
    fake_model_info = {
        "ok": True,
        "data": {
            "mode": mode,
            "manifest": {
                "champion": {
                    "registered_model_name": "ensemble_lr_model",
                    "version": version,
                    "run_id": run_id,
                }
            },
            "database": {"server_address": None, "server_port": None, "database_name": None},
        },
    }
    fake_resp = MagicMock()
    fake_resp.json.return_value = fake_model_info
    monkeypatch.setattr(drift.requests, "get", lambda _url, **__kwargs: fake_resp)


def test_no_champion_fails():
    client = _FakeMlflowClient(champion=None)
    with pytest.raises(RuntimeError, match="no champion found"):
        drift._validate_production(client)  # type: ignore[arg-type]


def test_champion_cutoff_prefers_training_data_tag():
    from datetime import date

    champion = _FakeModelVersion(
        version="3", run_id="champ-run-id", creation_timestamp=1700000000000
    )
    client = _FakeMlflowClient(
        champion=champion, tags={drift.TRAIN_DATA_MAX_DATE_KEY: "2025-01-10"}
    )
    assert drift._champion_cutoff_date(client) == date(2025, 1, 10)  # type: ignore[arg-type]


def test_champion_cutoff_falls_back_to_creation():
    from datetime import UTC, datetime

    champion = _FakeModelVersion(
        version="3", run_id="champ-run-id", creation_timestamp=1700000000000
    )
    client = _FakeMlflowClient(champion=champion, tags={})
    expected = datetime.fromtimestamp(1700000000000 / 1000, tz=UTC).date()
    assert drift._champion_cutoff_date(client) == expected  # type: ignore[arg-type]


def test_development_bento_is_valid_for_drift(monkeypatch):
    _setup_model_info_stub(monkeypatch, mode="development", version="2", run_id="other-run")

    client = _FakeMlflowClient(champion=_FakeModelVersion(version="3", run_id="champ-run-id"))

    champion = drift._validate_production(client)  # type: ignore[arg-type]
    assert champion.version == "3"


def test_vite_url_is_rejected_before_http_request(monkeypatch):
    monkeypatch.setattr(drift, "PRODUCTION_BENTO_URL", "http://localhost:5173")
    monkeypatch.setattr(
        drift.requests,
        "get",
        lambda *_args, **_kwargs: pytest.fail("HTTP call"),
    )

    client = _FakeMlflowClient(champion=_FakeModelVersion())
    with pytest.raises(RuntimeError, match="points to Vite"):
        drift._validate_production(client)  # type: ignore[arg-type]


def test_empty_population_insufficient_data(monkeypatch, tmp_path):
    monkeypatch.setattr(drift, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(drift, "run_dbt_build", lambda **__kwargs: None)
    monkeypatch.setattr(drift, "load_env", lambda: None)
    _setup_model_info_stub(monkeypatch)

    champion = _FakeModelVersion(
        version="3", run_id="champ-run-id", creation_timestamp=1700000000000
    )
    client = _FakeMlflowClient(champion=champion)
    monkeypatch.setattr(drift, "MlflowClient", lambda: client)

    monkeypatch.setattr(drift, "execute_df", lambda _sql, _params=None: pd.DataFrame())

    mlflow_runs = []

    def fake_start_run(experiment_id=None, run_name=None, tags=None, log_system_metrics=False):
        del experiment_id, log_system_metrics
        run_id = f"run-{len(mlflow_runs)}"
        mlflow_runs.append({"name": run_name, "tags": tags})
        return MagicMock(
            info=SimpleNamespace(run_id=run_id), __enter__=MagicMock(), __exit__=MagicMock()
        )

    monkeypatch.setattr(drift.mlflow, "start_run", fake_start_run)

    result = drift.drift_flow.fn()
    assert result == 0
    assert any(
        r.get("tags") and r["tags"].get("status") == "insufficient_data" for r in mlflow_runs
    )


def _champion_tags(pinned_metrics: dict[str, float]) -> dict[str, str]:
    """Champion model-version tags: training-data watermark + pinned metric tags."""
    tags = {
        drift.TRAIN_DATA_MAX_DATE_KEY: "2025-01-10",
        drift.METRIC_COMPOSITE_KEY: "0.12",
        drift.EVAL_SPLIT_SIZE_KEY: "125",
        drift.EVAL_MAX_DATE_KEY: "2025-01-10",
    }
    tags.update(
        {f"{drift.METRIC_PREFIX}{name}": str(value) for name, value in pinned_metrics.items()}
    )
    return tags


_PINNED_METRICS = {
    "roc_auc": 0.72,
    "pr_auc": 0.60,
    "accuracy": 0.63,
    "precision": 0.60,
    "recall": 0.62,
    "f1": 0.61,
    "mcc": 0.28,
    "brier": 0.19,
}


def _fake_window(n: int, *, seed: int, win_rate: float = 0.8) -> pd.DataFrame:
    """A gold.match_features window: context/identity columns + all FEATURE_COLS."""
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {col: rng.normal(0, 1, n) for col in drift.FEATURE_COLS}
    data.update(
        {
            "match_id": [f"m{i}" for i in range(n)],
            "match_date": pd.Timestamp("2025-01-15"),
            "player_id": [str(i + 100) for i in range(n)],
            "opponent_id": [str(i + 200) for i in range(n)],
            "surface": ["hard"] * n,
            "is_indoor": [0] * n,
            "tournament_level": [3] * n,
            "round_encoded": [4] * n,
            "match_won": [1] * int(n * win_rate) + [0] * (n - int(n * win_rate)),
        }
    )
    return pd.DataFrame(data)


def test_normal_flow_runs_evidently_and_logs_drift_check(monkeypatch, tmp_path):
    monkeypatch.setattr(drift, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(drift, "run_dbt_build", lambda **__kwargs: None)
    monkeypatch.setattr(drift, "load_env", lambda: None)
    _setup_model_info_stub(monkeypatch)

    champion = _FakeModelVersion(
        version="3", run_id="champ-run-id", creation_timestamp=1700000000000
    )
    client = _FakeMlflowClient(champion=champion, tags=_champion_tags(_PINNED_METRICS))
    monkeypatch.setattr(drift, "MlflowClient", lambda: client)

    current_frame = _fake_window(60, seed=1)
    reference_frame = _fake_window(60, seed=2, win_rate=0.5)
    sql_calls: list[tuple[str, list[object]]] = []

    def fake_execute_df(sql, params=None):
        sql_calls.append((sql, params or []))
        if "match_date > %s" in sql:
            return current_frame
        if "match_date < %s" in sql:
            return reference_frame
        raise AssertionError(f"unexpected drift SQL: {sql}")

    monkeypatch.setattr(drift, "execute_df", fake_execute_df)

    posts: list[list[dict[str, object]]] = []

    def fake_post_batch(url, json=None, headers=None, timeout=None):
        del url, headers, timeout
        ctxs = (json or {}).get("rows", [])
        posts.append(ctxs)
        fake_resp = MagicMock()
        fake_resp.raise_for_status = lambda: None
        fake_resp.json.return_value = _stub_batch_response(ctxs)
        return fake_resp

    monkeypatch.setattr(drift.requests, "post", fake_post_batch)

    mlflow_runs = []

    def fake_start_run(experiment_id=None, run_name=None, tags=None, log_system_metrics=False):
        del experiment_id, log_system_metrics
        run_id = f"run-{len(mlflow_runs)}-{run_name}"
        mlflow_runs.append({"name": run_name, "tags": tags})
        return MagicMock(
            info=SimpleNamespace(run_id=run_id), __enter__=MagicMock(), __exit__=MagicMock()
        )

    monkeypatch.setattr(drift.mlflow, "start_run", fake_start_run)

    result = drift.drift_flow.fn()
    assert result == 0

    assert [r["name"] for r in mlflow_runs] == ["drift_check"]
    check_run = mlflow_runs[0]
    recommendation = check_run["tags"]["recommendation"]
    assert recommendation in {"healthy", "investigate", "retrain"}
    assert check_run["tags"]["retrain_required"] == str(recommendation == "retrain")

    # Both windows pulled from gold, reference size-matched to the current count.
    assert len(sql_calls) == 2
    current_sql, reference_sql = sql_calls
    assert "match_date > %s" in current_sql[0]
    assert "match_date < %s" in reference_sql[0]
    assert reference_sql[1] == [date(2025, 1, 10), 60]

    # Both windows scored through the production Bento.
    assert len(posts) == 2
    assert len(posts[0]) == 60
    assert len(posts[1]) == 60

    summary_text = [text for _, text, name in client.logged_texts if name == "drift_summary.json"]
    assert summary_text
    summary = json.loads(summary_text[0])
    assert summary["recommendation"] == recommendation
    assert summary["retrain_required"] == (recommendation == "retrain")
    assert {"drift_share", "prediction_psi", "calibration_delta", "per_feature_drift"} <= set(
        summary
    )

    assert sorted(Path(path).name for _, path in client.logged_artifacts) == [
        "drift_report_2025-01-10_v3.html",
        "drift_report_2025-01-10_v3.json",
    ]
    assert (tmp_path / "drift_report_2025-01-10_v3.json").exists()
    assert (tmp_path / "drift_report_2025-01-10_v3.html").exists()


def test_calibration_shift_triggers_retrain_verdict(monkeypatch, tmp_path):
    monkeypatch.setattr(drift, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(drift, "run_dbt_build", lambda **__kwargs: None)
    monkeypatch.setattr(drift, "load_env", lambda: None)
    _setup_model_info_stub(monkeypatch)

    champion = _FakeModelVersion(
        version="3", run_id="champ-run-id", creation_timestamp=1700000000000
    )
    client = _FakeMlflowClient(champion=champion, tags=_champion_tags(_PINNED_METRICS))
    monkeypatch.setattr(drift, "MlflowClient", lambda: client)

    # Current window wins 80% of matches, reference 0%: a 0.8 calibration
    # delta, far past the 0.05 threshold, so the verdict is retrain no matter
    # what Evidently reports for distribution drift.
    current_frame = _fake_window(60, seed=3, win_rate=0.8)
    reference_frame = _fake_window(60, seed=4, win_rate=0.0)

    def fake_execute_df(sql, _params=None):
        if "match_date > %s" in sql:
            return current_frame
        if "match_date < %s" in sql:
            return reference_frame
        raise AssertionError(f"unexpected drift SQL: {sql}")

    monkeypatch.setattr(drift, "execute_df", fake_execute_df)

    def fake_post_batch(url, json=None, headers=None, timeout=None):
        del url, headers, timeout
        ctxs = (json or {}).get("rows", [])
        fake_resp = MagicMock()
        fake_resp.raise_for_status = lambda: None
        fake_resp.json.return_value = _stub_batch_response(ctxs)
        return fake_resp

    monkeypatch.setattr(drift.requests, "post", fake_post_batch)

    mlflow_runs = []

    def fake_start_run(experiment_id=None, run_name=None, tags=None, log_system_metrics=False):
        del experiment_id, log_system_metrics
        run_id = f"run-{len(mlflow_runs)}-{run_name}"
        mlflow_runs.append({"name": run_name, "tags": tags})
        return MagicMock(
            info=SimpleNamespace(run_id=run_id), __enter__=MagicMock(), __exit__=MagicMock()
        )

    monkeypatch.setattr(drift.mlflow, "start_run", fake_start_run)

    result = drift.drift_flow.fn()
    assert result == 0

    check_run = mlflow_runs[0]
    assert check_run["name"] == "drift_check"
    assert check_run["tags"]["recommendation"] == "retrain"
    assert check_run["tags"]["retrain_required"] == "True"

    summary_text = next(
        text for _, text, name in client.logged_texts if name == "drift_summary.json"
    )
    summary = json.loads(summary_text)
    assert summary["recommendation"] == "retrain"
    assert summary["calibration_delta"] > drift.DRIFT_CALIBRATION_DELTA
