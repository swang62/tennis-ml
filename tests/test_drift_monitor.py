"""Offline drift-monitoring tests with mocked MLflow client and HTTP."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlflow
import pandas as pd
import pytest

from src.flows import drift


class _FakeModelVersion:
    def __init__(self, version="3", run_id="champ-run-id", creation_timestamp=1710000000000):
        self.version = version
        self.run_id = run_id
        self.creation_timestamp = creation_timestamp


class _FakeRun:
    def __init__(self, run_id="run-1"):
        self.info = SimpleNamespace(run_id=run_id)


class _FakeExperiment:
    def __init__(self, experiment_id="exp-1"):
        self.experiment_id = experiment_id


class _FakeMlflowClient:
    def __init__(self, champion=None, runs=None, download_dir="/fake/artifact/dir"):
        self._champion = champion
        self._runs = runs or []
        self._download_dir = download_dir
        self.logged_params: dict[str, dict[str, object]] = {}
        self.logged_metrics: dict[str, dict[str, float]] = {}
        self.logged_texts: list[tuple[str, str, str]] = []

    def get_model_version_by_alias(self, name, alias):
        assert name == "ensemble_lr_model"
        assert alias == "champion"
        if self._champion is None:
            from mlflow.exceptions import MlflowException

            raise MlflowException("Alias 'champion' not found")
        return self._champion

    def get_experiment_by_name(self, _name):
        return _FakeExperiment()

    def search_runs(self, experiment_ids, filter_string, order_by, max_results=1):
        del experiment_ids, filter_string, order_by, max_results
        return self._runs

    def get_run(self, run_id):
        if self._runs:
            return self._runs[0]
        return _FakeRun(run_id)

    def log_param(self, run_id, key, value):
        self.logged_params.setdefault(run_id, {})[key] = value

    def log_metric(self, run_id, key, value):
        self.logged_metrics.setdefault(run_id, {})[key] = value

    def log_text(self, run_id, text, artifact_file):
        self.logged_texts.append((run_id, text, artifact_file))

    def download_artifacts(self, _run_id, _path):
        return self._download_dir


def _stub_batch_response(ctxs, base_prob=0.65):
    return [
        {"player_id": c["player_id"], "opponent_id": c["opponent_id"], "p_win": base_prob}
        for c in ctxs
    ]


def _setup_model_info_stub(monkeypatch, mode="production", version="3", run_id="champ-run-id"):
    monkeypatch.setattr(drift, "BENTO_API_KEY", "")
    monkeypatch.setattr(drift, "PRODUCTION_BENTO_URL", "http://127.0.0.1:8187")
    monkeypatch.setattr(drift, "MODEL_INFO_ROUTE", "/api/internal/model-info")
    monkeypatch.setattr(
        drift,
        "_db_conn_params",
        lambda: {"server_address": None, "server_port": None, "database_name": None},
    )
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


def test_production_identity_mismatch_fails(monkeypatch):
    _setup_model_info_stub(monkeypatch, mode="development", version="2", run_id="other-run")

    client = _FakeMlflowClient(champion=_FakeModelVersion(version="3", run_id="champ-run-id"))

    with pytest.raises(RuntimeError, match="production Bento is not in production mode"):
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

    monkeypatch.setattr(drift, "to_dataframe", lambda _sql: pd.DataFrame())

    mlflow_runs = []

    def fake_start_run(experiment_id=None, run_name=None, tags=None, log_system_metrics=False):
        del experiment_id, log_system_metrics
        run_id = f"run-{len(mlflow_runs)}"
        mlflow_runs.append({"name": run_name, "tags": tags})
        return MagicMock(
            info=SimpleNamespace(run_id=run_id), __enter__=MagicMock(), __exit__=MagicMock()
        )

    monkeypatch.setattr(drift.mlflow, "start_run", fake_start_run)

    result = drift.check_drift()
    assert result == 0
    assert any(
        r.get("tags") and r["tags"].get("status") == "insufficient_data" for r in mlflow_runs
    )


def test_normal_flow_creates_baseline_and_check(monkeypatch, tmp_path):
    monkeypatch.setattr(drift, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(drift, "run_dbt_build", lambda **__kwargs: None)
    monkeypatch.setattr(drift, "load_env", lambda: None)
    _setup_model_info_stub(monkeypatch)

    champion = _FakeModelVersion(
        version="3", run_id="champ-run-id", creation_timestamp=1700000000000
    )
    client = _FakeMlflowClient(champion=champion)
    monkeypatch.setattr(drift, "MlflowClient", lambda: client)

    y_true = [1, 0, 1, 1, 0]
    fake_df = pd.DataFrame(
        {
            "match_id": ["m1", "m2", "m3", "m4", "m5"],
            "match_date": pd.Timestamp("2025-01-15"),
            "player_id": ["101", "102", "103", "104", "105"],
            "opponent_id": ["201", "202", "203", "204", "205"],
            "surface": ["hard"] * 5,
            "is_indoor": [0] * 5,
            "tournament_level": [3] * 5,
            "round_encoded": [4] * 5,
            "match_won": y_true,
        }
    )
    monkeypatch.setattr(drift, "to_dataframe", lambda _sql: fake_df)

    batch_calls = []

    def fake_post_batch(url, json=None, headers=None, timeout=None):
        del url, headers, timeout
        batch_calls.append(json)
        fake_resp = MagicMock()
        fake_resp.raise_for_status = lambda: None
        fake_resp.json.return_value = _stub_batch_response(json or [], base_prob=0.65)
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

    result = drift.check_drift()
    assert result == 0

    run_names = [r["name"] for r in mlflow_runs]
    assert "drift_baseline" in run_names
    assert "drift_check" in run_names

    assert len(batch_calls) == 1
    assert len(batch_calls[0]) == 5


def test_repeat_check_reuses_existing_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(drift, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(drift, "run_dbt_build", lambda **__kwargs: None)
    monkeypatch.setattr(drift, "load_env", lambda: None)
    _setup_model_info_stub(monkeypatch)

    baseline_artifact = {
        "metrics": {"roc_auc": 0.72, "brier": 0.19},
        "probas": [0.70, 0.30, 0.65],
        "y_true": [1, 0, 1],
    }

    art_dir = tmp_path / "fake_artifacts"
    art_dir.mkdir()
    (art_dir / "baseline.json").write_text(json.dumps(baseline_artifact))

    baseline_run = _FakeRun("baseline-run-id")
    champion = _FakeModelVersion(
        version="3", run_id="champ-run-id", creation_timestamp=1700000000000
    )
    client = _FakeMlflowClient(champion=champion, runs=[baseline_run], download_dir=str(art_dir))
    monkeypatch.setattr(drift, "MlflowClient", lambda: client)

    y_true = [1, 0, 1, 1, 0, 0, 1]
    fake_df = pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(7)],
            "match_date": pd.Timestamp("2025-01-15"),
            "player_id": [str(i + 100) for i in range(7)],
            "opponent_id": [str(i + 200) for i in range(7)],
            "surface": ["hard"] * 7,
            "is_indoor": [0] * 7,
            "tournament_level": [3] * 7,
            "round_encoded": [4] * 7,
            "match_won": y_true,
        }
    )
    monkeypatch.setattr(drift, "to_dataframe", lambda _sql: fake_df)

    def fake_post_batch(url, json=None, headers=None, timeout=None):
        del url, headers, timeout
        fake_resp = MagicMock()
        fake_resp.raise_for_status = lambda: None
        fake_resp.json.return_value = _stub_batch_response(json or [], base_prob=0.68)
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

    result = drift.check_drift()
    assert result == 0

    run_names = [r["name"] for r in mlflow_runs]
    assert "drift_baseline" not in run_names  # baseline already exists
    assert "drift_check" in run_names
