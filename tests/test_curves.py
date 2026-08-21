"""Tests for champion reference-curve pinning and resolution."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.evaluate.curves import (
    compute_curve_data,
    dict_to_curve_data,
    resolve_champion_curves,
    serialize_curve_data,
)


def _fake_version(tags):
    return SimpleNamespace(tags=tags)


def _fake_champion():
    return SimpleNamespace(version="7", run_id="run-7")


def test_compute_curve_data_keys_and_auc_range():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=200) < 0.5).astype(float)
    p = np.clip(rng.normal(0.5, 0.2, 200), 0.01, 0.99)
    data = compute_curve_data(y, p)
    assert set(data.roc) == {"fpr", "tpr", "auc"}
    assert set(data.pr) == {"recall", "precision", "ap"}
    assert set(data.calibration) == {"prob_true", "prob_pred", "brier"}
    assert 0.0 <= data.roc["auc"] <= 1.0
    assert len(data.roc["fpr"]) == len(data.roc["tpr"])


def test_serialize_round_trip():
    rng = np.random.default_rng(1)
    y = (rng.uniform(size=100) < 0.5).astype(float)
    p = np.clip(rng.uniform(0.1, 0.9, 100), 0.01, 0.99)
    data = compute_curve_data(y, p)
    raw, digest = serialize_curve_data(data)
    assert isinstance(digest, str) and len(digest) == 64
    restored = dict_to_curve_data(__import__("json").loads(raw.decode("utf-8")))
    assert restored.roc["auc"] == data.roc["auc"]
    assert restored.calibration["prob_pred"] == data.calibration["prob_pred"]


def test_resolve_returns_none_without_champion():
    class _Client:
        def get_model_version(self, *_a, **_k):
            raise AssertionError("should not be called")

    assert resolve_champion_curves(_Client(), None) is None


def test_resolve_returns_none_without_curve_tag():
    client = type("C", (), {"get_model_version": lambda *_: _fake_version({})})()
    champion = _fake_champion()
    assert resolve_champion_curves(client, champion) is None


def test_resolve_loads_pinned_curve(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    y = (rng.uniform(size=100) < 0.5).astype(float)
    p = np.clip(rng.uniform(0.1, 0.9, 100), 0.01, 0.99)
    data = compute_curve_data(y, p)
    raw, _ = serialize_curve_data(data)
    artifact = tmp_path / "champion_curves.json"
    artifact.write_bytes(raw)

    captured = {}

    def _download(artifact_uri):
        captured["uri"] = artifact_uri
        return str(artifact)

    monkeypatch.setattr("mlflow.artifacts.download_artifacts", _download)
    client = type(
        "C",
        (),
        {"get_model_version": lambda *_: _fake_version({"champion_curve_uri": "runs:/r/c.json"})},
    )()
    champion = _fake_champion()
    result = resolve_champion_curves(client, champion)
    assert result is not None
    assert result.roc["auc"] == data.roc["auc"]
    assert captured["uri"] == "runs:/r/c.json"


def test_resolve_returns_none_on_download_failure(monkeypatch):
    def _download(_uri):
        raise RuntimeError("registry unreachable")

    monkeypatch.setattr("mlflow.artifacts.download_artifacts", _download)
    client = type(
        "C",
        (),
        {"get_model_version": lambda *_: _fake_version({"champion_curve_uri": "runs:/r/c.json"})},
    )()
    champion = _fake_champion()
    assert resolve_champion_curves(client, champion) is None
