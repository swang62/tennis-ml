"""Native BentoModel materialization tests (deploy flow, mocked MLflow/BentoML)."""

import importlib
import json
from types import SimpleNamespace

import lightgbm
import xgboost
from sklearn.linear_model import LogisticRegression


def _deploy():
    return importlib.import_module("src.flows.deploy")


# --- _materialize_native_model: XGBoost vs LightGBM framework detection ---


def _pin(name="gbdt_best", version="2"):
    return {
        "registered_model_name": name,
        "version": version,
        "model_uri": f"models:/{name}/{version}",
    }


def _materialize(raw_model, monkeypatch, name="gbdt_best"):
    """Run _materialize_native_model with faked bentoml/mlflow; return (model, framework, calls)."""
    import sys

    calls = {}

    def save_model(name_, model, metadata=None):
        calls["name"], calls["model"], calls["metadata"] = name_, model, metadata
        return SimpleNamespace(tag=f"{name_}:materialized")

    fake_bentoml = SimpleNamespace(
        models=SimpleNamespace(get=lambda _n: (_ for _ in ()).throw(Exception())),
        sklearn=SimpleNamespace(save_model=save_model),
        xgboost=SimpleNamespace(save_model=save_model),
        lightgbm=SimpleNamespace(save_model=save_model),
    )
    fake_mlflow = SimpleNamespace(
        pyfunc=SimpleNamespace(
            load_model=lambda _uri: SimpleNamespace(get_raw_model=lambda: raw_model)
        )
    )
    monkeypatch.setitem(sys.modules, "bentoml", fake_bentoml)
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    model, framework = _deploy()._materialize_native_model(_pin(name))
    return model, framework, calls


def test_materialize_native_model_detects_xgboost(monkeypatch):
    model, framework, calls = _materialize(xgboost.XGBClassifier(), monkeypatch)
    assert framework == "xgboost"
    assert calls["name"] == "gbdt_best"
    assert calls["metadata"]["framework"] == "xgboost"
    assert calls["metadata"]["mlflow_uri"] == "models:/gbdt_best/2"
    assert calls["metadata"]["mlflow_version"] == "2"
    assert str(model.tag) == "gbdt_best:materialized"


def test_materialize_native_model_detects_lightgbm(monkeypatch):
    _model, framework, calls = _materialize(lightgbm.LGBMClassifier(), monkeypatch)
    assert framework == "lightgbm"
    assert calls["name"] == "gbdt_best"
    assert calls["metadata"]["framework"] == "lightgbm"


def test_materialize_native_model_sklearn_path(monkeypatch):
    _model, framework, calls = _materialize(LogisticRegression(), monkeypatch, name="linear_best")
    assert framework is None
    assert calls["name"] == "linear_best"
    assert "framework" not in calls["metadata"]  # only GBDT records the framework
    assert calls["metadata"]["mlflow_uri"] == "models:/linear_best/2"


def test_materialize_native_model_rejects_non_estimator(monkeypatch):
    import pytest

    with pytest.raises(RuntimeError, match="not an sklearn estimator"):
        _materialize("not-a-model", monkeypatch, name="linear_best")


# --- model_info.json records the GBDT framework, preserving existing fields ---


def _lineage_tags():
    return {
        "base_linear_registered_model_name": "linear_best",
        "base_linear_version": "3",
        "base_linear_run_id": "run-linear",
        "base_linear_model_uri": "runs:/run-linear/linear_model",
        "base_gbdt_registered_model_name": "gbdt_best",
        "base_gbdt_version": "2",
        "base_gbdt_run_id": "run-gbdt",
        "base_gbdt_model_uri": "runs:/run-gbdt/gbdt_model",
        "base_nn_registered_model_name": "nn_best",
        "base_nn_version": "1",
        "base_nn_run_id": "run-nn",
        "base_nn_model_uri": "runs:/run-nn/nn_model",
        "aux_embeddings_uri": "runs:/run-aux/bio_embeddings.npz",
        "aux_embeddings_hash": "bbb",
        "aux_bio_feature_cols_uri": "runs:/run-aux/bio_feature_cols.json",
        "aux_bio_feature_cols_hash": "ccc",
    }


class _FakeModelVersion:
    def __init__(self, tags, version="7", run_id="run-prod"):
        self.tags = dict(tags)
        self.version = version
        self.run_id = run_id
        self.creation_timestamp = 123


class _FakeMlflowClient:
    def __init__(self, model_version):
        self._mv = model_version

    def get_model_version(self, name, version):
        assert name == "ensemble_lr_model"
        assert str(version) == str(self._mv.version)
        return self._mv


def test_lineage_pins_and_manifest_record_gbdt_framework(monkeypatch, tmp_path):
    d = _deploy()
    monkeypatch.setattr(d, "_gbdt_framework", lambda _uri: "lightgbm")
    manifest_file = tmp_path / "model_info.json"
    monkeypatch.setattr(d, "MODEL_INFO_FILE", manifest_file)

    client = _FakeMlflowClient(_FakeModelVersion(_lineage_tags()))
    production = SimpleNamespace(version="7", run_id="run-prod")
    pins = d._lineage_pins(client, production)
    assert pins["gbdt"]["framework"] == "lightgbm"
    assert pins["gbdt"]["model_uri"] == "runs:/run-gbdt/gbdt_model"

    d._write_model_info(client, production, pins, "fp")
    manifest = json.loads(manifest_file.read_text())
    # New field: the GBDT framework recorded for the serving adapter.
    assert manifest["bases"]["gbdt"]["framework"] == "lightgbm"
    # Existing lineage contract preserved.
    assert manifest["bases"]["gbdt"]["registered_model_name"] == "gbdt_best"
    assert manifest["bases"]["gbdt"]["version"] == "2"
    assert manifest["bases"]["linear"]["registered_model_name"] == "linear_best"
    assert manifest["aux_artifacts"]["embeddings_uri"] == "runs:/run-aux/bio_embeddings.npz"
    assert manifest["champion"]["version"] == "7"
    assert manifest["build_input_fingerprint"] == "fp"


# --- Deploy flow has no bentoml.mlflow import ---


def test_deploy_flow_does_not_import_bentoml_mlflow():
    src = (_deploy().ROOT / "src" / "flows" / "deploy.py").read_text()
    assert "import bentoml.mlflow" not in src


def test_serving_closure_has_no_mlflow_reference():
    service_src = (_deploy().ROOT / "src" / "serving" / "service.py").read_text()
    assert "bentoml.mlflow" not in service_src
    assert "import mlflow" not in service_src
