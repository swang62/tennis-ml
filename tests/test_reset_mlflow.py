import importlib.util
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location(
    "reset_mlflow", Path(__file__).parents[1] / "scripts" / "reset_mlflow.py"
)
assert spec and spec.loader
reset_mlflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reset_mlflow)


class FakeMlflowClient:
    def __init__(self):
        self.deleted_models: list[str] = []
        self.deleted_experiments: list[str] = []

    def search_registered_models(self):
        return [SimpleNamespace(name="champion")]

    def search_experiments(self, view_type):
        assert view_type is not None
        return [
            SimpleNamespace(experiment_id="0", name="Default"),
            SimpleNamespace(experiment_id="1", name="training"),
        ]

    def delete_registered_model(self, name):
        self.deleted_models.append(name)

    def delete_experiment(self, experiment_id):
        self.deleted_experiments.append(experiment_id)


def test_reset_deletes_models_and_nondefault_experiments(monkeypatch):
    client = FakeMlflowClient()
    monkeypatch.setattr("builtins.input", lambda _: "yes")

    reset_mlflow.full_reset_mlflow_registry(client)

    assert client.deleted_models == ["champion"]
    assert client.deleted_experiments == ["1"]
