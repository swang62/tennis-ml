from typing import Any

from mlflow.entities import ViewType
from mlflow.tracking.client import MlflowClient

from src.constants import load_env


def full_reset_mlflow_registry(client: Any | None = None) -> None:
    load_env()
    if client is None:
        client = MlflowClient()
    models = client.search_registered_models()
    experiments = [
        experiment
        for experiment in client.search_experiments(view_type=ViewType.ACTIVE_ONLY)
        if experiment.experiment_id != "0"
    ]
    print(
        f"MLflow reset target: {len(models)} registered model(s), "
        f"{len(experiments)} non-default experiment(s)"
    )
    try:
        answer = input("Delete all listed MLflow resources? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    if answer not in {"y", "yes"}:
        print("MLflow reset cancelled.")
        return

    for model in models:
        client.delete_registered_model(name=model.name)
        print(f"Deleted registered model: {model.name}")

    for experiment in experiments:
        client.delete_experiment(experiment.experiment_id)
        print(f"Deleted experiment: {experiment.name} ({experiment.experiment_id})")

    print("MLflow reset complete (the default experiment is retained by MLflow).")


if __name__ == "__main__":
    full_reset_mlflow_registry()
