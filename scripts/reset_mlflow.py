from datetime import UTC, datetime
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
        for experiment in client.search_experiments(view_type=ViewType.ALL)
        if experiment.experiment_id != "0"
        and not (experiment.lifecycle_stage == "deleted" and "__reset_" in experiment.name)
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

    for experiment in experiments:
        if experiment.lifecycle_stage == "deleted":
            client.restore_experiment(experiment.experiment_id)
        archived_name = f"{experiment.name}__reset_{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{experiment.experiment_id}"
        client.rename_experiment(experiment.experiment_id, archived_name)
        client.delete_experiment(experiment.experiment_id)
        print(f"Archived experiment: {experiment.name} ({experiment.experiment_id})")

    for model in models:
        client.delete_registered_model(name=model.name)
        print(f"Deleted registered model: {model.name}")

    print(
        "MLflow reset complete (the default experiment is retained; DagsHub retains archived "
        "experiments but their original names are available)."
    )


if __name__ == "__main__":
    full_reset_mlflow_registry()
