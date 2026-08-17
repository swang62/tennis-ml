from mlflow.tracking.client import MlflowClient

client = MlflowClient()


def full_reset_mlflow_registry() -> None:
    models = client.search_registered_models()
    print(f"MLflow registry target: {len(models)} registered model(s)")
    try:
        answer = input("Delete all registered models? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    if answer not in {"y", "yes"}:
        print("MLflow reset cancelled.")
        return

    for model in models:
        client.delete_registered_model(name=model.name)
        print(f"Deleted registered model: {model.name}")

    print("MLflow registry fully wiped.")


if __name__ == "__main__":
    full_reset_mlflow_registry()
