from mlflow.tracking.client import MlflowClient

client = MlflowClient()


def full_reset_mlflow_registry() -> None:
    for model in client.search_registered_models():
        client.delete_registered_model(name=model.name)
        print(f"Deleted registered model: {model.name}")

    print("MLflow registry fully wiped.")


if __name__ == "__main__":
    full_reset_mlflow_registry()
