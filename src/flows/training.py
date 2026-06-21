"""Prefect flow: orchestrate the training pipeline via Papermill."""

from datetime import datetime
from pathlib import Path

import papermill as pm
from prefect import flow, task

NOTEBOOKS = Path(__file__).resolve().parent.parent.parent / "notebooks"
OUTPUTS = NOTEBOOKS / "outputs"
PARAMS = NOTEBOOKS / "parameters"

NOTEBOOK_PARAMS: dict[str, dict] = {
    "01_feature_engineering.ipynb": {
        "gold_table": "gold.match_features",
        "profiles_table": "gold.player_profiles",
        "test_size": 0.2,
        "random_state": 42,
        "output_dir": "data/processed",
    },
    "02_tune_gbdt.ipynb": {
        "input_dir": "data/processed",
        "n_trials": 100,
        "random_state": 42,
        "cv_folds": 5,
    },
    "02_tune_linear.ipynb": {
        "input_dir": "data/processed",
        "n_trials": 50,
        "random_state": 42,
        "cv_folds": 5,
    },
    "02_tune_nn.ipynb": {
        "input_dir": "data/processed",
        "n_trials": 50,
        "max_epochs": 100,
        "patience": 10,
        "batch_size": 128,
        "seq_len": 10,
    },
    "03_pick_best.ipynb": {
        "input_dir": "data/processed",
        "output_dir": "data/processed",
        "cv_folds": 5,
        "random_state": 42,
        "experiments": {"linear": "linear_models", "gbdt": "gbdt_models", "nn": "nn_models"},
    },
    "04_stack_ensemble.ipynb": {
        "input_dir": "data/processed",
        "random_state": 42,
        "model_names": ["linear", "gbdt", "nn"],
    },
    "05_evaluate.ipynb": {
        "input_dir": "data/processed",
        "random_state": 42,
        "production_model_name": "production_model",
    },
}

NB_ORDER = [
    "01_feature_engineering.ipynb",
    "02_tune_gbdt.ipynb",
    "02_tune_linear.ipynb",
    "02_tune_nn.ipynb",
    "03_pick_best.ipynb",
    "04_stack_ensemble.ipynb",
    "05_evaluate.ipynb",
]


@task
def run_notebook(name: str, params: dict | None = None) -> None:
    src = PARAMS / name
    dst = OUTPUTS / f"{datetime.now():%Y%m%d_%H%M%S}_{name}"
    pm.execute_notebook(
        input_path=str(src),
        output_path=str(dst),
        parameters=params or {},
    )


@flow(log_prints=True)
def training_flow(
    rebuild: bool = True,
    notebook_overrides: dict[str, dict] | None = None,
) -> None:
    """Run the full training pipeline via Papermill.

    Args:
        rebuild: If True, trigger BentoML image rebuild on successful promotion.
        notebook_overrides: Per-notebook param overrides keyed by notebook name.
            E.g. {"02_tune_nn.ipynb": {"n_trials": 100, "max_epochs": 50}}.
    """
    overrides = notebook_overrides or {}
    rebuild_cmd = "just build-image" if rebuild else ""

    for name in NB_ORDER:
        defaults = NOTEBOOK_PARAMS[name]
        params = {**defaults, **(overrides.get(name, {}))}
        if name == "05_evaluate.ipynb":
            params["rebuild_cmd"] = rebuild_cmd
        run_notebook(name, params)

    print("Training pipeline complete.")


if __name__ == "__main__":
    training_flow()
