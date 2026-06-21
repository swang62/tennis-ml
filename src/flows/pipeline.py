#!/usr/bin/env python3
"""
Standalone pipeline runner — runs all Papermill notebooks in sequence.
Override any parameter via --params JSON or env var PIPELINE_PARAMS.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import papermill as pm

ROOT = Path(__file__).resolve().parent.parent.parent
NOTEBOOKS = ROOT / "notebooks"
PARAMS = NOTEBOOKS / "parameters"
OUTPUTS = NOTEBOOKS / "outputs"

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


def run_notebook(name: str, params: dict | None = None) -> None:
    src = PARAMS / name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = OUTPUTS / f"{timestamp}_{name}"

    print(f"\n{'=' * 60}")
    print(f"Running: {name}")
    print(f"Output:  {dst.name}")
    print(f"{'=' * 60}")

    pm.execute_notebook(
        input_path=str(src),
        output_path=str(dst),
        parameters=params or {},
    )
    print(f"  Done: {name}")


def run_pipeline(overrides: dict[str, dict] | None = None) -> None:
    overrides = overrides or {}
    for name in NB_ORDER:
        defaults = NOTEBOOK_PARAMS[name]
        params = {**defaults, **(overrides.get(name, {}))}
        run_notebook(name, params)


def parse_overrides(raw: str) -> dict[str, dict]:
    """Parse JSON keyed by notebook name → dict of param overrides."""
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise TypeError("--params must be a JSON object keyed by notebook name")
    return {k: v for k, v in parsed.items() if isinstance(v, dict)}


if __name__ == "__main__":
    overrides: dict[str, dict] = {}

    if "--params" in sys.argv:
        idx = sys.argv.index("--params")
        raw = sys.argv[idx + 1]
        overrides = parse_overrides(raw)

    env_raw = os.getenv("PIPELINE_PARAMS")
    if env_raw:
        overrides = {**overrides, **parse_overrides(env_raw)}

    print(f"Pipeline starting — {len(NB_ORDER)} notebooks")
    run_pipeline(overrides)

    print(f"\n{'=' * 60}")
    print(" Pipeline complete.")
    print(f" Check outputs: {OUTPUTS}")
    print(f"{'=' * 60}")
