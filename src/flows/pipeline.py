#!/usr/bin/env python3
"""
Standalone pipeline runner — runs all Papermill notebooks in sequence.
"""

from datetime import datetime

import papermill as pm

from src.constants import OUTPUTS, PARAMS
from src.utils import ensure_kernel, load_env

# Training notebooks (00-05), run in order.
NB_ORDER = [
    "00_embeddings.ipynb",
    "01_train_test_split.ipynb",
    "02_tune_gbdt.ipynb",
    "02_tune_linear.ipynb",
    "02_tune_nn.ipynb",
    "03_ensemble_split.ipynb",
    "04_ensemble_stack.ipynb",
    "05_evaluate.ipynb",
]

# Load the env file in this parent process so papermill kernels (which inherit
# os.environ) see it before their own load_env() cell runs.
load_env()


def run_notebook(name: str) -> None:
    src = PARAMS / name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = OUTPUTS / f"{timestamp}_{name}"

    OUTPUTS.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Running: {name}")
    print(f"Output:  {dst.name}")
    print(f"{'=' * 60}")

    pm.execute_notebook(
        input_path=str(src),
        output_path=str(dst),
        kernel_name=ensure_kernel(),
    )
    print(f"  Done: {name}")


if __name__ == "__main__":
    print(f"Pipeline starting — {len(NB_ORDER)} notebooks")
    for name in NB_ORDER:
        # Notebooks own their parameter defaults via their tagged parameter
        # cells; nothing is injected.
        run_notebook(name)

    print(f"\n{'=' * 60}")
    print(" Pipeline complete.")
    print(f" Check outputs: {OUTPUTS}")
    print(f"{'=' * 60}")
