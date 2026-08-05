#!/usr/bin/env python3
"""
Standalone pipeline runner — runs all Papermill notebooks in sequence.
"""

import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from typing import TextIO

import papermill as pm

from src.constants import LOGS, OUTPUTS, PARAMS
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


class _Tee:
    """Write to two streams: the real console and a capture file."""

    def __init__(self, console: TextIO, log: TextIO) -> None:
        self.console = console
        self.log = log

    def write(self, s: str) -> int:
        self.log.write(s)
        return self.console.write(s)

    def flush(self) -> None:
        self.console.flush()
        self.log.flush()


if __name__ == "__main__":
    LOGS.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS / f"pipeline_{timestamp}.log"
    # Tee the whole run (per-notebook prints + Papermill kernel progress) to a
    # log file under artifacts/logs while keeping console output streaming.
    with (
        log_path.open("w") as log,
        redirect_stdout(_Tee(sys.stdout, log)),
        redirect_stderr(_Tee(sys.stderr, log)),
    ):
        print(f"Pipeline starting — {len(NB_ORDER)} notebooks")
        for name in NB_ORDER:
            # Notebooks own their parameter defaults via their tagged parameter
            # cells; nothing is injected.
            run_notebook(name)

        print(f"\n{'=' * 60}")
        print(" Pipeline complete.")
        print(f" Check outputs: {OUTPUTS}")
        print(f"{'=' * 60}")
