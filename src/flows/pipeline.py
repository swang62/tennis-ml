#!/usr/bin/env python3
"""Run the training and evaluation notebooks in sequence."""

import argparse
import logging
import shutil
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from typing import TextIO

import papermill as pm

from src.config import ensure_kernel, suppress_insecure_tls_warning
from src.constants import (
    LOGS,
    OUTPUTS,
    PARAMS,
    load_env,
)

# Training and evaluation notebooks (01-04), run in order. The former 00
# embeddings step was removed: the pipeline is tabular-only and pins no
# auxiliary (bio) artifacts. Promotion assigns @champion directly in 04.
NB_ORDER = [
    "01_train_test_split.ipynb",
    "02_tune_gbdt.ipynb",
    "02_tune_linear.ipynb",
    "02_tune_nn.ipynb",
    "03_train_ensemble.ipynb",
    "04_evaluate.ipynb",
]

# Kernels inherit this environment before their own load_env() cell runs.
load_env()
suppress_insecure_tls_warning()


def run_notebook(name: str, parameters: dict | None = None) -> None:
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
        parameters=parameters,
        log_output=True,  # stream cell stdout/stderr live (progress bar is default)
    )
    latest = OUTPUTS / f"latest_{name}"
    shutil.copyfile(dst, latest)
    print(f"  Done: {name} (latest: {latest.name})")


def selected_notebooks(promote_only: bool) -> list[str]:
    """Return the evaluation notebook alone for a promotion-only run."""
    return ["04_evaluate.ipynb"] if promote_only else NB_ORDER


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-promote",
        action="store_true",
        help="always promote the candidate, bypassing the metric gate",
    )
    parser.add_argument(
        "--promote-only",
        action="store_true",
        help="run only 04_evaluate against the existing candidate artifacts",
    )
    args, ignored = parser.parse_known_args()
    if ignored:
        print(f"Ignoring unsupported pipeline arguments: {' '.join(ignored)}")

    LOGS.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS / f"pipeline_{timestamp}.log"
    # Keep notebook output streaming while capturing it in the run log.
    with (
        log_path.open("w") as log,
        redirect_stdout(_Tee(sys.stdout, log)),
        redirect_stderr(_Tee(sys.stderr, log)),
    ):
        # Route Papermill output through the redirected stderr.
        logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")

        print(
            "Promotion-only pipeline starting..." if args.promote_only else "Pipeline starting..."
        )
        for name in selected_notebooks(args.promote_only):
            parameters = (
                {"force_promote": args.force_promote} if name == "04_evaluate.ipynb" else None
            )
            # Tagged parameter cells own notebook defaults.
            run_notebook(name, parameters=parameters)

        print(f"\n{'=' * 60}")
        print(" Pipeline complete.")
        print(f" Check outputs: {OUTPUTS}")
        print(f"{'=' * 60}")
