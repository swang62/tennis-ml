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
from src.db import training
from src.db.snapshot import SNAPSHOT_PATH, refresh_snapshot
from src.utils import ensure_kernel, load_env, suppress_insecure_tls_warning

# Training notebooks (00-05), run in order.
NB_ORDER = [
    "00_embeddings.ipynb",
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


def build_similarity_index() -> None:
    """Rebuild the player similarity index from the fresh DuckDB snapshot.

    Always rebuilds from scratch; training never reuses a previously saved
    index, so a stale index can never leak into a training run.
    """
    from src.models.similarity import PlayerSimilarity

    PlayerSimilarity().build(query=training.to_dataframe)


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
    # Keep notebook output streaming while capturing it in the run log.
    with (
        log_path.open("w") as log,
        redirect_stdout(_Tee(sys.stdout, log)),
        redirect_stderr(_Tee(sys.stderr, log)),
    ):
        # Refresh first so training cannot read a stale snapshot.
        print("Refreshing training snapshot from PostgreSQL...")
        refresh_snapshot()
        print(f"  Snapshot refreshed: {SNAPSHOT_PATH}")

        # Build the player similarity index from the DuckDB snapshot so it is
        # always fresh and never depends on a running PostgreSQL.
        print("\nBuilding player similarity index (snapshot)...")
        build_similarity_index()
        print("  Similarity index built.")

        print(f"Pipeline starting — {len(NB_ORDER)} notebooks")
        for name in NB_ORDER:
            # Tagged parameter cells own notebook defaults.
            run_notebook(name)

        print(f"\n{'=' * 60}")
        print(" Pipeline complete.")
        print(f" Check outputs: {OUTPUTS}")
        print(f"{'=' * 60}")
