#!/usr/bin/env python3
"""
Standalone pipeline runner — runs all Papermill notebooks in sequence.

``--force-promote`` passes ``force_promote=True`` into 04_evaluate so it always
promotes the candidate (bypassing the metric gate), useful to refresh lineage
tags without re-beating production.
"""

import argparse
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import TextIO

import papermill as pm

from src.constants import (
    LOGS,
    OUTPUTS,
    PARAMS,
    PLAYSTYLE_CLUSTER_LABELS,
    PLAYSTYLE_N_CLUSTERS,
    PLAYSTYLE_RANDOM_STATE,
)
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
    )
    print(f"  Done: {name}")


def generate_cluster_artifacts() -> None:
    """Fit player-archetype clusters from the fresh DuckDB snapshot and write
    the deploy-time similarity build consumes (one-hot memberships + archetype
    labels). Runs after the snapshot refresh so assignments stay current.
    """
    from src.models.similarity import build_cluster_artifacts

    build_cluster_artifacts(
        n_clusters=PLAYSTYLE_N_CLUSTERS,
        labels=PLAYSTYLE_CLUSTER_LABELS,
        query=training.to_dataframe,
        random_state=PLAYSTYLE_RANDOM_STATE,
    )


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
    args = parser.parse_args()

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

        # Fit player-archetype clusters from the fresh snapshot. Deploy later
        # consumes these snapshot artifacts to build navigation assets.
        print("\nGenerating playstyle clusters...")
        generate_cluster_artifacts()
        print("  Cluster artifacts written.")

        print("Pipeline starting...")
        for name in NB_ORDER:
            parameters = (
                {"force_promote": args.force_promote} if name == "04_evaluate.ipynb" else None
            )
            # Tagged parameter cells own notebook defaults.
            run_notebook(name, parameters=parameters)

        print(f"\n{'=' * 60}")
        print(" Pipeline complete.")
        print(f" Check outputs: {OUTPUTS}")
        print(f"{'=' * 60}")
