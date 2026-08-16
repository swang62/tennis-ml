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
    DATA_PROCESSED,
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


def _file_hash(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_cluster_artifacts() -> None:
    """Fit player-archetype clusters from the fresh DuckDB snapshot and write
    the artifacts PlayerSimilarity.build consumes (one-hot memberships +
    archetype labels). Runs after the snapshot refresh so a stale assignment
    can never leak into the similarity index build.
    """
    from src.models.similarity import build_cluster_artifacts

    build_cluster_artifacts(
        n_clusters=PLAYSTYLE_N_CLUSTERS,
        labels=PLAYSTYLE_CLUSTER_LABELS,
        query=training.to_dataframe,
        random_state=PLAYSTYLE_RANDOM_STATE,
    )


def build_similarity_index() -> None:
    """Build snapshot-backed player similarity artifacts and pin them in one MLflow run.

    Only the similarity index and its metadata are pinned here; the web player
    directory is generated at deploy time from the same DuckDB snapshot (see
    src.flows.deploy.generate_directory_artifact), never champion-pinned.
    """
    import json

    import mlflow

    from src.models.similarity import DEFAULT_INDEX, DEFAULT_METADATA, PlayerSimilarity

    PlayerSimilarity().build(query=training.to_dataframe)
    mlflow.set_experiment("artifacts")
    with mlflow.start_run():
        mlflow.log_artifact(str(DEFAULT_INDEX))
        mlflow.log_artifact(str(DEFAULT_METADATA))

        run_id = mlflow.active_run().info.run_id  # type: ignore[union-attr]  # non-None inside start_run()
        pins = {
            "similarity_index_uri": f"runs:/{run_id}/player_similarity.index",
            "similarity_index_hash": _file_hash(DEFAULT_INDEX),
            "similarity_metadata_uri": f"runs:/{run_id}/player_metadata.json",
            "similarity_metadata_hash": _file_hash(DEFAULT_METADATA),
        }
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        (DATA_PROCESSED / "similarity_pins.json").write_text(json.dumps(pins, indent=2) + "\n")


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

        # Fit player-archetype clusters from the fresh snapshot before the
        # similarity index build so the index consumes freshly generated
        # cluster artifacts (memberships + labels).
        print("\nGenerating playstyle clusters...")
        generate_cluster_artifacts()
        print("  Cluster artifacts written.")

        # Build the player similarity index from the DuckDB snapshot so it is
        # always fresh and never depends on a running PostgreSQL.
        print("\nBuilding player similarity FAISS index...")
        build_similarity_index()
        print("  Similarity index built.")

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
