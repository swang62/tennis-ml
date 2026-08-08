"""Run the project's dbt build against the configured PostgreSQL database."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import TextIO

from src import constants
from src.db.conninfo import dbt_env

DBT_BUILD_CMD = ["uv", "run", "dbt", "build", "--project-dir", "dbt", "--profiles-dir", "dbt"]


def run_dbt_build(
    profiles_dir: str | Path = "dbt", log_file: Path | None = None
) -> subprocess.CompletedProcess:
    """Build dbt models, optionally streaming output to ``log_file``."""
    cmd = DBT_BUILD_CMD
    if str(profiles_dir) != "dbt":
        cmd = [*DBT_BUILD_CMD[:-2], "--profiles-dir", str(profiles_dir)]
    env = {**os.environ, **dbt_env(constants.build_database_url())}
    if log_file is None:
        return subprocess.run(cmd, cwd=constants.ROOT, check=True, env=env)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w") as log:
        return _run_streamed(cmd, log, env)


def _run_streamed(cmd: list[str], log: TextIO, env: dict[str, str]) -> subprocess.CompletedProcess:
    proc = subprocess.Popen(
        cmd, cwd=constants.ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        text = line.decode(errors="replace")
        sys.stdout.write(text)
        sys.stdout.flush()
        log.write(text)
        log.flush()
    returncode = proc.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)
    return subprocess.CompletedProcess(cmd, returncode)


if __name__ == "__main__":
    run_dbt_build()
