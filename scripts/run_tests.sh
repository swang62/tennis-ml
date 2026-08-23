#!/usr/bin/env bash
# Treat testmon's "no tests selected" exit code as success; preserve failures.
OMP_NUM_THREADS=1 uv run pytest --testmon
status=$?
if [ "$status" -eq 5 ]; then
  exit 0
fi
exit "$status"
