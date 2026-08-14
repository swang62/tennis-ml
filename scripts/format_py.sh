#!/usr/bin/env bash
# Silently format Python files and re-stage them so pre-commit never fails on a
# reformat. Only errors when ruff cannot format a file (e.g. syntax error).
set -euo pipefail
if [ "$#" -eq 0 ]; then
  exit 0
fi
ruff format --force-exclude "$@"
git add -- "$@"
