#!/usr/bin/env bash
uv run pytest --testmon

status=$?
if [ "$status" -eq 5 ]; then
  exit 0
fi
exit "$status"
