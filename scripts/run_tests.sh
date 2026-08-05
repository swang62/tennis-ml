#!/usr/bin/env bash
# Run pytest --testmon and normalize exit code 5 (no tests collected/selected)
# to success, so pre-commit passes when testmon selects nothing. Other exit
# codes (e.g. failures) are preserved.
uv run pytest --testmon
status=$?
if [ "$status" -eq 5 ]; then
  exit 0
fi
exit "$status"