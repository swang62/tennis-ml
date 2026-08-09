#!/bin/sh
# Fail fast when the runtime API key is missing: the rendered Nginx config
# would otherwise compare X-Drift-API-Key against an empty string and let
# unauthenticated requests through. compose.yaml also requires the key via
# ${DRIFT_API_KEY:?...}; this guard covers direct `docker run` too.
set -eu

if [ -z "${DRIFT_API_KEY:-}" ]; then
    echo "error: DRIFT_API_KEY must be set and non-empty (see README)" >&2
    exit 1
fi

exec /docker-entrypoint.sh "$@"
