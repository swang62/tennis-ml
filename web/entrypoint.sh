#!/bin/sh
# Fail fast when the runtime API key is missing: the rendered Nginx config
# would otherwise compare X-API-Key against an empty string and let
# unauthenticated requests through. compose.yaml also requires the key via
# ${BENTO_API_KEY:?...}; this guard covers direct `docker run` too.
set -eu

if [ -z "${BENTO_API_KEY:-}" ]; then
    echo "error: BENTO_API_KEY must be set and non-empty (see README)" >&2
    exit 1
fi

exec /docker-entrypoint.sh "$@"
