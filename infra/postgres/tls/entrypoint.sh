#!/bin/sh
# Start PostgreSQL with TLS when a cert + key are staged successfully, otherwise
# fall back to plain (non-SSL). Bind-mounted files keep the host UID, so we copy
# them to a postgres-owned location (PostgreSQL runs as uid 999).
set -u

SSL_ARGS=""
CERT=/etc/ssl/postgres/server.crt
KEY=/etc/ssl/postgres/server.key

if [ -r /mnt/tls/server.crt ] && [ -r /mnt/tls/server.key ]; then
  if mkdir -p /etc/ssl/postgres \
      && cp /mnt/tls/server.crt /mnt/tls/server.key /etc/ssl/postgres/ \
      && chown postgres:postgres "$CERT" "$KEY" \
      && chmod 600 "$KEY"; then
    SSL_ARGS="-c ssl=on -c ssl_cert_file=$CERT -c ssl_key_file=$KEY"
      echo "TLS certs found, starting with SSL"
  else
    echo "postgres entrypoint: failed to stage TLS certs; starting without SSL" >&2
  fi
else
  echo "postgres entrypoint: TLS cert/key missing or unreadable; starting without SSL" >&2
fi

# shellcheck disable=SC2086
exec /usr/local/bin/docker-entrypoint.sh postgres $SSL_ARGS
