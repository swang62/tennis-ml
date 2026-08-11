#!/bin/sh
# Start Bento and Vite against the Homebrew PostgreSQL target in .env.
# Preflight never prints credentials and verifies:
#   - .env exists and provides the single DATABASE_URL
#   - DRIFT_API_KEY exists in .env (generated high-entropy, never displayed)
#   - the database is reachable and is the configured/expected database
#   - the required application schemas/tables exist
#   - the target is not the Compose database (127.0.0.1:6543)
#   - ports 3000 (Bento) and 5173 (Vite) are free
#
# It never changes services, Compose, migrations, or seed data.

set -u

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT" || exit 1

# --- Load the existing .env without printing it ----------------------------
if [ ! -f .env ]; then
    echo "error: .env not found in $ROOT; configure DATABASE_URL for Homebrew PostgreSQL" >&2
    exit 1
fi

# --- Ensure DRIFT_API_KEY exists in .env (never printed) -------------------
# Compose requires a non-empty key at startup. Preserve every existing entry;
# only generate a high-entropy key when none is present (an empty placeholder
# is replaced). The value is never displayed or committed.
if ! grep -Eq '^DRIFT_API_KEY=.+' .env; then
    key=$(openssl rand -hex 32) || { echo "error: 'openssl rand' failed (is openssl installed?)" >&2; exit 1; }
    sed -i '' '/^DRIFT_API_KEY=$/d' .env
    printf '\n# Drift/operational API key for the production Nginx internal routes.\nDRIFT_API_KEY=%s\n' "$key" >> .env
    unset key
    echo "generated DRIFT_API_KEY in .env (value not displayed)"
fi

set -a
. ./.env
set +a

# --- Resolve the effective database target --------------------------------
# DATABASE_URL is used only to connect and is never echoed.
if [ -z "${DATABASE_URL:-}" ]; then
    echo "error: .env must set DATABASE_URL for the local workflow (see README)" >&2
    exit 1
fi
url=${DATABASE_URL#*://}
url=${url%%\?*}
creds=${url%%@*}
[ "$creds" = "$url" ] && creds=
[ -n "$creds" ] && url=${url##*@}
hostport=${url%%/*}
dbname=${url#*/}
case $hostport in
    *:*) DB_HOST=${hostport%%:*}; DB_PORT=${hostport##*:} ;;
    *) DB_HOST=$hostport; DB_PORT=5432 ;;
esac
DB_NAME=$dbname

# --- Database preflight ----------------------------------------------------
db_psql() {
    psql -X -w -tAq "$DATABASE_URL" -v ON_ERROR_STOP=1 -c "$1"
}

command -v psql >/dev/null 2>&1 || {
    echo "error: 'psql' not found; is Homebrew PostgreSQL installed and linked?" >&2
    exit 1
}

echo "checking database $DB_NAME at ${DB_HOST:-localhost}:$DB_PORT"
actual_db=$(db_psql "SELECT current_database()") || {
    echo "error: cannot connect to PostgreSQL at ${DB_HOST:-localhost}:$DB_PORT (database $DB_NAME)" >&2
    echo "  start Homebrew PostgreSQL and make sure .env points at it" >&2
    exit 1
}
if [ -n "$DB_NAME" ] && [ "$actual_db" != "$DB_NAME" ]; then
    echo "error: connected to database '$actual_db' but .env expects '$DB_NAME'" >&2
    exit 1
fi

# Missing schemas/tables are reported by name (never credentials).
missing=$(db_psql "
SELECT string_agg(t.t, ', ')
FROM (VALUES ('bronze.match_events'), ('silver.player_matches'),
             ('silver.rolling_features'), ('gold.player_profiles')) AS t(t)
WHERE to_regclass(t.t) IS NULL") || exit 1
if [ -n "$missing" ]; then
    echo "error: required table(s) missing in database '$DB_NAME': $missing" >&2
    echo "  run 'just db-init' then 'just db-seed' and 'just db-etl' to build them" >&2
    exit 1
fi

# --- Port preflight (fail before starting anything) ------------------------
port_pid() {
    lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null || true
}
conflicts=""
for spec in 3000:Bento 5173:Vite; do
    port=${spec%%:*}
    service=${spec#*:}
    pid=$(port_pid "$port")
    if [ -n "$pid" ]; then
        conflicts="${conflicts:+$conflicts, }port $port ($service) in use by PID $pid"
    fi
done
if [ -n "$conflicts" ]; then
    echo "error: cannot start dev servers: $conflicts" >&2
    echo "  stop the conflicting process(es) and re-run" >&2
    exit 1
fi

command -v uv >/dev/null 2>&1 || { echo "error: 'uv' not found (install it or run 'just deps')" >&2; exit 1; }
command -v pnpm >/dev/null 2>&1 || { echo "error: 'pnpm' not found (install it; see package.json)" >&2; exit 1; }

# --- Start both servers, stop both on exit/interrupt ----------------------
BENTO_PID= VITE_PID=
cleanup() {
    trap - INT TERM EXIT
    for pid in "$BENTO_PID" "$VITE_PID"; do
        [ -n "$pid" ] || continue
        pkill -P "$pid" 2>/dev/null
        kill "$pid" 2>/dev/null
    done
    sleep 1
    for pid in "$BENTO_PID" "$VITE_PID"; do
        [ -n "$pid" ] || continue
        kill -9 "$pid" 2>/dev/null
    done
    echo "dev servers stopped" >&2
}
STOPPED_BY_SIGNAL=0
trap 'STOPPED_BY_SIGNAL=1; cleanup' INT TERM
trap cleanup EXIT

echo "preflight ok: PostgreSQL at ${DB_HOST:-localhost}:$DB_PORT (database $DB_NAME), tables present, ports free"

# --- Import MLflow models into BentoML local store (idempotent) ----------------
echo "importing MLflow models into BentoML local store..."
uv run python -c '
from pathlib import Path
from src.constants import DATA_PROCESSED, load_env
from src.flows.deploy import (
    _lineage_pins, _import_or_reuse, _materialize_nn_onnx, _latest_production_version,
)
from mlflow.tracking.client import MlflowClient

load_env()

client = MlflowClient()
production = _latest_production_version(client)
if production is None:
    raise SystemExit("no champion found")
pins = _lineage_pins(client, production)

for key in ("production", "linear", "gbdt"):
    _import_or_reuse(pins[key])

nn_onnx = DATA_PROCESSED / "nn_best.onnx"
if nn_onnx.exists():
    print(f"[nn_best] ONNX already exists: {nn_onnx}")
else:
    _materialize_nn_onnx(pins["nn"])

print("dev import complete")
' || {
    echo "error: model import failed — are models registered in MLflow? Run 'just train' first." >&2
    exit 1
}

echo "starting Bento on http://127.0.0.1:3000 and Vite on http://127.0.0.1:5173; Ctrl-C stops both"
(
    cd "$ROOT" && exec uv run bentoml serve src/serving/service.py:TennisPredictor --host 127.0.0.1 --port 3000
) &
BENTO_PID=$!
(
    cd "$ROOT/web" && exec pnpm dev
) &
VITE_PID=$!

# Stop both servers if either exits; detect zombies to avoid hanging.
alive() {
    kill -0 "$1" 2>/dev/null && [ "$(ps -p "$1" -o stat= 2>/dev/null)" != "Z" ]
}
while alive "$BENTO_PID" && alive "$VITE_PID"; do
    sleep 1
done
cleanup
if [ "$STOPPED_BY_SIGNAL" -eq 1 ]; then
    exit 130
fi
echo "error: a dev server exited; see its output above" >&2
exit 1
