#!/bin/sh
# Start Bento (with backend-only auto-reload) and Vite against the Homebrew
# PostgreSQL target in .env.
# Preflight never prints credentials and verifies:
#   - .env exists and provides the single DATABASE_URL
#   - BENTO_API_KEY exists in .env (generated high-entropy, never displayed)
#   - the database is reachable and is the configured/expected database
#   - the required application schemas/tables exist
#   - the target is not the Compose database (127.0.0.1:6543)
#   - ports 3000 (Bento) and 5173 (Vite) are free
#
# It never changes services, Compose, migrations, or seed data.

set -u

if [ -t 1 ]; then
    COLOR_DB=$(printf '\033[34m')
    COLOR_MINISEARCH=$(printf '\033[35m')
    COLOR_BENTO=$(printf '\033[32m')
    COLOR_DEV=$(printf '\033[90m')
    COLOR_RESET=$(printf '\033[0m')
    export COURTSIDE_COLOR=1
else
    COLOR_DB=''
    COLOR_MINISEARCH=''
    COLOR_BENTO=''
    COLOR_DEV=''
    COLOR_RESET=''
    unset COURTSIDE_COLOR
fi

log() {
    category=$1
    message=$2
    case "$category" in
        db) color=$COLOR_DB ;;
        minisearch) color=$COLOR_MINISEARCH ;;
        bento) color=$COLOR_BENTO ;;
        *) color=$COLOR_DEV ;;
    esac
    printf '%s[%s]%s %s\n' "$color" "$category" "$COLOR_RESET" "$message"
}

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT" || exit 1

# --- Load the existing .env without printing it ----------------------------
if [ ! -f .env ]; then
    echo "[dev] error: .env not found in $ROOT; configure DATABASE_URL for Homebrew PostgreSQL" >&2
    exit 1
fi

# --- Ensure BENTO_API_KEY exists in .env (never printed) -------------------
# Compose requires a non-empty key at startup. Preserve every existing entry;
# only generate a high-entropy key when none is present (an empty placeholder
# is replaced). The value is never displayed or committed.
if ! grep -Eq '^BENTO_API_KEY=.+' .env; then
    key=$(openssl rand -hex 32) || { echo "[dev] error: 'openssl rand' failed (is openssl installed?)" >&2; exit 1; }
    sed -i '' '/^BENTO_API_KEY=$/d' .env
    printf '\n# Operational API key for the production Nginx internal routes.\nBENTO_API_KEY=%s\n' "$key" >> .env
    unset key
    echo "[dev] generated BENTO_API_KEY in .env (value not displayed)"
fi

set -a
. ./.env
set +a

# --- Resolve the effective database target --------------------------------
# DATABASE_URL is used only to connect and is never echoed.
if [ -z "${DATABASE_URL:-}" ]; then
    echo "[dev] error: .env must set DATABASE_URL for the local workflow (see README)" >&2
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

log db "checking $DB_NAME at ${DB_HOST:-localhost}:$DB_PORT"
actual_db=$(db_psql "SELECT current_database()") || {
    echo "[db] error: cannot connect to PostgreSQL at ${DB_HOST:-localhost}:$DB_PORT (database $DB_NAME)" >&2
    echo "[db] start Homebrew PostgreSQL and make sure .env points at it" >&2
    exit 1
}
if [ -n "$DB_NAME" ] && [ "$actual_db" != "$DB_NAME" ]; then
    echo "[db] error: connected to '$actual_db' but .env expects '$DB_NAME'" >&2
    exit 1
fi

# Missing schemas/tables are reported by name (never credentials).
missing=$(db_psql "
SELECT string_agg(t.t, ', ')
FROM (VALUES ('bronze.match_events'), ('silver.player_matches'),
             ('silver.rolling_features'), ('gold.player_profiles')) AS t(t)
WHERE to_regclass(t.t) IS NULL") || exit 1
if [ -n "$missing" ]; then
    echo "[db] error: required table(s) missing in '$DB_NAME': $missing" >&2
    echo "[db] run 'just migrate' then 'just seed' and 'just etl' to build them" >&2
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
    echo "[dev] error: cannot start dev servers: $conflicts" >&2
    echo "[dev] stop the conflicting process(es) and re-run" >&2
    exit 1
fi

command -v uv >/dev/null 2>&1 || { echo "[dev] error: 'uv' not found (install it or run 'just deps')" >&2; exit 1; }
command -v pnpm >/dev/null 2>&1 || { echo "[dev] error: 'pnpm' not found (install it; see package.json)" >&2; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "[dev] error: 'curl' not found" >&2; exit 1; }

# --- Rebuild the static player index from the local training snapshot -------
# Vite must never serve a stale or fixture directory (e.g. the old Player
# A/B/C test data): regenerate web/public/player-directory.json from the
# DuckDB training snapshot, then serialize it with the web index builder into
# the content-hashed payload + manifest. Both steps fail fast, so Vite starts
# only after the index reflects the snapshot players (run `just snapshot`
# after changing DATABASE_URL).
log minisearch "rebuilding static player index from training snapshot..."
uv run python -c '
from src.flows.deploy import generate_directory_artifact
generate_directory_artifact()
' || {
    echo "[minisearch] error: player-directory generation failed (missing training snapshot or artifact write)" >&2
    exit 1
}
node web/scripts/build-player-index.mjs || {
    echo "[minisearch] error: player index build failed (web dependencies missing? run pnpm install)" >&2
    exit 1
}

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
    # Preflight guaranteed 3000/5173 were free, so any listener still on them
    # is from this session (e.g. a worker the reloader respawned between the
    # pkill above and the arbiter's death). Sweep them so the next run starts
    # clean; the next preflight would otherwise report a stale port.
    for port in 3000 5173; do
        pid=$(port_pid "$port")
        [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null
    done
        echo "[dev] dev servers stopped" >&2
}
STOPPED_BY_SIGNAL=0
trap 'STOPPED_BY_SIGNAL=1; cleanup' INT TERM
trap cleanup EXIT

log db "PostgreSQL at ${DB_HOST:-localhost}:$DB_PORT (database $DB_NAME), tables present, ports free"

# --- Import MLflow models into BentoML local store (idempotent) ----------------
log bento "resolving @champion and importing cached models..."
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
    framework = pins[key].get("framework") if key == "gbdt" else None
    _import_or_reuse(pins[key], framework)

nn_onnx = DATA_PROCESSED / "nn_best.onnx"
if nn_onnx.exists():
    print(f"[bento] reusing nn_best ONNX: {nn_onnx}")
else:
    _materialize_nn_onnx(pins["nn"])
' || {
    echo "[bento] error: model import failed — are models registered in MLflow? Run 'just train' first." >&2
    exit 1
}

uv run python src/db/migrate_db.py migrate
log bento "starting Bento on http://127.0.0.1:3000 and Vite on http://127.0.0.1:5173; Ctrl-C stops both"
# --reload restarts only for files matched by bentofile.yaml include and
# .bentoignore (src/**, infra/postgres/schema.sql, data/processed/*); web/,
# notebooks/, tests/, and mlruns/ are ignored, so Vite HMR edits never restart
# Bento. The reloader is a circus plugin thread inside this process, so the
# process tree and cleanup are unchanged.
(
    cd "$ROOT" && exec uv run bentoml serve src/serving/service.py:TennisPredictor \
        --host 127.0.0.1 --port 3000 --reload --working-dir "$ROOT"
) &
BENTO_PID=$!

# Bento loads several models before it can answer requests. Wait for its
# readiness endpoint before starting Vite so the first browser request cannot
# race the backend startup.
BENTO_READY=0
attempts=0
while [ "$attempts" -lt 120 ]; do
    if ! kill -0 "$BENTO_PID" 2>/dev/null; then
        break
    fi
    if curl -fsS --max-time 1 http://127.0.0.1:3000/health >/dev/null 2>&1; then
        BENTO_READY=1
        break
    fi
    attempts=$((attempts + 1))
    sleep 1
done
if [ "$BENTO_READY" -ne 1 ]; then
    echo "[bento] error: did not become healthy within 120 seconds" >&2
    cleanup
    exit 1
fi

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
echo "[dev] error: a dev server exited; see its output above" >&2
exit 1
