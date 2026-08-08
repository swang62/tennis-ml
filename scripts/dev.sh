#!/bin/sh
# One-command local development against the Homebrew PostgreSQL instance
# configured in .env: starts Bento (127.0.0.1:3000, --reload) and the Vite
# dashboard (HMR) concurrently, and stops both on exit/interrupt.
#
# Preflight (fails before anything starts, never prints credentials):
#   - .env exists and provides the single DATABASE_URL
#   - the database is reachable and is the configured/expected database
#   - the required application schemas/tables exist
#   - the target is not the Compose database (127.0.0.1:6543)
#   - ports 3000 (Bento) and 5173 (Vite) are free
#
# Homebrew services, Compose, migrations, and seeding are never touched.

set -u

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT" || exit 1

# --- Load the existing .env without printing it ----------------------------
if [ ! -f .env ]; then
    echo "error: .env not found in $ROOT; configure DATABASE_URL for Homebrew PostgreSQL" >&2
    exit 1
fi
set -a
. ./.env
set +a

# --- Resolve the effective database target --------------------------------
# The single connection contract is DATABASE_URL (see README); the local
# Homebrew URL carries no password, and values are only used to connect, never
# echoed.
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

# --- Reject the Compose database target -----------------------------------
# The Compose stack publishes PostgreSQL on host port 6543; the local workflow
# targets Homebrew PostgreSQL. Report only the keys, never the values.
case "$DB_HOST:$DB_PORT" in
    127.0.0.1:6543 | localhost:6543 | ::1:6543)
        echo "error: database target is the Compose stack host port (6543)" >&2
        echo "  set DATABASE_URL in .env to the Homebrew PostgreSQL target" >&2
        exit 1 ;;
esac

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
echo "starting Bento on http://127.0.0.1:3000 (--reload) and Vite on http://127.0.0.1:5173; Ctrl-C stops both"
(
    cd "$ROOT" && exec uv run bentoml serve src/serving/service.py:TennisPredictor --host 127.0.0.1 --port 3000 --reload
) &
BENTO_PID=$!
(
    cd "$ROOT/web" && exec pnpm dev
) &
VITE_PID=$!

# Run until one child exits (detecting zombies via ps stat so a crashed
# server ends the session instead of hanging), then stop the other.
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
