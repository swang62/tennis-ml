setup: deps create setup-base db-init

deps:
    uv sync

create:
    ./infra/k3d/start.sh

setup-base: validate
    kubectl apply -f infra/manifests/default/

# The Prefect server runs in the cluster; the worker must run on the host
worker:
    uv run python infra/prefect/worker.py

# PostgreSQL bootstrap: schemas + base tables (structure only, idempotent).
# All db commands connect via the .env PostgreSQL contract (DATABASE_URL or
# POSTGRES_* components). Data loading is the explicit `just db-seed` /
# `just db-seed --all` step.
db-init:
    uv run python -m src.flows.init_db init

db-seed *args:
    uv run python -m src.flows.seed {{args}}

# Pass args (e.g. --enrich) directly: just db-etl --enrich
db-etl *args:
    uv run python src/flows/etl.py {{args}}

# dbt ETL over PostgreSQL: sources .env so dbt sees the shared POSTGRES_*
# credential contract (dbt does not load .env itself). silver/gold tables are
# dbt-owned and rebuilt in dependency order by `dbt build`.
db-dbt:
    set -a && source .env && set +a && uv run dbt build --project-dir dbt --profiles-dir dbt

# Pull an atomic PostgreSQL -> DuckDB training snapshot (gold.match_features +
# gold.player_profiles only, validated). Training (`just train`) refreshes it
# first; use this to inspect the snapshot ahead of a run.
db-snapshot:
    uv run python -m src.db.snapshot

# Destructive: drops the bronze/silver/gold schemas, then recreates structure.
# init_db refuses unless the ACTUAL connection target is a local host
# (127.0.0.1/localhost/::1) with the configured POSTGRES_PORT and POSTGRES_DB —
# an ENVIRONMENT value alone can never authorize resetting a non-local database.
db-reset:
    uv run python -m src.flows.init_db reset

# Local frontend dev dashboard with HMR
dashboard-local:
    npm --prefix web run dev

# Local backend API server on :3000
deploy-local:
    docker compose down
    uv run bentoml serve src/serving/service.py:TennisPredictor --host 0.0.0.0 --port 3000

# Build docker image locally, push to Docker Hub
deploy-bento *args:
    uv run python src/flows/deploy.py {{args}}

restart:
    kubectl rollout restart deployment
    kubectl rollout restart daemonset
    kubectl rollout restart statefulset

train:
    uv run python src/flows/pipeline.py

lint:
    uv run pre-commit run --all-files

test:
    uv run pytest

validate:
    kubeconform -ignore-missing-schemas -summary infra/manifests/

stop:
    k3d cluster stop tennis-ml

destroy:
    k3d cluster delete tennis-ml
