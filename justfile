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

db-init:
    uv run python infra/duckdb/initialize_schemas.py init

db-seed *args:
    uv run python infra/duckdb/seed.py {{args}}

# Pass args (e.g. --enrich) directly: just db-etl --enrich
db-etl *args:
    uv run python src/flows/etl.py {{args}}

# --- Production, host-side (against the running Quack server) ---
# The compose quack-db publishes :9494 on the host. These connect over the
# Quack protocol (quack:127.0.0.1:9494) with ENVIRONMENT=production and the
# QUACK_TOKEN from .env. The DB file is never opened on the host and seeding
# runs against the server (not inside its container). Copy args as above.
db-seed-prod *args:
    ENVIRONMENT=production QUACK_URI=quack:127.0.0.1:9494 uv run python infra/duckdb/seed.py {{args}}

db-etl-prod *args:
    ENVIRONMENT=production QUACK_URI=quack:127.0.0.1:9494 uv run python src/flows/etl.py {{args}}

db-dbt:
    uv run dbt build --project-dir dbt --profiles-dir dbt

db-reset:
    rm -f data/tennis.duckdb
    just db-init

# Build the Quack companion image that serves the production DuckDB remotely
quack-build:
    docker build -f infra/duckdb/Dockerfile -t tennis-quack-db:latest .

# Run the Quack companion server against a local production DB (dev/production
# data path); for local testing only — production runs it via Docker Compose.
quack-local:
    uv run python infra/duckdb/server.py

# Local frontend dev server for the React dashboard
dashboard-local:
    npm --prefix web run dev

# Local backend API server on :3000
deploy-local:
    uv run bentoml serve src/serving/service.py:TennisPredictor --host 0.0.0.0 --port 3000

# Build docker image locally, push to Docker Hub and boot via Docker Compose
# Pass args (e.g. --force) directly: just deploy-bento --force
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
