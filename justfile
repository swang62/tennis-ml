setup: deps create setup-base db-init

deps:
    uv sync

create:
    ./infra/k3d/start.sh

setup-base: validate
    kubectl apply -f infra/manifests/default/

# Start the host-local Prefect worker attached to the tennis-pool work pool.
# The Prefect server runs in the cluster; the worker must run on the host so it
# can reach local artifacts (DuckDB, models). Loads .env (PREFECT_API_URL).
worker:
    uv run python infra/prefect/worker.py

db-init:
    uv run python infra/duckdb/initialize_schemas.py init

db-seed:
    uv run python infra/duckdb/seed.py

db-etl:
    uv run python src/flows/etl.py

db-dbt:
    uv run dbt build --project-dir dbt --profiles-dir dbt

db-reset:
    rm -f data/tennis.duckdb
    just db-init

# Local dev server for the React dashboard (web/) with HMR. Requires the
# Bento API running locally on :3000 (see `just deploy-local`).
dashboard-local:
    npm --prefix web run dev

# Local dev server on :3000 — smoke-test /predict, /predict-from-ids, /health
# without k3d. Requires trained artifacts in data/processed/ (run `just train`).
deploy-local:
    uv run bentoml serve src/serving/service.py:TennisPredictor --host 0.0.0.0 --port 3000

# Build docker image locally, push to Docker Hub and boot via Docker Compose
deploy-bento:
    uv run python src/flows/deploy.py

restart:
    kubectl rollout restart deployment
    kubectl rollout restart daemonset
    kubectl rollout restart statefulset

train:
    uv run python src/flows/pipeline.py

# Lint + format + typecheck (pre-commit covers ruff, kubeconform, basedpyright)
test:
    uv run pre-commit run --all-files
    uv run pytest

validate:
    kubeconform -ignore-missing-schemas -summary infra/manifests/

stop:
    k3d cluster stop tennis-ml

destroy:
    k3d cluster delete tennis-ml
