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

db-seed:
    uv run python infra/duckdb/seed.py

db-etl:
    uv run python src/flows/etl.py

db-dbt:
    uv run dbt build --project-dir dbt --profiles-dir dbt

db-reset:
    rm -f data/tennis.duckdb
    just db-init

# Local frontend dev server for the React dashboard
dashboard-local:
    npm --prefix web run dev

# Local backend API server on :3000
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
