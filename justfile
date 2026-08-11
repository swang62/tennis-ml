# Create the local k3d cluster.
create:
    ./infra/k3d/start.sh

# Run bronze-to-gold ETL (dbt build).
db-etl:
    uv run tennis-db-etl

# Create PostgreSQL schemas and tables.
db-init:
    uv run tennis-db-init

# Drop and recreate PostgreSQL schemas.
db-reset:
    uv run tennis-db-reset

# Seed deterministic raw matches (--all: every ATP CSV; --enrich: Wikipedia bios).
db-seed *args:
    uv run tennis-db-seed {{ args }}

# Export an atomic PostgreSQL training snapshot.
db-snapshot:
    uv run tennis-db-snapshot

# Build and push all production docker images.
deploy *args:
    uv run tennis-deploy {{ args }}
    docker build -t swang62/tennis-web:latest web/
    docker push swang62/tennis-web:latest

# Install Python dependencies.
deps:
    uv sync

# Delete the local k3d cluster.
destroy:
    k3d cluster delete tennis-ml

# Run Bento and Vite with local preflight checks.
dev:
    ./scripts/dev.sh

# Start the Compose production stack.
docker-up:
    docker compose up -d --build

# Fetch missing weekly ATP rankings; register the Monday deployment with --deploy.
rankings-fetch *args:
    uv run tennis-rankings-fetch {{ args }}

# Run all configured linters.
lint:
    uv run pre-commit run --all-files

# Restart Kubernetes workloads.
restart:
    kubectl rollout restart deployment
    kubectl rollout restart daemonset
    kubectl rollout restart statefulset

# Create the cluster, manifests, and database.
setup: deps create setup-base db-init

# Apply Kubernetes manifests.
setup-base: validate
    kubectl apply -f infra/manifests/default/

# Stop the local k3d cluster.
stop:
    k3d cluster stop tennis-ml

# Run the Python test suite.
test:
    uv run pytest

# Run the notebook training pipeline.
train:
    uv run tennis-train

# Validate Kubernetes manifests.
validate:
    kubeconform -ignore-missing-schemas -summary infra/manifests/

# Start the host Prefect worker.
worker:
    uv run tennis-worker

# Run production drift monitoring.
check-drift:
    uv run tennis-check-drift
