# Create the local k3d cluster.
create:
    ./infra/k3d/start.sh

# Build silver and gold dbt models.
db-dbt:
    uv run python src/db/dbt.py

# Run bronze-to-gold ETL; pass --enrich to fetch bios.
db-etl *args:
    uv run python src/flows/etl.py {{args}}

# Create PostgreSQL schemas and tables.
db-init:
    uv run python src/flows/init_db.py init

# Drop and recreate PostgreSQL schemas.
db-reset:
    uv run python src/flows/init_db.py reset

# Seed deterministic raw matches.
db-seed *args:
    uv run python src/flows/seed.py {{args}}

# Export an atomic PostgreSQL training snapshot.
db-snapshot:
    uv run python src/db/snapshot.py

# Build and push the production Bento image.
deploy-bento *args:
    uv run python src/flows/deploy.py {{args}}

# Serve Bento locally on port 3000.
deploy-local:
    docker compose down
    uv run bentoml serve src/serving/service.py:TennisPredictor --host 0.0.0.0 --port 3000

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
    uv run python src/flows/pipeline.py

# Validate Kubernetes manifests.
validate:
    kubeconform -ignore-missing-schemas -summary infra/manifests/

# Start the host Prefect worker.
worker:
    uv run python infra/prefect/worker.py
