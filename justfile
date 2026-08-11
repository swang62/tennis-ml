# Create the local k3d cluster.
create:
    ./infra/k3d/start.sh

# Run bronze-to-gold ETL (dbt build).
db-etl:
    uv run python src/flows/etl.py

# Create PostgreSQL schemas and tables.
db-init:
    uv run python src/db/init_db.py init

# Drop and recreate PostgreSQL schemas.
db-reset:
    uv run python src/db/init_db.py reset

# Seed deterministic raw matches.
db-seed *args:
    uv run python src/db/seed.py {{ args }}

# Export an atomic PostgreSQL training snapshot.
db-snapshot:
    uv run python src/db/snapshot.py

# Build and push all production docker images.
deploy *args:
    uv run python src/flows/deploy.py {{ args }}
    docker build -t swang62/tennis-web:latest web/ --no-cache
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

# Register the Monday Prefect deployment with: uv run python src/flows/rankings.py --deploy
rankings-fetch:
    uv run python src/flows/rankings.py

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

# Run production drift monitoring.
check-drift:
    uv run python src/flows/check_drift.py
