# Create the local k3d cluster.
cluster-create:
    ./infra/k3d/start.sh

# Delete the local k3d cluster.
cluster-destroy:
    k3d cluster delete tennis-ml

# Restart Kubernetes workloads.
cluster-restart:
    kubectl rollout restart deployment
    kubectl rollout restart daemonset
    kubectl rollout restart statefulset

# Create the cluster, manifests, and database.
cluster-setup: deps cluster-create db-init

# Run bronze-to-gold ETL; pass --full-refresh to rebuild silver/gold from bronze.
db-etl *args:
    uv run python src/flows/etl.py {{ args }}

# Create PostgreSQL schemas and tables.
db-init:
    uv run python src/db/init_db.py init

# Drop and recreate PostgreSQL schemas.
db-reset:
    uv run python src/db/init_db.py reset

# Seed deterministic raw matches. --all (every CSV) --enrich (Wikipedia bios) --force (overwrite).
db-seed *args:
    uv run python src/db/seed.py {{ args }}

# Export an atomic PostgreSQL training snapshot, optional as the training pipeline always snapshots first.
db-snapshot:
    uv run python src/db/snapshot.py

# Build and push all production docker images.
deploy *args:
    uv run python src/flows/deploy.py {{ args }}
    docker build -t swang62/tennis-web:latest web/
    docker push swang62/tennis-web:latest

# Install Python dependencies.
deps:
    uv sync

# Run Bento and Vite with local preflight checks.
dev:
    ./scripts/dev.sh

# Run production drift monitoring.
drift:
    uv run python src/flows/check_drift.py

# Run all configured linters.
lint:
    uv run pre-commit run --all-files

# Trigger the Prefect scrape deployment; pass Prefect CLI args through unchanged (see scrape_flow docstring).
scrape *args:
    uv run prefect deployment run scrape-flow/scrape {{ args }}

# Run the Python and web test suites.
test:
    uv run pytest
    pnpm --dir web test

# Run the notebook training pipeline.
train:
    uv run python src/flows/pipeline.py
