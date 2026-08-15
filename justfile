# VITE_* build inputs: web/.env is the single source of truth (Vite reads it
# for pnpm build; just loads it here for the deploy build args). Shell env
# still overrides via env_var_or_default.
set dotenv-filename := "web/.env"

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
cluster-setup: deps cluster-create migrate

# Run bronze-to-gold ETL; pass --full-refresh to rebuild silver/gold from bronze.
etl *args:
    uv run python src/flows/etl.py {{ args }}

# Apply idempotent PostgreSQL schema migrations without dropping data.
migrate:
    uv run python src/db/migrate_db.py migrate

# Drop and recreate PostgreSQL schemas.
db-reset:
    uv run python src/db/migrate_db.py reset

# Seed deterministic raw matches. --all (every CSV) --enrich (Wikipedia bios) --force (overwrite).
seed *args:
    uv run python src/db/seed.py {{ args }}

# Insert deterministic cloned real bronze matches for drift testing. --dry-run --force --after.
seed-random *args:
    uv run python src/db/seed_drift.py {{ args }}

# Export an atomic PostgreSQL training snapshot, optional as the training pipeline always snapshots first.
snapshot:
    uv run python src/db/snapshot.py

# Build and push all production docker images.
deploy *args:
    uv run python src/flows/deploy.py {{ args }}
    docker buildx build --builder tennis-multiarch --platform linux/amd64,linux/arm64 \
        --build-arg VITE_SITE_URL={{ env_var_or_default('VITE_SITE_URL', '') }} \
        --build-arg VITE_SITE_ID={{ env_var_or_default('VITE_SITE_ID', '') }} \
        --tag swang62/tennis-web:latest --push web/

# Install Python dependencies.
deps:
    uv sync

# Run Bento and Vite with local preflight checks.
dev:
    ./scripts/dev.sh

# Run production drift monitoring.
drift:
    uv run python src/flows/drift.py

# Run all configured linters.
lint:
    uv run pre-commit run --all-files

# Delete every registered model and non-default experiment from MLflow.
mlflow-reset:
    uv run python scripts/reset_mlflow.py

# Trigger the Prefect scrape deployment; pass Prefect CLI args through unchanged (see scrape_flow docstring).
scrape *args:
    uv run prefect deployment run scrape-flow/scrape {{ args }}

# Run the Python and web test suites.
test:
    uv run pytest
    pnpm --dir web test

# Run the notebook training pipeline.
train *args:
    uv run python src/flows/pipeline.py {{ args }}
