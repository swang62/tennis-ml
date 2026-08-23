# VITE_* build inputs: web/.env is the single source of truth (Vite reads it
# for pnpm build; just loads it here for the deploy build args). Shell env
# still overrides via env_var_or_default.
set dotenv-filename := "web/.env"

# Run another recipe and notify on failure; preserves the original exit code.
notify-failure *args:
    #!/usr/bin/env bash
    set +e
    just {{ args }}
    status=$?
    if (( status != 0 )); then
        osascript -e 'display notification "just {{ args }} failed" with title "tennis-ml" sound name "Glass"'
    fi
    exit $status

# Tag for published application images. Shell env wins, then root .env
# (not covered by the web/.env dotenv above); default dev.
DOCKER_TAG := env_var_or_default('DOCKER_TAG', shell('grep -q "^DOCKER_TAG=" .env 2>/dev/null && sed -n "s/^DOCKER_TAG=//p" .env | head -n1 || printf dev'))

# Create the local k3d cluster.
cluster-create:
    ./infra/k3d/start.sh

# Delete the local k3d cluster.
cluster-destroy:
    bash -c 'printf "Delete k3d cluster tennis-ml and its data? [y/N] "; read -r answer; case "$$answer" in y|Y|yes|YES) k3d cluster delete tennis-ml ;; *) printf "%s\n" "Cluster deletion cancelled." ;; esac'

# Restart the Prefect server Deployment and wait for rollout readiness.
cluster-restart:
    kubectl rollout restart deployment/prefect-server
    kubectl rollout status deployment/prefect-server --timeout=300s

# Drop and recreate PostgreSQL schemas.
db-reset:
    uv run python src/db/migrate_db.py reset

# Build and push all production docker images, bento and web images.
# Pass --no-cache to force a full refresh of every deploy artifact (including
# the web image's Docker Buildx layer cache).
deploy *args:
    uv run python src/flows/deploy.py {{ args }}
    @cache_flag=""; \
    case " {{ args }} " in *" --no-cache "*) cache_flag="--no-cache";; esac; \
    docker buildx build --builder tennis-multiarch --platform linux/amd64,linux/arm64 \
        --build-arg VITE_SITE_URL={{ env_var_or_default('VITE_SITE_URL', '') }} \
        --build-arg VITE_SITE_ID={{ env_var_or_default('VITE_SITE_ID', '') }} \
        $cache_flag \
        --tag swang62/tennis-web:{{ DOCKER_TAG }} --push web/

# Install Python dependencies.
deps:
    uv sync

# Run Bento and Vite with local preflight checks.
dev:
    ./scripts/dev.sh

# Run production drift monitoring; pass --cutoff YYYY-MM-DD to override the champion cutoff date.
drift *args:
    uv run python src/flows/drift.py {{ args }}

# Run production docker compose detached with 10 minute health check wait
docker:
    docker compose up -d --force-recreate --build --wait --wait-timeout 600

# Run bronze-to-gold ETL; pass --incremental to process only latest matches.
etl *args:
    uv run python src/flows/etl.py {{ args }}

# End-to-end pipeline, targets the .env DATABASE_URL currently set
full-pipeline *args:
    just notify-failure _full-pipeline {{ args }}

# seed: --all, --enrich, --reset, etl: --incremental, train: --force-promote, deploy: --no-cache
_full-pipeline *args: deps lint test cluster-create probe migrate
    just seed {{ args }}
    just etl {{ args }}
    just snapshot
    just train {{ args }}
    just deploy

# Run all configured linters.
lint:
    uv run pre-commit run --all-files

# Apply idempotent PostgreSQL schema migrations without dropping data.
migrate:
    uv run python src/db/migrate_db.py migrate

# Scrape ATP match stats for a window; pass --start YYYY-MM-DD --end YYYY-MM-DD (runs standalone, no rankings scrape).
matches *args:
    uv run python src/flows/matches.py {{ args }}

# Delete every registered model and non-default experiment from MLflow.
mlflow-reset:
    uv run python scripts/reset_mlflow.py

# Fail-fast, non-mutating preflight for the host pipeline environment/
probe:
    uv run python scripts/probe.py

# Scrape ATP rankings for a window; pass --start YYYY-MM-DD --end YYYY-MM-DD (runs standalone, no matches scrape).
rankings *args:
    uv run python src/flows/rankings.py {{ args }}

# Seed deterministic raw matches. --all (every match) --enrich (Wikipedia bios) --reset (overwrite existing rows).
seed *args:
    uv run python src/db/seed.py {{ args }}

# Export/refresh the local DuckDB training snapshot used by `just train`.
snapshot:
    uv run python src/db/snapshot.py

# Run the Python and web test suites.
test:
    uv run pytest
    pnpm --dir web test

# Run the notebook training pipeline. --force-promote will always promote the candidate model as @champion
train *args:
    uv run python src/flows/pipeline.py {{ args }}

# Evaluate the existing candidate and always promote it as @champion.
promote:
    uv run python src/flows/pipeline.py --promote-only --force-promote

# Select a calibration temperature from existing OOF predictions; does not promote or deploy.
calibrate:
    uv run python src/flows/calibrate.py
