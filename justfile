setup: deps create setup-base db-init

deps:
    uv sync

create:
    ./infra/k3d/start.sh

setup-base: validate
    kubectl apply -f infra/manifests/default/

deploy: validate
    kubectl apply -f infra/manifests/deploy/

db-init:
    uv run python infra/duckdb/run_init.py init

db-seed:
    uv run python infra/duckdb/run_init.py seed

db-etl:
    uv run python src/flows/etl.py

db-dbt:
    uv run dbt build --project-dir dbt --profiles-dir dbt

db-reset:
    rm -f data/tennis.duckdb
    just db-init

dashboard-local:
    panel serve src/dashboard/app.py

dashboard-build:
    docker build -t tennis-dashboard:latest -f infra/manifests/deploy/Dockerfile .
    k3d image import tennis-dashboard:latest -c tennis-ml

bento-local:
    bentoml serve src/serving/service.py --reload

# Build the production Bento image in the local Docker engine. Does not require
# k3d to exist or be running.
bento-build:
    uv run python -c "from src.flows.deploy import build_bento_image; build_bento_image()"

# Build locally, push to the k3d-managed registry, and roll out when the cluster
# is running. Evaluation/promotion already ran in the training pipeline.
deploy-bento:
    uv run python -c "from src.flows.deploy import deploy_bento; deploy_bento()"

restart:
    kubectl rollout restart deployment
    kubectl rollout restart daemonset
    kubectl rollout restart statefulset

train:
    uv run python src/flows/pipeline.py

pipeline:
    uv run python src/flows/pipeline.py

validate:
    kubeconform -ignore-missing-schemas -summary infra/manifests/

stop:
    k3d cluster stop tennis-ml

destroy:
    k3d cluster delete tennis-ml
