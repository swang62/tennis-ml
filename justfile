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

dashboard-local:
    panel serve src/dashboard/app.py

dashboard-build:
    docker build -t tennis-dashboard:latest -f infra/manifests/deploy/Dockerfile .
    k3d image import tennis-dashboard:latest -c tennis-ml

deploy-local:
    bentoml serve src/serving/service.py --reload

# Build docker image locally, push to the k3d-managed registry and deploy on k3d 
# if the cluster is running AND the latest trained model has been promoted
deploy-bento:
    uv run python src/flows/deploy.py

restart:
    kubectl rollout restart deployment
    kubectl rollout restart daemonset
    kubectl rollout restart statefulset

train:
    uv run python src/flows/pipeline.py

validate:
    kubeconform -ignore-missing-schemas -summary infra/manifests/

stop:
    k3d cluster stop tennis-ml

destroy:
    k3d cluster delete tennis-ml
