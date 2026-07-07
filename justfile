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

bento-build:
    uv run bentoml build --bentofile bentofile.yaml
    uv run bentoml containerize tennis_prediction:latest -t bento-serving:latest
    k3d image import bento-serving:latest -c tennis-ml

restart:
    kubectl rollout restart deployment
    kubectl rollout restart daemonset
    kubectl rollout restart statefulset

train:
    uv run python src/flows/training.py

pipeline:
    uv run python src/flows/pipeline.py

validate:
    kubeconform -ignore-missing-schemas -summary infra/manifests/

stop:
    k3d cluster stop tennis-ml

destroy:
    k3d cluster delete tennis-ml
