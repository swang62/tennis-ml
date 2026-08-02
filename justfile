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

# Single deployment path: build and deploy the serving Bento for the latest
# promoted production model. No-ops when production_model has no version newer
# than the last deployed one. Invoked by the deploy flow (src/flows/deploy.py)
# or directly for a re-deploy. Evaluation/promotion (05) already ran in the
# training pipeline — this target never runs it.
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
