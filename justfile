setup: deps create helm-install setup-base init-clickhouse

deps:
    uv sync

create:
    ./infra/k3d/start.sh

helm-install:
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
      -n monitoring --create-namespace \
      -f infra/k3d/prometheus-overrides.yaml
    helm upgrade --install loki grafana/loki \
      -n monitoring --version 6.55.0 \
      -f infra/k3d/loki-overrides.yaml
    helm upgrade --install promtail grafana/promtail \
      -n monitoring \
      -f infra/k3d/promtail-overrides.yaml

setup-base: validate
    kubectl apply -f infra/manifests/default/
    kubectl apply -f infra/manifests/monitoring/

deploy: validate
    kubectl apply -f infra/manifests/deploy/

init-clickhouse:
    kubectl wait --for=condition=ready pod/clickhouse-0 --timeout=180s
    kubectl exec -i statefulset/clickhouse -- clickhouse-client --password password --multiquery < infra/clickhouse/init.sql

reset-clickhouse:
    kubectl exec statefulset/clickhouse -- clickhouse-client --password password --multiquery --query "DROP DATABASE IF EXISTS bronze; DROP DATABASE IF EXISTS gold"
    just init-clickhouse

seed-clickhouse:
    kubectl exec -i statefulset/clickhouse -- clickhouse-client --password password --multiquery < infra/clickhouse/seed.sql

dashboard-local:
    streamlit run src/dashboard/app.py

dashboard-build:
    docker build -t tennis-dashboard:latest -f infra/manifests/deploy/Dockerfile .
    k3d image import tennis-dashboard:latest -c tennis-ml

bento-local:
    bentoml serve src/serving/service.py --reload

bento-build:
    uv run bentoml build --bentofile bentofile.yaml
    uv run bentoml containerize tennis_prediction:latest -t bento-serving:latest
    k3d image import bento-serving:latest -c tennis-ml

reload:
    kubectl rollout restart deployment
    kubectl rollout restart daemonset
    kubectl rollout restart statefulset

train:
    uv run python src/flows/training.py

pipeline:
    uv run python src/flows/pipeline.py

validate:
    kubeconform -ignore-missing-schemas -summary infra/manifests/

destroy:
    k3d cluster delete tennis-ml
