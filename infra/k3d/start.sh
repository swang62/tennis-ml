#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME="tennis-ml"

MACHINE_ID="infra/k3d/machine-id"
if [ ! -f "$MACHINE_ID" ]; then
  echo -n "abc123abc123abc123abc123abc123ab" > "$MACHINE_ID"
fi

if k3d cluster list | grep -q "$CLUSTER_NAME"; then
  echo "Cluster '$CLUSTER_NAME' already exists. Skipping..."
else
  echo "Creating cluster '$CLUSTER_NAME'..."
  k3d cluster create --config infra/k3d/config.yaml \
    --volume "$PWD/$MACHINE_ID:/etc/machine-id"
fi

kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -
kubectl cluster-info
