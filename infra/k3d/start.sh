#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME="tennis-ml"

if k3d cluster list | grep -q "$CLUSTER_NAME"; then
  echo "Cluster '$CLUSTER_NAME' already exists. Skipping..."
else
  echo "Creating cluster '$CLUSTER_NAME'..."
  k3d cluster create --config infra/k3d/config.yaml
fi

kubectl cluster-info
