# Plan: Remove k8s MLflow (keeping DagsHub)

**Status: COMPLETE** — all tasks verified against live cluster state.

## Goal
Delete the local k3d MLflow deployment. The project already uses DagsHub MLflow — this is purely removing the now-unused local infrastructure.

## Scope

### In scope
- Delete k8s MLflow resources (runtime)
- Delete MLflow manifest file
- Update ingress (remove mlflow.macsteve.lan route)
- Update configmap (remove MLFLOW_TRACKING_URI)
- Update documentation (AGENTS.md, README.md)

### Out of scope
- `mlflow-clean` script and justfile recipe (kept — works against DagsHub too)
- `.env` or `.env.example` changes (already configured for DagsHub)
- Python code changes (MLflow client code is URI-agnostic)
- Data migration (already done)

## Tasks

### [x] Task 1: Delete k8s MLflow resources (runtime)

**Description**: Remove the running MLflow deployment from the cluster before editing manifests.

**Commands**:
```bash
# Delete MLflow deployment, service, and PVC
kubectl delete -f infra/manifests/default/mlflow.yaml

# Verify removal
kubectl get pods,svc,pvc -l app=mlflow
```

**Acceptance criteria**:
- No MLflow pods, services, or PVCs remain
- `mlflow-data` PVC is deleted (5Gi storage freed)

---

### [x] Task 2: Delete MLflow manifest file

**Description**: Remove the now-unused MLflow k8s manifest.

**Files**:
- Delete: `infra/manifests/default/mlflow.yaml`

**Acceptance criteria**:
- File no longer exists

---

### [x] Task 3: Update ingress manifest

**Description**: Remove the `mlflow.macsteve.lan` ingress rule.

**Files**:
- Edit: `infra/manifests/default/ingress.yaml`

**Changes**:
- Remove the entire `- host: mlflow.macsteve.lan` rule block (lines 8-15)
- Keep the `prefect.macsteve.lan` rule intact

**Acceptance criteria**:
- Ingress only routes `prefect.macsteve.lan`
- YAML is valid

---

### [x] Task 4: Update configmap manifest

**Description**: Remove the `MLFLOW_TRACKING_URI` entry from the cluster configmap.

**Files**:
- Edit: `infra/manifests/default/config-map.yaml`

**Changes**:
- Remove line 8: `MLFLOW_TRACKING_URI: "http://mlflow:5000"`

**Acceptance criteria**:
- Configmap no longer contains `MLFLOW_TRACKING_URI`
- Other entries (PREFECT_*) remain

---

### [x] Task 5: Apply updated manifests

**Description**: Reapply the edited ingress and configmap to the cluster.

**Commands**:
```bash
kubectl apply -f infra/manifests/default/ingress.yaml
kubectl apply -f infra/manifests/default/config-map.yaml
```

**Acceptance criteria**:
- Ingress no longer routes `mlflow.macsteve.lan`
- Configmap no longer contains `MLFLOW_TRACKING_URI`

---

### [x] Task 6: Update AGENTS.md

**Description**: Remove references to the local k8s MLflow deployment.

**Files**:
- Edit: `AGENTS.md`

**Changes**:
- Line 55: Remove the `mlflow.macsteve.lan` row from the DNS table
- Line 60: Update "MLflow is the registry of record" to clarify it's DagsHub-hosted

**Acceptance criteria**:
- No mention of `mlflow.macsteve.lan`
- Documentation reflects DagsHub as the registry

---

### [x] Task 7: Update README.md (if needed)

**Description**: Check and update any references to the local MLflow deployment.

**Files**:
- Edit: `README.md` (if it mentions k8s MLflow)

**Changes**:
- Remove or update any references to `mlflow.macsteve.lan` or local MLflow setup

**Acceptance criteria**:
- README reflects DagsHub as the MLflow backend

---

## Dependencies

- Tasks 1-5 must run in order (delete resources before editing manifests, apply after editing)
- Tasks 6-7 can run in parallel (independent doc updates)

## QA/Testing Scenarios

1. **Verify k8s cleanup**:
   ```bash
   kubectl get pods,svc,pvc -l app=mlflow  # should return nothing
   kubectl get ingress                      # should not list mlflow.macsteve.lan
   kubectl get configmap tennis-default-config -o yaml | grep MLFLOW  # should return nothing
   ```

2. **Verify MLflow client still works**:
   ```bash
   uv run python -c "from mlflow.tracking.client import MlflowClient; c = MlflowClient(); print(c.search_registered_models())"
   ```
   Should connect to DagsHub and return results.

3. **Verify dev.sh still works**:
   - `scripts/dev.sh` loads `.env` and uses `MLFLOW_TRACKING_URI` via Python
   - Should import models from DagsHub

## Audit Findings

Verified against live cluster + working tree after execution:

| Check | Result |
|---|---|
| `kubectl get pods,svc,pvc -l app=mlflow` | No resources found |
| `kubectl get ingress` | Only `prefect.macsteve.lan` |
| `kubectl get configmap tennis-default-config -o yaml \| grep MLFLOW` | Nothing |
| `infra/manifests/default/mlflow.yaml` | Deleted |
| `ingress.yaml` | Only prefect rule |
| `config-map.yaml` | No MLFLOW_TRACKING_URI |
| `AGENTS.md` DNS table | Only prefect row; "hosted on DagsHub" |
| `README.md` | "MLflow on DagsHub"; k3d line says "Prefect" only |

**Follow-up (out of scope, not blocking):** `.env.example` lines 22-23 still read `MLFLOW_TRACKING_URI=http://127.0.0.1:5000` with comment "default is the local ./mlruns store". This is a committed template, so a fresh clone defaults to local `mlruns/` instead of DagsHub. You said your real `.env` is already migrated, so this is cosmetic — update it if you want the template to match.

## Notes

- **`mlflow-clean` preserved**: The script and justfile recipe are URI-agnostic — they work against DagsHub just as well as the local server.
- **Python code unchanged**: `mlflow.set_tracking_uri()` and `MlflowClient()` automatically use the DagsHub URI from `.env`.
