# Plan: Phase 2 self-hosted single-node k3d data platform

## Goal

Run the entire tennis-ml data and ML platform locally inside the existing single-node k3d cluster, without Helm app charts or SaaS infrastructure dependencies. Phase 1 (dbt migration) is complete: the repo already builds `gold.match_features` from a DuckDB `bronze.match_events` source via `dbt/` models. Phase 2 adds a streaming bronze path (Redpanda -> Kafka Connect -> SeaweedFS Parquet) while keeping the existing Prefect, MLflow, BentoML, and training architecture.

```text
Host Python script --HTTPS--> Redpanda HTTP proxy (pandaproxy, built into broker)
       |
       v
Redpanda broker (Kafka-API compatible, single node)
       |
       v
Kafka Connect (one distributed worker, S3 sink)
       |
       v
SeaweedFS `weed mini` (local S3) -> immutable Parquet bronze
       |
       v
Prefect worker -> dbt-duckdb -> local DuckDB silver/gold
       |                              |
       v                              v
local MLflow                   training (standalone) / dashboard
       |
       v
BentoML production service
```

All host ingress goes through Traefik: host Caddy fronts TLS and routes `*.macsteve.lan` to the k3d load balancer on `localhost:8080`, which lands on Traefik. No host port-forwards and no broker port mappings:

| Hostname | Traefik destination |
|---|---|
| `prefect.macsteve.lan` | Prefect UI/API (live) |
| `mlflow.macsteve.lan` | MLflow UI/API (live) |
| `bento.macsteve.lan` | BentoML prediction API (live) |
| `registry.macsteve.lan` | k3d-managed container registry (live) |
| `kafka.macsteve.lan` | Redpanda HTTP proxy (host ingest endpoint) (new) |
| `redpanda.macsteve.lan` | Redpanda Console (new) |
| `storage.macsteve.lan` | SeaweedFS Admin UI (new) |
| `s3.macsteve.lan` | SeaweedFS S3 API (new) |

Host producers ingest through `https://kafka.macsteve.lan` (Traefik -> Redpanda's built-in HTTP proxy / pandaproxy -> broker). Self-signed TLS is accepted by clients using insecure-TLS flags, mirroring the existing `MLFLOW_TRACKING_INSECURE_TLS` and `PREFECT_API_TLS_INSECURE_SKIP_VERIFY` pattern. The Kafka API (9092), Connect REST, and the broker's admin API stay internal ClusterIP services; `redpanda.macsteve.lan` is the browser console, not the bootstrap endpoint.

## Current repo state (baseline for this plan)

- Phase 1 dbt migration is **complete**: models live under `dbt/models/{silver,gold}/`, dbt tests under `dbt/tests/gold/`, `just db-etl` runs the dbt-backed Prefect ETL path (`src/flows/etl.py` runs `dbt build`).
- Bronze is still the DuckDB `bronze.match_events` table; `dbt/models/sources.yml` sources it.
- Prefect server/worker (`prefecthq/prefect:3.7.3-python3.11`, process pool `tennis-pool`), MLflow (`ghcr.io/mlflow/mlflow:v2.20.2`, sqlite + artifacts PVC), BentoML, and Panel dashboard are already deployed as manifests in `infra/manifests/`.
- Training is **standalone** (`src/flows/pipeline.py` runs Papermill notebooks) and is NOT a Prefect flow. Deploy is a Prefect flow (`src/flows/deploy.py`) that builds the Bento locally, pushes to the local registry, and rolls out.
- MLflow registry uses **aliases** (`@best`, `@champion`), not stages. The NN is served via ONNX, not torch.
- `bentofile.yaml` is a template; `deploy.py` generates a pinned `data/processed/bentofile.pinned.yaml`.
- A k3d-managed container registry (`tennis-ml-registry`, hostPort 5000) is live; images push through `registry.macsteve.lan`.
- Ingress is single-entrypoint: host Caddy routes `*.macsteve.lan` to the k3d load balancer. `dashboard.macsteve.lan` is out of scope for Phase 2.
- Config map (`infra/manifests/default/config-map.yaml`) carries `MLFLOW_TRACKING_URI`, `PREFECT_API_URL`, and Prefect server DB settings.

## Self-hosted definition

- Redpanda, Kafka Connect, SeaweedFS, DuckDB, Prefect, MLflow, dbt, training, Redpanda Console, BentoML, and the container registry are self-hosted.
- Confluent Cloud, MotherDuck, Prefect Cloud, Databricks, DagsHub, or another SaaS control/data plane is not required.
- Internet access remains allowed for source-data collection, Wikipedia enrichment, Hugging Face/FastEmbed downloads, container images, package installation, and updates.
- Existing data persists across pod restarts. A host-mounted k3d data root preserves authoritative state across cluster recreation.

## Scope

### In scope

- Existing one-server, zero-agent k3d cluster from `infra/k3d/config.yaml`.
- Plain Kubernetes manifests applied with `kubectl`; no Helm app charts or operators (k3s built-in Traefik HelmChartConfig stays).
- One Redpanda broker (Kafka-API compatible) with its built-in HTTP proxy; no ZooKeeper.
- One Kafka Connect distributed-mode worker with an S3 sink connector writing Parquet to SeaweedFS.
- Single-container SeaweedFS `weed mini` for local S3-compatible object storage.
- Redpanda Console for topics, messages, consumer groups, and Kafka Connect visibility.
- A versioned JSON event contract and a host producer helper that publishes CSV rows to Redpanda over HTTPS.
- SeaweedFS Parquet as the dbt bronze source (new staging model), keeping gold parity with the Phase 1 baseline.
- A pinned project Prefect worker image containing the repo package and dbt-duckdb.
- Traefik host-based ingress for the new HTTP UIs; existing services stay as-is.
- A host-mounted k3d data root so state survives cluster recreation.
- README rewrite documenting the complete self-hosted platform.

### Out of scope

- Dashboard (`dashboard.macsteve.lan`) and dashboard port changes.
- Confluent Cloud, MotherDuck, Prefect Cloud, or another required SaaS dependency.
- Helm app charts, Strimzi, Confluent Operator, or another Kubernetes operator.
- Multi-broker Kafka, replication, high availability, autoscaling, or production SLAs.
- Delta Lake; Parquet is the bronze format.
- Schema Registry, Avro, Protobuf, Flink, Spark, or ksqlDB.
- Cloud GPU training.
- Globally accessible production serving.
- Air-gapped operation without internet access.

## Architecture decisions

- Redpanda runs as a one-replica StatefulSet (Kafka-API compatible; no ZooKeeper, no JVM). The broker bundles the Kafka API (internal `9092`), the HTTP proxy / pandaproxy (internal `8082`), and Schema Registry (internal `8081`) — Schema Registry is out of scope but its bundled port is simply not exposed. Only internal ClusterIP listeners exist; no broker port is mapped to the host.
- Host producers publish through the Redpanda HTTP proxy (pandaproxy) exposed at `https://kafka.macsteve.lan` via Traefik (no custom bridge). Clients accept the self-signed cert with insecure-TLS flags mirroring `MLFLOW_TRACKING_INSECURE_TLS`/`PREFECT_API_TLS_INSECURE_SKIP_VERIFY`.
- Kafka Connect runs in distributed mode even with one worker; configuration, status, and offsets live in internal topics with replication factor one. Connect REST stays internal ClusterIP.
- SeaweedFS runs as one StatefulSet using `weed mini`, one persistent volume, S3 endpoint `8333`, Admin UI `23646`. Development object store, not highly available storage.
- The bronze bucket stores immutable Parquet objects. dbt owns casting, validation, deduplication, and silver/gold materialization.
- Redpanda Console (separate `redpandadata/console` image) is the only broker browser UI; the Redpanda broker itself does not bundle a web UI.
- The existing local DuckDB database remains the analytical database. Kafka Connect must not write directly into the `.duckdb` file.
- Kubernetes Secrets contain runtime credentials; Git tracks only key names and setup instructions.
- Images, plugins, and Python dependencies are version-pinned; new images are pushed to the local registry (`tennis-ml-registry:5000`).
- A host directory mounted into the k3d server backs Redpanda, SeaweedFS, DuckDB, Prefect, and MLflow state so `k3d cluster delete` does not silently delete platform data.
- Configuration is declarative and Git-tracked as Kubernetes YAML, dbt YAML/SQL, or connector configuration templates. Mutable data lives only in SeaweedFS or stable host-backed volumes.
- Bootstrap Jobs reconcile desired state idempotently: create missing topics/buckets/connectors/schemas, update compatible configuration, never truncate or recreate existing durable data.

## Tasks

### [ ] Task 1: Inventory current workloads and define budgets, hostnames, and persistence

- **Description**: Confirm the Phase 1 baseline (already complete). Inventory currently deployed workloads (Prefect server/worker, MLflow, BentoML, registry) and reserve capacity for Redpanda, Connect, SeaweedFS, and Redpanda Console. Pin every new container image. Assign the new hostnames, and map one host-mounted data root into the k3d server with stable subdirectories for Redpanda, SeaweedFS, DuckDB, Prefect, and MLflow.
- **Files**:
  - `infra/k3d/config.yaml`
  - `infra/k3d/start.sh`
  - `infra/manifests/default/config-map.yaml`
  - `infra/manifests/default/ingress.yaml`
  - `justfile`
  - `README.md`
- **Acceptance Criteria**:
  - Planned additions fit within a documented host budget (~4 GiB additional RAM), leaving capacity for training outside the cluster.
  - New images use immutable versions and push to `tennis-ml-registry:5000`.
  - Hostnames assigned without conflicts: `kafka` (Redpanda HTTP proxy), `redpanda` (Console), `storage`, `s3` added to the existing `mlflow`, `prefect`, `bento`, `registry`.
  - One host-mounted data root is mapped into the k3d server with stable subdirectories; Redpanda cluster data survives a tested delete/recreate cycle.
  - Static host-backed PV/PVC definitions use stable paths and a `Retain` lifecycle; reapply reconciles, not resets.
- **Guardrails**:
  - Do not add Helm app charts, repositories, or operators.
  - Do not expose the Kafka API or Connect REST publicly through HTTP ingress.
  - Do not store persistent service data only inside the disposable k3d server container.

### [ ] Task 2: Add single-container SeaweedFS object storage

- **Description**: Deploy one SeaweedFS StatefulSet running `weed mini -dir=/data`, one PVC, resource limits, startup/readiness probes, S3 credentials from a Secret, and the `tennis-bronze` bucket created through supported environment configuration. Expose the Admin UI through Traefik; keep the S3 endpoint internal via ClusterIP.
- **Files**:
  - `infra/manifests/default/seaweedfs.yaml` (new)
  - `infra/manifests/default/ingress.yaml`
  - `infra/manifests/default/config-map.yaml`
  - `justfile`
- **Acceptance Criteria**:
  - SeaweedFS restarts without losing bucket contents.
  - `weed mini` initializes credentials and `tennis-bronze` idempotently.
  - A test Parquet object can be uploaded, listed, downloaded, and read with local DuckDB through `http://seaweedfs:8333`.
  - Credentials come from a Kubernetes Secret created outside Git.
  - SeaweedFS Admin UI loads through Traefik at `storage.macsteve.lan`.
  - Resource usage measured with a target limit no larger than 512 MiB unless testing proves that insufficient.
- **Guardrails**:
  - Single-node/single-drive SeaweedFS is for local development only.
  - Do not enable anonymous bucket writes.
  - Do not store credentials in ConfigMaps or tracked connector JSON.

### [ ] Task 3: Add one-node Redpanda broker with built-in HTTP proxy

- **Description**: Deploy one Redpanda StatefulSet (Kafka-API compatible, no ZooKeeper) with internal ClusterIP services only — Kafka API (`9092`), HTTP proxy / pandaproxy (`8082`), and admin API. PVC, resource requests/limits, and health probes. Configure development-safe topic retention and auto-topic behavior explicitly. The HTTP proxy is the host ingest endpoint and is exposed through Traefik at `https://kafka.macsteve.lan`.
- **Files**:
  - `infra/manifests/default/redpanda.yaml` (new)
  - `infra/manifests/default/config-map.yaml`
  - `infra/manifests/default/ingress.yaml`
  - `justfile`
- **Acceptance Criteria**:
  - Redpanda starts without ZooKeeper; Kafka-API clients work against it unchanged.
  - Topic `tennis.match_events` can be created, listed, produced to, and consumed from inside the cluster via the Kafka API.
  - The HTTP proxy accepts a `POST /topics/tennis.match_events` request from inside the cluster and the record lands on the topic.
  - Only internal ClusterIP listeners exist; no broker port is mapped to the host.
  - Broker events, consumer offsets, Connect internal topics, and broker identity survive pod restart and cluster recreation.
  - Internal topics and development topics use replication factor one.
  - Connect REST and the broker's internal listener stay internal ClusterIP services with no host port-forwards.
- **Guardrails**:
  - Do not represent this as highly available or production-grade.
  - Do not add multiple brokers simply to imitate production.
  - Keep the Kafka API and admin API internal to the host/cluster; only the HTTP proxy is exposed via Traefik.

### [ ] Task 4: Deploy Kafka Connect worker and S3 sink, add Redpanda Console

- **Description**: Use the stock `confluentinc/cp-kafka-connect` image, which already bundles the S3 sink connector and JSON/Parquet converters — no custom Dockerfile or `confluent-hub install` needed. Configure the worker entirely via ConfigMap (env) and register an idempotent S3 sink connector (connector JSON as a ConfigMap/applied via bootstrap Job) targeting SeaweedFS. Deploy Redpanda Console (`redpandadata/console`, a separate stateless Deployment) at `redpanda.macsteve.lan`. Confirm the bundled connector license/redistribution terms before adopting its output contract.
- **Files**:
  - `infra/manifests/default/kafka-connect.yaml` (new: Deployment/StatefulSet + internal ClusterIP service, env from ConfigMap)
  - `infra/kafka-connect/connector.json` (new: S3 sink connector config template)
  - `infra/manifests/default/config-map.yaml`
  - `infra/manifests/default/redpanda-console.yaml` (new)
  - `infra/manifests/default/ingress.yaml`
  - `justfile`
- **Acceptance Criteria**:
  - `GET /connector-plugins` reports the S3 sink connector after deployment.
  - Connector authenticates to SeaweedFS through the internal service name and writes valid Parquet (not JSON with a Parquet extension).
  - Worker and connector state survive Connect pod restart via Kafka internal topics; committed offsets resume without dropping acknowledged events.
  - Records land under a deterministic bronze prefix partitioned by event date or another documented stable convention.
  - Connector registration can be rerun without duplicates; Connect reports auth/serialization/SeaweedFS failures visibly.
  - Redpanda Console lists topics, messages, consumer groups, lag, and Connect connector status at `redpanda.macsteve.lan`.
  - The image tag is pinned; the container is not mutated at pod startup; bundled connector license restrictions documented.
- **Guardrails**:
  - Do not write a custom Dockerfile for the Connect image unless the stock image's bundled connector proves insufficient.
  - Do not rely on `confluent-hub install` at runtime.
  - Do not write a custom Kafka consumer unless the user explicitly rejects every compatible Connect sink.
  - Do not use standalone mode or a local Connect offset file.
  - Do not expose the Connect REST API through public ingress.
  - Do not deploy AKHQ, Kafdrop, Provectus Kafka UI, or Confluent Control Center.

### [ ] Task 5: Define the event contract and host producer path

- **Description**: Define a versioned JSON contract for validated match events and a host-side producer helper so existing CSV rows can be published to Redpanda through its built-in HTTP proxy at `https://kafka.macsteve.lan` — no custom bridge service. Use a deterministic key based on match/player identity and include event metadata required for traceability and deduplication.
- **Files**:
  - `src/flows/ingest.py`
  - `src/features/validate.py`
  - `src/features/columns.py`
  - `src/producers/` (new: thin HTTP-proxy client that validates then POSTs to pandaproxy) (or equivalent path)
  - `pyproject.toml`
  - `tests/`
- **Acceptance Criteria**:
  - Every `BRONZE_COLUMNS` field has a documented type and nullability rule (columns already centralized in `src/features/columns.py`).
  - A host Python script POSTs a JSON event to `https://kafka.macsteve.lan` (self-signed cert accepted via an insecure-TLS flag mirroring `MLFLOW_TRACKING_INSECURE_TLS`/`PREFECT_API_TLS_INSECURE_SKIP_VERIFY`) and the event lands on `tennis.match_events`.
  - Invalid events are rejected before publish with a visible 4xx response.
  - Re-publishing the same CSV produces the same logical event keys.
  - Delivery failures (Redpanda down) surface as 5xx rather than silently dropped.
  - Focused tests cover serialization, validation, and key stability.
  - Existing Wikipedia enrichment continues to work through its current HTTP path with explicit timeout/retry behavior.
- **Guardrails**:
  - Use JSON; do not introduce Schema Registry, Avro, or Protobuf.
  - Keep the direct CSV bootstrap path until end-to-end parity passes.
  - The producer only posts to the HTTP proxy; it must not touch Connect REST or broker internals.
  - Wikipedia failures must be reported or skipped deterministically without blocking durable ingestion.

### [ ] Task 6: Make SeaweedFS Parquet the dbt bronze source

- **Description**: Replace the production bronze-table source with external Parquet files in SeaweedFS. Add a staging model under `dbt/models/staging/` that casts connector output to the current typed bronze contract, retains Kafka metadata, and deduplicates replayed/retried events before `gold.match_features` runs.
- **Files**:
  - `dbt/models/sources.yml`
  - `dbt/models/staging/` (new staging model and tests)
  - `dbt/models/gold/match_features.sql`
  - `dbt/dbt_project.yml`
  - `dbt/profiles.yml`
  - `src/flows/etl.py`
- **Acceptance Criteria**:
  - dbt reads Parquet through SeaweedFS's internal S3 endpoint.
  - Staging output matches the existing bronze column contract.
  - Kafka retries or replay do not duplicate logical gold rows.
  - dbt tests cover required columns, allowed values, key uniqueness, and source freshness.
  - Gold output matches the Phase 1 baseline for identical source events.
- **Guardrails**:
  - Raw Parquet remains immutable.
  - Do not point gold directly at untyped connector output.
  - Do not add Delta Lake.

### [ ] Task 7: Containerize Prefect execution for dbt

- **Description**: Replace the generic Prefect worker image with a pinned project worker image containing the repository package and dbt-duckdb. Mount persistent DuckDB and pipeline artifact storage. Keep training standalone: the worker image must NOT bundle heavyweight training dependencies, and training stays a manual `just train` step, not a Prefect flow.
- **Files**:
  - `infra/images/worker.Dockerfile` (new)
  - `infra/manifests/default/prefect-worker.yaml`
  - `infra/manifests/default/config-map.yaml`
  - `src/flows/etl.py`
  - `src/flows/deploy.py`
  - `justfile`
- **Acceptance Criteria**:
  - Prefect schedules and runs the dbt build (ETL) and the deploy flow inside the local cluster.
  - dbt writes silver/gold to the persistent local DuckDB database.
  - Worker image contains pinned runtime dependencies rather than installing them during each flow run.
  - FastEmbed/Hugging Face assets may download on first use and then reuse a persistent local cache.
  - Flow retries do not cause duplicate logical gold rows.
  - Training can be manually triggered after a successful dbt run.
- **Guardrails**:
  - Do not use Prefect Cloud.
  - Do not run Kafka or Kafka Connect as Prefect flow tasks.
  - Do not convert training into a Prefect flow; it stays standalone (`src/flows/pipeline.py`).
  - Keep heavyweight training dependencies out of the worker and service images.

### [ ] Task 8: Consolidate ingress, rewrite README, and prove self-hosted operation

- **Description**: Extend the existing Traefik ingress for the new hostnames, rewrite the README for the complete platform, then run the full pipeline, test cluster recreation with host data preserved, and cut over the default local path to Redpanda/Parquet.
- **Files**:
  - `infra/manifests/default/ingress.yaml`
  - `infra/manifests/default/config-map.yaml`
  - `infra/k3d/config.yaml`
  - `infra/k3d/start.sh`
  - `justfile`
  - `README.md`
  - `AGENTS.md`
- **Acceptance Criteria**:
  - `prefect.macsteve.lan`, `mlflow.macsteve.lan`, `bento.macsteve.lan`, `registry.macsteve.lan`, `kafka.macsteve.lan`, `redpanda.macsteve.lan`, `storage.macsteve.lan`, and `s3.macsteve.lan` route through Traefik to the documented services.
  - Readiness prevents dependent jobs from starting before Redpanda, Connect, or SeaweedFS are usable.
  - Operators can inspect Kafka lag, connector state, SeaweedFS objects, Prefect runs, and MLflow runs; no internal credential is rendered in a browser UI or pod log.
  - Seeded CSV events flow through Redpanda, Connect, SeaweedFS Parquet, dbt, DuckDB, training, MLflow, and BentoML.
  - Host-mounted Redpanda/SeaweedFS/DuckDB/Prefect/MLflow state survives pod restart and cluster deletion/recreation; Connect resumes from preserved offsets without duplicating Parquet.
  - Reapplying all YAML and bootstrap Jobs twice produces no duplicate resources, events, Parquet data, or destructive changes.
  - Temporary Wikipedia/Hugging Face/network failure does not damage or roll back locally persisted state.
  - Backup and restore are tested for Redpanda PVC, SeaweedFS PVC, DuckDB file, Prefect DB, and MLflow state.
  - Default local commands use the new path only after parity succeeds; the legacy CSV/DuckDB bronze path remains until the observation window closes.
  - README documents: quick start/idempotent setup, DNS table, data lineage diagram, runtime/infrastructure diagram, framework and infrastructure stack tables, persistence map, trigger/reconciliation model, browser UI directory, and failure/recovery behavior.
  - README states `just destroy` removes the cluster but preserves host data, with a separate explicit purge operation that is destructive.
- **Guardrails**:
  - Do not add a separate ingress controller.
  - Do not use Traefik HTTP ingress for Kafka protocol traffic.
  - Do not expose admin UIs outside the local machine/network.
  - External enrichment stays isolated from durable ingestion so third-party downtime cannot lose accepted source events.
  - Do not delete the legacy local database or direct CSV path until the observation window closes.
  - Do not describe unfinished or unverified features as working.

## Dependencies

1. SeaweedFS and its bucket must be healthy before Kafka Connect starts.
2. Redpanda must be healthy before Connect or Redpanda Console starts.
3. The S3 connector compatibility/licensing spike must pass before adopting its output contract.
4. Connect must produce valid Parquet before dbt changes its bronze source.
5. dbt parity must pass before Prefect schedules or downstream training use the new path.
6. Host-mounted persistence must be in place before stateful services become authoritative.
7. The old ingestion path remains available until parallel-run validation finishes.

## QA / Testing Scenarios

- Produce one valid event from the host script and trace it through HTTP proxy request, Redpanda offset, Connect task, SeaweedFS object, staging row, gold row, Prefect run, and Redpanda Console.
- Publish the same event twice and verify one logical gold row.
- Restart Redpanda, Connect, and SeaweedFS; verify data, offsets, and bronze objects remain and acknowledged events are not lost.
- Replay a topic and verify raw bronze captures replay while staging/gold deduplicate correctly.
- Publish malformed JSON, invalid values, and missing identity fields; verify visible failures.
- Revoke SeaweedFS write credentials temporarily; verify Connect fails visibly and resumes after restoration.
- Fill or nearly fill a test PVC; verify failure is visible and does not silently corrupt output.
- Run dbt twice with no new events and verify identical gold output.
- Trigger training only after dbt tests pass; verify failed dbt runs block training.
- Delete/recreate k3d and verify host-mounted state is preserved (record Redpanda end offsets, Connect offsets, SeaweedFS object checksums, DuckDB checksums, Prefect runs, MLflow runs, registered models before deletion; verify after recreation).
- Apply all YAML and bootstrap Jobs twice; verify no duplicate or destructive changes.
- Block Wikipedia/Hugging Face temporarily and verify durable ingestion continues while only dependent enrichment steps degrade.
- Verify Traefik routes for every browser UI and confirm the Kafka API/Connect admin endpoints remain internal.

## Completion Criteria

- All platform services run as plain Kubernetes manifests in one local k3d server node.
- Redpanda, Kafka Connect, SeaweedFS, Redpanda Console, Prefect, DuckDB/dbt, MLflow, and BentoML operate without required cloud services.
- Kafka Connect writes immutable Parquet bronze data to SeaweedFS; it never writes directly to DuckDB.
- dbt produces baseline-compatible silver/gold tables in persistent local DuckDB; training remains standalone.
- Redpanda Console provides browser visibility into the broker and Connect at `redpanda.macsteve.lan`; host producers ingest through `https://kafka.macsteve.lan` (built-in HTTP proxy).
- Traefik exposes all intended HTTP UIs through predictable local hostnames.
- README documents the verified end-to-end data flow, infrastructure topology, DNS contract, persistence map, idempotent reconciliation, and recovery procedure.
- The cluster can be recreated while preserving host-mounted persistent data.
- Redpanda events, consumer offsets, Connect state, Parquet bronze, DuckDB tables, Prefect state, MLflow experiments/artifacts, and serving inputs all survive cluster deletion and recreation.
- Recovery, backup, resource limits, credentials, external-source failure behavior, and cluster-deletion behavior are documented.
