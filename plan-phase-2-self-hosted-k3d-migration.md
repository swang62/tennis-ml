# Plan: Phase 2 self-hosted single-node k3d data platform

## Goal

After the existing dbt migration is complete, run the entire tennis-ml data and ML platform locally inside the existing single-node k3d cluster, without Helm charts or SaaS infrastructure dependencies.

```text
CSV/event producer
       |
       v
Kafka (single KRaft broker)
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
local MLflow                   dashboard / training
       |
       v
BentoML production service

Browser visibility through Traefik:
Prefect UI | MLflow UI | Redpanda Console | SeaweedFS Admin | Dashboard | BentoML
```

Local DNS contract (the user manages DNS so every hostname resolves to the k3d/Traefik address):

| Hostname | Traefik destination |
|---|---|
| `prefect.macsteve.lan` | Prefect UI/API |
| `mlflow.macsteve.lan` | MLflow UI/API |
| `kafka.macsteve.lan` | Redpanda Console |
| `storage.macsteve.lan` | SeaweedFS Admin UI |
| `s3.macsteve.lan` | SeaweedFS S3 API |
| `dashboard.macsteve.lan` | Tennis dashboard |
| `bento.macsteve.lan` | BentoML prediction API |

Kafka protocol and Kafka Connect REST remain internal ClusterIP services; `kafka.macsteve.lan` is the Redpanda browser console, not the Kafka bootstrap endpoint.

## Self-hosted definition

“Fully local” means all durable platform services and orchestration run in k3d:

- Kafka, Kafka Connect, SeaweedFS, DuckDB, Prefect, MLflow, dbt, training, dashboard, Redpanda Console, and BentoML are self-hosted.
- Confluent Cloud, MotherDuck, Prefect Cloud, Databricks, DagsHub, or another SaaS control/data plane is not required.
- Internet access remains allowed for source-data collection, Wikipedia enrichment, Hugging Face/FastEmbed downloads, container images, package installation, and updates.
- A temporary internet or third-party API failure can delay the affected enrichment/download step, but must not corrupt locally persisted platform data.
- Existing data persists across pod restarts. A host-mounted k3d data root preserves authoritative state across cluster recreation.

## Dependency on Phase 1

Do not begin implementation until `plan-migrate-to-dbt.md` is complete and stable.

Phase 2 entry gate:

- `gold.match_features` is built by dbt rather than inline SQL in `src/flows/etl.py`.
- `just db-etl` invokes the dbt-backed Prefect path.
- dbt tests pass against `data/tennis.duckdb`.
- dbt output matches the captured legacy ETL baseline.
- The first training notebook still reads `gold.match_features` successfully.
- Phase 1 work is committed or otherwise stable so Phase 2 does not conflict with the other agent.

## Scope

### In scope

- Existing one-server, zero-agent k3d cluster from `infra/k3d/config.yaml`.
- Plain Kubernetes manifests applied with `kubectl`; no Helm charts or operators.
- One combined Kafka broker/controller using KRaft; no ZooKeeper.
- One Kafka Connect distributed-mode worker.
- An S3 sink connector that writes Parquet to SeaweedFS.
- Single-container SeaweedFS `weed mini` for local S3-compatible object storage.
- Redpanda Console for Kafka topics, messages, consumer groups, and Kafka Connect visibility.
- Existing local Prefect server/worker, MLflow, dashboard, and BentoML services.
- Local DuckDB for dbt silver/gold tables.
- Traefik host-based ingress for HTTP browser UIs.
- A host-mounted k3d data root so state survives cluster recreation.
- Internet-backed source ingestion and enrichment remain supported.
- Persistent volumes, resource limits, health checks, startup ordering, backups, and end-to-end verification.

### Out of scope

- Confluent Cloud, MotherDuck, Prefect Cloud, or another required SaaS dependency.
- Helm, Strimzi, Confluent Operator, or another Kubernetes operator.
- Multi-broker Kafka, replication, high availability, autoscaling, or production SLAs.
- Delta Lake; Parquet is the bronze format.
- Schema Registry, Avro, Protobuf, Flink, Spark, or ksqlDB.
- Cloud GPU training.
- Globally accessible production serving.
- Air-gapped operation without internet access.

## Architecture decisions

- Kafka runs as a one-replica StatefulSet in combined broker/controller KRaft mode.
- Kafka Connect runs in distributed mode even with one worker. Connector configuration, status, and offsets live in Kafka internal topics with replication factor one, avoiding a separate Connect offset volume.
- Kafka and Connect stay internal ClusterIP services. Host CLI access uses `kubectl port-forward`; Traefik HTTP ingress is not used for the Kafka TCP protocol.
- SeaweedFS runs as one StatefulSet using `weed mini`, one persistent volume, S3 endpoint `8333`, and Admin UI `23646`. It is a development object store, not a highly available storage system.
- The bronze bucket stores immutable Parquet objects. dbt owns casting, validation, deduplication, and silver/gold materialization.
- Redpanda Console is the only Kafka browser UI. Confluent Control Center, AKHQ, Kafdrop, and the abandoned Provectus Kafka UI are not added.
- The existing local DuckDB database remains the analytical database. Kafka Connect must not write directly into the `.duckdb` file.
- Kubernetes Secrets contain runtime credentials. Git tracks only required key names and setup instructions, never actual secret values.
- Images, plugins, and Python dependencies are version-pinned for reproducibility, but may be downloaded during setup and upgrades.
- A host directory mounted into the k3d server backs Kafka, SeaweedFS, DuckDB, Prefect, and MLflow state so `k3d cluster delete` does not silently delete platform data.
- Kafka KRaft data includes the event log, consumer offsets, Connect configuration/status/offset topics, broker metadata, and stable cluster identity; all must survive cluster deletion and recreation.
- Configuration is declarative and Git-tracked as Kubernetes YAML, dbt YAML/SQL, or connector configuration templates. Mutable data belongs only in SeaweedFS or stable host-backed volumes.
- Bootstrap Jobs reconcile desired state idempotently: create missing topics/buckets/connectors/schemas, update compatible configuration, and never truncate or recreate existing durable data.

## Tasks

### [ ] Task 1: Validate Phase 1 and capture the local baseline

- **Description**: Confirm the completed dbt migration before modifying infrastructure. Record bronze/gold counts, schemas, deterministic checksums, dbt test output, and representative notebook/dashboard queries.
- **Files**:
  - `plan-migrate-to-dbt.md`
  - `models/gold/match_features.sql`
  - `models/gold/match_features.yml`
  - `models/sources.yml`
  - `src/flows/etl.py`
- **Acceptance Criteria**:
  - All Phase 1 acceptance criteria pass.
  - Baseline output is reproducible from the seed database.
  - No unresolved Phase 1 changes are overwritten.
- **Guardrails**:
  - Do not change feature calculations during the infrastructure migration.
  - Do not edit Phase 1 files until the other agent has finished.

### [ ] Task 2: Define resource, storage, image, and ingress budgets

- **Description**: Inventory current k3d workloads and reserve capacity for Kafka, Connect, SeaweedFS, and Redpanda Console. Pin every container image and define persistent-volume sizes, internal ports, browser hostnames, health probes, and startup dependencies before adding manifests.
- **Files**:
  - `infra/k3d/config.yaml`
  - `infra/manifests/default/config-map.yaml`
  - `infra/manifests/default/ingress.yaml`
  - `README.md`
- **Acceptance Criteria**:
  - Planned additions fit within a documented host budget, targeting approximately 4 GiB additional RAM and leaving capacity for training outside the cluster.
  - Images use immutable versions or digests and `IfNotPresent` after local import.
  - Hostnames are assigned for Redpanda Console, SeaweedFS S3/Admin endpoints, and existing Prefect, MLflow, dashboard, and BentoML services without conflicts.
  - One host-mounted data root is mapped into the k3d server and assigned stable subdirectories for Kafka, SeaweedFS, DuckDB, Prefect, and MLflow.
  - Stateful data survives pod restart and a tested cluster delete/recreate cycle.
  - Kafka's KRaft cluster identity and `meta.properties` are preserved rather than regenerated over existing logs.
  - Static host-backed PV/PVC definitions use stable paths and a `Retain` lifecycle; reapplying manifests rebinds to the same data.
- **Guardrails**:
  - Do not add Helm repositories, charts, or operators.
  - Do not expose Kafka or Connect publicly through HTTP ingress.
  - Do not store persistent service data only inside the disposable k3d server container.
  - `just setup` and repeated `kubectl apply` operations must reconcile, not reset, storage.

### [ ] Task 3: Add single-container SeaweedFS object storage

- **Description**: Deploy one SeaweedFS StatefulSet running `weed mini -dir=/data`, with one PVC, resource limits, startup/readiness probes, S3 credentials, and the `tennis-bronze` bucket created through supported environment configuration. Expose the Admin UI through Traefik while keeping the S3 endpoint available primarily through its internal ClusterIP service.
- **Files**:
  - `infra/manifests/default/seaweedfs.yaml`
  - `infra/manifests/default/ingress.yaml`
  - `infra/manifests/default/config-map.yaml`
  - `justfile`
  - `README.md`
- **Acceptance Criteria**:
  - SeaweedFS restarts without losing bucket contents.
  - `weed mini` initializes credentials and `tennis-bronze` idempotently.
  - A test Parquet object can be uploaded, listed, downloaded, and read with local DuckDB through `http://seaweedfs:8333`.
  - Credentials come from a Kubernetes Secret created outside Git.
  - SeaweedFS Admin UI loads through Traefik.
  - Resource usage is measured with a target limit no larger than 512 MiB unless testing proves that insufficient.
- **Guardrails**:
  - Single-node/single-drive SeaweedFS is for local development only.
  - Do not enable anonymous bucket writes.
  - Do not store credentials in ConfigMaps or tracked connector JSON.

### [ ] Task 4: Add one-node Kafka in KRaft mode

- **Description**: Deploy one Kafka StatefulSet in combined broker/controller KRaft mode with a headless service, internal ClusterIP bootstrap service, PVC, bounded JVM heap, resource requests/limits, and health probes. Configure development-safe topic retention and auto-topic behavior explicitly.
- **Files**:
  - `infra/manifests/default/kafka.yaml`
  - `infra/manifests/default/config-map.yaml`
  - `justfile`
  - `README.md`
- **Acceptance Criteria**:
  - Kafka starts without ZooKeeper.
  - Topic `tennis.match_events` can be created, listed, produced to, and consumed from inside the cluster.
  - Broker events, consumer offsets, Connect internal topics, and broker identity survive pod restart and cluster recreation.
  - Internal topics and development topics use replication factor one.
  - Host CLI access works through a documented port-forward.
- **Guardrails**:
  - Do not represent this as highly available or production-grade Kafka.
  - Do not add multiple brokers simply to imitate production.
  - Keep the service internal to the host/cluster.

### [ ] Task 5: Build the Kafka Connect image

- **Description**: Build one pinned Kafka Connect image containing the selected S3 sink connector and all Parquet dependencies. Verify connector licensing, SeaweedFS endpoint override support, path-style access, and Parquet output before adopting it. Import the finished image into k3d for repeatable local deployment.
- **Files**:
  - `infra/kafka-connect/Dockerfile`
  - `infra/kafka-connect/connector.json`
  - `infra/manifests/default/kafka-connect.yaml`
  - `justfile`
  - `README.md`
- **Acceptance Criteria**:
  - `GET /connector-plugins` reports the S3 sink connector after deployment.
  - Connector can authenticate to SeaweedFS through the internal service name.
  - Connector writes valid Parquet rather than JSON files with a Parquet extension.
  - Image build pins the plugin version and does not mutate the container at pod startup.
  - License and redistribution restrictions are documented.
- **Guardrails**:
  - Do not rely on `confluent-hub install` at runtime.
  - Do not write a custom Kafka consumer unless the user explicitly rejects every compatible Connect sink.
  - Run a compatibility spike before building later tasks around the connector.

### [ ] Task 6: Deploy one distributed Kafka Connect worker and S3 sink

- **Description**: Deploy one Connect worker in distributed mode, configure its three internal Kafka topics with replication factor one, expose its REST API only inside the cluster, and register an idempotent S3 sink connector targeting SeaweedFS. Batch low-volume events to avoid excessive small Parquet files.
- **Files**:
  - `infra/manifests/default/kafka-connect.yaml`
  - `infra/kafka-connect/connector.json`
  - `infra/manifests/default/config-map.yaml`
  - `justfile`
- **Acceptance Criteria**:
  - Worker and connector state survive Connect pod restart via Kafka internal topics.
  - Connector resumes committed offsets without dropping acknowledged events.
  - Records land under a deterministic bronze prefix partitioned by event date or another documented stable convention.
  - Connect reports authentication, serialization, and SeaweedFS failures visibly.
  - Connector registration can be rerun without creating duplicates.
- **Guardrails**:
  - Do not use standalone mode or a local Connect offset file.
  - Do not expose the Connect REST API through public ingress.
  - Prefer larger low-frequency batches over one Parquet file per event.
  - Connector reconciliation must use idempotent create-or-update behavior and must not reset consumer offsets.

### [ ] Task 7: Add Redpanda Console for Kafka and Connect visibility

- **Description**: Deploy Redpanda Console as one stateless Deployment configured for the Kafka bootstrap service and Kafka Connect REST endpoint. Expose it through Traefik using the existing local hostname pattern.
- **Files**:
  - `infra/manifests/default/redpanda-console.yaml`
  - `infra/manifests/default/ingress.yaml`
  - `infra/manifests/default/config-map.yaml`
  - `README.md`
- **Acceptance Criteria**:
  - Browser UI lists topics, partitions, messages, consumer groups, lag, and Connect connector status.
  - A user can inspect `tennis.match_events` and the bronze sink without using CLI tools.
  - Console starts reliably from its pinned image.
  - Console has read/write permissions appropriate for local development and is not exposed beyond the local ingress.
- **Guardrails**:
  - Do not deploy AKHQ, Kafdrop, Provectus Kafka UI, or Confluent Control Center alongside it.
  - Treat Console as visibility/admin tooling, not the owner of connector configuration.

### [ ] Task 8: Define the event contract and local producer path

- **Description**: Define a versioned JSON contract for validated match events and extend ingestion so existing CSV rows can be published to Kafka. Use a deterministic key based on match/player identity and include event metadata required for traceability and deduplication.
- **Files**:
  - `src/flows/ingest.py`
  - `src/features/validate.py`
  - `pyproject.toml`
  - `tests/`
  - `README.md`
- **Acceptance Criteria**:
  - Every `EXPECTED_COLUMNS` field has a documented type and nullability rule.
  - Invalid events are rejected before publish.
  - Re-publishing the same CSV produces the same logical event keys.
  - Delivery failures are surfaced rather than silently ignored.
  - Focused tests cover serialization, validation, and key stability.
  - Existing Wikipedia enrichment continues to work through its current HTTP path with explicit timeout/retry behavior.
- **Guardrails**:
  - Use JSON; do not introduce Schema Registry, Avro, or Protobuf.
  - Keep the direct CSV bootstrap path until end-to-end parity passes.
  - Wikipedia failures must be reported or skipped deterministically without blocking durable Kafka ingestion.

### [ ] Task 9: Make SeaweedFS Parquet the dbt bronze source

- **Description**: Replace the production bronze-table source with external Parquet files in SeaweedFS. Add a staging model that casts connector output to the current typed bronze contract, retains Kafka metadata, and deduplicates replayed/retried events before `gold.match_features` runs.
- **Files**:
  - `models/sources.yml`
  - `models/staging/` (new staging model and tests)
  - `models/gold/match_features.sql`
  - `dbt_project.yml`
  - `profiles.yml`
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

### [ ] Task 10: Containerize Prefect execution for dbt and training orchestration

- **Description**: Replace the generic Prefect worker image with a pinned project worker image containing the repository package, dbt-duckdb, Papermill, and required runtime dependencies. Mount persistent DuckDB and pipeline artifact storage. Keep Prefect server and worker fully local.
- **Files**:
  - `infra/images/worker.Dockerfile` (or project-equivalent path)
  - `infra/manifests/default/prefect-worker.yaml`
  - `infra/manifests/default/config-map.yaml`
  - `src/flows/etl.py`
  - `src/flows/training.py`
  - `src/flows/pipeline.py`
  - `justfile`
- **Acceptance Criteria**:
  - Prefect schedules and runs the producer, dbt build, tests, and training flow inside the local cluster.
  - dbt writes silver/gold to the persistent local DuckDB database.
  - Worker image contains its pinned runtime dependencies rather than installing them during each flow run.
  - FastEmbed/Hugging Face assets may download on first use and then reuse a persistent local cache.
  - Flow retries do not cause duplicate logical gold rows.
  - Training can be manually triggered after a successful dbt run.
- **Guardrails**:
  - Do not use Prefect Cloud.
  - Do not run Kafka or Kafka Connect as Prefect flow tasks.
  - Keep heavyweight training dependencies out of unrelated service images.

### [ ] Task 11: Integrate existing MLflow, dashboard, and BentoML services

- **Description**: Preserve the existing local MLflow, dashboard, and BentoML architecture while switching their data dependencies to the Parquet/dbt/DuckDB path. Ensure model artifacts required for serving remain accessible after pod restart and that no service requires MotherDuck or another cloud endpoint.
- **Files**:
  - `infra/manifests/default/mlflow.yaml`
  - `infra/manifests/deploy/dashboard.yaml`
  - `infra/manifests/deploy/bentoml.yaml`
  - `src/db/client.py`
  - `src/dashboard/app.py`
  - `src/serving/service.py`
  - `bentofile.yaml`
- **Acceptance Criteria**:
  - Training logs experiments and artifacts to local MLflow.
  - Approved model can be built and loaded by BentoML from local MLflow artifacts.
  - Dashboard reads the persistent local gold table.
  - Existing browser endpoints remain accessible through Traefik.
- **Guardrails**:
  - Do not move MLflow artifacts to SeaweedFS unless the existing PVC proves insufficient; reuse current working storage first.
  - Do not redesign models, dashboard UX, or serving APIs during infrastructure migration.

### [ ] Task 12: Consolidate Traefik ingress and local observability

- **Description**: Extend the existing Traefik ingress so all HTTP UIs have predictable local hostnames. Add Kubernetes health probes, resource observations, logs, and operational commands for each service. Keep Kafka TCP and Connect REST internal.
- **Files**:
  - `infra/manifests/default/ingress.yaml`
  - `infra/manifests/deploy/ingress.yaml`
  - `infra/manifests/default/config-map.yaml`
  - `justfile`
  - `README.md`
- **Acceptance Criteria**:
  - `prefect.macsteve.lan`, `mlflow.macsteve.lan`, `kafka.macsteve.lan`, `storage.macsteve.lan`, `s3.macsteve.lan`, `dashboard.macsteve.lan`, and `bento.macsteve.lan` route through Traefik to the documented services.
  - Readiness prevents dependent jobs from starting before Kafka, Connect, or SeaweedFS are usable.
  - Operators can inspect Kafka lag, connector state, SeaweedFS objects, Prefect runs, MLflow runs, and pod resource usage.
  - No internal credential is rendered in a browser UI or pod log.
- **Guardrails**:
  - Do not add a separate ingress controller.
  - Do not use Traefik HTTP ingress for Kafka protocol traffic.
  - Do not expose admin UIs outside the local machine/network.

### [ ] Task 13: Rewrite README for the complete self-hosted platform

- **Description**: Replace the outdated CSV-to-local-DuckDB overview with documentation for the completed Kafka/Parquet/dbt/ML pipeline. Keep the README operational and visually scannable, with Mermaid diagrams that render on GitHub and plain-text fallbacks where commands depend on local tooling.
- **Files**:
  - `README.md`
  - `AGENTS.md`
- **Required README Sections**:
  - System purpose and self-hosted constraints.
  - Quick start, idempotent setup/reapply, stop, restart, cluster delete/recreate, backup, and explicit data purge commands.
  - Local DNS table listing every `*.macsteve.lan` hostname and stating that the user configures all records to point at Traefik.
  - End-to-end data lineage diagram: source/CSV → producer → Kafka → Connect → SeaweedFS Parquet bronze → dbt staging/gold → DuckDB → training → MLflow → BentoML/dashboard.
  - Runtime/infrastructure diagram: host → k3d → Traefik → UIs/services → host-backed persistence.
  - Separate framework/application stack table: Prefect, dbt, DuckDB, Papermill, MLflow, BentoML, Panel/dashboard, model libraries.
  - Separate infrastructure/platform stack table: k3d/k3s, Kubernetes manifests, Traefik, Kafka KRaft, Kafka Connect, SeaweedFS, Redpanda Console, PV/PVC storage.
  - Persistence map showing every stateful component, data path/type, owning PVC/host directory, backup method, and cluster-recreation behavior.
  - Trigger/reconciliation model showing which components are continuous services, Prefect flows, bootstrap Jobs, and manual operations.
  - Browser UI directory describing what each UI is for.
  - Failure and recovery behavior, including third-party source outages and full k3d recreation.
- **Acceptance Criteria**:
  - Diagrams agree with deployed manifests and actual service names.
  - Every documented command is executed successfully during final verification.
  - README clearly distinguishes `kafka.macsteve.lan` (Redpanda Console) from the internal Kafka bootstrap service.
  - README states that `just destroy` removes the cluster but preserves host data, while a separate explicit purge operation is destructive.
  - README explains that YAML/config templates are declarative configuration while event logs, Parquet, databases, artifacts, and state live only in SeaweedFS or host-backed PVs.
- **Guardrails**:
  - Do not describe unfinished or unverified features as working.
  - Do not include secrets, machine-specific credentials, or destructive cleanup commands without a prominent warning.
  - Keep diagrams focused; use separate data-flow and infrastructure diagrams rather than one unreadable graph.

### [ ] Task 14: Prove self-hosted operation, cluster recreation, recovery, and cut over

- **Description**: Restart every pod, run the complete pipeline, then delete and recreate the cluster while preserving host-mounted data. Document backup, restore, restart, external-source failure, and disaster behavior before making the Kafka/Parquet path the default.
- **Files**:
  - `infra/k3d/config.yaml`
  - `infra/k3d/start.sh`
  - `justfile`
  - `README.md`
  - `AGENTS.md`
- **Acceptance Criteria**:
  - Seeded CSV events flow through Kafka, Connect, SeaweedFS Parquet, dbt, DuckDB, training, MLflow, and BentoML.
  - Redpanda Console and all browser UIs remain available.
  - Kafka/SeaweedFS/DuckDB/MLflow state survives pod restart.
  - Host-mounted Kafka/SeaweedFS/DuckDB/Prefect/MLflow state survives cluster deletion and recreation.
  - Events produced before cluster deletion remain consumable afterward at their original topic/partition offsets.
  - Connect resumes from preserved internal offsets and does not republish the entire Kafka history into duplicate Parquet objects.
  - Reapplying all YAML and bootstrap Jobs repeatedly produces the same desired configuration without deleting topics, buckets, tables, experiments, or model artifacts.
  - Temporary Wikipedia/Hugging Face/network failure does not damage or roll back locally persisted Kafka, Parquet, DuckDB, Prefect, or MLflow state.
  - Backup and restore procedures are tested for Kafka PVC, SeaweedFS PVC, DuckDB file, Prefect DB, and MLflow state.
  - Default local commands use the new path only after parity succeeds.
- **Guardrails**:
  - External enrichment remains isolated from durable ingestion so third-party downtime cannot lose accepted source events.
  - Do not delete the legacy local database or direct CSV path until the observation window closes.
  - `just destroy` must warn that host-mounted data remains and provide a separate, explicitly destructive cleanup procedure; never delete it implicitly.

## Dependencies

1. Phase 1 dbt migration must pass before Phase 2 begins.
2. SeaweedFS and its bucket must be healthy before Kafka Connect starts.
3. Kafka must be healthy before Connect or Redpanda Console starts.
4. The S3 connector compatibility/licensing spike must pass before adopting its output contract.
5. Connect must produce valid Parquet before dbt changes its bronze source.
6. dbt parity must pass before Prefect schedules or downstream training use the new path.
7. Host-mounted persistence must be in place before stateful services become authoritative.
8. The old ingestion path remains available until parallel-run validation finishes.

## QA / Testing Scenarios

- Produce one valid event and trace it through Kafka offset, Connect task, SeaweedFS object, staging row, gold row, Prefect run, and Redpanda Console.
- Publish the same event twice and verify one logical gold row.
- Restart Kafka and verify topic data and committed offsets remain.
- Restart Connect between consumption and Parquet flush and verify acknowledged events are not lost.
- Restart SeaweedFS and verify existing bronze objects remain readable.
- Replay a Kafka topic and verify raw bronze captures replay while staging/gold deduplicate correctly.
- Publish malformed JSON, invalid surface/round/ranking, and missing identity fields; verify visible failures.
- Revoke SeaweedFS write credentials temporarily; verify Connect fails visibly and resumes after restoration.
- Fill or nearly fill a test PVC; verify failure is visible and does not silently corrupt output.
- Run dbt twice with no new events and verify identical gold output.
- Trigger training only after dbt tests pass; verify failed dbt runs block training.
- Restart cluster workloads, ingest seeded events, transform, train, register, and serve a prediction.
- Delete/recreate k3d and verify host-mounted service state is preserved.
- Before deletion, record Kafka end offsets, Connect offsets, SeaweedFS object checksums, DuckDB checksums, Prefect runs, MLflow runs, and registered models; verify them after recreation.
- Apply all YAML and bootstrap Jobs twice; verify the second application produces no duplicate resources, events, Parquet data, or destructive changes.
- Block Wikipedia/Hugging Face temporarily and verify durable ingestion continues while only dependent enrichment steps degrade.
- Verify Traefik routes for every browser UI and confirm Kafka/Connect admin endpoints remain internal.

## Completion Criteria

- All platform services run as plain Kubernetes manifests in one local k3d server node.
- Kafka, Kafka Connect, SeaweedFS, Redpanda Console, Prefect, DuckDB/dbt, MLflow, dashboard, and BentoML operate without required cloud services.
- Kafka Connect writes immutable Parquet bronze data to SeaweedFS; it never writes directly to DuckDB.
- dbt produces baseline-compatible silver/gold tables in persistent local DuckDB.
- Redpanda Console provides browser visibility into Kafka and Connect.
- Traefik exposes all intended HTTP UIs through predictable local hostnames.
- README documents the verified end-to-end data flow, infrastructure topology, DNS contract, persistence map, idempotent reconciliation, and recovery procedure.
- The complete pipeline runs without Confluent Cloud, MotherDuck, Prefect Cloud, Databricks, DagsHub, or another SaaS platform dependency.
- The cluster can be recreated while preserving host-mounted persistent data.
- Kafka events, consumer offsets, Connect state, Parquet bronze, DuckDB tables, Prefect state, MLflow experiments/artifacts, and serving inputs all survive cluster deletion and recreation.
- Recovery, backup, resource limits, credentials, external-source failure behavior, and cluster-deletion behavior are documented.
