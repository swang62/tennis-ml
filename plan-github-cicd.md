# Plan: GitHub CI/CD

## Goal

Add one GitHub Actions pipeline that validates every same-repository pull request to `main`, builds both Docker images without publishing on PRs, and, after a successful merge to `main`, publishes the current MLflow `@champion` Bento image plus the web image to Docker Hub as `latest`.

The pipeline will authenticate to DagsHub only to fetch and verify the immutable artifacts pinned by the current champion. It will not train models, mutate MLflow, access PostgreSQL, run Prefect, or deploy to a production host.

## Scope

### Included

- Python lint, type check, and hermetic test suite.
- Web test, type-check/build, and Docker build.
- Authenticated DagsHub/MLflow champion resolution and Bento Docker build.
- Docker Hub push only after a `main` push.
- GitHub-hosted caches for uv, pnpm, Buildx layers, and verified MLflow/Bento materialization.
- Branch-protection and repository-secret setup instructions.

### Explicitly excluded

- Deploying to the Compose host, SSH, Kubernetes, Prefect, database migrations, ETL, scraping, training, or MLflow promotion.
- Fork PR support. Repository settings should disallow/avoid fork-contributed workflow runs; no workflow uses `pull_request_target`.
- Immutable Docker tags. Per decision, main publishes only `latest`.

## Required GitHub Configuration

### Secrets

| Secret | Purpose | Minimum privilege |
| --- | --- | --- |
| `DOCKERHUB_TOKEN` | Push `swang62/tennis-bento:latest` and `swang62/tennis-web:latest` after a main merge. | Docker Hub access token restricted to those repositories with read/write access. |
| `DAGSHUB_TOKEN` | Read the current MLflow champion, its model versions, and lineage-pinned artifacts during Bento builds. | Dedicated CI account/token with read-only access to the DagsHub repository and MLflow artifacts. |

### Repository variables (not secrets)

| Variable | Value |
| --- | --- |
| `DOCKERHUB_USERNAME` | Docker Hub account that owns the two image repositories (currently `swang62`). |
| `DAGSHUB_USERNAME` | Username of the dedicated DagsHub CI account. |
| `MLFLOW_TRACKING_URI` | DagsHub MLflow URL for this repository. |

Map the DagsHub values only within the Bento-build step as `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`; never print them. No `DATABASE_URL`, `POSTGRES_PASSWORD`, `BENTO_API_KEY`, Prefect credential, or Docker Hub password is needed by CI.

Protect `main` by requiring the CI workflow's required check and prohibiting direct pushes. Grant the workflow only `contents: read`; Docker Hub and DagsHub use their dedicated secrets rather than a broad GitHub token.

## Tasks

### [ ] Task 1: Make dependency resolution valid on GitHub-hosted Linux runners

- **Description**: Extend uv's supported environments beyond the current macOS arm64-only resolution, then regenerate `uv.lock` with the Linux x86_64 resolution used by `ubuntu-latest`. Preserve Python 3.12 and the existing dependency-group layout.
- **Files**: `pyproject.toml`, `uv.lock`
- **Acceptance Criteria**:
  - A fresh Ubuntu x86_64 runner can install the `dev` dependency group from the committed lockfile.
  - macOS arm64 local resolution remains supported.
  - No runtime or dependency-group semantics change other than platform support.
- **Guardrails**: Do not introduce a second package manager, Docker-based test database, or CI-only dependency group unless the existing groups cannot install on Linux.

### [ ] Task 2: Separate local publish behavior from CI build-only behavior

- **Description**: Extend the existing Bento deploy entrypoint with one explicit build-only/no-push mode for CI. Keep `just deploy` and the default direct invocation behavior publishing the existing multi-architecture (`linux/amd64,linux/arm64`) `latest` image. In no-push mode, build a single runner-native Linux image with Buildx `--load` so the PR job performs a real container build but cannot publish.
- **Files**: `src/flows/deploy.py`, `justfile` (only if the existing recipe needs to expose the new explicit flag), relevant focused tests under `tests/`
- **Acceptance Criteria**:
  - Default local deploy still logs in through `DOCKER_TOKEN` and pushes the current Docker Hub `latest` Bento image.
  - The CI build-only mode resolves `ensemble_lr_model@champion`, validates all lineage tags/hashes, materializes the same artifacts, and runs a non-publishing Docker build.
  - The push mode retains multi-architecture output; the PR mode neither runs `docker login` nor includes `--push`.
  - Tests cover command construction/mode selection without contacting Docker, MLflow, or a database.
- **Guardrails**: Reuse the existing `build_bento_image`, hash validation, and lineage-pinning path. Do not add a generic CLI wrapper, change model aliases, or perform registry writes.

### [ ] Task 3: Add the unified PR/main GitHub Actions workflow

- **Description**: Create a single workflow triggered by `pull_request` targeting `main` and by pushes to `main`. Use concurrency to cancel superseded PR runs while keeping main builds independent. Pin every third-party action to a full commit SHA.

  Jobs/steps:
  1. Check out the repository, install Python 3.12 and uv, restore uv cache, and install the locked `dev` environment.
  2. Install Node 22/Corepack, restore pnpm cache from `web/pnpm-lock.yaml`, then run Python lint/type/test and web test/build. Run with `DATABASE_URL` unset so the repository's no-live-database contract remains enforced.
  3. Configure Buildx and its GitHub Actions cache backend. Restore a separate Actions cache containing `.bentoml`, `data/deploy`, and the minimal generated deploy state needed to reuse downloaded model material.
  4. Export DagsHub MLflow credentials only for the Bento step. On PRs, call the build-only Bento mode; on `main`, call the normal publishing mode with `DOCKER_TOKEN` mapped from `DOCKERHUB_TOKEN`.
  5. Build the web Docker image on every PR. On `main` only, log in to Docker Hub and push `swang62/tennis-web:latest`.
  6. Upload sanitized Bento build logs on failure. Never upload `.env`, credentials, or unredacted command environments.
- **Files**: `.github/workflows/ci.yml`
- **Acceptance Criteria**:
  - A same-repository PR to `main` runs validation and builds both images without pushing either.
  - A successful push to `main` reruns the same checks and pushes only `swang62/tennis-bento:latest` and `swang62/tennis-web:latest`.
  - Docker Hub login is conditional on main push; DagsHub credentials are masked and scoped to the authenticated Bento step.
  - Workflow uses `pull_request`, never `pull_request_target`, and has no deployment command.
- **Guardrails**: Do not use `latest` as an MLflow model input: the existing champion alias and immutable lineage tags remain the source of truth. Do not add PostgreSQL services to Actions.

### [ ] Task 4: Implement cache keys, invalidation, and integrity boundaries

- **Description**: Use the native setup-action caches for uv and pnpm, GitHub Actions cache backend for each Docker build scope (`bento` and `web`), and a restore/save cache for MLflow/Bento directories. Key the model-material cache by OS, lockfile, Bento/service/feature/model source inputs, and Bentofile inputs, with a conservative restore prefix for reuse across commits.

  Treat every restored MLflow/Bento file as an untrusted accelerator: the existing deploy path must still resolve the champion from DagsHub and verify URI/version metadata plus lineage artifact SHA-256 values before producing an image.
- **Files**: `.github/workflows/ci.yml`, `src/flows/deploy.py` (only where needed to pass Buildx cache parameters and preserve local behavior)
- **Acceptance Criteria**:
  - Repeated runs restore uv, pnpm, Docker layers, and model material without changing the resulting validated champion inputs.
  - A champion change or mismatched artifact hash forces redownload/re-materialization rather than using a stale cache.
  - Web and Bento Buildx layers do not collide.
  - Cache paths omit `.env`, logs, database files, and credential-bearing configuration.
- **Guardrails**: Model caches are intentionally accessible to repository workflow users, per the confirmed decision. Do not treat cache keys as authorization or skip remote lineage validation on a cache hit.

### [ ] Task 5: Document CI/CD operations and harden action upkeep

- **Description**: Add a concise CI/CD section explaining the triggers, no-fork constraint, registry behavior, required repository variables/secrets, cache contents, and that a main image publication does not deploy the Compose host. Add Dependabot configuration for GitHub Actions updates if the repository does not already manage action-version updates elsewhere.
- **Files**: `README.md`, `.github/dependabot.yml` (if absent)
- **Acceptance Criteria**:
  - A maintainer can configure the two secrets and three variables from the documentation without copying any secret into the repository.
  - Documentation states `latest` is mutable and deployment remains a separate/manual concern.
  - Action updates remain reviewable as grouped Dependabot PRs.
- **Guardrails**: Keep documentation operational and brief; do not alter local Compose, Prefect, or production-host instructions beyond accurately distinguishing them from CI.

## Dependencies

1. Task 1 must finish before a Linux-hosted Python CI job can install reproducibly.
2. Task 2 must finish before a PR can build the Bento without publishing it.
3. Task 3 depends on Tasks 1 and 2.
4. Task 4 is implemented with Task 3 and verified after the full workflow exists.
5. Task 5 follows the final workflow/secret names.

## QA/Testing Scenarios

1. Same-repository PR changing Python source: lint, type check, hermetic tests, web tests/build, authenticated champion artifact verification, and both Docker builds succeed; Docker Hub has no new image.
2. Same-repository PR with a broken frontend Docker build: workflow fails before merge and does not publish.
3. Main merge: both images publish only as `latest`; no SSH, Compose, Prefect, or deployment command runs.
4. Cached run: cache hits reduce downloads/layer rebuilding, but DagsHub is still queried and every cached lineage artifact is hash-verified.
5. Champion changes between runs: the workflow rejects stale cache inputs and builds from the newly resolved champion.
6. Missing/invalid DagsHub token or missing champion lineage tag: Bento job fails clearly, does not push an image, and exposes no secret in logs.
7. Python tests pass with `DATABASE_URL` absent and without a PostgreSQL service, preserving `tests/test_no_live_db.py` guarantees.
