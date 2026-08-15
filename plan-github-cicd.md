# Plan: GitHub CI/CD

## Goal

Add one GitHub Actions workflow that validates same-repository pull requests to `main`, builds the web image without publishing on PRs, and, after successful validation on a push to `main`, builds and publishes the current MLflow `@champion` Bento image plus the corresponding web image to Docker Hub as `latest`.

`just deploy` remains a fully independent manual build-and-publish path. GitHub Actions does not deploy the Compose host, run Prefect, train/promote models, or modify MLflow.

## Scope

### Included

- Python lint, type check, and hermetic test suite on PRs and `main`.
- Web tests and a non-publishing web Docker build on PRs.
- Main-only champion resolution, artifact materialization, Bento multi-architecture build/push, and web image build/push.
- GitHub-hosted caching only: native uv/pnpm caches, BuildKit's GitHub Actions cache backend, and a bounded Actions cache for reusable MLflow/Bento material.
- Docker Hub publication only for deployable images, never as a BuildKit cache registry.
- Repository-secret, permissions, cache-limit, branch-protection, and manual-release documentation.

### Explicitly excluded

- Bento image builds on PRs: those require trusted DagsHub/MLflow and production PostgreSQL access.
- Registry cache exports/imports (`type=registry`) or a separate Docker cache image/tag.
- Fork PR support, `pull_request_target`, SSH, Compose deployment, migrations, ETL, scraping, training, model promotion, or immutable Docker release tags.

## Required GitHub Configuration

### Secrets

| Secret | Purpose | Minimum privilege |
| --- | --- | --- |
| `DOCKERHUB_TOKEN` | Push `swang62/tennis-bento:latest` and `swang62/tennis-web:latest` on `main`. | Docker Hub token limited to those repositories with read/write access. |
| `DAGSHUB_TOKEN` | Resolve the champion and download its immutable lineage-pinned artifacts. | Dedicated read-only DagsHub/MLflow account. |
| `PRODUCTION_DATABASE_URL` | Read the canonical player directory while generating the production web artifact on `main`. | Read-only PostgreSQL credentials, network-restricted to the required tables. |

### Repository variables

| Variable | Value |
| --- | --- |
| `DOCKERHUB_USERNAME` | Docker Hub account that owns the image repositories (currently `swang62`). |
| `DAGSHUB_USERNAME` | Dedicated DagsHub CI account username. |
| `MLFLOW_TRACKING_URI` | DagsHub MLflow URL for this repository. |

Map credentials only into the steps requiring them. Never print them or cache `.env`, credential files, logs, database files, or generated environment configuration.

### Cache policy and cost

- Use BuildKit `type=gha` with separate `scope=web` and `scope=bento`; no Docker registry is used for cache storage.
- `type=gha` is GitHub Actions cache storage. It is free for public repositories; private repositories consume the account's included Actions minutes/cache storage and are subject to GitHub's cache quota (10 GB by default unless repository settings change).
- Use native setup-action caches keyed by `uv.lock` and `web/pnpm-lock.yaml`. Use a bounded `actions/cache` entry only for selected MLflow/Bento material that is safe to reuse after remote lineage/hash validation.
- Retain cache keys scoped by OS and image/component. Cache misses must fall back to a correct clean build; cache hits must never suppress champion lookup or artifact integrity checks.

## Tasks

### [ ] Task 1: Make locked Python dependencies installable on GitHub-hosted Linux runners

- **Description**: Extend uv's supported environments beyond the current macOS arm64-only resolution, then regenerate `uv.lock` with the Linux x86_64 resolution used by `ubuntu-latest`. Preserve Python 3.12 and the existing dependency-group layout.
- **Files**: `pyproject.toml`, `uv.lock`
- **Acceptance Criteria**:
  - A fresh Ubuntu x86_64 runner installs the committed `dev` environment from the lockfile.
  - macOS arm64 local resolution remains supported.
  - Dependency-group semantics do not otherwise change.
- **Guardrails**: Do not add a second package manager, Docker database, or CI-only dependency group unless the existing groups cannot install on Linux.

### [ ] Task 2: Add the unified PR/main GitHub Actions workflow

- **Description**: Create `.github/workflows/ci.yml`, triggered by `pull_request` targeting `main` and by pushes to `main`. Use concurrency to cancel superseded PR runs while keeping `main` publication runs independent. Pin every third-party action to a full commit SHA.

  PR path:
  1. Check out the repository; install Python 3.12/uv and Node 22/Corepack.
  2. Restore uv and pnpm dependency caches; install from the committed lockfiles.
  3. Run Python lint/type/hermetic tests with `DATABASE_URL` unset, plus web tests/type-check.
  4. Create a minimal deterministic `web/public/player-directory.json` test fixture only inside the runner, then build the web Docker image without login, `--push`, or production credentials.

  Main path:
  1. Repeat all validation above.
  2. Configure Buildx and expose the GitHub Actions runtime variables required by the `type=gha` backend to the existing command-driven Bento build.
  3. Export the DagsHub credentials, `PRODUCTION_DATABASE_URL`, and Docker Hub token only to `src/flows/deploy.py`; run its existing path to generate the real directory artifact, resolve the champion, build the multi-architecture Bento image, and publish it.
  4. Build and push `swang62/tennis-web:latest` from the artifact that `deploy.py` generated. Do not regenerate or replace that artifact between Bento and web builds.
  5. Upload sanitized build logs only on failure.
- **Files**: `.github/workflows/ci.yml`
- **Acceptance Criteria**:
  - Same-repository PRs run all validation and a web image build, without Docker Hub, DagsHub, or PostgreSQL credentials and without publishing either image.
  - Successful `main` pushes publish exactly `swang62/tennis-bento:latest` and `swang62/tennis-web:latest`.
  - The main path uses the real current champion and real generated player directory; neither image is built from the PR fixture.
  - `just deploy` still works unchanged for an on-demand manual publish.
  - The workflow uses `pull_request`, never `pull_request_target`, and has no host deployment command.
- **Guardrails**: Do not add PostgreSQL as a GitHub Actions service. Do not use `latest` as an MLflow input: the existing champion alias and immutable lineage tags remain the source of truth.

### [ ] Task 3: Wire GitHub-hosted caches without a registry cache

- **Description**: Configure uv and pnpm caches through their setup actions. Configure Buildx cache import/export as `type=gha,scope=web` for the web build and `type=gha,scope=bento` for the main-only Bento build, both with `mode=max`. Use the GitHub Actions runtime helper only as needed to make the GHA cache backend available to the existing raw `docker buildx` invocation from `deploy.py`.

  Add a deliberately bounded `actions/cache` entry for reusable MLflow/Bento material only if profiling confirms its contents fit the repository's Actions cache budget and it materially avoids downloads. Key it from OS plus the relevant Bento/service/feature/model and lockfile inputs; restore with a conservative prefix. The deploy path must always re-resolve `@champion` and verify immutable lineage metadata and SHA-256 hashes before reuse.
- **Files**: `.github/workflows/ci.yml`, `src/flows/deploy.py` (only if it needs narrowly scoped optional `type=gha` Buildx flags), focused tests under `tests/` if deploy command construction changes
- **Acceptance Criteria**:
  - Repeated PRs reuse the `web` BuildKit cache; repeated main runs reuse independent `bento` and `web` caches.
  - No `--cache-from type=registry`, `--cache-to type=registry`, cache image/tag, or Docker Hub cache repository exists.
  - Cache use is safe on a clean runner and does not require a persistent Buildx builder.
  - A stale or mismatched cached model artifact is rejected and rematerialized after remote lineage/hash checks.
  - Cache paths exclude secrets, `.env`, logs, and database files.
- **Guardrails**: Do not assume BuildKit cache mounts are exportable as ordinary Actions caches; dependency layers remain cacheable through their lockfile-first Dockerfile ordering. Do not retain an oversized model cache that thrashes GitHub's cache quota.

### [ ] Task 4: Document CI/CD operations and action maintenance

- **Description**: Add a concise CI/CD section describing PR vs. main behavior, required secrets/variables, Docker Hub publication, GitHub-only cache behavior and quota caveat, and the fact that publishing does not deploy the Compose host. Document `just deploy` as the retained independent manual release path. Add Dependabot configuration for GitHub Actions updates if absent.
- **Files**: `README.md`, `.github/dependabot.yml` (if absent)
- **Acceptance Criteria**:
  - A maintainer can configure the secrets and variables without committing a credential.
  - Documentation distinguishes PR fixture builds from main production builds and distinguishes image publication from host deployment.
  - Documentation states that `latest` is mutable and manual deploy remains supported.
  - Action updates are reviewable as Dependabot PRs.
- **Guardrails**: Keep documentation operational and brief; do not alter Compose, Prefect, or production-host setup beyond accurately distinguishing it from CI.

## Dependencies

1. Task 1 must finish before a Linux-hosted Python CI job can install reproducibly.
2. Task 2 depends on Task 1.
3. Task 3 is implemented with Task 2; confirm cache behavior after the complete workflow exists.
4. Task 4 follows the final workflow, secret, and cache names.

## QA/Testing Scenarios

1. Same-repository PR changing Python source: lint, type checks, hermetic tests, web tests, and a non-publishing web Docker build succeed with `DATABASE_URL` absent and no external services.
2. Same-repository PR with a broken frontend image build fails before merge and does not receive production credentials or publish an image.
3. Main push: validation succeeds; the workflow resolves the current champion, generates the real directory, and pushes Bento and web images only as `latest`.
4. Cached run: uv/pnpm and image-specific GHA BuildKit caches reduce downloads/rebuilds; the workflow remains correct after cache eviction or on a fresh runner.
5. Champion changes between runs: main ignores stale cached materials after remote lineage/hash verification and publishes the newly pinned champion.
6. Missing or invalid DagsHub/database/Docker Hub credentials: main build fails before its relevant image is pushed and exposes no secret in logs.
7. Manual `just deploy`: still independently generates artifacts and publishes both images outside GitHub Actions.
