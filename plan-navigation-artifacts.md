# Plan: Fresh Snapshot-Backed Navigation Artifacts

## Goal

Make player navigation artifacts independent of model training and champion lineage:

- The player directory, web MiniSearch payload, and FAISS similarity index are rebuilt at every deploy from the **current local DuckDB training snapshot**; playstyle clusters are refit at every train run from that same snapshot (see Task 2) and their labels are embedded into the deploy-built metadata and directory.
- The DuckDB snapshot remains the shared data boundary. Its freshness and source `DATABASE_URL` are operator-controlled by the snapshot refresh step.
- Navigation artifacts are never logged to MLflow, included in a candidate manifest, tagged on a champion, hash-verified as lineage, or used by promotion.
- Deploy bundles the fresh FAISS index/metadata with Bento and the fresh directory/MiniSearch output with the web image.

## Scope

### In scope

- Remove similarity, cluster, and directory artifacts from training, candidate, promotion, and champion-lineage paths.
- Build clusters, FAISS metadata/index, and directory from one immutable DuckDB snapshot during deploy.
- Keep cluster assignments and labels in memory while generating deploy artifacts; only their embedded effects ship.
- Document the artifact-boundary contract in the root `AGENTS.md`.
- Update affected hermetic tests.

### Out of scope

- Changing snapshot refresh behavior, `DATABASE_URL` selection, model features, predictive inference, promotion metrics, or drift monitoring.
- Adding live PostgreSQL reads to navigation deploy staging.
- Pinning/reproducing historical navigation results for a champion.

## Tasks

### [x] Task 1: Separate model lineage from navigation artifacts

- **Description:** Explicitly revert the in-progress player-directory champion-pinning implementation: remove its MLflow upload, URI/hash pin creation, candidate-manifest merge, champion tags, deploy download spec, staged pinned-file copy, and related tests. Then remove the pre-existing similarity index and similarity metadata URI/hash pairs from `LINEAGE_AUX_KEYS`, preserving only predictive-model inputs (embeddings, bio feature columns, scaler, model versions). Update lineage comments so MLflow/champion tags are explicitly model-prediction-only.
- **Files:** `src/constants.py`, `tests/test_deploy.py`, `tests/test_deploy_native_models.py`
- **Acceptance Criteria:**
  - Champion tags and `model_info.json` do not contain similarity, cluster, or directory artifacts.
  - No player-directory champion-pinning code introduced in the current worktree remains.
  - Existing model/scaler/embedding tag validation remains unchanged.
  - A champion created before/after this change is deployable without navigation-related MLflow tags.
- **Guardrails:** Do not weaken hash validation for model-affecting artifacts.

### [ ] Task 2: Remove navigation generation and pinning from training

- **Description:** Keep the implementation's snapshot cluster fitting — `generate_cluster_artifacts()` (pipeline) → `build_cluster_artifacts()` (similarity), run right after `refresh_snapshot()` — as the canonical producer of `cluster_assignments.parquet` + `cluster_descriptions.json`. Remove FAISS index generation, directory generation, MLflow artifact logging of the similarity index/metadata, and `similarity_pins.json` creation from the standalone training pipeline. Training refreshes DuckDB, refits playstyle clusters, and executes model notebooks only.
- **Files:** `src/flows/pipeline.py`, `tests/test_pipeline.py`
- **Acceptance Criteria:**
  - `just train` creates no similarity/directory artifacts and logs no navigation artifacts to MLflow; it continues to refit and persist the playstyle cluster artifacts (assignments + labels) from the fresh snapshot.
  - Training has no dependency on FAISS navigation outputs or cluster labels.
  - Tests prove the pipeline does not call FAISS/directory build or MLflow-navigation-pin code.
- **Guardrails:** Keep the DuckDB snapshot refresh and the cluster fit before notebook execution; do not change notebook execution order.

### [ ] Task 3: Remove navigation pins from candidate and promotion handoff

- **Description:** Stop notebook 03 from reading/merging `similarity_pins.json` into `aux_pins`; retain only model-feature auxiliary pins. Confirm notebook 04 promotion receives the reduced manifest and produces the reduced champion tags through the shared lineage helper.
- **Files:** `notebooks/parameters/03_train_ensemble.ipynb`, `notebooks/parameters/04_evaluate.ipynb` (only if its comments/assertions name navigation lineage), `tests/test_deploy.py`
- **Acceptance Criteria:**
  - Candidate manifests contain no FAISS, similarity metadata, cluster, or directory pins.
  - Promotion succeeds with model-only auxiliary pins.
  - No MLflow run uploads navigation artifacts.
- **Guardrails:** Do not alter model base-version pins, evaluation metrics, or promotion-gate logic.

### [ ] Task 4: Fit clusters in the pipeline; make similarity construction in-memory at deploy

- **Description:** Refactor the similarity/clustering boundary so the pipeline fits clusters from the fresh snapshot via `build_cluster_artifacts` (injected query function; writes `cluster_assignments.parquet` + `cluster_descriptions.json` as the shared interchange) and deploy passes assignments/labels directly into similarity construction, writing only the final FAISS index and player metadata to staged paths. The EDA notebook reviews exactly the pipeline's fit.
- **Files:** `src/models/similarity.py`, `notebooks/eda/02_playstyle_clusters.ipynb` (if required by the refactored API), `tests/test_similarity.py`
- **Acceptance Criteria:**
  - Deploy builds the similarity index without fitting its own clusters: it consumes the pipeline-generated assignments/labels (in memory) and writes only the FAISS index and player metadata to caller-provided paths.
  - FAISS vectors include the fresh one-hot cluster membership.
  - Returned/written player metadata embeds the fresh cluster label.
  - The EDA workflow retains an explicit way to persist/review clusters if it currently relies on saved artifacts.
- **Guardrails:** Preserve deterministic KMeans settings (fixed random_state, `n_init="auto"`) and the current calibrated block vector composition (identity/playstyle/surface/reputation/bio/cluster blocks with the `SIM_*` weights, PCA-reduced bio, exposure-shrunk surface). Do not make cluster artifacts model inputs.

### [ ] Task 5: Build and stage all navigation artifacts from the current DuckDB snapshot during deploy

- **Description:** Add one deploy pre-build step that uses `src.db.training.to_dataframe` exclusively. It must:
  1. query the snapshot once for the player/profile data needed by both directory and similarity;
  2. consume the pipeline-fitted cluster assignments/labels in memory;
  3. build FAISS and player metadata into `data/deploy/` for Bento;
  4. produce `web/public/player-directory.json` with the same in-memory cluster labels for the web image.

  The existing web build then transforms the raw directory into content-hashed directory and MiniSearch assets. Bento continues loading the staged FAISS/metadata from `data/deploy/`.
- **Files:** `src/flows/deploy.py`, `src/serving/directory.py`, `src/models/similarity.py`, `bentofile.yaml` (verify/update only if staged artifact inclusion needs adjustment), `tests/test_deploy.py`, `tests/test_directory.py`, `tests/test_service_similar.py`
- **Acceptance Criteria:**
  - Deploy has no `src.db.client`/`execute_df` dependency for navigation artifact generation.
  - The navigation build fails clearly when the DuckDB snapshot is absent.
  - It runs before Bento/Web images are built, so no image can contain missing navigation assets.
  - Directory player IDs exactly match similarity metadata player IDs for a test snapshot.
  - Directory cluster labels exactly match similarity metadata cluster labels for a test snapshot.
  - `data/deploy/player_similarity.index` and `data/deploy/player_metadata.json` are bundled in Bento; the raw web directory is available to the existing MiniSearch build script.
- **Guardrails:** Never download navigation artifacts from MLflow. Never read live PostgreSQL at deploy. Do not add cluster parquet/JSON to the deploy image.

### [ ] Task 6: Remove obsolete deploy staging/download behavior

- **Description:** Delete navigation entries from MLflow auxiliary-download specs, similarity-directory tag/hash checks, and champion-manifest serialization. Remove stale comments that call navigation data serving lineage. Retain local staging paths and Bento includes for the freshly generated FAISS/metadata.
- **Files:** `src/flows/deploy.py`, `src/constants.py`, `tests/test_deploy.py`, `tests/test_deploy_native_models.py`
- **Acceptance Criteria:**
  - `_download_aux_artifacts()` handles only model-prediction artifacts.
  - A deploy with valid model tags does not require navigation tags or an MLflow navigation download.
  - Each deploy overwrites/rebuilds its local navigation assets from the current snapshot.
- **Guardrails:** Do not delete user data or old MLflow artifacts; simply stop producing/consuming them.

### [ ] Task 7: Document the artifact boundary

- **Description:** Add a concise section to the root repository `AGENTS.md` distinguishing champion-pinned model artifacts from fresh snapshot-backed navigation artifacts. State sources, lifecycle, packaging destination, and the operator responsibility for refreshing the snapshot after changing `DATABASE_URL`.
- **Files:** `AGENTS.md`
- **Acceptance Criteria:**
  - Documentation says model artifacts affecting match probabilities are MLflow/champion-pinned.
  - Documentation says directory, MiniSearch, FAISS similarity, and clusters are rebuilt at deploy from the current DuckDB snapshot, are not MLflow/champion lineage, and are embedded into web/Bento outputs as applicable.
  - Documentation states snapshot freshness is explicit/operator-controlled.
- **Guardrails:** Keep it short; do not duplicate the full pipeline documentation.

## Dependencies

1. Task 1 and Task 3 remove lineage contracts before deploy stops expecting them.
2. Task 4 provides the in-memory construction API required by Task 5.
3. Task 5 replaces deploy staging before Task 6 deletes navigation downloads.
4. Task 7 documents the final, verified behavior.

## QA / Testing Scenarios

1. **Fresh navigation deployment:** Given a DuckDB snapshot fixture, deploy-stage generation writes FAISS/metadata and directory; all player IDs and cluster labels agree.
2. **No snapshot:** Navigation deploy staging fails with the existing actionable snapshot-missing error; no image build is started.
3. **No navigation lineage:** A champion fixture with only model/scaler/embedding tags deploys successfully and performs no MLflow navigation download.
4. **No training coupling:** Training/candidate/promotion fixtures neither create nor require similarity/directory pins.
5. **Web packaging:** The existing MiniSearch script consumes the freshly staged raw directory, writes hashed assets/manifest, and removes only its raw web input as it does today.
6. **Bento packaging:** The staged FAISS index and player metadata are included and `/similar_players` loads them without MLflow access at runtime.
7. **Regression suite:** Run targeted navigation/deploy tests, then `just lint`.
