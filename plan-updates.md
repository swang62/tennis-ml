# Tennis-ML — PLANNING.md Drift Alignment

Alignment pass: PLANNING.md → current repo. Six updates pulled from the old
doc, plus a full evaluation overhaul for `05_evaluate`. All decisions locked
with the user.

## Decisions (locked)

| Topic | Choice |
|---|---|
| Promotion decision | Weighted relative-delta composite score over 8 metrics |
| Metric weights | `{roc_auc:.30, f1:.20, accuracy:.15, pr_auc:.10, precision:.10, recall:.10, mcc:.05, brier:0.00}` (Brier reported, not scored) |
| Similarity vector | Keep backhand/handedness one-hots + style stats; drop height/years_pro |
| Serve/return stat | `break_pct_10` (break points converted; no schema change) |
| `weighted_form` window | 10-match, repo-consistent (`weighted_form_10`) |
| Dashboard | New "Player Comparison" tab: 2 columns, profile info + basic stats + rank movement |
| Dashboard deploy | Dockerized dashboard pushed to the k3d registry (`just dashboard-deploy`), exposed at `dashboard.macsteve.lan:8501` |

---

## Change 1 — Similarity vector: style stats

**File:** `src/models/similarity.py` (`PlayerSimilarity.build`)

- Profile query drops `height` / `turned_pro`; keeps
  `player_id, display_name, backhand, handedness, summary`.
- New query — latest snapshot per player from `gold.rolling_features`
  (DuckDB `QUALIFY ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY snapshot_date DESC, match_id DESC) = 1`):
  `ace_rate_10, first_serve_pct_10, break_pct_10, clay_win_rate_10, grass_win_rate_10, hard_win_rate_10`.
- Vector = `[backhand/handedness one-hot] + [style stats] + [bio embeddings]`.
- Style cells NULL-filled to `0.0` for players with no eligible snapshot or
  no matches on a surface.
- FAISS vector dim changes accordingly.

**Tests:** `tests/test_similarity.py`
- `to_dataframe` stub must dispatch by table (profiles vs rolling_features).
- Extend fixture frames with the style-stat columns.
- `test_embed_bio_summaries_*` untouched.

---

## Change 2 — Bio embeddings: PCA → 10 dims into the NN

**File:** `notebooks/parameters/00_embeddings.ipynb`

- After `embed_bio_summaries`, fit `sklearn.decomposition.PCA(n_components=10)`
  on the `bio_*` columns, transform, and persist the 10-col
  `data/processed/bio_embeddings.parquet` + `bio_feature_cols.json` (10 names).
- **No `PCA.pkl` needed** — all bios are pre-reduced statically at artifact build
  time; consumers read the dim dynamically:
  - `02_tune_nn.ipynb` reads `bio_dim = len(bio_feat_cols)` from the parquet.
  - `src/serving/service.py:86-90` loads the parquet + cols generically.
  - `src/flows/deploy.py:215` ONNX export introspects `raw.bio_mlp[0].in_features`.
- Similarity keeps **full-dim** bios (separate `embed_bio_summaries` path).

---

## Change 3 — `weighted_form_10` (exponential decay)

Rolling form with exponential decay over the last 10 matches (most recent
weight `0.9^0 = 1`), consistent with the repo's inclusive window convention.

**Files:**
- `dbt/models/gold/rolling_features.sql`
  - Add reverse row-number per player (newest = 0) via a CTE.
  - `SUM(match_won * POW(0.9, rn)) OVER w10 / NULLIF(SUM(POW(0.9, rn)) OVER w10, 0) AS weighted_form_10`.
- `dbt/models/gold/rolling_features.yml` — document the new column.
- `dbt/models/gold/match_features.sql`
  - `pr.weighted_form_10` in `player_match_enriched`.
  - Expose `player_weighted_form_10` / `opponent_weighted_form_10`.
- `dbt/models/gold/match_features.yml` — document the new columns.
- `src/features/columns.py` — add `"weighted_form_10"` to `GOLD_ROLLING_COLS`
  (auto-propagates to `FEATURE_COLS`).
- `src/features/inference.py`
  - `_POOL_AGG_SQL`: `AVG(weighted_form_10) AS weighted_form_10` (imputation).
  - `_side_values`: `"weighted_form_10": cell("weighted_form_10", _DEFAULT_RATE)`.

**Tests:** `test_rolling_contract.py`, `test_e2e_ingest_to_inference.py`,
`test_inference_features.py` — update for the new column.

**Ops:** requires `dbt build` + re-ETL so the column exists in DuckDB.

---

## Change 4 — Full evaluation & reporting (`05_evaluate.ipynb`)

Replace the pure-AUC promotion gate with a multi-axis weighted comparison and
add an unconditional visualization section for human oversight.

**Metrics (candidate vs production on the SAME `test_preds` matrix):**
`roc_auc, pr_auc, accuracy, precision, recall, f1, mcc, brier` — per-axis deltas
printed and logged. Production via `models:/ensemble_lr_model@champion`;
graceful "first promotion" path when it doesn't exist.

**Weighted decision:** composite `Σ wᵢ·(candᵢ − prodᵢ)/max(prodᵢ, ε)`, promote
if `> 0` (or no production). Idempotency guard (`@champion` already on this run)
kept.

**Plot section — always runs, promoted or not:**
- ROC curves: 3 base models + candidate + production, one figure.
- PR curves: candidate + production.
- Calibration (reliability) curves.
- Confusion matrices side-by-side.
- SHAP: `shap.TreeExplainer` on `gbdt_best@best`, ~1–2k subsample of
  `X_test.parquet` (columns from `feature_cols.json`); beeswarm summary +
  waterfall on misclassified examples.
- Surface error analysis (existing) + base-model comparison table.
- All figures logged via `mlflow.log_figure`.
- Accuracy added to the base-model ROC-AUC loop.

---

## Change 5 — Local dashboard: 2-column player comparison

**File:** `src/dashboard/app.py` (run via existing `just dashboard-local` → `panel serve src/dashboard/app.py`)

Current state: a Panel app already exists but is partly broken and shows no
profile info:

- `get_player_match_history` (`app.py:69-79`) SELECTs `ace_rate,
  double_fault_rate, first_serve_pct` — none of these columns exist in
  `gold.match_features` (real names `player_ace_rate_5` /
  `player_first_serve_pct_10`; no `double_fault_rate` anywhere). The Player
  Explorer and Matchup "Match History" sections crash on load.
- `gold.player_profiles` (display_name, handedness, backhand, height,
  turned_pro, birthplace, summary) is never queried — no profile info shown.

Work:

- Fix `get_player_match_history`: drop the nonexistent columns from the
  SELECT (keep match_date, opponent, surface, tournament, round, ranking,
  result).
- New queries:
  - `get_profile(player_id)` → `gold.player_profiles` (display_name,
    handedness, backhand, height, turned_pro, birthplace, summary).
  - `get_latest_stats(player_id)` → newest `gold.rolling_features` snapshot
    per player (`QUALIFY ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY
    snapshot_date DESC, match_id DESC) = 1`): win_rate_5/10/20, ace_rate_10,
    first_serve_pct_10, break_pct_10, clay/grass/hard_win_rate_10, win_streak.
- New **"Player Comparison"** tab: two `pn.widgets.Select` (Player A / Player B);
  each column renders a player card = profile info (Markdown) + basic stats
  table + small rank-movement line chart (reuse `get_player_rank_history` +
  `px.line`, reversed axis).

**Verification:** `panel serve src/dashboard/app.py` starts clean; both tabs
render without query errors against the seeded DuckDB.

---

## Change 6 — Dockerize & deploy the dashboard to k3d

**Files:** `infra/manifests/deploy/dashboard.yaml`, `infra/manifests/deploy/Dockerfile`,
`infra/manifests/deploy/dashboard-requirements.txt` (new), `infra/manifests/default/ingress.yaml`,
`src/dashboard/deploy.py` (new), `justfile`

The dashboard was half-wired: a Deployment/Service manifest existed but was
unreachable and would have crashed in the container.

- `dashboard.yaml`: image `tennis-ml-registry:5000/tennis-dashboard:latest`
  with `imagePullPolicy: Always` (was bare `tennis-dashboard:latest` with
  `IfNotPresent` — k3d nodes can't see the local Docker image without an
  import, and `IfNotPresent` would never re-pull).
- `Dockerfile`: serve panel on `8501` (was `5006`, mismatched with the
  manifest/ingress), `COPY data/tennis.duckdb data/tennis.duckdb` (the
  dashboard queries DuckDB at import time, so the container crashed without
  the DB file), and drop the `pip install -e ".[dev]"` (pulled
  torch/lightning/prefect/mlflow into the image — the build timed out). New
  `dashboard-requirements.txt` installs only the runtime deps the dashboard
  imports (`panel, plotly, duckdb, pandas, numpy, python-dotenv`); the
  copied `src/` package resolves via cwd, no editable install needed.
- `ingress.yaml`: uncomment the `dashboard.macsteve.lan` host → service
  `tennis-dashboard` port `8501` (single-entrypoint ingress, no port-forwards).
- `src/dashboard/deploy.py` (new, plain script — no Prefect): builds the image,
  checks cluster/registry (`k3d` list helpers mirrored from `src/flows/deploy.py`),
  tags+pushes to `{REGISTRY_PUSH_URL}/tennis-dashboard:latest`, applies the
  deploy manifest + ingress, then `kubectl rollout restart` + status.
- `justfile`: `dashboard-deploy` → `uv run python src/dashboard/deploy.py`.

**Verification:** `just dashboard-deploy` builds and (with the cluster up)
serves the dashboard at `https://dashboard.macsteve.lan`.

---

## Sequencing & verification

1. Change 2 → Change 3 → Change 1 → Change 4 (cheapest/independent first;
   Changes 1–3 touch tests, Change 4 is a papermill re-run). Change 5
   (dashboard) is independent — can go anytime.
2. `dbt build` + re-ETL after Change 3; re-run `00_embeddings` after Change 2.
3. Verify: `just` lint/typecheck + affected test files (`test_similarity`,
   `test_rolling_contract`, `test_e2e_ingest_to_inference`,
   `test_inference_features`); re-run `pipeline.py` for Change 4.
4. Retrain + redeploy (`just pipeline`, `uv run python src/flows/deploy.py`)
   so the ONNX (`bio_dim=10`) and new feature column ship.
5. Change 5: `just dashboard-local` and visually confirm the two-column
   player comparison renders.
6. Change 6: `just dashboard-deploy` builds the image and (cluster up) pushes
   to the k3d registry and serves at `https://dashboard.macsteve.lan`.