# tennis-ml — AGENTS.md

Commands run via `just` (not make). `uv` is the package manager (not pip/poetry).

## Quick reference

| Action | Command |
|---|---|
| Install deps | `uv sync` |
| Full local setup | `just setup` |
| Lint | `ruff check src/` |
| Format | `ruff format src/` |
| Lint + format all | `uv run pre-commit run --all-files` |
| Ingest CSV | `uv run python -m src.flows.ingest data/matches.csv` |
| Run ETL | `just db-etl` |
| Train pipeline | `just pipeline` |
| Run training only | `just train` |
| Reset DB | `just db-reset` |
| Dashboard (local) | `just dashboard-local` |
| BentoML dev server | `just bento-local` |
| Deploy serving Bento | `just deploy-bento` |
| Teardown | `just destroy` |

## Key facts

- **DuckDB** is the data warehouse — embedded, file-based local database.
- **Prefect** orchestrates the ETL flow.
- **Papermill** all runs via parameterized Jupyter notebooks under, logging all artifiacts to MLflow.
- **Model serving** is BentoML, pulls models/features from MLflow registry, serves via FastAPI.
- **Container runtime** K3d/K3s single-node kubernetes cluster to handle all deployments
- **Visualization** is via custom Panel dashboard (src/dashboard/app.py), also deployed as docker image

## Architecture overview

**Data flow:** CSV → bronze.match_events (ingest) → gold.match_features (DuckDB SQL rolling transforms in ETL) → training notebooks → MLflow model registry → BentoML serving.

**Model strategy:** Three model classes compete independently via Optuna (linear, GBDT, neural net). The best from each is stacked via a simple logistic regression meta-model on their probability outputs. Architecturally designed for 80k match samples.

**Two-tower match sequence predictor:** Each player tower merges a sequence pathway (LSTM/GRU/TCN encoding match history) with a static pathway (rank, age, style, bio embeddings). The pairwise head uses `concat(a, b, a-b, a*b)` followed by a small MLP classifier.

**Player similarity (src/models/similarity.py):** Separate unsupervised FAISS index built from player bios via fastembed + one-hot encoded categoricals. Not part of match prediction — purely a content-based retrieval system.

## src/flows/

- `ingest.py` — validate and insert CSV match data into DuckDB bronze layer
- `etl.py` — Prefect flow: bronze-to-gold transforms (DuckDB SQL), player profile enrichment
- `pipeline.py` — standalone training runner (no Prefect): runs all training notebooks in sequence (tune → stack → evaluate → promote)

# Bento inference contract

`src/serving/service.py` (`TennisPredictor.predict`) serves the stacked
ensemble: `linear_best` + `gbdt_best` + `nn_best` base models combined by the
`production_model` logistic-regression head.

**The contract is: preprocessing computes everything; Bento only runs the
models.**

## Layer 1 — raw inputs upstream must know (never sent to Bento)

- Both player ids (`player_id`, `opponent_id`).
- Current match surface (raw `surface`) — one-hot it upstream into
  `is_clay` / `is_grass` / `is_hard`. Raw `surface` is NOT part of the payload.
- Current tournament / round context: `tournament` -> `tournament_level`
  (`grand_slam=4, masters=3, atp_500=2, atp_250=1, else 0`),
  `round` -> `round_encoded` (`r128=1, r64=2, r32=3, r16=4, qf=5, sf=6, f=7,
  else 0`). Same mapping as `dbt/models/gold/match_features.sql:222-233`.
- Per-player rolling stats computed from each player's prior matches up to
  (not including) the current match: `win_rate_5/10/20`, `ace_rate_5/10`,
  `first_serve_pct_5/10`, `break_pct_5/10`, `avg_opp_rank_10/20`,
  `rank_trend_10/20`, `win_streak`, `days_since_last_match`, `matches_30d`,
  `surface_win_rate_10` — for the player and, prefixed `opp_`, for the
  opponent, plus `player_ranking` / `opponent_ranking`.
- Differentials between the two sides: `rank_diff`, `win_rate_diff`,
  `ace_rate_diff`, `break_diff`, `streak_diff`, `matches_30d_diff`,
  `surface_win_diff`, `rank_trend_diff`.

## Layer 2 — the finalized Bento payload

Send one row per match as a pandas DataFrame (the API is batchable,
`batch_dim=0`) with exactly `FEATURE_COLS` plus the two ids:

```text
FEATURE_COLS (49, in order) + "player_id" + "opponent_id"
```

`FEATURE_COLS = PLAYER_COLS + OPPONENT_COLS + DIFF_COLS + CONTEXT_COLS`,
defined canonically in `src/features/rolling.py` (mirrors
`data/processed/feature_cols.json`). Any missing column is rejected with a
`MissingColumnsError`.

Over HTTP this is split-orientation JSON:

```json
{
  "columns": [
    "player_ranking", "win_rate_5", "... all 49 FEATURE_COLS ...",
    "player_id", "opponent_id"
  ],
  "data": [[1900, 0.72, "...", "novak-djokovic", "carlos-alcaraz"]]
}
```

The row must be canonical: the lower/stable ATP id on the `player_*` side
(see `build_inference_features` in `src/features/rolling.py`). Bento does NOT
re-canonicalize — `p_win` is P(canonical player wins) as you send it.

## What Bento does internally

At init: loads the persisted train-fit scaler (`linear_scaler.pkl`), bio
embeddings (`bio_embeddings.parquet`, `bio_feature_cols.json`), and the 4
models from the BentoML store.

Per row:

1. `scaler.transform(FEATURE_COLS)` — used by the linear and NN paths (the
   GBDT path uses the raw row, as in training).
2. `p_linear` = `linear_best.predict_proba(scaled)[:, 1]`
3. `p_gbdt` = `gbdt_best.predict_proba(raw)[:, 1]`
4. `p_nn` = sigmoid of `nn_best(scaled, bio(player_id), bio(opponent_id))`;
   unknown ids map to a zero bio vector.
5. `p_win` = `production_model.predict_proba([[p_linear, p_gbdt, p_nn]])[:, 1]`

Response: a DataFrame with `player_id`, `opponent_id`, `p_win`, `p_linear`,
`p_gbdt`, `p_nn`.

Bento does NOT derive rolling/diff/context features from raw match rows —
compute them all upstream (Layer 1 -> Layer 2) before calling.
