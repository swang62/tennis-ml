# Plan: Feature Enrichment + Dashboard

## Goal

Close the raw-feature gaps in the pipeline (features the ATP CSVs carry but the
pipeline ignores), derive high-signal serve/rank/age features from them, retrain
the models, expose player data through Bento endpoints, and rebuild the
dashboard as a local Vite + React app.

## Audit summary (validated against data/column_glossary.md + raw CSV headers)

Raw ATP columns NOT used anywhere today, and confirmed useful:
- `w_1stWon` / `l_1stWon` — first-serve points won (1st-serve win % impossible)
- `w_2ndWon` / `l_2ndWon` — second-serve points won (2nd-serve win % impossible)
- `w_SvGms` / `l_SvGms` — service games (per-game ace/DF rates impossible)
- `w_bpSaved` / `l_bpSaved` — stored as the WRONG derivation (`faced - saved`); the saved value is lost
- `winner_rank_points` / `loser_rank_points` — Elo-like strength signal, ignored (0 files)
- `winner_age` / `loser_age` — no age feature anywhere

Explicitly excluded (decision): `indoor`, `seed`, `score`, `best_of`, and all
descriptive tournament columns (`tourney_name`, `draw_size`, `entry`). Player
identity columns (`name`, `hand`, `ht`, `ioc`) are already covered by
`gold.player_profiles`.

## Tasks

### [x] Task 1: Bronze — ingest unused raw features + fix break-point semantics

First thing. New columns on `bronze.match_events` (winner/loser -> player1/player2):

| Raw column | Bronze column | Type |
| --- | --- | --- |
| `w_1stWon` | `player1_first_serve_points_won` | UTINYINT |
| `l_1stWon` | `player2_first_serve_points_won` | UTINYINT |
| `w_2ndWon` | `player1_second_serve_points_won` | UTINYINT |
| `l_2ndWon` | `player2_second_serve_points_won` | UTINYINT |
| `w_SvGms` | `player1_service_games` | UTINYINT |
| `l_SvGms` | `player2_service_games` | UTINYINT |
| `winner_rank_points` | `player1_rank_points` | INTEGER |
| `loser_rank_points` | `player2_rank_points` | INTEGER |
| `winner_age` | `player1_age` | DOUBLE |
| `loser_age` | `player2_age` | DOUBLE |

Break-point semantic fix (user correction): bronze currently stores
`break_points_won = bpFaced - bpSaved` (break points broken against the player).
Rename and store the raw values instead:

| Raw column | Bronze column (new name) | Old name |
| --- | --- | --- |
| `w_bpSaved` | `player1_break_points_saved` | `player1_break_points_won` |
| `l_bpSaved` | `player2_break_points_saved` | `player2_break_points_won` |
| `w_bpFaced` | `player1_break_points_faced` | `player1_break_points_total` |
| `l_bpFaced` | `player2_break_points_faced` | `player2_break_points_total` |

- **Files**:
  - `src/features/columns.py` — `BRONZE_COLUMNS_INT` (UTINYINT 0..255): rename the 4 bp columns, add the 6 serve/stat columns (first_serve_points_won, second_serve_points_won, service_games x2). Add `BRONZE_COLUMNS_INT32` (rank_points x2) and `BRONZE_COLUMNS_FLOAT` (age x2).
  - `src/flows/ingest.py` — extend `RAW_ATP_COLUMNS` with the 10 raw columns; map them in `atp_rows_to_bronze` (seed.py reuses this transform); add a float parser for `age` (raw values like `24.41`); drop the `faced - saved` derivation.
  - `infra/duckdb/init.sql` — `bronze.match_events` DDL: rename 4 columns, add 10 (UTINYINT / INTEGER / DOUBLE).
  - `src/features/validate.py` — rename bp checks to saved/faced; add range checks for rank_points (0..20000) and age (0..100).
  - `dbt/models/sources.yml` — document renamed + new bronze source columns.
  - `dbt/models/silver/player_matches.sql` + `.yml` — rename bp passthrough to `break_points_saved` / `break_points_faced`; pass through the 6 new per-player columns.
  - `tests/test_rolling_contract.py` — update the hard-coded `len(BRONZE_COLUMNS_INT) == 16` count and add assertions for the new column sets.
- **Guardrails**: do NOT ingest `indoor`, `seed`, `score`, `best_of`, `tourney_name`, `draw_size`, `entry`, or player identity columns (covered by profiles). Keep raw age as DOUBLE (fractional years), do not round.
- **Rebuild**: `just db-reset && just db-seed && just db-dbt` — deletes and rebuilds `data/tennis.duckdb` from raw CSVs (destructive; approved).
- **Acceptance criteria**:
  - `pytest` green (updated contract tests).
  - `bronze.match_events` has all 10 new columns + 4 renamed bp columns with correct types.
  - Spot-check a sampled match: `break_points_saved` == raw `w_bpSaved`, `age` parsed as float, `rank_points` matches the CSV.
  - `dbt build` clean; `silver.player_matches` carries the new columns.

### [x] Task 2: Silver/Gold — derive high-signal features

> **User overrides (locked, supersede plan text):** all new rolling features use
> `_5/_10` windows only (no `_20` variants). Rank points is exposed ONLY as
> `rank_points_diff` — no per-side `player/opponent_rank_points` features, no
> rolling/trend rank-points machinery (rank already embeds that signal
> implicitly). `latest_player_rank_points` remains in rolling_features as
> internal backing for the inference-time diff. Final FEATURE_COLS: 80.

- **Per-match (silver)**: pass through the new serve/rank/age columns.
- **Per-match (gold, both players joined)**: `first_serve_win_pct` = 1stWon/1stIn, `second_serve_win_pct` = 2ndWon/(svpt-1stIn), `serve_win_pct` = (1stWon+2ndWon)/svpt, `aces_per_svc_game`, `df_per_svc_game`, `break_points_saved_pct` = saved/faced, `rank_points_diff`, `age`, `age_diff`.
- **Rolling (gold.rolling_features, windows 5/10)**: rolling `first_serve_win_pct`, `second_serve_win_pct`, `serve_win_pct`, `break_points_saved_pct`, `aces_per_svc_game`, `rank_points` avg + trend.
- **Contract**: every new feature added to `src/features/columns.py` FEATURE_COLS AND `src/features/inference.py` (as-of-date lookups); `tests/test_rolling_contract.py` keeps FEATURE_COLS <-> gold.match_features in sync.

### [ ] Task 3: Retrain + promote + redeploy

Feature set changed, so old models are invalid: `just pipeline` (3 base models + ensemble), promotion sets `@champion`, `just deploy-bento`.

### [ ] Task 4: Bento data endpoints

`/players`, `/player_profile` (bio + recent + career stats incl. 1st/2nd-serve win %, save rate, rank-points trend), `/rank_history`, `/match_history`, `/head_to_head` — parameterized SQL in `src/serving/service.py`.

### [ ] Task 5: Dashboard

`web/` Vite + React 19 + TS, TanStack Router/Query/Table, echarts-for-react, Tailwind v4. Pure local dev + HMR; `/api` proxy -> local `bentoml serve` (:3000). No wrangler/Cloudflare, no Dockerfile, no deploy step. Pages: Player Profile (bio, career-vs-recent stat bars, rank chart, form strip, matches table), Head-to-Head (scoreboard, surface split, rank lines, model overlay).

## Dependencies

1 -> 2 -> 3 -> 4 -> 5. Task 1 must land and the DB rebuild must verify before any downstream feature work.

## QA/Testing Scenarios

- Task 1: sampled match spot-check vs raw CSV; pytest; dbt build; `just validate` (kubeconform untouched).
- Task 2: FEATURE_COLS contract test; feature distribution sanity (no div-by-zero -> NULL imputed at train).
- Task 3: pipeline runs end-to-end; deployment smoke via live `/predict_from_ids`.
- Task 4: curl each endpoint against local `bentoml serve`; JSON shape matches frontend types.
- Task 5: `npm run build` clean; profile + H2H flows against the local Bento.
