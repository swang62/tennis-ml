# Plan: Feature Enrichment + Leakage-Safe Match-Day Features + Dashboard

Merged single plan (was `plan-feature-enrichment.md`; merged with
`plan-match-day-features.md` on 2026-08-04 and the old file deleted). Goal:
close the raw-feature gaps (ATP CSVs carry features the pipeline ignored),
derive high-signal serve/rank/age features, add leakage-safe match-day
features (rolling form, strength of schedule, perspective-explicit H2H),
retrain + redeploy, expose player data through Bento endpoints, and rebuild
the dashboard as a local Vite + React app.

## Decisions (locked)

| Topic | Choice |
| --- | --- |
| New rolling windows | `_5/_10` only — no `_20` variants for any new feature |
| Rank points | `rank_points_diff` only; no per-side lookup, no rolling/trend (rank embeds the signal) |
| `avg_rank_faced` | 5/10 windows: rename existing `avg_opponent_rank_10` -> `avg_rank_faced_10`, add a w5 variant, drop the `_20` |
| df / loss diffs | Not needed — keep `df_rate_5/10` and `loss_streak` features, no `df_rate_diff` / `loss_streak_diff` |
| Age | Keep as-of-date raw ATP age (landed in Task 2) as `player_age`/`opponent_age` + `age_diff`; NO birthdate-derived age |
| Activity windows | `matches_30d` only — no 7d/90d windows |
| H2H win rate | Last-5-meeting recency; neutral `0.5` when zero prior meetings |
| Current-match serve stats | Gold-only enrichment columns in `gold.match_features`, never in FEATURE_COLS |
| Dashboard | Local `web/` Vite + React app (no Docker, no deploy step) |

## Completed

### [x] Task 1: Bronze — ingest unused raw features + fix break-point semantics

New `bronze.match_events` columns: `player{1,2}_first_serve_points_won`,
`player{1,2}_second_serve_points_won`, `player{1,2}_service_games`
(UTINYINT), `player{1,2}_rank_points` (INTEGER), `player{1,2}_age` (DOUBLE,
fractional years preserved). Break points renamed to raw semantics:
`player{1,2}_break_points_saved` / `player{1,2}_break_points_faced` (the
`faced - saved` derivation is gone). `RAW_ATP_COLUMNS` extended, `columns.py`
gains `BRONZE_COLUMNS_INT32`/`BRONZE_COLUMNS_FLOAT`, validate.py range
checks, silver passthrough, DB rebuilt (`just db-reset && db-seed &&
db-dbt`). Done + committed.

### [x] Task 2: Silver/Gold — derive high-signal features

Per-side as-of-date `player_age`/`opponent_age` + `age_diff`; rolling
`first_serve_win_pct_5/10`, `second_serve_win_pct_5/10`, `serve_win_pct_5/10`,
`aces_per_svc_game_5/10`, `break_points_saved_pct_5/10` (renamed from
`break_pct_5/10`; `src/models/similarity.py` updated) + diffs; `rank_points_diff`
only for rank points (per locked decision; `latest_player_rank_points` kept as
internal rolling backing for the inference diff). Current-match serve stats
(`*_first_serve_win_pct`, `*_serve_win_pct`, `*_df_per_svc_game`,
`*_break_points_saved_pct`) are gold-only enrichment, not FEATURE_COLS.
FEATURE_COLS = 80. Done + committed.

## New (merged from plan-match-day-features)

### [x] Task 3: Leakage-safe rolling form + strength of schedule

- **Files**: `dbt/models/gold/rolling_features.sql` + `.yml`,
  `dbt/models/gold/match_features.sql` + `.yml`, `src/features/columns.py`,
  `src/features/inference.py`, `tests/test_rolling_contract.py`,
  `tests/test_inference_features.py`, `tests/test_inference_units.py`
  (if present).
- `df_rate_5/10` = rolling `SUM(double_faults) / NULLIF(SUM(total_serve_points), 0)`
  over w5/w10, consistent with existing serve rates.
- `loss_streak` = consecutive losses ending at the snapshot match (mirror of
  existing `win_streak`; a win resets to 0). No diff column.
- `avg_rank_faced`: rename existing `avg_opponent_rank_10/20` in
  `rolling_features.sql` -> `avg_rank_faced_10` and add `avg_rank_faced_5`
  (both `AVG(opponent_ranking)` over w5/w10); drop the `_20`. Expose
  `player/opponent_avg_rank_faced_5/10` + `avg_rank_faced_diff` (10-window) in
  `gold.match_features` and FEATURE_COLS.
- Mirror everything in `src/features/inference.py` (pool aggregates, side
  values, diffs, fallbacks) — SQL remains the feature source of truth.
- **No** `df_rate_diff`, **no** `loss_streak_diff`, **no** `_20` windows.
- **Acceptance**: every training row gets these only from snapshot N-1;
  first-match rows NULL where no history; inference outputs the new FEATURE_COLS
  contract without NaNs after fallback; `df_rate_5/10` NULL when historical
  serve points are 0; `win_streak`/`loss_streak` never both positive.

### [x] Task 4: Perspective-explicit H2H features (last-5 recency)

- **Files**: `dbt/models/gold/match_features.sql` + `.yml`,
  `src/features/columns.py`, `src/features/inference.py`,
  `dbt/tests/gold/match_features_h2h_no_current_match.sql`,
  `tests/test_inference_features.py`, `tests/test_e2e_ingest_to_inference.py`.
- Prior-meeting aggregates for the canonical pair, only matches with
  `match_date` strictly before the target date, restricted to the **most
  recent 5 meetings** (recency window — per locked decision).
- Expose `player_h2h_matches`, `player_h2h_wins`, `player_h2h_win_rate`,
  `opponent_h2h_matches`, `opponent_h2h_wins`, `opponent_h2h_win_rate`.
  `player_h2h_*` always describes the canonical `player_id` perspective.
- Zero meetings -> `0.5` win rate (neutral, locked), zero counts stay zero so
  models can distinguish uncertainty.
- One parameterized inference query for the canonical pair; reverse
  perspective derived from the same prior meetings (no duplicate queries).
  Prepared parameters only — never interpolate ids/dates.
- No surface-specific or recent-five *extra* H2H variants (recency IS the
  last-5 window).
- **Acceptance**: first meeting has zero priors + neutral rate; second meeting
  sees exactly one prior; same-date meetings excluded; `player_h2h_matches ==
  opponent_h2h_matches`; with priors, wins sum to matches and win rates sum to
  1 (float tolerance); reversed raw ids -> identical canonical row.

### [x] Task 5: Strengthen leakage + feature-contract tests

- **Files**: `dbt/tests/gold/match_features_no_current_match_leakage.sql`
  (extend), `tests/test_rolling_contract.py`,
  `tests/test_inference_features.py`, `tests/test_e2e_ingest_to_inference.py`.
- Extend the existing N-1 leakage dbt test beyond `win_rate_10` to cover all
  snapshot-backed fields from Tasks 2-3.
- Assert no current-match aces / double faults / first-serve totals /
  break-point totals / outcome values enter FEATURE_COLS (current-match serve
  stats live in gold only).
- Add a train/inference parity fixture: one historical gold row's features vs
  an inference row built at that match date from only earlier history.
- Keep per-side ordering, differential ordering, no-NaN inference, canonical id
  symmetry, cold start, missing profiles, missing ranks assertions.

## Then

### [x] Task 6: Retrain + promote + redeploy

Feature contract changed -> old models invalid. `just train` (3 base models +
ensemble via `src/flows/pipeline.py`; promotion sets `@champion`), then
`just deploy-bento` (`src/flows/deploy.py` — rebuilds ONNX + snapshots DuckDB
gold into the image). Smoke: live `/predict_from_ids` for known players,
cold-start players, no-H2H pairs, previously-met pairs. Do not reuse/deploy a
model trained against the old contract.

### [x] Task 7: Bento data endpoints

`/players`, `/player_profile` (bio + recent + career stats incl. 1st/2nd-serve
win %, save rate, rank-points trend), `/rank_history`, `/match_history`,
`/head_to_head` — parameterized SQL in `src/serving/service.py`. Curl each
against local `bentoml serve`; JSON shape matches the web dashboard types.

### [ ] Task 8: Dashboard (web/ Vite + React)

`web/` Vite + React 19 + TS, TanStack Router/Query/Table, echarts-for-react,
Tailwind v4. Pure local dev + HMR; `/api` proxy -> local `bentoml serve`
(:3000). No Dockerfile, no deploy step. Pages:
- **Player Profile** — bio, career-vs-recent stat bars (incl. 1st/2nd-serve
  win %, break save %), rank chart, form strip, matches table, and
  **all-time clay/grass/hard win rates** (from all completed matches in
  DuckDB; unplayed surface shows `n/a (n=0)`, not 0% — per locked decision;
  the old `src/dashboard/app.py` Panel implementation is deleted).
- **Head-to-Head** — scoreboard, surface split, rank lines, model overlay.
Verify `npm run build` clean; profile + H2H flows against the local Bento.

## Dependencies

1. Tasks 3-4 modify the shared feature contract; run sequentially (same files:
   `columns.py`, `inference.py`, `match_features.sql`). 2. Task 5 depends on
   the final SQL/Python names from Tasks 3-4. 3. Task 6 depends on all feature
   work + tests. 4. Task 7 depends on the rebuilt gold layer. 5. Task 8
   depends on Task 7's endpoint shapes + Task 6's model overlay.

## QA/Testing Scenarios

- Player with no match history: honest activity/H2H counts, valid inference
  fallbacks, no NaNs.
- First and second meeting between a pair; split and one-sided H2H records.
- Missing opponent ranks inside rolling windows: `avg_rank_faced` skips NULL
  ranks consistently.
- Zero historical serve points: `df_rate_5/10` NULL in gold, imputed at train.
- Win-to-loss and loss-to-win transitions: only one streak direction positive.
- Raw player input order reversed: canonical feature row unchanged.
- `just etl` green with all dbt tests; full `pytest` green; `just train`
  completes and logs the expanded feature contract; `/predict_from_ids` smoke
  green; dashboard `npm run build` clean.
