# Plan: Simplify Dashboard Queries

## Goal

Simplify Bento's dashboard-facing reads so match facts come directly from
`bronze.match_events`, identity and biography come from
`gold.player_profiles`, and derived silver/gold relations are used only where
their semantics are intrinsic. Reduce joins, database round trips, and repeated
player-orientation logic. Extend the player directory with latest rank points
and a deterministic point-derived current-rank estimate so every empty player
picker can offer the top 20 players instead of a blank dropdown.

Because another refactor is active, implementation must start by reconciling
this plan with the then-current worktree and `plan-tour-averages.md`. In
particular, any newly added profile benchmark fields backed by the
`gold.tour_averages` singleton remain an explicit allowed exception rather than
being silently recomputed from bronze.

## Scope

### In scope

- Rewrite `/players`, `/player_profile`, `/match_history`, and `/head_to_head`
  around bronze match facts and profile identity data.
- Extend `/players` entries with nullable `latest_rank_points` and
  `estimated_rank`, derived from each player's latest positive points
  observation rather than ATP-provided rank fields.
- Carry those fields through MiniSearch, show estimated rank at the right of
  every option, and show the estimated top 20 when a picker has no selection or
  query.
- Use the same directory estimate for H2H's `Current rank` comparison and remove
  the two rank-history requests made only for that row.
- Retain `/rank_history` as a bronze-only endpoint while consolidating its
  player-orientation projection with the other dashboard queries where useful.
- Reduce `/player_profile` from its current five sequential queries to one
  dashboard-fact query, plus only an explicit singleton benchmark read if the
  ongoing tour-averages refactor requires it.
- Preserve current JSON envelopes, existing field names, null behavior, result
  orientation, and parameter validation, except for the intentional `/players`
  field additions and rank-first directory ordering.
- Add focused endpoint/query tests and inspect resulting PostgreSQL plans.

### Out of scope

- Changing `/predict`, `/predict_from_ids`, bulk prediction, inference feature
  construction, rolling snapshot semantics, or model defaults.
- Changing `/similar_players` ranking or rebuilding its packaged FAISS artifact.
- Redesigning page layouts beyond a compact right-aligned rank label in picker
  options.
- Persisting a calculated rank or rank-points column in
  `gold.player_profiles`; the checked-in schema has no trustworthy current-rank
  field and remains identity/biography only.
- Rebuilding historical rank charts from point-derived tour-wide snapshots;
  `/rank_history` remains the existing raw-rank time series for the profile
  chart.
- Adding score, set, tiebreak, or fifth-set data to bronze.
- Adding a dbt model or materialized dashboard table solely to simplify these
  request-time queries.
- Adding indexes without evidence from `EXPLAIN` on the rewritten queries.

## Endpoint Source Contract

| Endpoint | Planned request-time sources | Rationale |
|---|---|---|
| `/players` | `bronze.match_events` + `gold.player_profiles` | Match counts and each player's latest positive rank points come from raw matches; profiles provide names and preserve players with zero matches. Point ordering estimates current rank. |
| `/player_profile` | Bronze + profiles; optionally one `gold.tour_averages` singleton read if already added by the active refactor | Career, surface, recent-form, and rank-point trends are derivable from raw matches; biography is profile data. |
| `/rank_history` | Bronze | Remains the profile chart's historical raw-rank series; H2H no longer calls it for current rank. |
| `/match_history` | Bronze + one profile lookup join | Bronze has match/result/stat fields; profiles are needed only for opponent display names. |
| `/head_to_head` | Bronze | Bronze already has both player IDs, winner, date, surface, tournament, and round. |
| `/similar_players` | Packaged FAISS artifact | No request-time SQL; behavior should remain unchanged. |
| `/model_info` | Packaged manifest/environment | No match query. |
| Prediction endpoints | Existing silver/gold inference sources | Strict as-of rolling state and finalized defaults are genuinely derived data. |

## Rank Estimation Contract

1. Expand both bronze player sides into `player_id`, `match_id`, `match_date`,
   and `rank_points` observations.
2. Treat null and zero points as missing.
3. For each player, select the positive-points observation with the greatest
   `(match_date, match_id)`; no cross-player recency cutoff is applied.
4. Assign unique estimates with `ROW_NUMBER()` ordered by
   `latest_rank_points DESC, player_id ASC`.
5. Left-join those estimates to all profile rows. Players without positive
   points receive `latest_rank_points = null` and `estimated_rank = null`.
6. Do not read ATP-provided `player*_ranking` values or persist the estimate in
   `gold.player_profiles` for the current-rank use case.

The `/players` item contract becomes:

| Field | Type | Source |
|---|---|---|
| `player_id` | `string` | `gold.player_profiles` |
| `display_name` | `string` | `gold.player_profiles` |
| `matches_played` | `number` | Count of oriented bronze matches |
| `latest_rank_points` | `number \| null` | Latest positive oriented bronze points |
| `estimated_rank` | `number \| null` | Directory-wide point ordering |

## Picker Behavior Contract

| State | Options |
|---|---|
| No selection, empty trimmed query, directory loaded | First 20 ranked players after applying `exclude` |
| Non-empty trimmed query | Existing name-based matching/MiniSearch relevance, including unranked players |
| Directory/index still loading | Loading state; do not flash an empty/no-results message |
| Non-empty query with no matches | Existing `No matching players` state |

Default options come directly from the rank-ordered `/players` array; MiniSearch
remains responsible only for typed queries. A default option is a suggestion,
not an automatic selection.

## Tasks

### [x] Task 1: Reconcile the plan with the active refactor

- **Description**:
  - Inspect the current worktree diff and re-explore all SQL constants, handlers,
    player-picker/MiniSearch code, frontend response types, and tests before
    editing.
  - Compare current work against `plan-tour-averages.md`, especially any changes
    to `/player_profile`, shared tour-average loading, or response fields.
  - Record the final endpoint source matrix in the implementation PR/commit
    description before rewriting SQL.
  - Preserve active, unrelated work rather than reverting or overwriting it.
- **Files**:
  - `src/serving/service.py`
  - `src/features/tour_averages.py`
  - `web/src/api.ts`
  - `plan-tour-averages.md`
- **Acceptance Criteria**:
  - Every dashboard response field has an identified source before edits begin,
    including `latest_rank_points` and `estimated_rank` on `/players`.
  - Any field that truly requires `gold.tour_averages`, a packaged artifact, or
    rolling inference data is documented as an explicit exception.
  - No active refactor changes are reverted, duplicated, or silently bypassed.
- **Guardrails**:
  - Do not implement this plan against stale line numbers or the pre-refactor
    handler shape.
  - Do not fold prediction or similarity work into this query cleanup.

### [ ] Task 2: Centralize the bronze player-perspective projection

- **Description**:
  - Define one reusable SQL fragment in the serving module that expands each
    bronze match into winner and loser perspectives with consistent columns:
    player/opponent IDs, result, rank/rank points, date, match metadata, and the
    correctly oriented serve statistics.
  - Reuse that projection only in queries that need player-perspective
    aggregation; query bronze directly for pair-level H2H data.
  - Keep all request values parameterized with `%s` placeholders.
- **Files**:
  - `src/serving/service.py`
- **Acceptance Criteria**:
  - Player-side CASE/UNION orientation is defined once for dashboard aggregate
    queries rather than repeated independently.
  - Winner and loser perspectives expose identical column names and preserve
    one row per player per match.
  - The shared fragment contains no request values and introduces no new module
    or abstraction layer.
- **Guardrails**:
  - Do not create a new dbt model, view, ORM, query builder, or helper package.
  - Do not use the player-perspective expansion for H2H, where one bronze row is
    already the correct grain.

### [ ] Task 3: Rewrite player directory and profile reads

- **Description**:
  - Rewrite `/players` to aggregate match counts from both bronze player-ID
    columns and left-join the result to profiles for display names.
  - In the same query, orient both bronze sides into `(player_id, match_date,
    match_id, rank_points)`, discard missing/zero points, and select each
    player's newest observation deterministically by `match_date DESC,
    match_id DESC`.
  - Rank only players with a positive latest observation using
    `ROW_NUMBER() OVER (ORDER BY latest_rank_points DESC, player_id)`; left-join
    the ranked set back to all profiles so players without points remain in the
    directory with null point/rank fields.
  - Return directory rows by `estimated_rank NULLS LAST, display_name,
    player_id` so the response itself provides a stable top-player default.
  - Rewrite `/player_profile` as one bronze/profile query that returns biography,
    career totals and weighted serve rates, hard/grass/clay splits, latest-10
    form, and earliest/latest rank-point observations.
  - Derive latest-10 form from the ten newest oriented bronze matches ordered
    deterministically by `match_date DESC, match_id DESC`; expose the newest
    match date as `snapshot_date`.
  - Preserve zero-match behavior: count `0`, nullable denominator-derived rates,
    three fixed surface entries with `0` matches and `null` rates, and no recent
    form or rank-point trend.
  - If the active refactor adds tour-average comparison fields, keep the shared
    singleton loader/read rather than duplicating those global aggregates in the
    profile query.
- **Files**:
  - `src/serving/service.py`
  - `src/features/tour_averages.py` (only if the active refactor already makes it
    part of the profile contract)
- **Acceptance Criteria**:
  - `/players` no longer references `silver.player_matches`.
  - Every `/players` entry has `latest_rank_points: number | null` and
    `estimated_rank: number | null` in addition to its existing fields.
  - Equal points produce unique, reproducible positions via the player-id
    tiebreaker; no ATP-provided ranking or profile rank is used.
  - A player with no positive points remains searchable but cannot appear in
    the ranked top 20.
  - Core `/player_profile` match facts no longer reference
    `silver.player_matches` or `silver.rolling_features`.
  - `/player_profile` performs one core data query instead of five sequential
    queries; at most one additional validated singleton read is allowed for an
    already-required tour-average response.
  - Career ratios use summed numerators divided by summed denominators with
    `DOUBLE PRECISION` and `NULLIF`, not averages of per-match percentages.
  - Existing API response fields and frontend behavior remain unchanged.
- **Guardrails**:
  - Do not change profile biography ownership or copy profile fields into
    bronze.
  - Do not add or update a physical current-rank column on
    `gold.player_profiles`, and do not add a dbt model just to cache the
    directory estimate.
  - Do not change recent-form meaning accidentally; verify the rewritten latest
    ten against the current newest rolling snapshot on representative players.
  - Do not recompute global tour benchmarks per request.

### [ ] Task 4: Update player search, picker defaults, and H2H rank use

- **Description**:
  - Extend the frontend `Player` contract with nullable `latest_rank_points`
    and `estimated_rank`.
  - Add both fields to MiniSearch `storeFields`, map them back into every search
    result, and increment the local-storage index key so an old serialized index
    cannot silently omit the new metadata.
  - Make the shared `PlayerPicker` use the first 20 ranked directory players
    when there is no selection and the trimmed query is empty. Apply `exclude`
    before the limit so the second H2H picker still receives up to 20 options.
  - Show the existing loading state for an empty picker while directory/index
    data is unavailable, then replace it with defaults when ready.
  - Keep typed search relevance driven by player name; rank points are stored
    result metadata, not an additional searchable field or ranking boost.
  - Render option content as a name plus a right-aligned `#estimated_rank` when
    available. Leave the rank area absent or neutral for unranked typed-search
    results; do not display points in the row.
  - On H2H, read both current ranks from the selected `Player` records and remove
    `lastRank`, the two rank-history queries, and imports made unused by that
    change.
- **Files**:
  - `web/src/api.ts`
  - `web/src/pages/Home.tsx`
  - `web/src/components.tsx`
  - `web/src/pages/H2H.tsx`
  - `web/src/index.css`
- **Acceptance Criteria**:
  - Opening any unselected player picker with an empty input shows up to 20
    ranked players in ascending estimated-rank order; it never shows the former
    empty black list when ranked data is available.
  - Loading an empty picker does not briefly announce `No matching players`.
  - Both keyboard and pointer selection continue to work, with the first
    default option active and `aria-activedescendant` pointing at a real option.
  - Every ranked option shows the name at left and `#N` at right without
    changing the accessible option semantics.
  - Typed MiniSearch results retain `latest_rank_points` and `estimated_rank`
    after both a fresh index build and a local-storage reload.
  - H2H's `Current rank` values equal the selected players' directory estimates,
    and selecting a pair no longer triggers rank-history requests.
- **Guardrails**:
  - Do not add a frontend test framework or new search dependency for this
    change.
  - Do not search or boost on rank points, display raw points in picker rows, or
    turn the top-20 default into a preselected player.
  - Do not change the profile rank-history chart in this task.

### [ ] Task 5: Rewrite match history and H2H at their natural grain

- **Description**:
  - Rewrite `/match_history` directly from bronze using CASE expressions to
    orient opponent ID, player ranking, result, and player-side serve stats.
  - Retain one profile join solely for opponent display name.
  - Preserve the current deepest-round-per-tournament selection, deterministic
    ordering, and limit clamp.
  - Rewrite `/head_to_head` as a direct bronze pair filter with no silver join,
    grouping, or deduplication.
  - Preserve canonical lower-ID `player1_id` orientation and derive
    `player1_won` from `winner_id = lower_id`.
- **Files**:
  - `src/serving/service.py`
- **Acceptance Criteria**:
  - `/match_history` references bronze and profiles only and returns the same
    selected matches and oriented values as before.
  - `/head_to_head` references bronze only and returns one meeting per bronze
    match in newest-first order.
  - H2H summary totals, last-five rate, surface breakdown inputs, and no-meeting
    fallback values remain unchanged.
  - All IDs remain bound parameters; none are interpolated into SQL text.
- **Guardrails**:
  - Do not join profiles into H2H; the handler/frontend already resolves names
    from the player directory.
  - Do not add set-score parsing or new H2H statistics in this refactor.

### [ ] Task 6: Add contract and equivalence coverage

- **Description**:
  - Add focused service-data endpoint tests covering response construction,
    parameter binding, and source-table restrictions.
  - Add PostgreSQL-backed equivalence checks using the deterministic seed for
    representative winner/loser perspectives, ranked/unranked players, empty
    surfaces, zero-H2H pairs, and players with fewer than ten matches.
  - Add directory-query cases for newest-point selection, zero/missing points,
    equal-point deterministic ordering, and players with different latest
    observation dates.
  - Assert dashboard SQL no longer names silver relations, except outside the
    dashboard handlers for prediction/inference.
  - Preserve or update any profile benchmark tests introduced by the active
    tour-averages refactor.
- **Files**:
  - `tests/test_service_data_endpoints.py` (new)
  - `tests/test_e2e_ingest_to_inference.py`
  - `tests/test_service_similar.py`
- **Acceptance Criteria**:
  - Tests demonstrate correct orientation for both player sides of a bronze
    match.
  - Rewritten endpoint JSON matches the established frontend contract plus the
    two documented `/players` additions.
  - `/players` contract coverage asserts point-derived rank values and stable
    null behavior without relying on raw ranking columns.
  - Error responses for missing IDs, invalid limits, and database failures are
    unchanged.
  - Similar-player tests remain unchanged and passing.
- **Guardrails**:
  - Do not lock tests to incidental SQL whitespace.
  - Do not weaken existing inference or temporal-leakage tests to accommodate
    dashboard query changes.

### [ ] Task 7: Verify query simplicity and performance

- **Description**:
  - Run focused service tests, `just test`, `just lint`, and
    `pnpm --dir web build`.
  - Manually verify Home and both H2H pickers with an empty query, typed ranked
    and unranked results, exclusion/backfill, mouse selection, arrow keys,
    Enter, Escape, clear, and stale local-storage data.
  - Run `EXPLAIN (ANALYZE, BUFFERS)` for the rewritten player directory, profile,
    match-history, and H2H queries against representative full data.
  - Compare query count, joins, row expansion, and latency with the pre-refactor
    implementation.
  - Add an index only if a rewritten query shows a measured scan problem not
    covered by the existing `(player1_id, match_date)` and
    `(player2_id, match_date)` bronze indexes.
- **Files**:
  - `src/serving/service.py`
  - `infra/postgres/init.sql` (only if measurement proves an index is required)
- **Acceptance Criteria**:
  - `/player_profile` core data round trips fall from five to one.
  - `/head_to_head` has no silver scan, player-perspective expansion, grouping,
    or deduplication.
  - `/match_history` has only the profile lookup join required for the opponent
    name.
  - Relevant tests, lint, and `web` TypeScript/Vite build pass.
  - Manual picker checks confirm top-20 order, right-aligned labels,
    accessibility state, and responsive layout in both themes.
  - Query-plan evidence is recorded; any new index has a demonstrated plan or
    latency benefit.
- **Guardrails**:
  - Do not add caches, materialized views, or indexes preemptively.
  - Do not claim performance improvement from query shape alone; record measured
    plans or timings.

## Dependencies

1. The ongoing refactor must reach a stable diff or provide a clear integration
   point before Task 2 begins.
2. Task 1 determines whether `gold.tour_averages` is an allowed profile-response
   dependency.
3. Task 2 supplies the shared orientation used by Task 3; Task 5 may use direct
   bronze CASE expressions where that is simpler.
4. Task 3 defines the `/players` rank contract required by Task 4.
5. Tasks 3 through 5 must complete before endpoint equivalence and query-plan
   verification in Tasks 6 and 7.

## QA/Testing Scenarios

- Profiled player with matches on all three surfaces.
- Profiled player with no matches and nullable biography values.
- Player with fewer than ten matches and exactly ten-or-more matches.
- Match viewed from both winner and loser perspectives, verifying ranks and all
  side-specific service statistics.
- Tournament containing multiple rounds, verifying only the deepest reached row
  remains in recent tournament history.
- H2H pair in both request orders, verifying stable canonical response
  orientation.
- H2H pair with no meetings and pair with more than five meetings.
- Rank `0` rows excluded from rank history while positive ranks remain ordered.
- Players whose newest positive point observations occur on different dates,
  ranked solely by those per-player latest values with no recency cutoff.
- Two players with equal latest points, receiving stable unique row numbers by
  player-id tiebreak.
- Profiled player with no positive rank points: present in typed search with no
  rank label, absent from the default top 20.
- Empty Home and H2H pickers show at most 20 ranked options; H2H exclusion is
  applied before limiting and backfills the list.
- Fresh and cached MiniSearch indexes return the same point/rank metadata, with
  old cache data invalidated by the versioned key.
- H2H Current rank uses `/players` estimates and performs no rank-history fetches
  for the comparison row.
- Missing/zero serve denominators return `null`, never division errors or false
  zero percentages.
- Existing similarity artifact behavior and prediction/inference results remain
  unchanged.
