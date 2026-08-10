# Plan: Simplify Dashboard Queries

## Goal

Reduce repeated dashboard database and HTTP requests while keeping match facts
normalized. Permit one deliberate player-grain denormalization: materialize
expensive, reusable directory and career aggregates into dbt-owned
`gold.player_profiles` so requests never rescan hundreds of thousands of match
rows.

Execution must follow the numbered task order. Do not bulk-complete tasks. If
delegating, give one numbered task to one agent; parallelize only tasks that do
not touch the same files or contracts.

## Audited Current State

- `silver.player_matches` already materializes the reusable two-row-per-match
  player perspective. It is normalized and should not be widened merely to
  avoid simple joins.
- `gold.player_profiles` is currently an ingest-owned one-row-per-player identity
  and biography table. It is the natural public relation for materialized
  player-grain aggregates, but dbt must become its sole owner to avoid dual
  writers.
- `/players` already performs one database call, but its requested rank/count
  expansion would repeatedly scan the match fact unless materialized at ETL.
- `/player_profile` is the primary backend problem: five sequential player-data
  queries plus one separate tour-average query.
- `/match_history` already performs one database call. Its silver, bronze, and
  profile joins are acceptable because they preserve normalized ownership.
- `/head_to_head` performs one database call but uses a more complicated
  silver-expansion/deduplication path than the natural one-row bronze grain.
- Home's profile, rank-history, match-history, and similarity requests are
  distinct, parallel, independently cached resources—not row-by-row N+1.
- H2H issues two redundant rank-history requests solely to display current rank.
- `gold.tour_averages` is a legitimate one-row reusable aggregate and remains
  materialized.

## Design Rules

1. Optimize query count first; simple joins are acceptable.
2. Do not copy tournament, round, indoor, opponent identity, or additional
   oriented counters into silver solely to shorten SQL.
3. Materialize expensive reusable player-level aggregates only in
   `gold.player_profiles`; do not create a second player summary relation.
4. Derive detailed return statistics during that gold ETL by joining a player's
   silver row to the opponent's existing perspective for the same `match_id`.
5. Keep histories and event context in their natural normalized relations and
   join them on demand.
6. Keep `gold.tour_averages` separate rather than duplicating singleton values
   across every player row.
7. Add any further index, cache, view, materialization, or denormalized field only
   after measured evidence.

## API and Metric Contracts

### Player directory rank

- `/players` adds nullable `latest_rank_points` and `estimated_rank`.
- Select each player's latest positive rank-points observation by
  `match_date DESC, match_id DESC`.
- Assign unique ranks with `ROW_NUMBER() OVER (
  ORDER BY latest_rank_points DESC, player_id ASC)`.
- Profiles without positive points remain in the directory with null rank data.
- Directory ordering is `estimated_rank NULLS LAST, display_name, player_id`.
- This estimate supplies current rank in pickers, the profile header, and H2H.
- The same `estimated_rank` is exposed by `/player_profile` from the materialized
  gold row.
- `/rank_history` remains raw positive ATP ranking history for the chart only.

### Career service analysis

All rates use sums of numerators divided by sums of denominators with
`DOUBLE PRECISION` and `NULLIF`:

| Metric | Formula |
|---|---|
| First serve in % | `SUM(first_serves_made) / SUM(total_serve_points)` |
| Aces per first serve | `SUM(aces) / SUM(first_serves_made)` |
| First-serve points won % | `SUM(first_serve_points_won) / SUM(first_serves_made)` |
| Second-serve points won % | `SUM(second_serve_points_won) / SUM(total_serve_points - first_serves_made)` |
| Overall serve points won % | `SUM(first_serve_points_won + second_serve_points_won) / SUM(total_serve_points)` |
| Double faults per serve point | `SUM(double_faults) / SUM(total_serve_points)` |
| Aces per service game | `SUM(aces) / SUM(service_games)` |
| Break points saved % | `SUM(break_points_saved) / SUM(break_points_faced)` |

Do not report service hold percentage; service games held are not present.

### Career return analysis

Join each filtered player row `pm` to opponent perspective `opp` on
`opp.match_id = pm.match_id AND opp.player_id = pm.opponent_id`:

| Metric | Formula from opponent perspective |
|---|---|
| Overall return points won % | Existing `SUM(pm.return_points_won) / SUM(pm.return_points_available)` |
| First-serve return points won % | `SUM(opp.first_serves_made - opp.first_serve_points_won) / SUM(opp.first_serves_made)` |
| Second-serve return points won % | `SUM((opp.total_serve_points - opp.first_serves_made) - opp.second_serve_points_won) / SUM(opp.total_serve_points - opp.first_serves_made)` |
| Break-point conversion % | `SUM(opp.break_points_faced - opp.break_points_saved) / SUM(opp.break_points_faced)` |
| Break-point opportunities per return game | `SUM(opp.break_points_faced) / SUM(opp.service_games)` |

Do not report return games won; the source does not contain that outcome.

### Tour comparisons

- Reuse existing singleton service benchmarks.
- Reuse `tour_return_points_won_pct` for overall return quality.
- Derive first-serve return, second-serve return, and break-point conversion tour
  benchmarks as `1 -` their corresponding singleton serve benchmark; preserve
  nulls.
- Add only one missing singleton aggregate:
  `tour_break_point_opportunities_per_return_game =
  SUM(break_points_faced) / NULLIF(SUM(service_games), 0)`.
- Cross join the singleton inside the consolidated profile query so the endpoint
  performs one total database call. Shared inference singleton loading remains
  unchanged.

## Scope

### In scope

- Move raw profile identity/enrichment upstream and let dbt materialize enriched
  one-row-per-player `gold.player_profiles`.
- Materialize directory rank, career, service, return, surface, recent-form, and
  rank-point-trend aggregates at ETL.
- Serve player directory and profile data through one point/simple table query
  each rather than request-time match aggregation.
- Keep rank and match history as one-query normalized reads.
- Simplify H2H to bronze's natural match grain.
- Add complete serve/return career analysis and tour comparisons.
- Add top-20 picker defaults and remove H2H's two rank-history requests.
- Add focused contract/equivalence tests and compare query plans/timings.

### Out of scope

- Widening `silver.player_matches` with redundant context or return columns.
- Any second player-level aggregate relation beyond enriched
  `gold.player_profiles`.
- Duplicating match context, histories, opponent names, or singleton benchmark
  values into player rows.
- Bundling Home's distinct resources into one endpoint.
- Prediction, inference, rolling-feature, training-feature, similarity-artifact,
  or deployment changes.
- Service hold %, return games won, caches, views, or speculative indexes.

## Tasks

### [x] Task 1: Record current query and request baselines

- **Description**:
  - Record SQL call counts for all five dashboard endpoints.
  - Capture `EXPLAIN (ANALYZE, BUFFERS)` for current directory, each profile
    subquery, match history, rank history, and H2H on available representative
    data.
  - Record Home and H2H browser request counts, explicitly distinguishing
    distinct parallel resources from repeated requests.
- **Files**:
  - `src/serving/service.py` (read only)
  - `web/src/pages/Home.tsx` (read only)
  - `web/src/pages/H2H.tsx` (read only)
- **Acceptance Criteria**:
  - Evidence confirms `/player_profile` performs six database calls and H2H
    selection adds two rank-history HTTP/database calls.
  - Baseline timings, buffers, and row counts are available for Task 7.
- **Guardrails**:
  - Do not modify queries, add indexes, or add caches during baseline capture.

### [x] Task 2: Add the one missing tour return benchmark

- **Description**:
  - Extend `gold.tour_averages` with weighted break-point opportunities per
    return game using existing `silver.player_matches` columns.
  - Do not add complement metrics that can be derived from existing singleton
    service fields.
  - Update singleton schema documentation and contract/rate tests.
- **Files**:
  - `dbt/models/gold/tour_averages.sql`
  - `dbt/models/gold/tour_averages.yml`
  - `dbt/tests/gold/tour_averages_contract.sql`
  - `dbt/tests/gold/tour_averages_rate_bounds.sql`
  - `dbt/tests/gold/tour_averages_weighted_rates.sql`
- **Acceptance Criteria**:
  - Singleton remains exactly one row.
  - New value equals weighted `SUM(break_points_faced) /
    NULLIF(SUM(service_games), 0)` and is null only for a zero denominator.
  - Existing default and inference fields are unchanged.
- **Guardrails**:
  - Do not alter inference fallback columns or add detailed return values already
    derivable as complements.

### [x] Task 3: Materialize enriched gold player profiles

- **Description**:
  - Move the current ingest-owned identity/biography table to
    `bronze.player_profiles`, preserving its primary key, ATP fields, Wikipedia
    fields, and upsert behavior.
  - Declare that bronze profile source in dbt and make dbt the sole owner of
    `gold.player_profiles`.
  - Reorder `etl.py --enrich` so optional Wikipedia enrichment finishes in bronze
    before `dbt build`; the same ETL invocation must publish it to gold.
  - Add a PostgreSQL-backed deterministic miniset test that loads every seeded
    profile, runs the complete Wikipedia phase with mocked search/page responses
    for every seeded player, then runs dbt and verifies the enriched fields reach
    gold during that same pipeline execution.
  - Add a `gold.player_profiles` dbt model that preserves all profile rows and
    identity/biography fields, including players with zero matches.
  - Aggregate `silver.player_matches` once per ETL for match count, latest match,
    deterministic latest positive points, estimated rank, all eight service
    metrics, all five return metrics, and hard/grass/clay counts and win rates.
  - Derive return metrics by self-joining the two existing silver perspectives;
    do not add return counters to silver.
  - Join each player's newest `silver.rolling_features` row for recent snapshot
    date and `win_rate_10`.
  - Materialize deterministic earliest/latest positive points and point delta.
  - Use summed numerators/denominators with `DOUBLE PRECISION` and `NULLIF` for
    every rate; never average per-match percentages.
  - Update dbt consumers to use `ref('player_profiles')` and provide an explicit
    one-time local reset/migration instruction. Never silently drop existing
    profile data.
- **Files**:
  - `infra/postgres/init.sql`
  - `src/constants.py`
  - `src/flows/ingest.py`
  - `src/flows/etl.py`
  - `dbt/models/sources.yml`
  - `dbt/dbt_project.yml`
  - `dbt/models/gold/player_profiles.sql` (new)
  - `dbt/models/gold/player_profiles.yml` (new)
  - `dbt/models/gold/match_features.sql`
  - `dbt/models/gold/tour_averages.sql`
  - `tests/test_ingest.py`
  - `tests/test_e2e_ingest_to_inference.py`
  - `tests/test_dbt_helper.py`
- **Acceptance Criteria**:
  - Ingest/enrichment writes only `bronze.player_profiles`; dbt writes only
    `gold.player_profiles`.
  - Gold contains exactly one row per bronze profile, including zero-match
    players.
  - Directory/profile fields are fully materialized at player grain and match
    deterministic fixture calculations.
  - Detailed return aggregates reconcile to opponent silver counts without
    widening silver.
  - Existing inference, training, snapshot, and similarity consumers continue
    reading the stable gold relation.
  - Every deterministic miniset player passes through enrichment-before-dbt
    coverage without automated tests depending on live Wikipedia availability.
- **Guardrails**:
  - Do not copy histories, opponent names, match context, or tour singleton
    values into profile rows.
  - Do not create a second player summary table or allow dual table ownership.
  - Do not perform destructive migration without explicit operator action.

### [x] Task 4: Consolidate and simplify service queries

- **Description**:
  - Rewrite `/players` as a simple ordered read from enriched
    `gold.player_profiles`; perform no request-time match aggregation.
  - Replace the five profile queries with one point query from
    `gold.player_profiles` cross joined to the one-row `gold.tour_averages`
    singleton. Preserve the nested response, three fixed surfaces, null behavior,
    and unknown-player `404`.
  - Map the eight materialized service metrics, five return metrics, recent form,
    rank-point trend, and estimated rank into the response.
  - Rewrite `/rank_history` against `silver.player_matches`; it already has
    normalized player perspective and null ranks.
  - Keep `/match_history` as one normalized query joining silver to bronze for
    event context and profiles for opponent display name. Preserve deepest-round
    selection, deterministic ordering, and limit clamp.
  - Rewrite `/head_to_head` as one direct bronze pair query with no silver join,
    expansion, grouping, or deduplication. Preserve lower-ID response orientation
    and newest-first meetings.
  - Remove obsolete SQL constants, profile singleton-loader call, and unused
    imports. Do not alter the shared singleton loader used by inference.
- **Files**:
  - `src/serving/service.py`
  - `tests/test_service_profile.py`
  - `tests/test_service_data_endpoints.py` (new)
- **Acceptance Criteria**:
  - Every dashboard endpoint performs exactly one database call.
  - `/players` and `/player_profile` perform no request-time match-fact
    aggregation.
  - Profile response contains all service/return metrics and matching tour
    comparisons from one point query.
  - Match history retains normalized joins but performs no extra calls.
  - H2H reads one bronze row per meeting and remains parameterized.
  - Existing envelopes, validation, errors, ordering, and canonical orientation
    remain unchanged except documented additions.
- **Guardrails**:
  - Do not add further materialized tables/columns, caches, SQL-fragment
    abstractions, ORMs, or query builders.
  - Do not optimize away ordinary joins by copying their data.
  - Do not change prediction or inference code paths.

### [x] Task 5: Update frontend rank and profile analysis

- **Description**:
  - Extend `Player` with nullable `latest_rank_points` and `estimated_rank`.
  - Extend `CareerStats` and `TourAverages` with the documented service/return
    metrics.
  - Add rank metadata to MiniSearch stored fields/result mapping and increment
    the serialized local-storage key. Keep name as the only searchable field.
  - Make empty unselected pickers show up to 20 ranked directory entries. Apply
    `exclude` before limiting; suggestions never auto-select.
  - Render right-aligned `#N` labels while preserving listbox, active-descendant,
    keyboard, and pointer behavior.
  - Pass the selected directory player's `estimated_rank` to the profile view
    only as a loading fallback; extend `PlayerProfile` with the same materialized
    `estimated_rank` and use it for the final profile current-rank label. Keep
    `/rank_history` for the chart only.
  - Remove H2H's `lastRank`, two rank-history queries, and unused imports; use
    selected directory records for both current ranks.
  - Replace the narrow career display with `Serve game` and `Return game`
    sections. Show percentage-point deltas for rates and numeric deltas for
    per-game metrics. Render unavailable denominators as `n/a`, never false zero.
- **Files**:
  - `web/src/api.ts`
  - `web/src/pages/Home.tsx`
  - `web/src/components.tsx`
  - `web/src/pages/Profile.tsx`
  - `web/src/pages/H2H.tsx`
  - `web/src/index.css`
- **Acceptance Criteria**:
  - Empty Home and H2H pickers show ordered ranked top-20 defaults with exclusion
    backfill and no loading/no-results flash.
  - Fresh and cached MiniSearch results retain rank metadata.
  - Profile and H2H current-rank labels agree and use directory estimates.
  - Selecting an H2H pair no longer issues rank-history requests.
  - All eight service and five return metrics display with correct units,
    benchmark deltas, null behavior, and responsive layout.
- **Guardrails**:
  - Do not search/boost by rank points, add a frontend test framework, bundle
    Home endpoints, or remove the rank-history chart request.

### [x] Task 6: Add contract and equivalence coverage

- **Description**:
  - Test `/players` latest-positive selection, tie ordering, null rank behavior,
    all-profile preservation, parameter safety, and stable ordering.
  - Test the profile point query and formatter for known/unknown players,
    zero matches, fewer than ten matches, all surfaces, missing denominators,
    latest rolling snapshot, earliest/latest points, and one-call enforcement.
  - Test all service metrics as weighted ratios rather than averages of match
    percentages.
  - Test return metrics from both winner and loser perspectives, including split
    reconciliation, break conversion, opportunities per game, zero denominators,
    and rate bounds.
  - Test rank history, deepest-round match history, H2H both request orders,
    zero meetings, more than five meetings, parameter binding, invalid limits,
    and database failures.
  - Exercise the full deterministic miniset profile path: seed raw ATP profiles,
    mock Wikipedia search and page extraction for every seeded player, enrich
    bronze, run dbt, and assert identity, Wikipedia fields, and materialized
    aggregates in gold.
  - Preserve similarity, inference, leakage, and singleton tests unchanged except
    the additive singleton field fixture.
- **Files**:
  - `tests/test_service_profile.py`
  - `tests/test_service_data_endpoints.py`
  - `tests/test_e2e_ingest_to_inference.py`
  - `tests/test_service_similar.py`
  - `tests/test_inference_features.py`
  - dbt tour-average tests from Task 2
- **Acceptance Criteria**:
  - Tests prove one DB call per dashboard endpoint.
  - Both oriented match sides produce correct service and return output.
  - Response contracts match frontend types and null semantics.
  - Existing prediction, inference, and similarity behavior remains passing.
- **Guardrails**:
  - Do not assert incidental SQL whitespace or weaken existing temporal/model
    tests.

### [x] Task 7: Verify query count, latency, and behavior

- **Description**:
  - Run focused checks during each task, then `just test`, `just lint`, and
    `pnpm --dir web build`.
  - Run dbt build/tests against the deterministic seed and representative full
    data where available.
  - Re-run `EXPLAIN (ANALYZE, BUFFERS)` for all rewritten queries and compare to
    Task 1. Record total calls and endpoint/request latency, not only SQL shape.
  - Manually verify Home/Profile/H2H request counts, ranked defaults, typed
    ranked/unranked search, stale local storage, exclusion/backfill, pointer and
    keyboard operation, rank chart, metric formatting, null states, both themes,
    and responsive layouts.
  - Add an index only if a measured final query problem is not covered by
    existing indexes; record before/after evidence.
- **Files**:
  - `src/serving/service.py`
  - `web/src/**` (verification only unless fixing a discovered regression)
  - `dbt/dbt_project.yml` or `infra/postgres/init.sql` only if an index is proven
    necessary
- **Acceptance Criteria**:
  - Profile database calls fall from six to one.
  - H2H selection removes two rank-history HTTP/database calls.
  - Every dashboard endpoint uses one database call; only the audited
    player-grain profile aggregate and tour singleton are denormalized.
  - Final timings/plans are recorded and no endpoint regresses materially.
  - Relevant dbt tests, Python tests, lint, web build, manual accessibility, and
    visual checks pass.
- **Guardrails**:
  - Do not claim performance from fewer lines or fewer joins.
  - Do not add caches, further denormalized columns, views, materializations, or
    indexes without measured evidence and explicit scope review.

## Dependencies and Execution Order

1. Task 1 captures the baseline before changes.
2. Task 2 establishes the one additive benchmark needed by profile responses.
3. Task 3 establishes profile ownership, enrichment order, and materialized
   player-grain contracts.
4. Task 4 consumes those contracts in backend endpoints.
5. Task 5 consumes final API contracts in the frontend.
6. Task 6 completes cross-layer and full-miniset enrichment coverage after model
   and API contracts stabilize; focused tests still accompany Tasks 2–5.
7. Task 7 is final measured verification only.

## QA Scenarios

- Known profile with zero matches returns identity, zero counts, and null rates
  from one query.
- Latest positive rank points ignore newer zero/null observations.
- Equal points produce deterministic unique estimates by player ID.
- Profile, H2H, and picker current ranks agree; historical chart remains raw ATP
  rank.
- Career rates use weighted summed counts and null zero denominators.
- Winner and loser perspectives produce complementary first/second return rates
  and correct break-point conversion.
- Return split numerators reconcile to overall return points won.
- No service hold or return-games-won metric is fabricated.
- Match history preserves deepest tournament round and deterministic limit.
- H2H remains canonical in both request orders with zero and many meetings.
- Empty pickers show ranked top 20; typed search includes unranked players.
- Fresh and cached MiniSearch indexes preserve rank metadata.
- Prediction, inference, similarity, and training behavior remain unchanged.
- Every deterministic seed profile receives mocked full Wikipedia enrichment in
  bronze before dbt, and the same run publishes it to gold.
