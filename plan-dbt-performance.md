# dbt transformation performance plan

Goal: reduce dbt runtime for both full-refresh and incremental runs without
changing row grain, feature values, leakage semantics, or H2H ordering.

## Tasks

- [ ] Establish a reproducible performance baseline and parity fixtures
  - Initial timing reference supplied by the user: rolling_features 36.81s,
    tour_averages 35.38s, match_features 194.44s, and the leakage test
    270.39s.
  - Remaining: row counts, key uniqueness, representative checksums, and
    EXPLAIN (ANALYZE, BUFFERS) for each operation when the database is idle.
  - Do not run a full ETL for this task.
  - Capture row counts, key uniqueness, and representative feature checksums.
  - Capture `EXPLAIN (ANALYZE, BUFFERS)` for each model/test exceeding 30 seconds
    when the database is not running another ETL.
  - Record full-refresh and incremental timings separately.

- [ ] Simplify the full leakage test
  - Replace the all-row re-derivation with a deterministic, representative
    physical-match sample that covers cold starts, same-date matches, repeated
    H2H pairs, early history, and latest history.
  - Preserve both directional rows for sampled matches.
  - Carry `(match_id, player_id)` through comparisons and joins.
  - Keep the assertions and tolerance semantics unchanged for sampled rows.

- [ ] Optimize match feature lookups and H2H construction
  - Design indexes and query shapes that help full-refresh and incremental runs.
  - Avoid the broad OR-based historical pair join where an indexed bounded
    lookup can preserve the exact five-meeting contract.
  - Preserve strict prior-date filtering, deterministic tie ordering, and
    directional symmetry.
  - Verify output parity against the current implementation.

- [ ] Redesign rolling feature SQL for full and incremental efficiency
  - Reduce repeated scans/sorts of the full player-match relation.
  - Consolidate compatible surface/window work where possible.
  - Do not rely only on filtering affected players; retain a better full-refresh
    query shape as well as incremental correctness.
  - Verify all rolling values and affected-player rebuild behavior remain exact.

- [ ] Redesign tour-average aggregation
  - Avoid joining every rolling snapshot before reducing to player-level
    activity statistics.
  - Reduce duplicate scans of rolling/player-match inputs where safe.
  - Preserve singleton output, fallback values, weighted rates, and metadata.

- [ ] Run final parity and performance verification
  - Run full-refresh and incremental dbt builds/tests.
  - Confirm all existing tests pass without weakening production contracts.
  - Compare before/after row counts, keys, checksums, and query plans.
  - Document remaining bottlenecks and measured improvements.
