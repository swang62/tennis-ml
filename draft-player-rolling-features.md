# Draft: Player Rolling Features Architecture

## Requirements

- Accept simple inference inputs centered on player IDs and match context.
- Fetch each player's latest rolling state from DuckDB/dbt models.
- Assemble the canonical match-level `FEATURE_COLS` row automatically.
- Keep dbt as the source of truth for feature transformations.
- Preserve leakage-free pre-match features for training.

## Technical Decisions

- Recommend a historical player-centric rolling-feature model plus a one-row-per-player latest view.
- Keep `gold.match_features` as the final training matrix, derived by joining two player snapshots and adding matchup/context features.
- Do not reconstruct player rolling windows in Python.
- Python inference helper should query snapshots, canonicalize sides, and assemble pair/context columns only.
- Inference requires player ID, opponent ID, and surface; match date defaults to today, while tournament and round default to unknown/0 but remain optional inputs.
- Unknown players use neutral fallback values rather than rejecting the prediction.

## Research Findings

- Current `gold.match_features` expands matches to player rows internally, computes rolling windows excluding the current match, then collapses to one canonical match row.
- Selecting a player's latest row from `gold.match_features` is insufficient and one event stale because those rolling values describe state before that row's match.
- Surface-specific form requires the requested surface when selecting/assembling current features.
- Current model also requires `tournament_level` and `round_encoded`; player IDs plus surface alone cannot derive those without defaults or retraining.

## Open Questions

- Define the exact neutral fallback policy, especially ranking and days-since-last-match values.
- Confirm that `gold.match_features` should be rebuilt from the player-feature history rather than retaining duplicate rolling-window SQL.

## Scope Boundaries

- In scope: dbt data-model design, match-feature derivation, inference query/assembly contract.
- Out of scope until decisions are confirmed: implementation, model retraining, serving API changes.
