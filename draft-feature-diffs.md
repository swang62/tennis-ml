# Draft: feature-diffs

## Requirements
- Plan four additional match-prediction feature groups: surface-specific form, rest/fatigue, opponent-quality-adjusted form, and decisiveness/competitiveness.
- Keep the same 10-match lookback window.
- Avoid adding many feature columns.

## Technical Decisions
- Pending: confirm compact representation, candidate definitions, and data-quality boundaries.

## Research Findings
- Current model has 16 directional difference features, shared 10-match rolling snapshots, and a two-column H2H representation.
- Feature contract is shared across dbt training rows, inference, deployment, and serving.

## Open Questions
- Which compact metrics should represent each requested group?
- Whether rest means days since previous match only or also match load.
- Exact rank-adjusted-form construction.
- Score/retirement-data completeness and inclusion policy for competitiveness.

## Scope Boundaries
- No implementation in this planning session.
- Do not widen rolling window or add speculative granular feature families.
