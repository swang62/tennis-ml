# Draft: symmetric matchup modeling

## Requirements

- Remove ATP player-id ordering as a source of model-side bias.
- A requested `player_id` must be the `player_*` side and `p_win` must mean that player wins.
- Preserve strict pre-match feature semantics and prevent mirrored rows crossing data splits.
- Treat this as a cross-pipeline refactor, not a localized inference remap.
- Repair every identified training/evaluation integrity issue; do not add a weak/inverted-base promotion gate.
- Remove frontend probability remapping for lower-id canonical model output.
- Produce a full executable plan after the unresolved design choices below are agreed.

## Technical Decisions

- Retain signed differences, paired absolute side features, and shared context; do not reduce to differences only.
- Expand `gold.match_features` to two directional rows per physical match with `(match_id, player_id)` as its identity. `match_id` remains the immutable grouping key.
- All splits, OOF folds, NN early-stopping partitions, artifacts, and metrics are grouped by `match_id`; each physical match contributes total sample weight 1 across its two orientations.
- Per base, score both directions and apply logit antisymmetrization with symmetric probability clipping (`epsilon=1e-6` candidate): `sigmoid((logit(p_ab) - logit(p_ba)) / 2)`.
- Feed antisymmetric base logits to a zero-intercept logistic stacker. The serving result is therefore exactly complementary under a player swap.
- Do not use arithmetic probability averaging as the primary path; it is a valid fallback only if OOF calibration shows the logit method is worse.
- Keep final evaluation to one deterministic orientation per physical match (or paired rows at weight 0.5); do not count mirrors as independent matches.
- Correct NN tuning to use validation, correct its test-tensor source, fit preprocessing inside OOF folds, preserve row/fold identity, pin exact trained base versions, and validate stack column names/order.
- Remove lower-id canonicalization only from model feature orientation and probability/UI mappings. Retain unordered pair canonicalization internally only if needed to perform one shared H2H lookup / two-direction serving calculation.
- Confirmed: exact symmetry applies to each displayed base model and the ensemble; API returns only requested-player probability; existing artifacts require a complete retrain; historical fallback/bio leakage is explicitly accepted and out of scope.
- Confirmed feature scope: correctness repairs plus return strength. No feature-ablation work is included.
- Confirmed: use fixed empirical-Bayes smoothing plus exposure features. Use a neutral 50% Beta(1,1) prior: `(wins + 1) / (opportunities + 2)`, and retain the relevant opportunity/exposure count. Replace raw H2H count/wins with H2H exposure plus smoothed directional advantage.

## Research Findings

- `silver.player_matches` already contains two complementary player perspectives per match.
- `gold.match_features` currently collapses those perspectives to the lower-id side and has a one-row-per-match unique key.
- Inference currently sorts ids before snapshot/profile/H2H assembly.
- The current NN and GBDT are not intrinsically antisymmetric; mirrored training alone can only encourage, not guarantee, complementary probabilities.
- Difference-only features are unnecessary and would discard useful absolute matchup context.
- Current training has integrity bugs beyond orientation: NN uses test data during tuning/early stopping and writes test predictions from training tensors; linear and NN OOF scalers leak fold-validation distributions; OOF rows are ungrouped; artifacts have no row/fold identity; and stacker input order/lineage is insufficiently verified.
- `gold.match_features` currently has one-row `match_id` grain in SQL, dbt contract/PK, snapshot validation, test fixtures, and training notebooks.
- Most current 36 features are defensible but need temporal ablation, not intuition-only removal. Clear immediate feature work: add `return_points_won_pct_diff`; fix same-day train/inference snapshot mismatch; correct carpet surface fallback; add history exposure/smoothing; fix aggregate contract coverage; validate indoor binary input.
- Rank and rank points are correlated but not interchangeable; age and years-pro overlap but are not duplicates; raw H2H wins/counts should be evaluated against smoothed directional H2H advantage plus exposure.
- Existing all-history `tour_averages` fallbacks and current biography embeddings create documented historical leakage for old/cold-start examples. This requires an explicit scope decision because fixing it changes the feature-data architecture.

## Open Questions

None. Planning assumptions above can be refined before implementation if desired.

## Scope Boundaries

- In scope: dbt gold grain/keys, training split/grouping, OOF/stacking behavior, inference orientation, serving response semantics, UI remapping cleanup, hermetic tests, retraining/deployment migration.
- Out of scope unless requested: replacing the three model classes with a new pairwise-ranking architecture.
- Explicitly excluded: new rank-as-of data architecture; time-versioned player biographies; a promotion-quality gate for weak/inverted OOF base models.
