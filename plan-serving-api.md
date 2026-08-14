# Plan: Serving API — symmetric orientation, accurate OpenAPI, strict schemas

## Goal

Make the Bento serving API honest about its symmetric model. Training is already
fully symmetric (antisymmetric-evidence stacker), so every endpoint must
preserve the caller's exact `(player, opponent)` order — never sort, canonicalize,
or re-orient ids. Replace the leaked README in the OpenAPI description with an
accurate manual endpoint reference, and give the bulk prediction endpoint a
strict schema instead of `dict[str, object]`.

## Step 1 (critical) — remove player-id ordering / canonicalization

### Principle

The ensemble is antisymmetric: `p_win(player, opponent) = 1 - p_win(opponent,
player)`. A request in either orientation is valid and swappable. Contract for
every endpoint:

- The **first-supplied id** is the `player` side and the **second-supplied id** is
  the `opponent` side, exactly as received.
- `p_win` is always `P(first-supplied id wins)`.
- H2H meetings, summaries, and `player1_*` fields are reported in the **supplied
  order**, never a sorted order.
- No `sorted([p1, p2])`, no `LEAST/GREATEST`-driven re-orientation of responses,
  no "lower lexicographic id" anywhere user-visible.

### Audit (every reference)

**User-facing — must change:**

| Location | What it does | Fix |
| --- | --- | --- |
| `src/serving/service.py` `_head_to_head` (`lower, higher = sorted([p1, p2])`) | Sorts the two ids; response reports `player1_id=lower` | Keep `p1, p2` as supplied end-to-end |
| `src/serving/service.py` `_H2H_MEETINGS_SQL` call (`[lower, higher, higher, lower]`) | Canonicalizes the unordered-pair lookup | Bind `[p1, p2, p2, p1]` (both branches already cover the pair) |
| `src/serving/service.py` `_head_to_head` (`winner_id == lower`, `loser_id = higher if ...`) | Orients meetings against the lower id | Orient against `p1` |
| `web/src/lib/h2hOrientation.ts` `orientH2H` + `web/src/pages/H2H.tsx` call site | Frontend un-sorts the backend's lower-id canonical response into picker order | Delete the re-orientation; backend now preserves order |
| `tests/test_service_data_endpoints.py` `test_head_to_head_canonical_orientation_both_param_conventions` | Asserts the lower id lands on `player1_id` regardless of request order | Rewrite to assert order preservation (response echoes supplied order) |

**Docs — stale:**

| Location | Fix |
| --- | --- |
| `README.md` "Extra Notes — Canonicalization (lower lexicographic player id becomes `player_*` side)" | Replace with the symmetric/orientation-preserving contract |

**Internal H2H lookups — re-point to bronze (eliminates id comparisons entirely):**

The H2H feature only needs `match_id`, `match_date`, `player1_id`, `player2_id`,
and `winner_id` — all present in `bronze.match_events`, which is one row per
physical match. The current H2H SQL pulls from `silver.player_matches` (bronze
expanded into two directional rows) and then re-collapses it with
`LEAST`/`GREATEST` + `GROUP BY` — a pointless round-trip that is the only source
of id comparisons. Re-pointing both lookups at bronze removes the dedup and the
id ordering, and `winner_id` makes orientation explicit.

| Location | What it does now | Fix |
| --- | --- | --- |
| `src/features/inference.py` `_H2H_PRIOR_SQL` / `_H2H_PRIOR_BULK_SQL` + `_h2h_wins_for_requested` | `CASE WHEN player_id < opponent_id` + re-orient | Query `bronze.match_events` with `winner_id = requested_player`; drop `_h2h_wins_for_requested` |
| `dbt/models/gold/match_features.sql` `pair_meetings` / `prior_h2h` | `LEAST/GREATEST` dedup + `a_won` convention | Join `source('bronze','match_events')` directly; `winner_id = current_match.player_id` |

**Training evaluation orientation — arbitrary deterministic choice, not user-facing:**

| Location | Note |
| --- | --- |
| `notebooks/parameters/03_train_ensemble.ipynb` `chosen_mask = (player_id <= opponent_id)` | Picks one deterministic orientation per physical match to score the symmetric model once. Arbitrary for AUC; not an endpoint contract. Keep unless a non-id tiebreak is preferred |

**Validation join — not ordering:**

| Location | Note |
| --- | --- |
| `src/db/snapshot.py` (`a.player_id < b.player_id`) | Joins each match's two directional rows once for the reciprocal-pair validation; never user-visible |

### Tasks

- [ ] Rewrite `_head_to_head` to preserve `p1`/`p2` order (SQL params, winner/loser
      orientation, response ids, summary counts all keyed on `p1`).
- [ ] Delete `orientH2H` and its call site; frontend H2H reads `player1_id`/`player2_id`
      directly as the picker order.
- [ ] Update `test_head_to_head_*` to assert order preservation.
- [ ] Update `README.md` canonicalization note.
- [ ] Verify `predict_from_ids` / `predict_from_ids_bulk` already preserve order
      (`_normalize_inputs` does — no change needed) and add a regression test if absent.

## Step 2 — replace the leaked README in the OpenAPI description

BentoML auto-copies the build context's `README.md` into the bento as its
`doc`, which becomes the OpenAPI `info.description` (visible at `/api/` via the
nginx proxy). The repo README is not an API reference and should not appear there.

- Set `description=` on `@bentoml.service(...)` (in `src/serving/service.py`) to a
  concise markdown reference that documents every `/api/` endpoint manually,
  including the Starlette GET routes that BentoML's spec generator does not
  introspect (plain `starlette.applications.Starlette` mounts are skipped — only
  FastAPI mounts are merged).
- Accept that the Starlette GET routes will not appear as OpenAPI *paths*; the
  `description` is the manual documentation for them.
- Endpoint list to document:
  - `POST /predict_from_ids` — scalar ids-based prediction.
  - `POST /predict_from_ids_bulk` — bulk ids-based prediction (API-key gated).
  - `GET /players`, `GET /directory_info`, `GET /player_profile`,
    `GET /rank_history`, `GET /match_history`, `GET /head_to_head`,
    `GET /similar_players` — read-only dashboard data.
  - `GET /model_info` (API-key gated), `GET /health`.

## Step 3 — enforce a schema on the bulk prediction endpoint

`predict_from_ids_bulk(self, rows: list[dict[str, object]])` generates
`{"rows": {"type": "array", "items": {"type": "object"}}}` — any dict is accepted.
Replace with a Pydantic row model mirroring `predict_from_ids`'s fields so the
OpenAPI spec documents and validates the shape:

- Define `PredictFromIdsRow(BaseModel)` with `player_id: str`, `opponent_id: str`,
  `surface: str`, and the optional `tournament_level`, `round_encoded`,
  `tournament`, `round`, `as_of_date`, `indoor`.
- Annotate `rows: list[PredictFromIdsRow]` (BentoML keeps the `{"rows": [...]}`
  envelope, so the drift/nginx contract is unchanged, but each item now validates).
- Decide `extra` handling (`forbid` rejects unknown params) so "any random params"
  are refused; confirm BentoML's `IOMixin` preserves the model's `extra` config.

## Step 4 — model-only raw-feature endpoint for drift (deferred)

The ids-based bulk endpoint re-derives features from ids; drift must instead score
the **exact `gold.match_features` rows** (the test set). This needs a model-only
endpoint that accepts finalized `FEATURE_COLS` rows. The stacked ensemble requires
both orientations per match (antisymmetric evidence), so the endpoint must accept
and pair both directional rows (or derive the mirror). This is the service-side
enabler for the Evidently drift rewrite and is tracked here for the drift work.

## Out of scope

- The drift.py rewrite itself (Evidently) — handled separately.
- Converting the Starlette `DATA_APP` to FastAPI to get first-class OpenAPI paths
  (adds a serving-image dependency; manual `description` docs are the accepted
  fallback).
- Removing internal `LEAST/GREATEST` H2H lookup keys (orientation-safe; see audit).
