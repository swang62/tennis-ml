"""Pure promotion-gate logic shared by 05_evaluate and its focused tests.

The candidate is scored in the notebook (candidate head over the exact
candidate base matrix). The incumbent is never loaded from MLflow artifacts:
it is scored from the deployed production Bento's private bulk endpoint using
the same ordered raw match contexts, so the Bento rebuilds features with its
own baked contract and runs its own head + bases. This module makes no MLflow
or HTTP calls and mutates nothing; the notebook owns both.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from datetime import date
from typing import Any

import pandas as pd

from src.constants import (
    BATCH_MAX_SIZE_ROWS,
    CHAMPION_ALIAS,
    LINEAGE_MODEL_NAME_KEY,
    LINEAGE_RUN_ID_KEY,
    LINEAGE_VERSION_KEY,
    PRODUCTION_MODEL,
)

# ── Frozen weighted metric/composite policy (unchanged by the incumbent path) ──
METRIC_NAMES = [
    "roc_auc",
    "pr_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "mcc",
    "brier",
]
METRIC_WEIGHTS = {
    "roc_auc": 0.30,
    "f1": 0.20,
    "accuracy": 0.15,
    "pr_auc": 0.10,
    "precision": 0.10,
    "recall": 0.10,
    "mcc": 0.05,
    "brier": 0.00,
}
EPS = 1e-9


def compute_metrics(y_true: object, proba: object, pred: object) -> dict[str, float]:
    """The 8 gate metrics for one stack on one (y, proba, pred) triple."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred)),
        "recall": float(recall_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


def promotion_composite(cand_metrics: dict[str, float], prod_metrics: dict[str, float]) -> float:
    """Weighted relative-delta composite used by the promotion gate.

    `composite = sum(w_i * (cand_i - prod_i) / max(abs(prod_i), EPS))`.
    Positive means the candidate beats the incumbent on the same test matrix.
    """
    return sum(
        w * (cand_metrics[m] - prod_metrics[m]) / max(abs(prod_metrics[m]), EPS)
        for m, w in METRIC_WEIGHTS.items()
    )


def ordered_incumbent_contexts(info: pd.DataFrame) -> list[dict[str, object]]:
    """Minimal raw inference contexts in the exact held-out row order.

    Each context carries only what the production inference builder needs —
    requested-orientation ids, match date, surface, tournament/round encodings,
    and indoor state — never candidate feature rows, so the incumbent
    independently rebuilds every row against its own baked contract.
    """
    if info.empty:
        raise ValueError("holding out an empty evaluation set — nothing to compare")
    contexts: list[dict[str, object]] = []
    for rec in info.to_dict("records"):
        match_date = rec["match_date"]
        if isinstance(match_date, pd.Timestamp):
            match_date = match_date.date()
        contexts.append(
            {
                "player_id": str(rec["player_id"]),
                "opponent_id": str(rec["opponent_id"]),
                "surface": str(rec["surface"]),
                "as_of_date": match_date.isoformat(),
                "tournament_level": int(rec["tournament_level"]),
                "round_encoded": int(rec["round_encoded"]),
                "is_indoor": int(rec["is_indoor"]),
            }
        )
    return contexts


def verify_production_identity(model_info: object, champion: Any) -> None:
    """Require the production Bento to report the exact champion identity.

    `/model-info` returns the baked manifest of the image that is actually
    deployed. Before any incumbent scoring the gate demands production mode and
    an exact match on registered model name, version, and run; anything else is
    a stale or development Bento and fails the evaluation before promotion.
    """
    if champion is None:
        raise RuntimeError("cannot verify production identity without a champion")
    outer = model_info if isinstance(model_info, dict) else {}
    data = outer.get("data")
    if outer.get("ok") is not True or not isinstance(data, dict):
        raise RuntimeError(f"production /model-info unavailable or malformed: {model_info!r}")
    if data.get("mode") != "production":
        raise RuntimeError(
            f"production Bento is not in production mode (mode={data.get('mode')!r}) — "
            "deploy the champion image before evaluating"
        )
    manifest = data.get("manifest")
    if manifest is None:
        raise RuntimeError(
            "production Bento has no baked champion manifest — deploy before evaluating"
        )
    if not isinstance(manifest, dict) or not isinstance(manifest.get("champion"), dict):
        raise TypeError(
            f"production Bento manifest is malformed: expected a dict with a champion "
            f"dict, got {manifest!r}"
        )
    champ = manifest["champion"]
    expected = {
        LINEAGE_MODEL_NAME_KEY: PRODUCTION_MODEL,
        LINEAGE_VERSION_KEY: str(getattr(champion, "version", None)),
        LINEAGE_RUN_ID_KEY: getattr(champion, "run_id", None),
    }
    for key, want in expected.items():
        got = champ.get(key)
        if str(got) != str(want):
            raise RuntimeError(
                f"production Bento identity mismatch ({key}): deployed {got!r}, "
                f"expected {want!r} — evaluate against the deployed champion only"
            )


def check_candidate_cutoff(max_match_date: date, today: date) -> None:
    """Reject a candidate whose data extends beyond the current UTC date."""
    if max_match_date > today:
        raise RuntimeError(
            f"candidate data max match date {max_match_date} is after the current UTC "
            f"date {today} — cannot register a candidate trained with future data"
        )


def score_incumbent(
    contexts: Sequence[dict[str, object]],
    post_batch: Callable[[list[dict[str, object]]], list[dict[str, object]]],
    *,
    chunk_size: int = BATCH_MAX_SIZE_ROWS,
) -> list[float]:
    """Score raw contexts through the incumbent's private bulk endpoint.

    `post_batch` posts one chunk of contexts and returns the parsed response
    records. Each chunk is verified: exact row count, per-row identity and
    input order against the REQUESTED ids (player_id, opponent_id, in order —
    no sorting), and finite `p_win`. Any mismatch raises, so the gate fails
    before promotion instead of comparing shifted probabilities. Probability
    columns can be missing (old served shape), in which case `p_win` is
    required.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    probas: list[float] = []
    for start in range(0, len(contexts), chunk_size):
        chunk = list(contexts[start : start + chunk_size])
        records = post_batch(chunk)
        if not isinstance(records, list):
            raise TypeError(f"incumbent batch returned {type(records).__name__}, expected a list")
        if len(records) != len(chunk):
            raise RuntimeError(
                f"incumbent row-count mismatch: sent {len(chunk)} contexts, got {len(records)} rows"
            )
        for i, (ctx, rec) in enumerate(zip(chunk, records, strict=False)):
            if not isinstance(rec, dict):
                raise TypeError(f"incumbent row {i} is {type(rec).__name__}, expected a dict")
            expected_ids = (str(ctx["player_id"]), str(ctx["opponent_id"]))
            got_ids = (str(rec["player_id"]), str(rec["opponent_id"]))
            if expected_ids != got_ids:
                raise RuntimeError(
                    f"incumbent orientation/identity mismatch at row {i}: "
                    f"expected requested ids {expected_ids!r}, got {got_ids!r}"
                )
            p_win = rec.get("p_win")
            if isinstance(p_win, bool) or not isinstance(p_win, (int, float)):
                raise TypeError(f"incumbent row {i} has an invalid p_win {p_win!r}")
            value = float(p_win)
            if not math.isfinite(value):
                raise RuntimeError(f"incumbent row {i} returned a non-finite p_win {value!r}")
            probas.append(value)
    return probas


def decide_promotion(
    *,
    cand_metrics: dict[str, float],
    prod_metrics: dict[str, float] | None,
    champion_run_id: object,
    candidate_run_id: object,
    force: bool = False,
) -> int:
    """1 promote / 0 skip; metric-only, idempotent, first-promotion aware.

    ``force`` overrides every gate — composite score, idempotency, and
    first-promotion checks all yield 1 — so a manual ``--force-promote`` always
    registers a fresh version (refreshing lineage tags).
    """
    if force:
        return 1
    if champion_run_id is not None and str(champion_run_id) == str(candidate_run_id):
        return 0  # already promoted
    if prod_metrics is None:
        return 1  # first promotion: no incumbent to beat
    return 1 if promotion_composite(cand_metrics, prod_metrics) > 0 else 0


def resolve_champion(client: Any) -> Any | None:
    """Return the champion ModelVersion or None when the alias does not exist.

    Read-only MLflow resolution: evaluation never mutates the registry on the
    incumbent path, and registration happens only after every gate passes.
    """
    try:
        return client.get_model_version_by_alias(PRODUCTION_MODEL, CHAMPION_ALIAS)
    except Exception:
        return None
