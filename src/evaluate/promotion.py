"""Pure promotion-gate logic for candidate and champion metrics."""

from __future__ import annotations

import json
import math
from datetime import date
from typing import Any

from src.constants import (
    CHAMPION_ALIAS,
    PRODUCTION_MODEL,
    PROMOTION_TOLERANCE,
)

# ── Frozen promotion-metric policy (unchanged by the incumbent path) ──
METRIC_NAMES = [
    "log_loss",
    "roc_auc",
    "accuracy",
    "brier",
]


def compute_metrics(y_true: object, proba: object, pred: object) -> dict[str, float]:
    """The 4 gate metrics for one stack on one (y, proba, pred) triple."""
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    return {
        "log_loss": float(log_loss(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


def check_candidate_cutoff(max_match_date: date, today: date) -> None:
    """Reject a candidate whose data extends beyond the current UTC date."""
    if max_match_date > today:
        raise RuntimeError(
            f"candidate data max match date {max_match_date} is after the current UTC "
            f"date {today} — cannot register a candidate trained with future data"
        )


def decide_promotion(
    *,
    cand_metrics: dict[str, float],
    prod_metrics: dict[str, float] | None,
    champion_run_id: object,
    candidate_run_id: object,
    force: bool = False,
    champion_feature_hash: object = None,
    candidate_feature_hash: object = None,
) -> int:
    """Return 1 when force/contract/first-promotion passes or log loss/AUC improves within tolerance."""
    if force:
        return 1
    if champion_feature_hash is None or str(champion_feature_hash) != str(candidate_feature_hash):
        return 1  # feature contract changed or legacy champion: refresh the contract
    if champion_run_id is not None and str(champion_run_id) == str(candidate_run_id):
        return 0  # already promoted
    if prod_metrics is None:
        return 1  # first promotion: no incumbent to beat
    log_loss_improves = cand_metrics["log_loss"] < prod_metrics["log_loss"]
    auc_improves = cand_metrics["roc_auc"] > prod_metrics["roc_auc"]
    log_loss_within_tolerance = (
        cand_metrics["log_loss"] <= prod_metrics["log_loss"] + PROMOTION_TOLERANCE
    )
    auc_within_tolerance = cand_metrics["roc_auc"] >= prod_metrics["roc_auc"] - PROMOTION_TOLERANCE
    return int(
        (log_loss_improves and auc_within_tolerance) or (auc_improves and log_loss_within_tolerance)
    )


def resolve_champion(client: Any) -> Any | None:
    """Resolve the champion alias read-only, returning None when it is absent."""
    try:
        return client.get_model_version_by_alias(PRODUCTION_MODEL, CHAMPION_ALIAS)
    except Exception:
        return None


def resolve_champion_feature_contract(
    client: Any,
    champion: Any,
    feature_cols: list[str],
    feature_cols_hash: str,
) -> tuple[bool, Any]:
    """Compare the champion's recorded feature contract against the local one."""

    if champion is None:
        return True, None
    version = client.get_model_version(PRODUCTION_MODEL, champion.version)
    tags = dict(getattr(version, "tags", None) or {})
    raw_cols = tags.get("feature_cols")
    raw_hash = tags.get("feature_cols_hash")
    if raw_cols is None or raw_hash is None:
        return True, None
    try:
        champion_cols = json.loads(raw_cols)
    except (TypeError, ValueError):
        return True, None
    if (
        not isinstance(champion_cols, list)
        or [str(c) for c in champion_cols] != list(feature_cols)
        or str(raw_hash) != str(feature_cols_hash)
    ):
        return True, None
    return False, str(raw_hash)


def read_champion_metrics(client: Any, champion: Any) -> dict[str, float]:
    """Read all finite gate metrics from the champion version's tags."""
    if champion is None:
        raise RuntimeError("cannot read incumbent metrics without a champion")
    version = client.get_model_version(PRODUCTION_MODEL, champion.version)
    tags = dict(getattr(version, "tags", None) or {})
    metrics: dict[str, float] = {}
    for metric in METRIC_NAMES:
        tag = f"metric_{metric}"
        raw = tags.get(tag)
        if raw is None:
            raise RuntimeError(
                f"champion v{champion.version} ({PRODUCTION_MODEL}@{CHAMPION_ALIAS}) is "
                f"missing metric tag {tag!r} — incumbent has no recorded metric to beat"
            )
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"champion v{champion.version} record metric tag {tag!r} is malformed: {raw!r}"
            ) from exc
        if not math.isfinite(value):
            raise RuntimeError(
                f"champion v{champion.version} record metric tag {tag!r} is not finite: {raw!r}"
            )
        metrics[metric] = value
    return metrics
