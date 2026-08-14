"""Production drift monitor against the deployed champion Bento.

Scheduled 30 minutes after the weekly scrape (Monday 06:30 UTC, right after the
06:00 scrape). Runs dbt build to refresh gold data, resolves the MLflow
champion, and compares two size-matched windows from gold.match_features —
the post-cutoff current window and the most recent pre-cutoff reference
window, both scored through the production Bento — using Evidently's
DataDriftPreset (per-feature PSI, prediction PSI, drift share). Current-window
performance is compared against the champion's promotion-pinned metric tags,
and the combined signals map to a healthy / investigate / retrain verdict that
is printed and logged to MLflow. The full drift summary is printed to stdout
so it appears in both Prefect logs and the CLI.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import mlflow
import pandas as pd
import requests
from evidently import Report
from evidently.presets.drift import DataDriftPreset
from mlflow.tracking import MlflowClient
from prefect import flow

from src.constants import (
    ARTIFACTS,
    BATCH_MAX_SIZE_ROWS,
    BENTO_API_KEY,
    BENTO_API_KEY_HEADER,
    CHAMPION_ALIAS,
    DRIFT_AUC_DROP,
    DRIFT_CALIBRATION_DELTA,
    DRIFT_MIN_N_FOR_AUC,
    DRIFT_PRED_PSI_THRESHOLD,
    DRIFT_PSI_MODERATE,
    DRIFT_PSI_SIGNIFICANT,
    DRIFT_REF_MAX,
    DRIFT_REF_MIN,
    DRIFT_SHARE_THRESHOLD,
    EVAL_MAX_DATE_KEY,
    EVAL_SPLIT_SIZE_KEY,
    GOLD_TABLE,
    METRIC_COMPOSITE_KEY,
    METRIC_PREFIX,
    MODEL_INFO_ROUTE,
    PREDICT_BATCH_ROUTE,
    PRODUCTION_BENTO_URL,
    PRODUCTION_MODEL,
    TRAIN_DATA_MAX_DATE_KEY,
    WORK_POOL_NAME,
)
from src.db.client import execute_df
from src.evaluate.promotion import (
    METRIC_NAMES,
    compute_metrics,
    ordered_incumbent_contexts,
    resolve_champion,
    verify_production_identity,
)
from src.features.columns import FEATURE_COLS
from src.flows.etl import run_dbt_build
from src.utils import load_env, suppress_insecure_tls_warning

LOCK_FILE = ARTIFACTS / ".check_drift.lock"
EXPERIMENT_NAME = "drift_monitoring"

DRIFT_DEPLOYMENT_NAME = "drift"
# 30 minutes after the scrape cron (Monday 06:00 UTC), so ETL has finished.
DRIFT_CRON = "30 6 * * 1"

# Both drift windows are pulled from gold with the same column set: identity
# and context (for scoring through the Bento), the label, and every FEATURE_COL.
_DRIFT_WINDOW_COLUMNS = (
    "match_id, match_date, player_id, opponent_id, surface, is_indoor, "
    "tournament_level, round_encoded, match_won, " + ", ".join(FEATURE_COLS)
)


@contextmanager
def _file_lock() -> Any:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    try:
        import portalocker  # type: ignore[import-untyped]

        with portalocker.Lock(str(LOCK_FILE), timeout=300) as fh:
            yield fh
    except ImportError:
        try:
            import fcntl

            fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_RDWR)
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield None
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
        except ImportError:
            fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_RDWR)
            try:
                yield None
            finally:
                os.close(fd)


def _ensure_experiment(client: MlflowClient) -> str:
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is not None:
        return experiment.experiment_id
    return client.create_experiment(EXPERIMENT_NAME)


def _champion_cutoff_date(client: MlflowClient) -> date | None:
    """Latest training-data match date pinned on the champion, from MLflow.

    Falls back to the model-registration timestamp for champions promoted
    before the training-data watermark tag existed.
    """
    champion = resolve_champion(client)
    if champion is None:
        return None
    version = client.get_model_version(PRODUCTION_MODEL, champion.version)
    raw = version.tags.get(TRAIN_DATA_MAX_DATE_KEY)
    if raw:
        return date.fromisoformat(raw)
    return datetime.fromtimestamp(champion.creation_timestamp / 1000, tz=UTC).date()


def _pinned_metrics(client: MlflowClient, champion: Any) -> dict[str, Any]:
    """Champion metric tags pinned at promotion: the 8 gate metrics + eval notes.

    Returns only the tags that exist; the 8 metrics are floats, eval_max_date is
    kept as its stored string.
    """
    tags = client.get_model_version(PRODUCTION_MODEL, champion.version).tags or {}
    pinned: dict[str, Any] = {}
    for name in METRIC_NAMES:
        raw = tags.get(f"{METRIC_PREFIX}{name}")
        if raw is not None:
            pinned[name] = float(raw)
    for key, label in (
        (METRIC_COMPOSITE_KEY, "promotion_composite"),
        (EVAL_SPLIT_SIZE_KEY, "eval_split_size"),
    ):
        raw = tags.get(key)
        if raw is not None:
            pinned[label] = float(raw)
    raw_max_date = tags.get(EVAL_MAX_DATE_KEY)
    if raw_max_date is not None:
        pinned["eval_max_date"] = raw_max_date
    return pinned


def _post_batch(contexts: list[dict[str, object]]) -> list[dict[str, object]]:
    url = f"{PRODUCTION_BENTO_URL}{PREDICT_BATCH_ROUTE}"
    headers = {"Content-Type": "application/json"}
    if BENTO_API_KEY:
        headers[BENTO_API_KEY_HEADER] = BENTO_API_KEY
    # The bulk endpoint's Pydantic schema wraps the list under `rows`; a bare
    # array is rejected with a 400 "Input should be an object".
    resp = requests.post(url, json={"rows": contexts}, headers=headers, timeout=120)  # type: ignore[arg-type]
    resp.raise_for_status()
    body: object = resp.json()
    if not isinstance(body, list):
        raise TypeError(f"{PREDICT_BATCH_ROUTE} returned {type(body).__name__}, expected list")
    return body  # type: ignore[return-value]


def _score_batches(contexts: list[dict[str, object]]) -> list[float]:
    probas: list[float] = []
    for start in range(0, len(contexts), BATCH_MAX_SIZE_ROWS):
        chunk = contexts[start : start + BATCH_MAX_SIZE_ROWS]
        records = _post_batch(chunk)
        if len(records) != len(chunk):
            raise RuntimeError(f"row-count mismatch: sent {len(chunk)}, got {len(records)}")
        for rec in records:
            p_win = rec.get("p_win")
            if isinstance(p_win, bool) or not isinstance(p_win, (int, float)):
                raise TypeError(f"non-finite p_win in batch response: {rec!r}")
            probas.append(float(p_win))
    return probas


def _validate_production(client: MlflowClient) -> Any:
    champion = resolve_champion(client)
    if champion is None:
        raise RuntimeError(
            f"no champion found ({PRODUCTION_MODEL}@{CHAMPION_ALIAS}) — deploy a model first"
        )
    model_info_url = f"{PRODUCTION_BENTO_URL}{MODEL_INFO_ROUTE}"
    headers = {}
    if BENTO_API_KEY:
        headers[BENTO_API_KEY_HEADER] = BENTO_API_KEY
    resp = requests.get(model_info_url, headers=headers, timeout=30)
    resp.raise_for_status()
    model_info: object = resp.json()
    verify_production_identity(model_info, champion)
    return champion


def _pull_windows(cutoff_date: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Current post-cutoff window + size-matched pre-cutoff reference window.

    `reference` is the most recent `n_reference` matches strictly before the
    cutoff, where `n_reference` is the current-window size clamped to the
    DRIFT_REF_MIN/DRIFT_REF_MAX bounds. It is fetched newest-first then reversed
    back into chronological order so both windows share the same row order.
    """
    current = execute_df(
        f"SELECT {_DRIFT_WINDOW_COLUMNS} FROM {GOLD_TABLE} "
        "WHERE match_date > %s ORDER BY match_date, match_id",
        [cutoff_date],
    )
    n_reference = max(DRIFT_REF_MIN, min(len(current), DRIFT_REF_MAX))
    reference = execute_df(
        f"SELECT {_DRIFT_WINDOW_COLUMNS} FROM {GOLD_TABLE} "
        "WHERE match_date < %s ORDER BY match_date DESC, match_id DESC LIMIT %s",
        [cutoff_date, n_reference],
    )
    if reference is not None and not reference.empty:
        reference = reference.iloc[::-1].reset_index(drop=True)
    return current, reference


def _score_window(window: pd.DataFrame) -> pd.DataFrame:
    """Append `p_win` (champion Bento) to a window of FEATURE_COLS + match_won."""
    contexts = ordered_incumbent_contexts(window)
    probas = _score_batches(contexts)
    frame = window[[*FEATURE_COLS, "match_won"]].copy()
    frame["p_win"] = probas
    return frame


def _window_metrics(current_df: pd.DataFrame) -> dict[str, float]:
    """The 8 gate metrics for the current window via the shared promotion helper."""
    probas = current_df["p_win"].tolist()
    return compute_metrics(
        y_true=current_df["match_won"].tolist(),
        proba=probas,
        pred=[1 if p >= 0.5 else 0 for p in probas],
    )


def _as_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _evidently_drift(
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    json_path: Path,
    html_path: Path,
) -> tuple[dict[str, float], float, float]:
    """Run Evidently DataDriftPreset over the two windows; save JSON + HTML.

    All numeric columns are compared with PSI (categorical included), so every
    per-feature score is directly comparable against the PSI threshold table.
    Returns (per-FEATURE_COL PSI, drift share, prediction p_win PSI); drift
    share and prediction PSI come straight from the report payload.
    """
    report = Report(
        metrics=[
            DataDriftPreset(
                drift_share=DRIFT_SHARE_THRESHOLD,
                num_method="psi",
                num_threshold=DRIFT_PSI_SIGNIFICANT,
                cat_method="psi",
                cat_threshold=DRIFT_PSI_SIGNIFICANT,
            )
        ],
        include_tests=True,
    )
    snapshot = report.run(current_data=current_df, reference_data=reference_df)
    payload = json.loads(snapshot.json())
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    snapshot.save_html(str(html_path))

    drift_share = 0.0
    per_column: dict[str, float] = {}
    for metric in payload["metrics"]:
        name = metric["metric_name"]
        if name.startswith("DriftedColumnsCount"):
            drift_share = _as_float(metric["value"]["share"])  # type: ignore[index]
        elif name.startswith("ValueDrift(column="):
            column = name[len("ValueDrift(column=") :].split(",", 1)[0]
            per_column[column] = _as_float(metric["value"])
    per_feature = {column: score for column, score in per_column.items() if column in FEATURE_COLS}
    return per_feature, drift_share, per_column.get("p_win", 0.0)


def _recommendation(
    *,
    drift_share: float,
    prediction_psi: float,
    calibration_delta: float,
    n_current: int,
    auc_drop: float | None,
    per_feature_drift: dict[str, float],
) -> str:
    """Map drift signals to healthy / investigate / retrain (plan thresholds).

    retrain: drift share >= 0.5, prediction PSI >= 0.2, |Δcalibration_rate|
    > 0.05, or (n >= 30 and roc_auc drop > 0.05). investigate: any feature PSI
    in the moderate band. healthy: otherwise.
    """
    if (
        drift_share >= DRIFT_SHARE_THRESHOLD
        or prediction_psi >= DRIFT_PRED_PSI_THRESHOLD
        or calibration_delta > DRIFT_CALIBRATION_DELTA
        or (n_current >= DRIFT_MIN_N_FOR_AUC and auc_drop is not None and auc_drop > DRIFT_AUC_DROP)
    ):
        return "retrain"
    if any(
        DRIFT_PSI_MODERATE <= psi <= DRIFT_PSI_SIGNIFICANT for psi in per_feature_drift.values()
    ):
        return "investigate"
    return "healthy"


@flow(log_prints=True, retries=2)
def drift_flow() -> int:
    """Drift monitor: dbt build, champion validation, score windows, verdict.

    The smoke test counts gold matches newer than the champion's cutoff; when
    there are none the drift check is skipped entirely (insufficient_data).
    """
    load_env()
    suppress_insecure_tls_warning()
    client = MlflowClient()
    experiment_id = _ensure_experiment(client)
    start_ts = datetime.now(UTC).isoformat()

    print("Running dbt build...")
    run_dbt_build()
    print("dbt build complete.")

    with _file_lock():
        champion = _validate_production(client)
        print(f"Champion: {PRODUCTION_MODEL} v{champion.version} (run {champion.run_id})")

        cutoff_date = _champion_cutoff_date(client)
        if cutoff_date is None:
            raise RuntimeError("could not resolve champion cutoff date")
        print(f"Cutoff date (champion training-data watermark): {cutoff_date}")

        current, reference = _pull_windows(cutoff_date)
        if current is None or current.empty:
            print("insufficient_data: no matches found after champion cutoff")
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name="drift_insufficient_data",
                tags={"status": "insufficient_data", "timestamp": start_ts},
                log_system_metrics=False,
            ) as run:
                rid = run.info.run_id
                client.log_param(rid, "champion_version", str(champion.version))
                client.log_param(rid, "champion_run_id", champion.run_id)
                client.log_param(rid, "cutoff_date", str(cutoff_date))
                client.log_param(rid, "n_matches", 0)
            return 0

        if reference is None or reference.empty:
            raise RuntimeError(
                f"no reference matches before cutoff {cutoff_date} — cannot compare drift"
            )

        pinned_metrics = _pinned_metrics(client, champion)
        print(f"Pinned champion metrics: {pinned_metrics!r}")

        print(
            f"Scoring {len(current)} current + {len(reference)} reference "
            "matches through production Bento..."
        )
        current_df = _score_window(current)
        reference_df = _score_window(reference)
        print("Scoring complete.")

        report_json = ARTIFACTS / f"drift_report_{cutoff_date.isoformat()}_v{champion.version}.json"
        per_feature_drift, drift_share, prediction_psi = _evidently_drift(
            current_df, reference_df, report_json, report_json.with_suffix(".html")
        )
        print(f"Evidently report saved: {report_json}")

        current_metrics = _window_metrics(current_df)
        calibration_rate_current = float(current_df["match_won"].mean())
        calibration_rate_reference = float(reference_df["match_won"].mean())
        calibration_delta = abs(calibration_rate_current - calibration_rate_reference)
        metric_deltas = {
            name: current_metrics[name] - pinned_metrics[name]
            for name in METRIC_NAMES
            if name in pinned_metrics
        }
        pinned_roc_auc = pinned_metrics.get("roc_auc")
        auc_drop = (
            pinned_roc_auc - current_metrics["roc_auc"] if pinned_roc_auc is not None else None
        )
        recommendation = _recommendation(
            drift_share=drift_share,
            prediction_psi=prediction_psi,
            calibration_delta=calibration_delta,
            n_current=len(current_df),
            auc_drop=auc_drop,
            per_feature_drift=per_feature_drift,
        )
        retrain_required = recommendation == "retrain"

        summary: dict[str, Any] = {
            "n_current": len(current_df),
            "n_reference": len(reference_df),
            "cutoff_date": cutoff_date.isoformat(),
            "champion_version": str(champion.version),
            "pinned_metrics": pinned_metrics,
            "current_metrics": current_metrics,
            "metric_deltas": metric_deltas,
            "drift_share": drift_share,
            "prediction_psi": prediction_psi,
            "calibration_rate_current": calibration_rate_current,
            "calibration_rate_reference": calibration_rate_reference,
            "calibration_delta": calibration_delta,
            "recommendation": recommendation,
            "retrain_required": retrain_required,
            "per_feature_drift": per_feature_drift,
        }
        print(json.dumps(summary, indent=2, default=str))
        print(f"VERDICT: {recommendation} (retrain_required={retrain_required})")

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name="drift_check",
            tags={
                "stage": "check",
                "timestamp": start_ts,
                "recommendation": recommendation,
                "retrain_required": str(retrain_required),
            },
            log_system_metrics=False,
        ) as check_run:
            rid = check_run.info.run_id
            client.log_param(rid, "champion_version", str(champion.version))
            client.log_param(rid, "champion_run_id", champion.run_id)
            client.log_param(rid, "cutoff_date", str(cutoff_date))
            client.log_param(rid, "n_current", len(current_df))
            client.log_param(rid, "n_reference", len(reference_df))
            for name, value in current_metrics.items():
                client.log_metric(rid, name, value)
            for name, value in metric_deltas.items():
                client.log_metric(rid, f"{name}_delta", value)
            client.log_metric(rid, "drift_share", drift_share)
            client.log_metric(rid, "prediction_psi", prediction_psi)
            client.log_metric(rid, "calibration_rate_current", calibration_rate_current)
            client.log_metric(rid, "calibration_rate_reference", calibration_rate_reference)
            client.log_metric(rid, "calibration_delta", calibration_delta)
            client.log_text(
                rid,
                json.dumps(summary, indent=2, default=str),
                "drift_summary.json",
            )
            client.log_artifact(rid, str(report_json))
            client.log_artifact(rid, str(report_json.with_suffix(".html")))

        print(f"Drift check complete. Run: {check_run.info.run_id}")
        return 0


def register_deployment() -> None:
    """Create/update the Monday drift deployment (idempotent by name).

    Scheduled 30 minutes after the scrape cron (Monday 06:30 UTC), on the host
    ``tennis-pool`` work pool alongside scrape and ETL.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    deployment = cast(
        Any,
        drift_flow.from_source(
            source=str(repo_root),
            entrypoint="src/flows/drift.py:drift_flow",
        ),
    )
    deployment.deploy(
        name=DRIFT_DEPLOYMENT_NAME,
        work_pool_name=WORK_POOL_NAME,
        cron=DRIFT_CRON,
        build=False,
        ignore_warnings=True,
        print_next_steps=False,
    )
    print(f"Registered deployment {DRIFT_DEPLOYMENT_NAME!r} (cron {DRIFT_CRON})")


if __name__ == "__main__":
    drift_flow()
