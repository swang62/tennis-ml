"""Production drift monitor against the deployed champion Bento.

Entry point for `just check-drift` (or `uv run python src/flows/check_drift.py`).
Runs dbt build to refresh gold data, resolves the MLflow champion, scores all
new matches through the production Bento, and logs drift metrics to MLflow.

Exit 0 on success (including insufficient_data), 1 on failure.
"""

from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import requests
from mlflow.tracking import MlflowClient
from scipy import stats
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

from src import constants
from src.constants import (
    ARTIFACTS,
    BATCH_MAX_SIZE_ROWS,
    BENTO_API_KEY,
    BENTO_API_KEY_HEADER,
    CHAMPION_ALIAS,
    GOLD_TABLE,
    MODEL_INFO_ROUTE,
    PREDICT_BATCH_ROUTE,
    PRODUCTION_BENTO_URL,
    PRODUCTION_MODEL,
)
from src.db.client import to_dataframe
from src.evaluate.promotion import resolve_champion, verify_production_identity
from src.flows.etl import run_dbt_build
from src.utils import load_env, suppress_insecure_tls_warning

LOCK_FILE = ARTIFACTS / ".check_drift.lock"
EXPERIMENT_NAME = "drift_monitoring"


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


def _db_conn_params() -> dict[str, str | int | None]:
    from urllib.parse import unquote, urlsplit

    raw = constants.get_database_url()
    parts = urlsplit(raw)
    db_name = unquote(parts.path.lstrip("/")) if parts.path else None
    return {
        "server_address": parts.hostname,
        "server_port": parts.port,
        "database_name": db_name or None,
    }


def _verify_db_identity(db_meta: dict[str, Any]) -> None:
    expected = _db_conn_params()
    mismatches = []
    for key in ("server_address", "server_port", "database_name"):
        if str(db_meta.get(key)) != str(expected[key]):
            mismatches.append(f"{key}: deployed {db_meta.get(key)!r}, expected {expected[key]!r}")
    if mismatches:
        raise RuntimeError("production Bento database identity mismatch: " + "; ".join(mismatches))


def _champion_cutoff_date(client: MlflowClient) -> date | None:
    champion = resolve_champion(client)
    if champion is None:
        return None
    return datetime.fromtimestamp(champion.creation_timestamp / 1000, tz=UTC).date()


def _post_batch(contexts: list[dict[str, object]]) -> list[dict[str, object]]:
    url = f"{PRODUCTION_BENTO_URL}{PREDICT_BATCH_ROUTE}"
    headers = {"Content-Type": "application/json"}
    if BENTO_API_KEY:
        headers[BENTO_API_KEY_HEADER] = BENTO_API_KEY
    resp = requests.post(url, json=contexts, headers=headers, timeout=120)  # type: ignore[arg-type]
    resp.raise_for_status()
    body: object = resp.json()
    if not isinstance(body, list):
        raise TypeError(
            f"/api/internal/predict-batch returned {type(body).__name__}, expected list"
        )
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
    if isinstance(model_info, dict):
        db_meta = (model_info.get("data") or {}).get("database") or {}
    else:
        db_meta = {}
    _verify_db_identity(db_meta)
    return champion


def _compute_metrics(y_true: list[int], probas: list[float]) -> dict[str, float]:
    pred_list = [1 if p >= 0.5 else 0 for p in probas]
    return {
        "roc_auc": float(roc_auc_score(y_true, probas)) if len(set(y_true)) > 1 else 0.5,
        "pr_auc": float(average_precision_score(y_true, probas)),
        "accuracy": float(accuracy_score(y_true, pred_list)),
        "precision": float(precision_score(y_true, pred_list, zero_division=0.0)),  # type: ignore[arg-type]
        "recall": float(recall_score(y_true, pred_list, zero_division=0.0)),  # type: ignore[arg-type]
        "f1": float(f1_score(y_true, pred_list, zero_division=0.0)),  # type: ignore[arg-type]
        "mcc": float(matthews_corrcoef(y_true, pred_list)),
        "brier": float(brier_score_loss(y_true, probas)),
    }


def _baseline_run_id(client: MlflowClient, experiment_id: str) -> str | None:
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.stage = 'baseline'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    return runs[0].info.run_id if runs else None


def _baseline_artifacts(client: MlflowClient, run_id: str) -> dict[str, Any]:
    artifacts_dir = client.download_artifacts(run_id, "")
    baseline_path = Path(artifacts_dir) / "baseline.json"
    if baseline_path.exists():
        return json.loads(baseline_path.read_text())
    return {}


def _psi(expected: list[float], observed: list[float], bins: int = 10) -> float:
    if not expected or not observed:
        return 0.0
    combined = pd.Series(list(expected) + list(observed))
    if combined.nunique() <= 1:
        return 0.0
    try:
        edges = pd.qcut(combined, min(bins, combined.nunique()), duplicates="drop", retbins=True)[1]  # type: ignore[arg-type]
    except ValueError:
        return 0.0
    e_series = pd.Series(expected)
    o_series = pd.Series(observed)
    e_bins = pd.cut(e_series, bins=edges, include_lowest=True)  # type: ignore[call-overload]
    o_bins = pd.cut(o_series, bins=edges, include_lowest=True)  # type: ignore[call-overload]
    e_dist = e_series.groupby(e_bins, observed=False).size() / len(expected)
    o_dist = o_series.groupby(o_bins, observed=False).size() / len(observed)
    aligned = pd.DataFrame({"e": e_dist, "o": o_dist}).fillna(0.0)
    aligned["e"] = aligned["e"].replace(0.0, 1e-9)
    aligned["o"] = aligned["o"].replace(0.0, 1e-9)
    return float(sum(aligned["o"] * (aligned["o"] / aligned["e"]).apply(math.log)))


def _drift_summary(
    y_true: list[int], probas: list[float], baseline: dict[str, Any]
) -> dict[str, Any]:
    metrics = _compute_metrics(y_true, probas)
    mean_prob = float(pd.Series(probas).mean())
    std_prob = float(pd.Series(probas).std())
    cal_mean = float(pd.Series(y_true).mean())
    result: dict[str, Any] = {
        "n_matches": len(y_true),
        "metrics": metrics,
        "mean_prediction": mean_prob,
        "std_prediction": std_prob,
        "calibration_rate": cal_mean,
    }
    baseline_metrics: dict[str, float] | None = baseline.get("metrics")
    baseline_probas: list[float] | None = baseline.get("probas")
    if baseline_metrics:
        deltas = {
            f"{k}_delta": metrics[k] - baseline_metrics[k] for k in metrics if k in baseline_metrics
        }
        result["metric_deltas"] = deltas
    if baseline_probas:
        ks_result = stats.ks_2samp(probas, baseline_probas)
        result["ks_test_stat"] = float(ks_result.statistic)  # type: ignore[attr-defined]
        result["ks_test_pval"] = float(ks_result.pvalue)  # type: ignore[attr-defined]
        result["psi"] = _psi(baseline_probas, probas)
    return result


def check_drift() -> int:
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
        print(f"Cutoff date (champion creation): {cutoff_date}")

        df = to_dataframe(
            f"SELECT match_id, match_date, player_id, opponent_id, surface, "
            f"is_indoor, tournament_level, round_encoded, match_won "
            f"FROM {GOLD_TABLE} "
            f"WHERE match_date > %s "
            f"ORDER BY match_date, match_id",
        )
        if df is None or df.empty:
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

        print(f"Found {len(df)} drift matches since cutoff")

        contexts: list[dict[str, object]] = []
        y_true: list[int] = []
        for _idx, row in df.iterrows():
            match_date_val: Any = row["match_date"]
            if isinstance(match_date_val, pd.Timestamp):
                match_date_val = match_date_val.date()
            contexts.append(
                {
                    "player_id": str(row["player_id"]),
                    "opponent_id": str(row["opponent_id"]),
                    "surface": str(row["surface"]),
                    "as_of_date": match_date_val.isoformat(),
                    "tournament_level": int(row["tournament_level"]),
                    "round_encoded": int(row["round_encoded"]),
                    "is_indoor": int(row["is_indoor"]),
                }
            )
            y_true.append(int(row["match_won"]))

        print(f"Scoring {len(contexts)} contexts through production Bento...")
        probas = _score_batches(contexts)
        print("Scoring complete.")

        baseline_run_id = _baseline_run_id(client, experiment_id)
        baseline: dict[str, Any] = {}
        if baseline_run_id is not None:
            baseline = _baseline_artifacts(client, baseline_run_id)

        summary = _drift_summary(y_true, probas, baseline)
        print(json.dumps(summary, indent=2, default=str))

        if baseline_run_id is None:
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name="drift_baseline",
                tags={"stage": "baseline", "timestamp": start_ts},
                log_system_metrics=False,
            ) as base_run:
                rid = base_run.info.run_id
                client.log_param(rid, "champion_version", str(champion.version))
                client.log_param(rid, "champion_run_id", champion.run_id)
                client.log_param(rid, "n_matches", len(y_true))
                for key, val in summary["metrics"].items():
                    client.log_metric(rid, key, val)
                client.log_metric(rid, "mean_prediction", summary["mean_prediction"])
                client.log_metric(rid, "std_prediction", summary["std_prediction"])
                client.log_metric(rid, "calibration_rate", summary["calibration_rate"])
                baseline_artifact = {
                    "metrics": summary["metrics"],
                    "probas": probas,
                    "y_true": y_true,
                    "cutoff_date": str(cutoff_date),
                    "champion_version": str(champion.version),
                    "champion_run_id": champion.run_id,
                }
                client.log_text(
                    rid,
                    json.dumps(baseline_artifact, default=str),
                    "baseline.json",
                )

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name="drift_check",
            tags={
                "stage": "check",
                "timestamp": start_ts,
                "baseline_run_id": baseline_run_id or "",
            },
            log_system_metrics=False,
        ) as check_run:
            rid = check_run.info.run_id
            client.log_param(rid, "champion_version", str(champion.version))
            client.log_param(rid, "champion_run_id", champion.run_id)
            client.log_param(rid, "cutoff_date", str(cutoff_date))
            client.log_param(rid, "n_matches", len(y_true))
            for key, val in summary["metrics"].items():
                client.log_metric(rid, key, val)
            client.log_metric(rid, "mean_prediction", summary["mean_prediction"])
            client.log_metric(rid, "std_prediction", summary["std_prediction"])
            client.log_metric(rid, "calibration_rate", summary["calibration_rate"])
            for key, val in summary.get("metric_deltas", {}).items():
                client.log_metric(rid, key, val)
            for key in ("ks_test_stat", "ks_test_pval", "psi"):
                if key in summary:
                    client.log_metric(rid, key, summary[key])
            client.log_text(
                rid,
                json.dumps(summary, indent=2, default=str),
                "drift_summary.json",
            )

        print(f"Drift check complete. Run: {check_run.info.run_id}")
        return 0


if __name__ == "__main__":
    raise SystemExit(check_drift())
