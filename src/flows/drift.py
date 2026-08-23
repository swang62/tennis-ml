"""Monitor drift and current-window performance against the deployed champion."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import mlflow
import numpy as np
import pandas as pd
import requests
from evidently import Report
from evidently.presets.drift import DataDriftPreset
from mlflow.tracking import MlflowClient
from prefect import flow

from src.config import suppress_insecure_tls_warning
from src.constants import (
    ARTIFACTS,
    BATCH_MAX_SIZE_ROWS,
    BENTO_API_KEY,
    BENTO_API_KEY_HEADER,
    CHAMPION_ALIAS,
    DRIFT_AUC_DROP,
    DRIFT_CALIBRATION_DELTA,
    DRIFT_MIN_N_FOR_AUC,
    DRIFT_MIN_N_FOR_CHECK,
    DRIFT_PRED_PSI_THRESHOLD,
    DRIFT_PSI_MODERATE,
    DRIFT_PSI_SIGNIFICANT,
    DRIFT_REF_MAX,
    DRIFT_REF_MIN,
    DRIFT_SHARE_THRESHOLD,
    EVAL_MAX_DATE_KEY,
    EVAL_SPLIT_SIZE_KEY,
    METRIC_PREFIX,
    MODEL_INFO_ROUTE,
    PREDICT_BATCH_ROUTE,
    PRODUCTION_BENTO_URL,
    PRODUCTION_MODEL,
    TRAIN_DATA_MAX_DATE_KEY,
    WORK_POOL_NAME,
    load_env,
)
from src.db.client import execute_df
from src.evaluate.promotion import (
    METRIC_NAMES,
    compute_metrics,
    resolve_champion,
)
from src.serving.service import (
    PredictFromIdsRow,
    Round,
    Surface,
    TournamentLevel,
)

LOCK_FILE = ARTIFACTS / ".check_drift.lock"
EXPERIMENT_NAME = "drift-monitor"

DRIFT_DEPLOYMENT_NAME = "drift"
DRIFT_CRON = "0 20 1 * *"

DRIFT_FEATURE_COLS = [
    "player1_first_serve_pct",
    "player1_serve_win_pct",
    "player1_ace_rate",
    "player1_df_rate",
    "player2_first_serve_pct",
    "player2_serve_win_pct",
    "player2_ace_rate",
    "player2_df_rate",
]
DRIFT_ANALYSIS_COLUMNS = [*DRIFT_FEATURE_COLS, "match_won", "p_win"]

_BRONZE_WINDOW_COLUMNS: tuple[str, ...] = (
    "match_id",
    "match_date",
    "player1_id",
    "player2_id",
    "winner_id",
    "surface",
    "tournament",
    "round",
    "best_of",
    "COALESCE(is_indoor, 0) AS is_indoor",
    "CAST(player1_first_serves_made AS DOUBLE PRECISION) / NULLIF(player1_total_serve_points, 0) AS player1_first_serve_pct",
    "CAST(player1_first_serve_points_won + player1_second_serve_points_won AS DOUBLE PRECISION) / NULLIF(player1_total_serve_points, 0) AS player1_serve_win_pct",
    "CAST(player1_aces AS DOUBLE PRECISION) / NULLIF(player1_first_serves_made, 0) AS player1_ace_rate",
    "CAST(player1_double_faults AS DOUBLE PRECISION) / NULLIF(player1_total_serve_points, 0) AS player1_df_rate",
    "CAST(player2_first_serves_made AS DOUBLE PRECISION) / NULLIF(player2_total_serve_points, 0) AS player2_first_serve_pct",
    "CAST(player2_first_serve_points_won + player2_second_serve_points_won AS DOUBLE PRECISION) / NULLIF(player2_total_serve_points, 0) AS player2_serve_win_pct",
    "CAST(player2_aces AS DOUBLE PRECISION) / NULLIF(player2_first_serves_made, 0) AS player2_ace_rate",
    "CAST(player2_double_faults AS DOUBLE PRECISION) / NULLIF(player2_total_serve_points, 0) AS player2_df_rate",
)


@contextmanager
def _file_lock() -> Any:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    import fcntl

    fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        yield None
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _ensure_experiment(client: MlflowClient) -> str:
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else client.create_experiment(EXPERIMENT_NAME)
    )
    client.set_experiment_tag(experiment_id, "pipeline", "drift")
    return experiment_id


def _champion_cutoff_date(client: Any) -> date | None:
    """Return the champion's pinned training-data cutoff, with a timestamp fallback."""
    champion = resolve_champion(client)
    if champion is None:
        return None
    version = client.get_model_version(PRODUCTION_MODEL, champion.version)
    raw = version.tags.get(TRAIN_DATA_MAX_DATE_KEY)
    if raw:
        return date.fromisoformat(raw)
    return datetime.fromtimestamp(champion.creation_timestamp / 1000, tz=UTC).date()


def _pinned_metrics(client: Any, champion: Any) -> dict[str, Any]:
    """Return available promotion-pinned metrics and evaluation tags."""
    tags = client.get_model_version(PRODUCTION_MODEL, champion.version).tags or {}
    pinned: dict[str, Any] = {}
    for name in METRIC_NAMES:
        raw = tags.get(f"{METRIC_PREFIX}{name}")
        if raw is not None:
            pinned[name] = float(raw)
    for key, label in ((EVAL_SPLIT_SIZE_KEY, "eval_split_size"),):
        raw = tags.get(key)
        if raw is not None:
            pinned[label] = float(raw)
    raw_max_date = tags.get(EVAL_MAX_DATE_KEY)
    if raw_max_date is not None:
        pinned["eval_max_date"] = raw_max_date
    return pinned


def _post_batch(contexts: list[dict[str, str | int | None]]) -> list[dict[str, object]]:
    url = f"{PRODUCTION_BENTO_URL}{PREDICT_BATCH_ROUTE}"
    headers = {"Content-Type": "application/json"}
    if BENTO_API_KEY:
        headers[BENTO_API_KEY_HEADER] = BENTO_API_KEY
    # The bulk schema requires the list under `rows`.
    resp = requests.post(url, json={"rows": contexts}, headers=headers, timeout=120)
    resp.raise_for_status()
    body: object = resp.json()
    if not isinstance(body, list):
        raise TypeError(f"{PREDICT_BATCH_ROUTE} returned {type(body).__name__}, expected list")
    return [dict(record) for record in body if isinstance(record, dict)]


def _score_batches(contexts: list[dict[str, str | int | None]]) -> list[float]:
    probas: list[float] = []
    for start in range(0, len(contexts), BATCH_MAX_SIZE_ROWS):
        chunk = contexts[start : start + BATCH_MAX_SIZE_ROWS]
        records = _post_batch(chunk)
        if len(records) != len(chunk):
            raise RuntimeError(f"row-count mismatch: sent {len(chunk)}, got {len(records)}")
        for i, rec in enumerate(records):
            # Refuse reordered responses because they pair probabilities with wrong labels.
            request = chunk[i]
            if (
                rec.get("player_id") != request["player_id"]
                or rec.get("opponent_id") != request["opponent_id"]
            ):
                raise RuntimeError(
                    f"bulk response row {i} mismatches request row {start + i}: sent "
                    f"{request['player_id']} vs {request['opponent_id']}, got "
                    f"{rec.get('player_id')} vs {rec.get('opponent_id')} — "
                    "response order/identity cannot be trusted, refusing to score"
                )
            p_win = rec.get("p_win")
            if isinstance(p_win, bool) or not isinstance(p_win, (int, float)):
                raise TypeError(f"non-finite p_win in batch response: {rec!r}")
            probas.append(float(p_win))
    return probas


def _validate_production(client: Any) -> Any:
    """Resolve the champion and verify that the configured Bento responds."""
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
    return champion


def _pull_windows(cutoff_date: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return current and size-matched pre-cutoff physical-match windows."""
    current = execute_df(
        f"SELECT {', '.join(_BRONZE_WINDOW_COLUMNS)} FROM bronze.match_events "
        "WHERE match_date > %s ORDER BY match_date, match_id",
        [cutoff_date],
    )
    n_reference = max(DRIFT_REF_MIN, min(len(current), DRIFT_REF_MAX))
    reference = execute_df(
        f"SELECT {', '.join(_BRONZE_WINDOW_COLUMNS)} FROM bronze.match_events "
        "WHERE match_date < %s ORDER BY match_date DESC, match_id DESC LIMIT %s",
        [cutoff_date, n_reference],
    )
    if reference is not None and not reference.empty:
        reference = reference.iloc[::-1].reset_index(drop=True)
    return current, reference


def _validate_physical_matches(window: pd.DataFrame) -> None:
    """Require each physical match to identify exactly one winning player."""
    if window.empty:
        return
    missing = {"player1_id", "player2_id", "winner_id"} - set(window.columns)
    if missing:
        raise ValueError(f"bronze window missing required columns: {sorted(missing)}")
    missing_player_rows = int(window[["player1_id", "player2_id"]].isna().sum().sum())
    if missing_player_rows:
        raise ValueError(
            f"{missing_player_rows} bronze rows have a missing player id; "
            "both sides of a physical match must carry valid ids"
        )
    winner = window["winner_id"]
    missing_rows = winner.isna()
    if missing_rows.any():
        raise ValueError(
            f"{int(missing_rows.sum())} bronze rows have a missing winner_id; "
            "every physical match must have exactly one winner"
        )
    equals_p1 = winner == window["player1_id"]
    equals_p2 = winner == window["player2_id"]
    ambiguous = (equals_p1 & equals_p2).sum()
    if ambiguous:
        raise ValueError(
            f"{int(ambiguous)} bronze rows have winner_id equal to both players; "
            "a match cannot be won by both sides"
        )
    invalid = int((~(equals_p1 | equals_p2)).sum())
    if invalid:
        raise ValueError(
            f"{invalid} bronze rows have winner_id equal to neither player; "
            "winner_id must be exactly player1_id or player2_id"
        )


def _validate_expanded_window(frame: pd.DataFrame) -> None:
    """Require adjacent orientations to be complementary and the batch balanced."""
    if frame.empty:
        raise ValueError("expanded drift frame is empty")
    if "match_won" not in frame.columns:
        raise ValueError("expanded drift frame missing required column 'match_won'")
    labels = pd.to_numeric(frame["match_won"], errors="coerce")
    if labels.isna().any():
        raise ValueError(
            f"expanded drift frame has {int(labels.isna().sum())} non-numeric match_won labels"
        )
    if not set(labels) <= {0, 1}:
        raise ValueError(
            f"expanded drift frame has match_won labels outside {{0, 1}}: "
            f"{sorted(set(labels) - {0, 1})[:5]}"
        )
    n = len(frame)
    if n % 2 != 0:
        raise ValueError(f"expanded drift frame has odd row count {n}")
    label_arr = labels.to_numpy()
    pair_sums = label_arr[0::2] + label_arr[1::2]
    bad_pairs = np.where(pair_sums != 1)[0]
    if len(bad_pairs):
        raise ValueError(
            f"{len(bad_pairs)} adjacent orientation pairs do not sum to 1 "
            f"(first at row {int(bad_pairs[0]) * 2}); each physical match must "
            "yield exactly one match_won=1 and one match_won=0"
        )
    if float(labels.mean()) != 0.5:
        raise ValueError(
            f"expanded drift frame labels are not balanced: mean {float(labels.mean()):.4f} != 0.5"
        )


def _expand_orientations(window: pd.DataFrame) -> pd.DataFrame:
    """Expand each physical match into adjacent, complementary orientations."""
    _validate_physical_matches(window)
    if window.empty:
        return window.copy()
    first = window.copy()
    first["player_id"] = window["player1_id"]
    first["opponent_id"] = window["player2_id"]
    first["match_won"] = (window["winner_id"] == window["player1_id"]).astype(int)
    second = window.copy()
    second["player_id"] = window["player2_id"]
    second["opponent_id"] = window["player1_id"]
    second["match_won"] = (window["winner_id"] == window["player2_id"]).astype(int)
    expanded = pd.concat([first, second], ignore_index=True)
    interleave = np.repeat(np.arange(len(window)), 2) + np.tile([0, len(window)], len(window))
    expanded = expanded.iloc[interleave].reset_index(drop=True)
    _validate_expanded_window(expanded)
    return expanded


def _observation_contexts(window: pd.DataFrame) -> list[dict[str, object]]:
    """Build public Bento contexts from bronze-only drift observations."""
    contexts: list[dict[str, object]] = []
    for rec in window.to_dict("records"):
        match_date = rec["match_date"]
        if isinstance(match_date, pd.Timestamp):
            match_date = match_date.date()
        contexts.append(
            {
                "player_id": str(rec["player_id"]),
                "opponent_id": str(rec["opponent_id"]),
                "surface": str(rec["surface"]),
                "as_of_date": match_date.isoformat(),
                "tournament": rec["tournament"] or None,
                "round": None if pd.isna(rec["round"]) else rec["round"],
                "best_of": int(rec["best_of"]),
                "is_indoor": int(rec["is_indoor"]),
            }
        )
    return contexts


def _validated_contexts(contexts: list[dict[str, object]]) -> list[dict[str, str | int | None]]:
    """Validate and normalize contexts at the Bento request boundary."""
    accepted_rounds = {r.value for r in Round}
    accepted_tournaments = {t.value for t in TournamentLevel}
    accepted_surfaces = {s.value for s in Surface}
    validated: list[dict[str, str | int | None]] = []
    for context in contexts:
        normalized = dict(context)
        if normalized.get("round") is not None and normalized["round"] not in accepted_rounds:
            normalized["round"] = None
        if (
            normalized.get("tournament") is not None
            and normalized["tournament"] not in accepted_tournaments
        ):
            normalized["tournament"] = None
        if normalized.get("surface") is not None and normalized["surface"] not in accepted_surfaces:
            normalized["surface"] = "hard"
        row = PredictFromIdsRow.model_validate(normalized)
        validated.append(
            {
                "player_id": row.player_id,
                "opponent_id": row.opponent_id,
                "surface": row.surface.value,
                "tournament": row.tournament.value if row.tournament else None,
                "round": row.round.value if row.round else None,
                "best_of": row.best_of.value,
                "as_of_date": row.as_of_date.isoformat(),
                "is_indoor": row.is_indoor,
            }
        )
    return validated


def _scored_diagnostics(frame: pd.DataFrame) -> str:
    """Return diagnostic JSON with ids, labels, probabilities, and predictions."""
    diag = frame[["match_id", "player_id", "opponent_id", "match_won", "p_win"]].copy()
    diag["pred"] = (pd.to_numeric(diag["p_win"], errors="coerce") >= 0.5).astype(int)
    return json.dumps({"scored_rows": diag.to_dict("records")}, indent=2, default=str)


def _validate_scored_frame(frame: pd.DataFrame) -> None:
    """Require finite in-range probabilities and reject an all-tie scoring batch."""
    required = {"match_id", "player_id", "opponent_id", "match_won", "p_win"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"scored frame missing required columns: {sorted(missing)}")
    p_win = pd.to_numeric(frame["p_win"], errors="coerce")
    non_finite = int((~np.isfinite(p_win)).sum())
    if non_finite:
        raise ValueError(
            f"{non_finite} scored rows have a non-finite p_win; refusing to compute "
            f"metrics\n{_scored_diagnostics(frame)}"
        )
    out_of_range = int(((p_win < 0) | (p_win > 1)).sum())
    if out_of_range:
        raise ValueError(
            f"{out_of_range} scored rows have p_win outside [0, 1]; refusing to compute "
            f"metrics\n{_scored_diagnostics(frame)}"
        )
    if len(frame) and bool((p_win == 0.5).all()):
        raise ValueError(
            "every scored p_win is exactly 0.5 (constant-tie batch); a model with no "
            f"signal on any match cannot be drift-scored\n{_scored_diagnostics(frame)}"
        )


def _score_window(window: pd.DataFrame) -> pd.DataFrame:
    """Append champion ``p_win`` to a validated drift-observation window."""
    _validate_expanded_window(window)
    contexts = _validated_contexts(_observation_contexts(window))
    probas = _score_batches(contexts)
    traceable = window[["match_id", "player_id", "opponent_id", "match_won"]].copy()
    traceable["p_win"] = probas
    _validate_scored_frame(traceable)
    frame = window[[c for c in DRIFT_ANALYSIS_COLUMNS if c != "p_win"]].copy()
    frame["p_win"] = probas
    return frame[DRIFT_ANALYSIS_COLUMNS]


def _window_metrics(current_df: pd.DataFrame) -> dict[str, float]:
    """The 4 gate metrics for the current window via the shared promotion helper."""
    probas = current_df["p_win"].tolist()
    return compute_metrics(
        y_true=current_df["match_won"].tolist(),
        proba=probas,
        pred=[1 if p >= 0.5 else 0 for p in probas],
    )


def _as_float(value: object) -> float:
    if not isinstance(value, (str, int, float)):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _evidently_drift(
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    json_path: Path,
    html_path: Path,
) -> tuple[dict[str, float], float, float]:
    """Run Evidently drift checks and return feature, share, and prediction PSI."""
    evidently_columns = [*DRIFT_FEATURE_COLS, "p_win"]
    report = Report(
        metrics=[
            DataDriftPreset(
                columns=evidently_columns,
                drift_share=DRIFT_SHARE_THRESHOLD,
                num_method="psi",
                num_threshold=DRIFT_PSI_SIGNIFICANT,
            )
        ],
        include_tests=True,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        snapshot = report.run(
            current_data=current_df[evidently_columns],
            reference_data=reference_df[evidently_columns],
        )
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
    per_feature = {
        column: score for column, score in per_column.items() if column in DRIFT_FEATURE_COLS
    }
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
    """Map drift signals to ``healthy``, ``investigate``, or ``retrain``."""
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


@flow(log_prints=True)
def drift_flow(cutoff: date | None = None) -> int:
    """Build ETL, score current/reference windows, and publish a drift verdict."""
    load_env()
    suppress_insecure_tls_warning()
    client = MlflowClient()
    experiment_id = _ensure_experiment(client)
    start_ts = datetime.now(UTC).isoformat()

    with _file_lock():
        champion = _validate_production(client)
        print(f"Champion: {PRODUCTION_MODEL} v{champion.version} (run {champion.run_id})")

        cutoff_date = cutoff if cutoff is not None else _champion_cutoff_date(client)
        if cutoff_date is None:
            raise RuntimeError("could not resolve champion cutoff date")
        cutoff_source = "override" if cutoff is not None else "champion training-data watermark"
        print(f"Cutoff date ({cutoff_source}): {cutoff_date}")

        current, reference = _pull_windows(cutoff_date)
        if current is None or len(current) < DRIFT_MIN_N_FOR_CHECK:
            n_matches = 0 if current is None else len(current)
            print(
                f"insufficient_data: {n_matches} matches found after cutoff "
                f"(minimum {DRIFT_MIN_N_FOR_CHECK})"
            )
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name="skip-insufficient-data",
                tags={
                    "pipeline": "drift",
                    "status": "insufficient_data",
                    "timestamp": start_ts,
                },
                log_system_metrics=False,
            ) as run:
                rid = run.info.run_id
                client.log_param(rid, "champion_version", str(champion.version))
                client.log_param(rid, "champion_run_id", champion.run_id)
                client.log_param(rid, "cutoff_date", str(cutoff_date))
                client.log_param(rid, "n_matches", n_matches)
            return 0

        if reference is None or reference.empty:
            raise RuntimeError(
                f"no reference matches before cutoff {cutoff_date} — cannot compare drift"
            )

        pinned_metrics = _pinned_metrics(client, champion)
        print(f"Pinned champion metrics: {pinned_metrics!r}")

        print(
            f"Scoring {len(current)} current + {len(reference)} reference "
            "physical matches through production Bento..."
        )
        current_df = _score_window(_expand_orientations(current))
        reference_df = _score_window(_expand_orientations(reference))
        print("Scoring complete.")

        drift_artifacts = ARTIFACTS / "drift"
        drift_artifacts.mkdir(parents=True, exist_ok=True)
        report_json = (
            drift_artifacts / f"drift_report_{cutoff_date.isoformat()}_v{champion.version}.json"
        )
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
            run_name="drift-check",
            tags={
                "pipeline": "drift",
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
    """Create or update the drift deployment."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cutoff",
        type=date.fromisoformat,
        help="override the champion training-data watermark cutoff (YYYY-MM-DD)",
    )
    drift_flow(cutoff=parser.parse_args().cutoff)
