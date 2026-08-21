#!/usr/bin/env python3
"""Select a calibration temperature from existing cross-fitted ensemble predictions."""

import json

import pandas as pd

from src.constants import CALIBRATION_STATE, DATA_PROCESSED
from src.evaluate.calibration import select_temperature


def calibrate() -> None:
    """Write the selection-approved temperature without training or promoting."""
    oof = pd.read_parquet(DATA_PROCESSED / "oof_stack_cv.parquet")
    info = pd.read_parquet(DATA_PROCESSED / "info_train.parquet").reset_index(drop=True)
    labels = pd.read_parquet(DATA_PROCESSED / "y_train.parquet")["y"].reset_index(drop=True)
    if len(info) != len(labels):
        raise ValueError("info_train and y_train must be row-aligned")
    chosen = info["player_id"].astype(str) <= info["opponent_id"].astype(str)
    by_match = info.loc[chosen, ["match_id", "match_date"]].assign(y=labels.loc[chosen].to_numpy())
    frame = (
        oof.assign(match_id=oof["match_id"].astype(str))
        .merge(
            by_match.assign(match_id=by_match["match_id"].astype(str)),
            on="match_id",
            validate="one_to_one",
        )
        .sort_values("match_date")
    )
    result = select_temperature(frame["stack_pred_cv"], frame["y"], frame["fold"])

    CALIBRATION_STATE.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_STATE.write_text(json.dumps({"temperature": result.temperature}) + "\n")
    outcome = "accepted" if result.accepted else "rejected; using uncalibrated candidate (t=1.0)"
    print(f"Calibration {outcome}")
    print(f"  fitted temperature: {result.fitted_temperature:.6f}")
    print(
        f"  walk-forward log loss: raw {result.raw_log_loss:.6f} -> calibrated {result.calibrated_log_loss:.6f}"
    )
    print(
        f"  walk-forward Brier:    raw {result.raw_brier:.6f} -> calibrated {result.calibrated_brier:.6f}"
    )


if __name__ == "__main__":
    calibrate()
