"""Shared recency-weight contract for training sample weighting.

Weights decay exponentially with match age relative to an explicit snapshot
cutoff. The weight for a physical match date is shared by both of its
directional rows, and weights are normalized to mean 1.0 so the overall sample
scale is preserved. Wall-clock time is never consulted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import RECENCY_HALF_LIFE_DAYS


def recency_weights(match_dates, cutoff) -> np.ndarray:
    """Return positive, mean-1.0 exponential recency weights for match dates.

    ``match_dates`` is one date per training row; rows sharing a physical match
    date receive one shared weight by construction. ``cutoff`` is the explicit
    full-snapshot date (no wall-clock time). Raises on dates after ``cutoff``,
    empty input, or non-date values.
    """
    dates = pd.to_datetime(pd.Series(match_dates), errors="coerce", format="mixed")
    if dates.size == 0:
        raise ValueError("recency_weights: empty match_dates")
    if dates.isna().any():
        raise ValueError("recency_weights: match_dates contains NaT/non-date values")

    cutoff_ts = pd.Timestamp(cutoff)
    ages = (cutoff_ts - dates).dt.days.to_numpy()
    if np.any(ages < 0):
        raise ValueError("recency_weights: match date after cutoff (leakage)")

    # One shared weight per unique physical match date, then map back per row.
    unique_ages = np.unique(ages)
    raw = np.exp(-np.log(2.0) * unique_ages / RECENCY_HALF_LIFE_DAYS)
    if not np.all(np.isfinite(raw)):
        raise ValueError("recency_weights: non-finite weight produced")
    lut = dict(zip(unique_ages.tolist(), raw.tolist(), strict=True))
    weights = np.array([lut[int(a)] for a in ages], dtype=float)
    return weights / weights.mean()
