"""Shared recency-weight contract.

Weights are positive, mean-normalized to 1.0, monotonic by age, deterministic,
and shared by both directional rows of one physical match date. Half-life is
eight years: a row one half-life older carries half the pre-normalization weight
of the cutoff-date row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.constants import RECENCY_HALF_LIFE_DAYS
from src.training import recency


def test_weights_are_positive_and_mean_one():
    dates = pd.date_range("1995-01-01", periods=40, freq="QE")
    weights = recency.recency_weights(dates, cutoff=pd.Timestamp("2020-01-01"))
    assert np.all(weights > 0)
    assert float(weights.mean()) == pytest.approx(1.0, rel=1e-9)


def test_weights_monotonic_increasing_by_recency():
    # Newer matches (later dates, smaller age) get strictly larger weights.
    dates = pd.date_range("2000-01-01", periods=20, freq="YE")
    weights = recency.recency_weights(dates, cutoff=pd.Timestamp("2020-01-01"))
    assert np.all(np.diff(weights) > 0)


def test_same_physical_date_shares_weight_across_orientations():
    # Both directional rows of one match report the same date, so they receive
    # one identical weight; a different date yields a different weight.
    dates = [
        pd.Timestamp("2015-06-01"),
        pd.Timestamp("2015-06-01"),
        pd.Timestamp("2018-03-15"),
    ]
    weights = recency.recency_weights(dates, cutoff=pd.Timestamp("2020-01-01"))
    assert weights[0] == pytest.approx(weights[1])
    assert weights[2] != pytest.approx(weights[0])


def test_half_life_property():
    cutoff = pd.Timestamp("2020-01-01")
    recent = pd.Timestamp("2020-01-01")
    older = cutoff - pd.Timedelta(days=RECENCY_HALF_LIFE_DAYS)
    weights = recency.recency_weights([recent, older], cutoff=cutoff)
    assert weights[1] == pytest.approx(0.5 * weights[0], rel=1e-6)


def test_deterministic_for_same_inputs():
    dates = pd.date_range("2000-01-01", periods=20, freq="YE")
    a = recency.recency_weights(dates, cutoff=pd.Timestamp("2020-01-01"))
    b = recency.recency_weights(dates, cutoff=pd.Timestamp("2020-01-01"))
    assert np.allclose(a, b)


def test_raises_on_future_match_date_leakage():
    with pytest.raises(ValueError):
        recency.recency_weights([pd.Timestamp("2021-01-01")], cutoff=pd.Timestamp("2020-01-01"))


def test_raises_on_empty_input():
    with pytest.raises(ValueError):
        recency.recency_weights([], cutoff=pd.Timestamp("2020-01-01"))


def test_raises_on_non_date_values():
    with pytest.raises(ValueError):
        recency.recency_weights(["not-a-date", "also-bad"], cutoff=pd.Timestamp("2020-01-01"))
