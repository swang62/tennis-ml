"""Hermetic tests for inference helpers and pre-database validation."""

from datetime import date, datetime
from typing import cast

import pandas as pd
import pytest

from src.constants import TOUR_AVERAGES_TABLE
from src.features.columns import TOUR_AVERAGES_FALLBACK_COLS
from src.features.inference import _to_date, build_inference_features

# ── _to_date ──


def test_to_date_coerces_pandas_timestamp():
    result = _to_date(pd.Timestamp("2026-01-15"))
    assert result == date(2026, 1, 15)
    assert type(result) is date


def test_to_date_coerces_datetime():
    result = _to_date(datetime(2026, 1, 15, 23, 59, 59))
    assert result == date(2026, 1, 15)
    assert type(result) is date  # truncated to the date component, not datetime


def test_to_date_passes_plain_date_through():
    assert _to_date(date(2026, 1, 15)) == date(2026, 1, 15)


def test_to_date_parses_iso_string():
    result = _to_date("2026-01-15")
    assert result == date(2026, 1, 15)
    assert type(result) is date


def test_to_date_rejects_non_coercible_value():
    with pytest.raises(TypeError):
        _to_date(123)


# ── build_inference_features boundary validation (no DB access) ──


def test_empty_player_id_raises():
    with pytest.raises(ValueError):
        build_inference_features("", "B", "clay")


def test_whitespace_player_id_raises():
    with pytest.raises(ValueError):
        build_inference_features("   ", "B", "clay")


def test_non_str_player_id_raises():
    with pytest.raises(ValueError):
        build_inference_features(cast(str, 123), "B", "clay")


def test_empty_opponent_id_raises():
    with pytest.raises(ValueError):
        build_inference_features("A", "", "clay")


def test_whitespace_opponent_id_raises():
    with pytest.raises(ValueError):
        build_inference_features("A", "   ", "clay")


def test_non_str_opponent_id_raises():
    with pytest.raises(ValueError):
        build_inference_features("A", cast(str, 456), "clay")


def test_invalid_surface_raises():
    with pytest.raises(ValueError):
        build_inference_features("A", "B", "sponge")


def test_int_as_of_date_raises():
    with pytest.raises(TypeError):
        build_inference_features("A", "B", "clay", as_of_date=cast(date, 123))


@pytest.mark.parametrize("tournament_level", [-1, 5, True])
def test_invalid_tournament_level_raises(tournament_level):
    with pytest.raises(ValueError):
        build_inference_features("A", "B", "clay", tournament_level=tournament_level)


@pytest.mark.parametrize("round_encoded", [-1, 8, True])
def test_invalid_round_encoded_raises(round_encoded):
    with pytest.raises(ValueError):
        build_inference_features("A", "B", "clay", round_encoded=round_encoded)


def test_tournament_alias_non_str_raises():
    with pytest.raises(TypeError):
        build_inference_features("A", "B", "clay", tournament=cast(str, 4))


def test_round_alias_non_str_raises():
    with pytest.raises(TypeError):
        build_inference_features("A", "B", "clay", round=cast(str, 7))


def test_tournament_int_and_alias_conflict_raises():
    with pytest.raises(ValueError):
        build_inference_features("A", "B", "clay", tournament_level=4, tournament="grand_slam")


def test_round_int_and_alias_conflict_raises():
    with pytest.raises(ValueError):
        build_inference_features("A", "B", "clay", round_encoded=7, round="f")


@pytest.fixture
def empty_pool(monkeypatch):
    """Patch the DB so every player lookup is a cold start: the tour-averages
    singleton lookup returns one (all-constant) row, and every snapshot/profile
    query returns nothing."""
    singleton_row: dict[str, object] = dict.fromkeys(TOUR_AVERAGES_FALLBACK_COLS, 0.0)
    singleton_row["singleton_id"] = 1
    singleton_row["pool_as_of_date"] = date(2026, 8, 9)
    singleton_row["snapshot_pool_rows"] = 0
    singleton_row["snapshot_pool_players"] = 0
    singleton_row["profile_rows"] = 0
    singleton_row["player_match_rows"] = 0
    singleton_df = pd.DataFrame([singleton_row])

    def fake_execute_df(sql, params=None):  # noqa: ARG001 — generic DB stand-in
        return pd.DataFrame()

    monkeypatch.setattr("src.features.inference.execute_df", fake_execute_df)
    monkeypatch.setattr(
        "src.features.tour_averages.execute_df",
        lambda _sql, _params=None: singleton_df,
    )


def test_valid_string_aliases_map_and_build_with_empty_pool(empty_pool):  # noqa: ARG001 — fixture applied for its side effects only
    out = build_inference_features("A", "B", "clay", tournament="grand_slam", round="f")
    assert len(out) == 1
    row = out.iloc[0]
    assert row["tournament_level"] == 4
    assert row["round_encoded"] == 7
    assert row["player_id"] == "A"
    assert row["opponent_id"] == "B"
    assert not out.isnull().to_numpy().any()
