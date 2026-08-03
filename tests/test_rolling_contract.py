"""Contract tests for src/features/columns.py (no DB access).

columns.py is the single source of truth for the feature contract:
54 features -> 56-column rows when the two ids are prepended.
"""

import json

import pytest

from src.constants import ROOT
from src.features.columns import (
    BRONZE_COLUMNS,
    BRONZE_COLUMNS_INT,
    CONTEXT_COLS,
    DIFF_COLS,
    FEATURE_COLS,
    GOLD_ROLLING_COLS,
    OPPONENT_COLS,
    PLAYER_COLS,
    PROFILE_COLS,
)


def test_feature_cols_is_the_ordered_concatenation():
    assert FEATURE_COLS == PLAYER_COLS + OPPONENT_COLS + DIFF_COLS + CONTEXT_COLS


def test_feature_col_counts():
    assert len(FEATURE_COLS) == 54
    assert len(PLAYER_COLS) == 19
    assert len(OPPONENT_COLS) == 19
    assert len(DIFF_COLS) == 11
    assert len(CONTEXT_COLS) == 5


def test_side_cols_are_ranking_plus_rolling_plus_profile():
    assert len(GOLD_ROLLING_COLS) == 15
    assert len(PROFILE_COLS) == 3

    assert ["player_ranking"] + [f"player_{c}" for c in GOLD_ROLLING_COLS] + [
        f"player_{c}" for c in PROFILE_COLS
    ] == PLAYER_COLS
    assert ["opponent_ranking"] + [f"opponent_{c}" for c in GOLD_ROLLING_COLS] + [
        f"opponent_{c}" for c in PROFILE_COLS
    ] == OPPONENT_COLS


def test_no_duplicate_names_across_the_56_column_row():
    assert len({*FEATURE_COLS, "player_id", "opponent_id"}) == 56


def test_naming_conventions():
    assert all(c == "player_ranking" or c.startswith("player_") for c in PLAYER_COLS)
    assert all(c == "opponent_ranking" or c.startswith("opponent_") for c in OPPONENT_COLS)
    assert all(c.endswith("_diff") for c in DIFF_COLS)


def test_bronze_column_order_and_uniqueness():
    assert (
        "match_id",
        "match_date",
        "player1_id",
        "player2_id",
        "tournament",
        "round",
        "surface",
        "player1_ranking",
        "player2_ranking",
        *BRONZE_COLUMNS_INT,
        "winner_id",
    ) == BRONZE_COLUMNS
    assert len(BRONZE_COLUMNS_INT) == 16
    assert len(set(BRONZE_COLUMNS)) == len(BRONZE_COLUMNS)


def test_feature_cols_json_matches_columns_py():
    path = ROOT / "data" / "processed" / "feature_cols.json"
    if not path.exists():
        pytest.skip("feature_cols.json not present")
    with open(path, encoding="utf-8") as fh:
        assert json.load(fh) == FEATURE_COLS
