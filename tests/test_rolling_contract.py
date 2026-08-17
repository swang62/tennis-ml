"""Database-free contract tests for feature columns."""

import re

import pytest

from src.features.columns import (
    BRONZE_COLUMNS,
    BRONZE_COLUMNS_FLOAT,
    BRONZE_COLUMNS_INT,
    BRONZE_COLUMNS_INT32,
    CONTEXT_COLS,
    DIFF_COLS,
    FEATURE_COLS,
    H2H_COLS,
    PROFILE_COLS,
    RATE_EXPOSURE_COLS,
    SILVER_ROLLING_COLS,
    SIMILARITY_COLS,
)


def test_no_duplicate_names_across_the_row():
    assert len({*FEATURE_COLS, "player_id", "opponent_id"}) == len(FEATURE_COLS) + 2


def test_feature_contract_has_dynamic_shape_and_valid_names():
    assert FEATURE_COLS
    assert all(isinstance(name, str) and name for name in FEATURE_COLS)
    assert all(re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", name) for name in FEATURE_COLS)
    assert set(FEATURE_COLS) == {
        *DIFF_COLS,
        *RATE_EXPOSURE_COLS,
        *PROFILE_COLS,
        *H2H_COLS,
        *CONTEXT_COLS,
    }


def test_naming_conventions():
    assert all(c.endswith("_diff") for c in DIFF_COLS)


def test_bronze_column_order_and_uniqueness():
    assert (
        "match_id",
        "match_date",
        "player1_id",
        "player2_id",
        "tournament",
        "tournament_name",
        "round",
        "surface",
        "score",
        "is_indoor",
        "player1_ranking",
        "player2_ranking",
        *BRONZE_COLUMNS_INT,
        *BRONZE_COLUMNS_INT32,
        *BRONZE_COLUMNS_FLOAT,
        "winner_id",
    ) == BRONZE_COLUMNS
    assert len(BRONZE_COLUMNS_INT) == 18
    assert len(BRONZE_COLUMNS_INT32) == 2
    assert len(BRONZE_COLUMNS_FLOAT) == 2
    assert len(set(BRONZE_COLUMNS)) == len(BRONZE_COLUMNS)
