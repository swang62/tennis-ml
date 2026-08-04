"""Contract tests for src/features/columns.py (no DB access).

columns.py is the single source of truth for the feature contract:
80 features -> 82-column rows when the two ids are prepended.
"""

import json

import pytest

from src.constants import ROOT
from src.features.columns import (
    BRONZE_COLUMNS,
    BRONZE_COLUMNS_FLOAT,
    BRONZE_COLUMNS_INT,
    BRONZE_COLUMNS_INT32,
    CONTEXT_COLS,
    DIFF_COLS,
    FEATURE_COLS,
    GOLD_ROLLING_COLS,
    MATCH_STATS_COLS,
    OPPONENT_COLS,
    PLAYER_COLS,
    PROFILE_COLS,
)


def test_feature_cols_is_the_ordered_concatenation():
    assert FEATURE_COLS == PLAYER_COLS + OPPONENT_COLS + DIFF_COLS + CONTEXT_COLS


def test_feature_col_counts():
    assert len(FEATURE_COLS) == 80
    assert len(PLAYER_COLS) == 29
    assert len(OPPONENT_COLS) == 29
    assert len(DIFF_COLS) == 17
    assert len(CONTEXT_COLS) == 5


def test_side_cols_are_ranking_plus_rolling_plus_profile():
    assert len(GOLD_ROLLING_COLS) == 24
    assert len(PROFILE_COLS) == 3

    assert (
        ["player_ranking", "player_age"]
        + [f"player_{c}" for c in GOLD_ROLLING_COLS]
        + [f"player_{c}" for c in PROFILE_COLS]
    ) == PLAYER_COLS
    assert (
        ["opponent_ranking", "opponent_age"]
        + [f"opponent_{c}" for c in GOLD_ROLLING_COLS]
        + [f"opponent_{c}" for c in PROFILE_COLS]
    ) == OPPONENT_COLS


def test_match_stats_cols_are_gold_only():
    """Current-match serve/break stats are enriched columns only: never in
    FEATURE_COLS (no as-of-date inference source), always per side in gold."""
    assert len(MATCH_STATS_COLS) == 6
    assert all(c not in FEATURE_COLS for c in MATCH_STATS_COLS)


def test_new_rolling_and_current_match_names():
    # break_pct_* renamed to break_points_saved_pct_* at the rolling level.
    for col in ("break_pct_5", "break_pct_10", "break_pct_diff", "player_break_pct_5"):
        assert col not in FEATURE_COLS
    for col in (
        "break_points_saved_pct_5",
        "break_points_saved_pct_10",
        "first_serve_win_pct_5",
        "first_serve_win_pct_10",
        "second_serve_win_pct_5",
        "second_serve_win_pct_10",
        "serve_win_pct_5",
        "serve_win_pct_10",
        "aces_per_svc_game_5",
        "aces_per_svc_game_10",
    ):
        assert col in GOLD_ROLLING_COLS
        assert f"player_{col}" in PLAYER_COLS
        assert f"opponent_{col}" in OPPONENT_COLS
    # Rank points are a pure differential (no per-side/trend features).
    assert "player_rank_points" not in PLAYER_COLS
    assert "player_rank_points_trend_10" not in PLAYER_COLS
    assert "player_rank_points_trend_20" not in PLAYER_COLS
    assert "rank_points_trend_10" not in GOLD_ROLLING_COLS
    assert "rank_points_trend_20" not in GOLD_ROLLING_COLS
    # Per-side as-of-date age is a model feature.
    assert "player_age" in PLAYER_COLS
    # Diff columns for the per-match and rolling features.
    for col in (
        "rank_points_diff",
        "age_diff",
        "break_points_saved_pct_diff",
        "first_serve_win_pct_diff",
        "second_serve_win_pct_diff",
        "serve_win_pct_diff",
        "aces_per_svc_game_diff",
    ):
        assert col in DIFF_COLS
    assert "rank_points_trend_diff" not in DIFF_COLS


def test_no_duplicate_names_across_the_82_column_row():
    assert len({*FEATURE_COLS, "player_id", "opponent_id"}) == 82


def test_naming_conventions():
    assert all(
        c in ("player_ranking", "player_age") or c.startswith("player_") for c in PLAYER_COLS
    )
    assert all(
        c in ("opponent_ranking", "opponent_age") or c.startswith("opponent_")
        for c in OPPONENT_COLS
    )
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
        *BRONZE_COLUMNS_INT32,
        *BRONZE_COLUMNS_FLOAT,
        "winner_id",
    ) == BRONZE_COLUMNS
    assert len(BRONZE_COLUMNS_INT) == 22
    assert len(BRONZE_COLUMNS_INT32) == 2
    assert len(BRONZE_COLUMNS_FLOAT) == 2
    assert len(set(BRONZE_COLUMNS)) == len(BRONZE_COLUMNS)


def test_bronze_int_column_sets():
    # UTINYINT serve/break stats: 6 serve columns + 4 break-point columns per pair.
    for col in (
        "player1_first_serve_points_won",
        "player1_second_serve_points_won",
        "player1_service_games",
        "player2_first_serve_points_won",
        "player2_second_serve_points_won",
        "player2_service_games",
    ):
        assert col in BRONZE_COLUMNS_INT
    for col in (
        "player1_break_points_saved",
        "player1_break_points_faced",
        "player2_break_points_saved",
        "player2_break_points_faced",
    ):
        assert col in BRONZE_COLUMNS_INT
    # The old derived names are gone.
    for col in (
        "player1_break_points_won",
        "player1_break_points_total",
        "player2_break_points_won",
        "player2_break_points_total",
    ):
        assert col not in BRONZE_COLUMNS
    assert set(BRONZE_COLUMNS_INT32) == {"player1_rank_points", "player2_rank_points"}
    assert set(BRONZE_COLUMNS_FLOAT) == {"player1_age", "player2_age"}
    for col in (*BRONZE_COLUMNS_INT32, *BRONZE_COLUMNS_FLOAT):
        assert col in BRONZE_COLUMNS


def test_feature_cols_json_matches_columns_py():
    path = ROOT / "data" / "processed" / "feature_cols.json"
    if not path.exists():
        pytest.skip("feature_cols.json not present")
    with open(path, encoding="utf-8") as fh:
        assert json.load(fh) == FEATURE_COLS
