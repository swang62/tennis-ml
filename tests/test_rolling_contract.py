"""Contract tests for src/features/columns.py (no DB access).

columns.py is the single source of truth for the feature contract:
99 features -> 101-column rows when the two ids are prepended.
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
    H2H_COLS,
    MATCH_STATS_COLS,
    OPPONENT_COLS,
    PLAYER_COLS,
    PROFILE_COLS,
)


def test_feature_cols_is_the_ordered_concatenation():
    assert FEATURE_COLS == PLAYER_COLS + OPPONENT_COLS + DIFF_COLS + CONTEXT_COLS


def test_feature_col_counts():
    assert len(FEATURE_COLS) == 99
    assert len(PLAYER_COLS) == 37
    assert len(OPPONENT_COLS) == 37
    assert len(DIFF_COLS) == 18
    assert len(CONTEXT_COLS) == 7


def test_side_cols_are_ranking_plus_rolling_plus_profile_plus_h2h():
    assert len(GOLD_ROLLING_COLS) == 29
    assert len(PROFILE_COLS) == 3
    assert len(H2H_COLS) == 3

    assert (
        ["player_ranking", "player_age"]
        + [f"player_{c}" for c in GOLD_ROLLING_COLS]
        + [f"player_{c}" for c in PROFILE_COLS]
        + [f"player_{c}" for c in H2H_COLS]
    ) == PLAYER_COLS
    assert (
        ["opponent_ranking", "opponent_age"]
        + [f"opponent_{c}" for c in GOLD_ROLLING_COLS]
        + [f"opponent_{c}" for c in PROFILE_COLS]
        + [f"opponent_{c}" for c in H2H_COLS]
    ) == OPPONENT_COLS


def test_match_stats_cols_are_gold_only():
    """Current-match serve/break stats are enriched columns only: never in
    FEATURE_COLS (no as-of-date inference source), always per side in gold."""
    assert len(MATCH_STATS_COLS) == 6
    for c in MATCH_STATS_COLS:
        assert c not in FEATURE_COLS
        assert f"player_{c}" not in FEATURE_COLS
        assert f"opponent_{c}" not in FEATURE_COLS


def test_gold_rolling_cols_exact_order():
    """Locks the per-side rolling feature order — the SQL window order: form,
    then serve/break rates as _5/_10 pairs, then rank/strength, streaks,
    activity, surface. Downstream code slices FEATURE_COLS by these names."""
    assert GOLD_ROLLING_COLS == [
        "win_rate_5",
        "win_rate_10",
        "win_rate_20",
        "weighted_form_10",
        "ace_rate_5",
        "ace_rate_10",
        "first_serve_pct_5",
        "first_serve_pct_10",
        "break_points_saved_pct_5",
        "break_points_saved_pct_10",
        "first_serve_win_pct_5",
        "first_serve_win_pct_10",
        "second_serve_win_pct_5",
        "second_serve_win_pct_10",
        "serve_win_pct_5",
        "serve_win_pct_10",
        "df_rate_5",
        "df_rate_10",
        "aces_per_svc_game_5",
        "aces_per_svc_game_10",
        "rank_trend_10",
        "rank_trend_20",
        "avg_rank_faced_5",
        "avg_rank_faced_10",
        "win_streak",
        "loss_streak",
        "days_since_last_match",
        "matches_30d",
        "surface_win_rate_10",
    ]


def test_diff_cols_exact_order():
    """Locks the differential ordering: canonical side minus opponent side,
    rank/rank_points/age first, then the rolling _10-window diffs in
    GOLD_ROLLING_COLS order, then the profile diffs."""
    assert DIFF_COLS == [
        "rank_diff",
        "rank_points_diff",
        "age_diff",
        "win_rate_diff",
        "ace_rate_diff",
        "break_points_saved_pct_diff",
        "first_serve_win_pct_diff",
        "second_serve_win_pct_diff",
        "serve_win_pct_diff",
        "aces_per_svc_game_diff",
        "win_streak_diff",
        "matches_30d_diff",
        "surface_win_rate_diff",
        "rank_trend_diff",
        "avg_rank_faced_diff",
        "height_diff",
        "handedness_diff",
        "years_pro_diff",
    ]


def test_no_current_match_raw_stats_in_feature_cols():
    """Regression: current-match raw stats and the outcome must never enter
    the model contract. FEATURE_COLS is as-of-date / N-1-snapshot only; the
    raw values (aces, double faults, first-serve totals, break-point totals,
    serve-points-won, service games, rank points, the winner) exist in
    bronze/silver and the gold-only enrichment columns, never in FEATURE_COLS.
    player_ranking / rank_points_diff / age ARE legitimate (as-of-date)."""
    raw_stems = [
        "aces",
        "double_faults",
        "first_serves_made",
        "total_serve_points",
        "first_serve_points_won",
        "second_serve_points_won",
        "service_games",
        "break_points_saved",
        "break_points_faced",
        "rank_points",
        "match_won",
        "winner_id",
    ]
    for stem in raw_stems:
        assert stem not in FEATURE_COLS
        assert f"player_{stem}" not in FEATURE_COLS
        assert f"opponent_{stem}" not in FEATURE_COLS
    # rank_points exists ONLY as the legitimate as-of-date differential.
    assert "rank_points_diff" in DIFF_COLS
    assert not any(c != "rank_points_diff" and "rank_points" in c for c in FEATURE_COLS)
    # The outcome (match_won / winner) never appears anywhere in the contract.
    assert not any("match_won" in c or "winner" in c for c in FEATURE_COLS)


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
        "avg_rank_faced_diff",
    ):
        assert col in DIFF_COLS
    assert "rank_points_trend_diff" not in DIFF_COLS
    # Task 3 locked decision: df_rate and loss_streak are per-side only — no
    # df_rate_diff, no loss_streak_diff, and no _20 variants of the new cols.
    for col in ("df_rate_diff", "loss_streak_diff", "df_rate_20", "loss_streak_20"):
        assert col not in DIFF_COLS + GOLD_ROLLING_COLS
    for col in ("df_rate_5", "df_rate_10", "loss_streak"):
        assert col in GOLD_ROLLING_COLS
        assert f"player_{col}" in PLAYER_COLS
        assert f"opponent_{col}" in OPPONENT_COLS
    for col in ("avg_rank_faced_5", "avg_rank_faced_10"):
        assert col in GOLD_ROLLING_COLS
        assert f"player_{col}" in PLAYER_COLS
        assert f"opponent_{col}" in OPPONENT_COLS
    assert "player_avg_rank_faced_20" not in PLAYER_COLS
    assert "opponent_avg_rank_faced_20" not in OPPONENT_COLS


def test_no_duplicate_names_across_the_101_column_row():
    assert len({*FEATURE_COLS, "player_id", "opponent_id"}) == 101


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
        "is_indoor",
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


def test_h2h_cols_are_per_side_only():
    """H2H is a per-side pair-level feature group: 3 names mirrored on each
    side, present in FEATURE_COLS, no diff/rolling/context variants, and no
    surface-specific variants (recency IS the last-5 window)."""
    assert H2H_COLS == ["h2h_matches", "h2h_wins", "h2h_win_rate"]
    for col in H2H_COLS:
        assert f"player_{col}" in PLAYER_COLS
        assert f"opponent_{col}" in OPPONENT_COLS
        assert f"player_{col}" in FEATURE_COLS
        assert f"opponent_{col}" in FEATURE_COLS
    # No H2H differentials or variants.
    assert not any("h2h" in c for c in DIFF_COLS)
    assert not any("h2h" in c for c in GOLD_ROLLING_COLS)
    assert not any("h2h" in c for c in CONTEXT_COLS)
    for col in ("player_h2h_diff", "h2h_diff", "h2h_win_rate_5", "player_clay_h2h_win_rate"):
        assert col not in FEATURE_COLS
    assert len([c for c in FEATURE_COLS if "h2h" in c]) == 6


def test_feature_cols_json_matches_columns_py():
    path = ROOT / "data" / "processed" / "feature_cols.json"
    if not path.exists():
        pytest.skip("feature_cols.json not present")
    with open(path, encoding="utf-8") as fh:
        assert json.load(fh) == FEATURE_COLS
