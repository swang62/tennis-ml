"""Database-free contract tests for feature columns."""

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
    MATCH_STATS_COLS,
    PROFILE_COLS,
    RATE_EXPOSURE_COLS,
    SILVER_ROLLING_COLS,
    SIMILARITY_COLS,
)

FINAL_DIFFS = [
    "rank_diff",
    "rank_points_diff",
    "age_diff",
    "win_rate_diff",
    "ace_rate_diff",
    "first_serve_pct_diff",
    "break_points_saved_pct_diff",
    "first_serve_win_pct_diff",
    "second_serve_win_pct_diff",
    "serve_win_pct_diff",
    "return_points_won_pct_diff",
    "df_rate_diff",
    "aces_per_svc_game_diff",
    "rank_trend_diff",
    "avg_rank_faced_diff",
    "streak_diff",
]


def test_feature_col_counts():
    assert len(FEATURE_COLS) == 39
    assert len(DIFF_COLS) == 16
    assert len(CONTEXT_COLS) == 7


def test_gold_rolling_cols_exact_order():
    """Retained 10-match rolling values in SQL order."""
    assert SILVER_ROLLING_COLS == [
        "weighted_form_10",
        "win_rate_10",
        "ace_rate_10",
        "first_serve_pct_10",
        "break_points_saved_pct_10",
        "first_serve_win_pct_10",
        "second_serve_win_pct_10",
        "serve_win_pct_10",
        "return_points_won_pct_10",
        "df_rate_10",
        "aces_per_svc_game_10",
        "streak",
        "avg_rank_faced_10",
    ]


def test_diff_cols_exact_order():
    assert DIFF_COLS == FINAL_DIFFS


def test_no_5_or_20_rolling_variants():
    """Task 6: every `_5`/`_20` output and the separate win/loss streaks are
    removed from the rolling contract."""
    for col in SILVER_ROLLING_COLS:
        assert not col.endswith("_5") and not col.endswith("_20"), col
    assert "win_streak" not in SILVER_ROLLING_COLS
    assert "loss_streak" not in SILVER_ROLLING_COLS
    assert "win_rate_5" not in SILVER_ROLLING_COLS
    assert "win_rate_20" not in SILVER_ROLLING_COLS
    assert "rank_trend_20" not in SILVER_ROLLING_COLS
    assert "avg_rank_faced_5" not in SILVER_ROLLING_COLS
    assert "matches_30d" not in SILVER_ROLLING_COLS  # computed on demand
    assert "surface_win_rate_10" not in SILVER_ROLLING_COLS  # derived on demand
    assert "days_since_last_match" not in SILVER_ROLLING_COLS  # computed on demand


def test_diff_cols_removed_obsolete():
    """Obsolete differentials are absent; streak_diff replaces win_streak_diff."""
    for col in (
        "win_streak_diff",
        "loss_streak_diff",
        "matches_30d_diff",
        "surface_win_rate_diff",
        "height_diff",
        "handedness_diff",
        "years_pro_diff",
        "df_rate_diff_5",
    ):
        assert col not in DIFF_COLS
    assert "streak_diff" in DIFF_COLS


def test_match_stats_cols_removed_from_contract():
    """Task 6: current-match per-side serve/break analysis rates are removed
    from the gold contract entirely (derived on demand from bronze)."""
    assert MATCH_STATS_COLS == []
    for c in ("first_serve_win_pct", "serve_win_pct", "aces_per_svc_game", "df_per_svc_game"):
        assert c not in FEATURE_COLS
        assert f"player_{c}" not in FEATURE_COLS
        assert f"opponent_{c}" not in FEATURE_COLS


def test_profile_cols_keep_handedness_and_years_pro_only():
    """Only handedness and time-aware years_pro are model profile features."""
    assert PROFILE_COLS == ["is_left_handed", "years_pro"]
    assert "player_height" not in FEATURE_COLS
    assert "opponent_height" not in FEATURE_COLS
    assert "player_is_left_handed" in FEATURE_COLS
    assert "opponent_is_left_handed" in FEATURE_COLS
    assert "player_years_pro" in FEATURE_COLS
    assert "opponent_years_pro" in FEATURE_COLS


def test_h2h_pair_level_exposure_and_advantage():
    """H2H is pair-level: shared strictly-prior exposure + signed smoothed
    advantage; no player_/opponent_ prefixed variants remain."""
    assert H2H_COLS == ["h2h_exposure", "h2h_advantage"]
    assert "h2h_exposure" in FEATURE_COLS
    assert "h2h_advantage" in FEATURE_COLS
    for col in (
        "player_h2h_matches",
        "player_h2h_wins",
        "opponent_h2h_matches",
        "opponent_h2h_wins",
        "h2h_matches",
        "h2h_wins",
        "player_h2h_win_rate",
        "opponent_h2h_win_rate",
        "h2h_win_rate",
    ):
        assert col not in FEATURE_COLS
    # Pair-level names never carry a player_/opponent_ prefix.
    for col in H2H_COLS:
        assert not col.startswith(("player_", "opponent_"))


def test_rate_exposure_cols_minimal_pair():
    """One per-side 10-match window count backs all smoothed 10-match rates;
    no per-rate exposure columns exist."""
    assert RATE_EXPOSURE_COLS == ["player_matches_10", "opponent_matches_10"]
    for col in RATE_EXPOSURE_COLS:
        assert col in FEATURE_COLS
    assert len(RATE_EXPOSURE_COLS) == 2
    assert "player_matches_30d" in FEATURE_COLS  # distinct 30-day count kept
    assert "matches_10" not in FEATURE_COLS  # never a pair-level shared count


def test_return_strength_diff_position():
    """return_points_won_pct_diff sits immediately after serve_win_pct_diff,
    matching silver rolling order (return_points_won_pct_10 follows
    serve_win_pct_10)."""
    i = DIFF_COLS.index("serve_win_pct_diff")
    assert DIFF_COLS[i + 1] == "return_points_won_pct_diff"
    assert "return_points_won_pct_diff" in FEATURE_COLS


def test_feature_cols_exact_final_contract():
    """The exact 39-column FEATURE_COLS contract."""
    assert [
        *DIFF_COLS,
        "player_weighted_form_10",
        "opponent_weighted_form_10",
        "player_days_since_last_match",
        "opponent_days_since_last_match",
        "player_matches_30d",
        "opponent_matches_30d",
        "player_surface_win_rate_10",
        "opponent_surface_win_rate_10",
        *RATE_EXPOSURE_COLS,
        "player_is_left_handed",
        "opponent_is_left_handed",
        "player_years_pro",
        "opponent_years_pro",
        *H2H_COLS,
        *CONTEXT_COLS,
    ] == FEATURE_COLS


def test_swap_behavior_classification():
    """Every FEATURE_COL is signed, paired, or invariant, with no gaps."""
    signed = {*DIFF_COLS, "h2h_advantage"}
    invariant = {"h2h_exposure", *CONTEXT_COLS}
    paired = set(FEATURE_COLS) - signed - invariant
    assert all(c.startswith(("player_", "opponent_")) for c in paired)
    assert {c.removeprefix("player_") for c in paired if c.startswith("player_")} == {
        c.removeprefix("opponent_") for c in paired if c.startswith("opponent_")
    }


def test_no_current_match_raw_stats_in_feature_cols():
    """Regression: current-match raw stats and the outcome must never enter
    the model contract. FEATURE_COLS is as-of-date / N-1-snapshot only."""
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
    assert not any("match_won" in c or "winner" in c for c in FEATURE_COLS)


def test_no_old_99_column_shape_columns():
    """Regression: no column from the old 99-feature shape survives."""
    for col in (
        "player_win_rate_5",
        "player_win_rate_20",
        "player_win_streak",
        "player_loss_streak",
        "player_rank_trend_20",
        "player_avg_rank_faced_5",
        "player_height",
        "player_h2h_win_rate",
        "opponent_height",
        "weighted_form_5",
        "win_streak_diff",
        "surface_win_rate_diff",
        "height_diff",
        "handedness_diff",
        "years_pro_diff",
        "matches_30d_diff",
    ):
        assert col not in FEATURE_COLS, col


def test_no_duplicate_names_across_the_row():
    assert len({*FEATURE_COLS, "player_id", "opponent_id"}) == len(FEATURE_COLS) + 2


def test_naming_conventions():
    assert all(c.endswith("_diff") for c in DIFF_COLS)


def test_similarity_cols_exact_order_and_not_model_features():
    """Similarity serve/return columns never enter model features or contexts."""
    assert SIMILARITY_COLS == [
        "player_first_serve_pct_10",
        "opponent_first_serve_pct_10",
        "player_first_serve_win_pct_10",
        "opponent_first_serve_win_pct_10",
        "player_second_serve_win_pct_10",
        "opponent_second_serve_win_pct_10",
        "player_serve_win_pct_10",
        "opponent_serve_win_pct_10",
        "player_return_points_won_pct_10",
        "opponent_return_points_won_pct_10",
    ]
    for col in SIMILARITY_COLS:
        assert col not in FEATURE_COLS
        assert col not in DIFF_COLS
        assert col not in CONTEXT_COLS
        assert col not in SILVER_ROLLING_COLS
    # The serving-side save rate is no longer a similarity column.
    assert "player_break_points_saved_pct_10" not in SIMILARITY_COLS
    assert "opponent_break_points_saved_pct_10" not in SIMILARITY_COLS


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


def test_bronze_int_column_sets():
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
    assert set(BRONZE_COLUMNS_INT32) == {"player1_rank_points", "player2_rank_points"}
    assert set(BRONZE_COLUMNS_FLOAT) == {"player1_age", "player2_age"}
    for col in (*BRONZE_COLUMNS_INT32, *BRONZE_COLUMNS_FLOAT):
        assert col in BRONZE_COLUMNS


def test_old_99_column_shape_rejected():
    """Old 99-column payloads miss required finalized columns and are rejected."""
    old_99_cols = [
        "player_win_rate_5",
        "player_win_rate_10",
        "player_win_rate_20",
        "player_weighted_form_10",
        "player_ace_rate_5",
        "player_first_serve_pct_5",
        "player_break_points_saved_pct_5",
        "player_first_serve_win_pct_5",
        "player_second_serve_win_pct_5",
        "player_serve_win_pct_5",
        "player_df_rate_5",
        "player_aces_per_svc_game_5",
        "player_rank_trend_10",
        "player_rank_trend_20",
        "player_avg_rank_faced_5",
        "player_win_streak",
        "player_loss_streak",
        "player_days_since_last_match",
        "player_matches_30d",
        "player_surface_win_rate_10",
        "player_height",
        "player_h2h_win_rate",
        "win_streak_diff",
        "matches_30d_diff",
        "surface_win_rate_diff",
        "height_diff",
        "handedness_diff",
        "years_pro_diff",
    ]
    # The rejection property is NOT per-column absence: several old names were
    # retained verbatim (player_weighted_form_10, player_days_since_last_match,
    # player_matches_30d, player_surface_win_rate_10). The rejection comes from
    # required-columns-missing: an old-shape row drops at least one finalized
    # column the serving endpoint demands, so its required-column check fires.
    missing = [c for c in FEATURE_COLS if c not in old_99_cols]
    assert missing, "old-shape rows must drop at least one required FEATURE_COL"
    # And the removed/`_5`/`_20`/streak/height/diff variants are gone from the
    # materialized contract.
    retained_overlap = {
        "player_weighted_form_10",
        "player_days_since_last_match",
        "player_matches_30d",
        "player_surface_win_rate_10",
    }
    for col in old_99_cols:
        if col not in retained_overlap:
            assert col not in FEATURE_COLS
