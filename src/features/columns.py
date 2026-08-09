"""Column contracts for bronze ingestion and canonical match features."""

from __future__ import annotations

# ── Bronze ingestion schema (raw rows, validated before insert) ──

BRONZE_COLUMNS_INT: tuple[str, ...] = (
    "player1_aces",
    "player1_double_faults",
    "player1_first_serves_made",
    "player1_total_serve_points",
    "player1_first_serve_points_won",
    "player1_second_serve_points_won",
    "player1_service_games",
    "player1_break_points_saved",
    "player1_break_points_faced",
    "player2_aces",
    "player2_double_faults",
    "player2_first_serves_made",
    "player2_total_serve_points",
    "player2_first_serve_points_won",
    "player2_second_serve_points_won",
    "player2_service_games",
    "player2_break_points_saved",
    "player2_break_points_faced",
)

# Raw rank points at match time (ATP ranking points; 0 is the missing marker).
BRONZE_COLUMNS_INT32: tuple[str, ...] = (
    "player1_rank_points",
    "player2_rank_points",
)

# Raw age in fractional years at match time (0 is the missing marker).
BRONZE_COLUMNS_FLOAT: tuple[str, ...] = (
    "player1_age",
    "player2_age",
)

BRONZE_COLUMNS: tuple[str, ...] = (
    "match_id",
    "match_date",
    "player1_id",
    "player2_id",
    "tournament",
    "tournament_name",
    "round",
    "surface",
    "is_indoor",
    "player1_ranking",
    "player2_ranking",
    *BRONZE_COLUMNS_INT,
    *BRONZE_COLUMNS_INT32,
    *BRONZE_COLUMNS_FLOAT,
    "winner_id",
)

# round may be a legitimate non-draw stage and encodes to 0 in gold.
_REQUIRED_STRING_COLUMNS: tuple[str, ...] = (
    "match_id",
    "player1_id",
    "player2_id",
    "tournament",
    "surface",
    "winner_id",
)

# ── Rolling features computed in SQL (silver.rolling_features, post-match) ──
#
# Retain only required 10-match values; raw counts remain in player_matches.

SILVER_ROLLING_COLS: list[str] = [
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


# ── Per-side feature columns of the canonical match-level table ──

# Current-match rates lack an as-of source, so they are never model features.
MATCH_STATS_COLS: list[str] = []

# Time-aware profile features; height remains profile-only.
PROFILE_COLS: list[str] = [
    "is_left_handed",
    "years_pro",
]

# Last-five, strictly-prior canonical H2H counts; win rate is derived on demand.
H2H_COLS: list[str] = [
    "h2h_matches",
    "h2h_wins",
]


# ── Similarity-analysis serve/return percentages (NOT model features) ──
#
# Appended 10-match style signals for PlayerSimilarity, never FEATURE_COLS.
SIMILARITY_COLS: list[str] = [
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


DIFF_COLS: list[str] = [
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
    "df_rate_diff",
    "aces_per_svc_game_diff",
    "rank_trend_diff",
    "avg_rank_faced_diff",
    "streak_diff",
]

CONTEXT_COLS: list[str] = [
    "is_clay",
    "is_grass",
    "is_hard",
    "is_carpet",
    "is_indoor",
    "tournament_level",
    "round_encoded",
]

# ── Final model feature contract (36 numeric columns) ──
#
# Exact consumer contract: differentials, absolute state, H2H, then context.
FEATURE_COLS: list[str] = [
    # Matchup differences (rolling values are all 10-match values).
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
    "df_rate_diff",
    "aces_per_svc_game_diff",
    "rank_trend_diff",
    "avg_rank_faced_diff",
    "streak_diff",
    # Values where the absolute state of both players matters.
    "player_weighted_form_10",
    "opponent_weighted_form_10",
    "player_days_since_last_match",
    "opponent_days_since_last_match",
    "player_matches_30d",
    "opponent_matches_30d",
    "player_surface_win_rate_10",
    "opponent_surface_win_rate_10",
    "player_is_left_handed",
    "opponent_is_left_handed",
    "player_years_pro",
    "opponent_years_pro",
    # Canonical-player head-to-head history.
    "player_h2h_matches",
    "player_h2h_wins",
    # Numeric match context; keep one-hot surface for linear and neural models.
    "is_clay",
    "is_grass",
    "is_hard",
    "is_carpet",
    "is_indoor",
    "tournament_level",
    "round_encoded",
]

# ── Tour averages singleton (gold.tour_averages) ──
#
# Single full-pool row (singleton_id = 1) materialized by dbt; gold.match_features
# and scalar/bulk inference impute missing side values from it instead of
# computing AVG/PERCENTILE on demand.
#
# Median is used for rank/rank-points/streak-like values, mean for rates, age,
# years-pro, and handedness rate; `rate_default` is the fixed constant for
# unknown/0 surface and empty-pool rates. Observability counts
# (snapshot_pool_rows, snapshot_pool_players, profile_rows, player_match_rows),
# `pool_as_of_date`, and the tour_* benchmarks live in the same singleton row
# but are not model fallbacks.
TOUR_AVERAGES_FALLBACK_COLS: list[str] = [
    "latest_player_ranking",
    "latest_player_rank_points",
    "latest_player_age",
    "streak",
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
    "avg_player_rank_10",
    "avg_rank_faced_10",
    "clay_win_rate_10",
    "grass_win_rate_10",
    "hard_win_rate_10",
    "days_since_default",
    "matches_30d_default",
    "rate_default",
    "left_handed_rate",
    "avg_years_pro",
]

# Weighted tour-wide benchmarks (SUM / SUM from silver.player_matches), used
# for player-profile comparisons; may be NULL only when the denominator is 0.
TOUR_BENCHMARK_COLS: list[str] = [
    "tour_ace_rate",
    "tour_first_serve_pct",
    "tour_break_points_saved_pct",
    "tour_first_serve_win_pct",
    "tour_second_serve_win_pct",
    "tour_serve_win_pct",
    "tour_return_points_won_pct",
    "tour_df_rate",
    "tour_aces_per_svc_game",
]
