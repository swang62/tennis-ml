"""Column contracts for bronze ingestion and directional match features."""

from __future__ import annotations

CANONICAL_SURFACES: frozenset[str] = frozenset({"clay", "grass", "hard", "carpet"})

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

BRONZE_COLUMNS_INT32: tuple[str, ...] = (
    "player1_rank_points",
    "player2_rank_points",
)

BRONZE_COLUMNS_FLOAT: tuple[str, ...] = (
    "player1_age",
    "player2_age",
)

BRONZE_COLUMNS: tuple[str, ...] = (
    "match_id",
    "match_date",
    "match_num",
    "player1_id",
    "player2_id",
    "tournament",
    "tournament_name",
    "round",
    "best_of",
    "surface",
    "score",
    "is_indoor",
    "player1_ranking",
    "player2_ranking",
    *BRONZE_COLUMNS_INT,
    *BRONZE_COLUMNS_INT32,
    *BRONZE_COLUMNS_FLOAT,
    "winner_id",
)

_REQUIRED_STRING_COLUMNS: tuple[str, ...] = (
    "match_id",
    "player1_id",
    "player2_id",
    "tournament",
    "surface",
    "winner_id",
)

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
    "dominance",
    "df_rate_10",
    "aces_per_svc_game_10",
    "streak",
    "avg_rank_faced_10",
]


PROFILE_COLS: list[str] = [
    "player_is_left_handed",
    "opponent_is_left_handed",
    "player_years_pro",
    "opponent_years_pro",
]

H2H_COLS: list[str] = [
    "h2h_exposure",
    "h2h_advantage",
    "h2h_surface_advantage",
]


RATE_EXPOSURE_COLS: list[str] = [
    "player_weighted_form_10",
    "opponent_weighted_form_10",
    "player_matches_10",
    "opponent_matches_10",
]


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
    "return_points_won_pct_diff",
    "dominance_diff",
    "df_rate_diff",
    "aces_per_svc_game_diff",
    "rank_trend_diff",
    "avg_rank_faced_diff",
    "streak_diff",
    "surface_form_diff",
    "days_since_last_match_diff",
]

CONTEXT_COLS: list[str] = [
    "is_clay",
    "is_grass",
    "is_hard",
    "is_indoor",
    "best_of",
    "tournament_level",
    "round_encoded",
]

# FEATURE_COLS order is shared by training, validation, inference, and serving.
FEATURE_COLS: list[str] = [
    *DIFF_COLS,
    *RATE_EXPOSURE_COLS,
    *PROFILE_COLS,
    *H2H_COLS,
    *CONTEXT_COLS,
]

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
    "dominance",
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
