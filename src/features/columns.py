"""Single source of truth for every column definition in the pipeline.

Holds the canonical match-level training feature columns (`gold.match_features`;
player/opponent rolling stats, profile-derived identity, differentials, context)
and the bronze ingestion schema. Consumer modules import from here instead of
re-declaring column lists.

Rolling-feature *formulas* stay in dbt SQL only; Python never re-implements
them. The ID-based inference row builder lives in `src.features.inference`
(`build_inference_features`); row validation lives in `src.features.validate`.
"""

from __future__ import annotations

# ── Bronze ingestion schema (raw rows, validated before insert) ──

BRONZE_COLUMNS_INT: tuple[str, ...] = (
    "player1_wins_last_10",
    "player1_matches_last_10",
    "player1_aces",
    "player1_double_faults",
    "player1_first_serves_made",
    "player1_total_serve_points",
    "player1_first_serve_points_won",
    "player1_second_serve_points_won",
    "player1_service_games",
    "player1_break_points_saved",
    "player1_break_points_faced",
    "player2_wins_last_10",
    "player2_matches_last_10",
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

# String bronze columns that must be non-blank at ingestion time. `round` is
# excluded: non-draw stages (Davis Cup, round robins, blank) are legitimate
# and encode as round_encoded 0 in gold.
_REQUIRED_STRING_COLUMNS: tuple[str, ...] = (
    "match_id",
    "player1_id",
    "player2_id",
    "tournament",
    "surface",
    "winner_id",
)

# ── Rolling features computed in SQL (gold.match_features, canonical side) ──


GOLD_ROLLING_COLS: list[str] = [
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


# ── Per-side feature columns of the canonical match-level table ──

# Current-match per-side serve/break stats exposed ONLY in gold.match_features
# (dashboard/analysis value). The rolling versions in GOLD_ROLLING_COLS are the
# model features — a current match's own stats have no as-of-date inference
# source, so they are never in FEATURE_COLS.
MATCH_STATS_COLS: list[str] = [
    "first_serve_win_pct",
    "second_serve_win_pct",
    "serve_win_pct",
    "aces_per_svc_game",
    "df_per_svc_game",
    "break_points_saved_pct",
]

# Profile-derived identity features (from gold.player_profiles), exposed
# per side in the canonical row. years_pro is time-aware: years pro as of the
# match date, not the raw turned_pro year.
PROFILE_COLS: list[str] = [
    "height",
    "is_left_handed",
    "years_pro",
]

# Pair-level head-to-head history, exposed per side. NOT a rolling snapshot
# feature: it aggregates prior meetings between the canonical pair (strictly
# before the current match date, deduped to distinct match_ids, restricted to
# the most recent 5) straight from silver.player_matches. player_h2h_* always
# describes the canonical player_id side; opponent_h2h_* the opponent side.
# With zero prior meetings the counts are 0 and both win rates are 0.5
# (neutral, per locked decision). Wins sum to matches; rates sum to 1.
H2H_COLS: list[str] = [
    "h2h_matches",
    "h2h_wins",
    "h2h_win_rate",
]


PLAYER_COLS: list[str] = [
    "player_ranking",
    "player_age",
    *[f"player_{c}" for c in GOLD_ROLLING_COLS],
    *[f"player_{c}" for c in PROFILE_COLS],
    *[f"player_{c}" for c in H2H_COLS],
]

OPPONENT_COLS: list[str] = [
    "opponent_ranking",
    "opponent_age",
    *[f"opponent_{c}" for c in GOLD_ROLLING_COLS],
    *[f"opponent_{c}" for c in PROFILE_COLS],
    *[f"opponent_{c}" for c in H2H_COLS],
]

DIFF_COLS: list[str] = [
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

CONTEXT_COLS: list[str] = [
    "is_clay",
    "is_grass",
    "is_hard",
    "is_carpet",
    "is_indoor",
    "tournament_level",
    "round_encoded",
]

FEATURE_COLS: list[str] = PLAYER_COLS + OPPONENT_COLS + DIFF_COLS + CONTEXT_COLS
