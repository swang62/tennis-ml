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

# ── Rolling features computed in SQL (silver.rolling_features, post-match) ──
#
# Task 6 reductions: only the retained `_10`-window values the final match
# contract and inference need. Every `_5`/`_20` output, the separate win/loss
# streaks (replaced by a single signed `streak`), and intermediate/source
# outputs not consumed by gold.match_features or inference are removed.
# Raw serve/break counts stay in silver.player_matches because the retained
# rolling rates are computed from them.

SILVER_ROLLING_COLS: list[str] = [
    "weighted_form_10",
    "win_rate_10",
    "ace_rate_10",
    "first_serve_pct_10",
    "break_points_saved_pct_10",
    "first_serve_win_pct_10",
    "second_serve_win_pct_10",
    "serve_win_pct_10",
    "df_rate_10",
    "aces_per_svc_game_10",
    "streak",
    "avg_rank_faced_10",
]


# ── Per-side feature columns of the canonical match-level table ──

# Current-match per-side serve/break analysis rates are REMOVED from the gold
# contract (Task 6). Where the dashboard/analysis needs them, they derive on
# demand from bronze raw counts with the existing NULLIF zero-denominator
# behavior. The rolling versions in SILVER_ROLLING_COLS are the model features —
# a current match's own stats have no as-of-date inference source, so they are
# never in FEATURE_COLS.
MATCH_STATS_COLS: list[str] = []

# Profile-derived identity features (from gold.player_profiles), exposed per
# side in the canonical row. years_pro is time-aware: years pro as of the
# match date, not the raw turned_pro year. height is retained in
# gold.player_profiles and the profile API (dashboard data), but is NOT a model
# feature — the final contract keeps only is_left_handed and years_pro.
PROFILE_COLS: list[str] = [
    "is_left_handed",
    "years_pro",
]

# Pair-level head-to-head history, exposed per side. NOT a rolling snapshot
# feature: it aggregates prior meetings between the canonical pair (strictly
# before the current match date, deduped to distinct match_ids, restricted to
# the most recent 5) straight from silver.player_matches. Final contract keeps
# only the counts (matches + wins); the win rate is derived on demand.
H2H_COLS: list[str] = [
    "h2h_matches",
    "h2h_wins",
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
# Task 6 finalized order (plan lines 186-231): matchup differentials first,
# then the absolute player/opponent state values, then canonical-player H2H
# counts, then numeric match context. This is the exact, ordered contract every
# consumer (gold.match_features, inference, notebooks, feature_cols.json,
# serving) must emit. Rolling values are all 10-match values.
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
