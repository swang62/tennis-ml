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
    "player1_break_points_won",
    "player1_break_points_total",
    "player2_wins_last_10",
    "player2_matches_last_10",
    "player2_aces",
    "player2_double_faults",
    "player2_first_serves_made",
    "player2_total_serve_points",
    "player2_break_points_won",
    "player2_break_points_total",
)

BRONZE_COLUMNS: tuple[str, ...] = (
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
    "ace_rate_5",
    "ace_rate_10",
    "first_serve_pct_5",
    "first_serve_pct_10",
    "break_pct_5",
    "break_pct_10",
    "rank_trend_10",
    "rank_trend_20",
    "win_streak",
    "days_since_last_match",
    "matches_30d",
    "surface_win_rate_10",
]


# ── Per-side feature columns of the canonical match-level table ──

# Profile-derived identity features (from gold.player_profiles), exposed
# per side in the canonical row. years_pro is time-aware: years pro as of the
# match date, not the raw turned_pro year.
PROFILE_COLS: list[str] = [
    "height",
    "is_left_handed",
    "years_pro",
]


PLAYER_COLS: list[str] = [
    "player_ranking",
    *[f"player_{c}" for c in GOLD_ROLLING_COLS],
    *[f"player_{c}" for c in PROFILE_COLS],
]

OPPONENT_COLS: list[str] = [
    "opponent_ranking",
    *[f"opponent_{c}" for c in GOLD_ROLLING_COLS],
    *[f"opponent_{c}" for c in PROFILE_COLS],
]

DIFF_COLS: list[str] = [
    "rank_diff",
    "win_rate_diff",
    "ace_rate_diff",
    "break_pct_diff",
    "win_streak_diff",
    "matches_30d_diff",
    "surface_win_rate_diff",
    "rank_trend_diff",
    "height_diff",
    "handedness_diff",
    "years_pro_diff",
]

CONTEXT_COLS: list[str] = [
    "is_clay",
    "is_grass",
    "is_hard",
    "tournament_level",
    "round_encoded",
]

FEATURE_COLS: list[str] = PLAYER_COLS + OPPONENT_COLS + DIFF_COLS + CONTEXT_COLS
