"""Feature definitions for the canonical match-level training table.

`gold.match_features` is one row per match, canonicalized by ATP id ordering:
the lower/stable ATP id is the `player_*` side, the other is `opponent_*`,
with the full balanced feature set (player + opponent rolling stats,
profile-derived identity, differentials, context) and `match_won` relative to
the canonical player side.

This module holds the shared column definitions (training and serving).
The ID-based inference row builder lives in `src.features.inference`
(`build_inference_features`).
"""

from __future__ import annotations

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
