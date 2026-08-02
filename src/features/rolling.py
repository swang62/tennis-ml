"""Feature definitions for the canonical match-level training table.

`gold.match_features` is one row per match, canonicalized by ATP id ordering:
the lower/stable ATP id is the `player_*` side, the other is `opponent_*`,
with the full balanced feature set (player + opponent rolling stats,
differentials, context) and `match_won` relative to the canonical player side.

This module holds the shared column definitions (training and serving) plus
the order-invariant inference helper.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

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
    "avg_opp_rank_10",
    "avg_opp_rank_20",
    "rank_trend_10",
    "rank_trend_20",
    "win_streak",
    "days_since_last_match",
    "matches_30d",
    "surface_win_rate_10",
]


# ── Per-side feature columns of the canonical match-level table ──


PLAYER_COLS: list[str] = [
    "player_ranking",
    *GOLD_ROLLING_COLS,
]

OPPONENT_COLS: list[str] = [
    "opponent_ranking",
    *[f"opp_{c}" for c in GOLD_ROLLING_COLS],
]

DIFF_COLS: list[str] = [
    "rank_diff",
    "win_rate_diff",
    "ace_rate_diff",
    "break_diff",
    "streak_diff",
    "matches_30d_diff",
    "surface_win_diff",
    "rank_trend_diff",
]

CONTEXT_COLS: list[str] = [
    "is_clay",
    "is_grass",
    "is_hard",
    "tournament_level",
    "round_encoded",
]

FEATURE_COLS: list[str] = PLAYER_COLS + OPPONENT_COLS + DIFF_COLS + CONTEXT_COLS


# ── Inference helpers ──


def build_inference_features(
    player_row: dict[str, Any],
    opponent_row: dict[str, Any],
) -> pd.DataFrame:
    """Build one canonical FEATURE_COLS row from two per-player stat dicts.

    Order-invariant: the lower ATP id is treated as the canonical player, so
    swapping player_row/opponent_row yields the same feature row (same
    prediction). The prediction is always P(canonical player wins).
    """
    if str(player_row["player_id"]) > str(opponent_row["player_id"]):
        player_row, opponent_row = opponent_row, player_row

    row = {}
    for col in PLAYER_COLS:
        row[col] = player_row.get(col, 0)
    for col in OPPONENT_COLS:
        # opponent_ranking is stored un-prefixed as player_ranking in the
        # opponent's own perspective row; the rest carry the opp_ prefix.
        opp_col = "player_ranking" if col == "opponent_ranking" else col.removeprefix("opp_")
        row[col] = opponent_row.get(opp_col, 0)

    # Differentials
    row["rank_diff"] = player_row.get("player_ranking", 0) - opponent_row.get("player_ranking", 0)
    row["win_rate_diff"] = player_row.get("win_rate_10", 0) - opponent_row.get("win_rate_10", 0)
    row["ace_rate_diff"] = player_row.get("ace_rate_10", 0) - opponent_row.get("ace_rate_10", 0)
    row["break_diff"] = player_row.get("break_pct_10", 0) - opponent_row.get("break_pct_10", 0)
    row["streak_diff"] = player_row.get("win_streak", 0) - opponent_row.get("win_streak", 0)
    row["matches_30d_diff"] = player_row.get("matches_30d", 0) - opponent_row.get("matches_30d", 0)
    row["surface_win_diff"] = player_row.get("surface_win_rate_10", 0) - opponent_row.get(
        "surface_win_rate_10", 0
    )
    row["rank_trend_diff"] = player_row.get("rank_trend_10", 0) - opponent_row.get(
        "rank_trend_10", 0
    )

    # Context — caller must set these manually
    for c in CONTEXT_COLS:
        row[c] = 0

    return pd.DataFrame([row])[FEATURE_COLS]
