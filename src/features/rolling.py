"""Cross-player feature construction.

Per-player rolling features (win_rate_5/10/20, ace_rate, serve_pct, etc.)
are pre-computed in the DuckDB ETL. This module handles the rest:

- Head-to-head rolling (per player-opponent pair)
- Pairwise match construction (self-join by match_id)
- Differential features (player - opponent)
- Match context encoding (ordinal/OHE)

Shared between training (notebook) and inference (serving).
"""

from __future__ import annotations

import pandas as pd

# ── Rolling features computed in SQL (gold.match_features) ──


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


# ── Per-player features for the pairwise model ──


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

SEQ_FEATURE_COLS: list[str] = [
    "player_ranking",
    "opponent_ranking",
    "aces",
    "double_faults",
    "first_serves_made",
    "total_serve_points",
    "break_points_won",
    "break_points_total",
    "match_won",
]


# ── Head-to-head features ──


def compute_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-player-opponent rolling H2H features (window 10, exclude current)."""
    results = []
    for (_player, _opponent), group in df.sort_values("match_date").groupby(
        ["player_id", "opponent_id"]
    ):
        group = group.sort_values("match_date").reset_index(drop=True)
        shifted = group["match_won"].shift(1)
        group["h2h_wins"] = shifted.rolling(10, min_periods=1).sum().fillna(0)
        group["h2h_matches"] = (
            shifted.notna().astype(int).rolling(10, min_periods=1).sum().fillna(0)
        )
        group["h2h_win_rate"] = (group["h2h_wins"] / group["h2h_matches"].clip(lower=1)).fillna(0)
        results.append(group)
    return pd.concat(results, ignore_index=True)


# ── Pairwise match construction ──


def build_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    """Convert per-player-per-match rows into one-row-per-match for GBDT.

    Each match_id must have exactly 2 rows (A's perspective, B's perspective).
    Produces: A_features + B_features + differentials + match context.
    """
    a_cols = [
        "match_id",
        "player_id",
        "opponent_id",
        "match_date",
        "tournament",
        "round",
        "surface",
        "match_won",
        *PLAYER_COLS,
    ]
    b_cols = ["match_id", "player_id", *PLAYER_COLS]

    a_rows = df.groupby("match_id").nth(0).reset_index()[a_cols]
    b_rows = df.groupby("match_id").nth(1).reset_index()[b_cols]

    b_rename = {"player_id": "opponent_id"}
    for col in PLAYER_COLS:
        b_rename[col] = f"opp_{col}"

    paired = a_rows.merge(
        b_rows.rename(columns=b_rename),
        on=["match_id", "opponent_id"],
        how="inner",
        suffixes=("", "_dup"),
    )
    paired = paired.loc[:, ~paired.columns.str.endswith("_dup")]

    # Differential features
    paired["rank_diff"] = paired["player_ranking"] - paired["opponent_ranking"]
    paired["win_rate_diff"] = paired["win_rate_10"] - paired["opp_win_rate_10"]
    paired["ace_rate_diff"] = paired["ace_rate_10"] - paired["opp_ace_rate_10"]
    paired["break_diff"] = paired["break_pct_10"] - paired["opp_break_pct_10"]
    paired["streak_diff"] = paired["win_streak"] - paired["opp_win_streak"]
    paired["matches_30d_diff"] = paired["matches_30d"] - paired["opp_matches_30d"]
    paired["surface_win_diff"] = paired["surface_win_rate_10"] - paired["opp_surface_win_rate_10"]
    paired["rank_trend_diff"] = paired["rank_trend_10"] - paired["opp_rank_trend_10"]

    # Match context encoding
    # Surface: OHE (3 values, no natural order)
    paired["is_clay"] = (paired["surface"] == "clay").astype(int)
    paired["is_grass"] = (paired["surface"] == "grass").astype(int)
    paired["is_hard"] = (paired["surface"] == "hard").astype(int)

    # Tournament level: ordinal (natural hierarchy)
    tournament_levels = {"grand_slam": 4, "masters": 3, "atp_500": 2, "atp_250": 1}
    paired["tournament_level"] = paired["tournament"].map(tournament_levels).fillna(0)

    # Round: ordinal (natural progression)
    round_order = {
        "r128": 1,
        "r64": 2,
        "r32": 3,
        "r16": 4,
        "qf": 5,
        "sf": 6,
        "f": 7,
    }
    paired["round_encoded"] = paired["round"].map(round_order).fillna(0)

    return paired


# ── Inference helpers ──


def build_inference_features(
    player_row: dict,
    opponent_row: dict,
) -> pd.DataFrame:
    """Build a single-row DataFrame for prediction from two gold rows."""
    row = {}
    for col in PLAYER_COLS:
        row[col] = player_row.get(col, 0)
    for col in OPPONENT_COLS:
        opp_col = col.replace("opp_", "")
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
