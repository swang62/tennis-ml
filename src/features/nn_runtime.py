"""Dependency-free GRU feature contract shared by training and serving."""

from __future__ import annotations

import numpy as np
import pandas as pd

HISTORY_LEN = 10
N_RAW = 18
STORE_WIDTH = N_RAW

GRU_RAW_NAMES: list[str] = [
    "match_won",
    "log_player_ranking",
    "log_player_rank_points",
    "log_opponent_ranking",
    "log_opponent_rank_points",
    "is_clay",
    "is_grass",
    "is_hard",
    "ace_per_svc_game",
    "fs_in_pct",
    "fs_win_pct",
    "second_serve_win_pct",
    "return_won_pct",
    "double_faults",
    "break_points_saved",
    "break_points_faced",
    "log_gap_days",
    "log_total_svc_pts",
]

GRU_CONTEXT_NAMES: list[str] = [
    "is_clay",
    "is_grass",
    "is_hard",
    "is_indoor",
    "best_of",
    "tournament_level",
    "round_encoded",
    "elo_diff",
    "age_diff",
    "h2h_exposure",
    "h2h_advantage",
    "h2h_surface_advantage",
]


def _safe_rate(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Return ``num / den`` where both values are present and denominator is positive."""
    out = np.full(len(num), np.nan, dtype=np.float32)
    ok = (den > 0) & (~np.isnan(den)) & (~np.isnan(num))
    out[ok] = (num[ok] / den[ok]).astype(np.float32)
    return out


def transform_history_rows(df: pd.DataFrame) -> np.ndarray:
    """Vectorize raw GRU timestep values for player-match rows."""
    ts = df["total_serve_points"].to_numpy(dtype=np.float32)
    fsm = df["first_serves_made"].to_numpy(dtype=np.float32)
    svc_g = df["service_games"].to_numpy(dtype=np.float32)
    rpa = df["return_points_available"].to_numpy(dtype=np.float32)
    prev_date = df.groupby("player_id")["match_date"].shift(1)
    gap = (df["match_date"] - prev_date).dt.days.to_numpy(dtype=np.float32)

    raw = np.zeros((len(df), N_RAW), dtype=np.float32)
    raw[:, 0] = df["match_won"].to_numpy(dtype=np.float32)
    raw[:, 1] = np.log1p(df["player_ranking"].to_numpy(dtype=np.float32))
    raw[:, 2] = np.log1p(df["player_rank_points"].to_numpy(dtype=np.float32))
    raw[:, 3] = np.log1p(df["opponent_ranking"].to_numpy(dtype=np.float32))
    raw[:, 4] = np.log1p(df["opponent_rank_points"].to_numpy(dtype=np.float32))
    surf = df["surface"].to_numpy()
    raw[:, 5] = (surf == "clay").astype(np.float32)
    raw[:, 6] = (surf == "grass").astype(np.float32)
    raw[:, 7] = (surf == "hard").astype(np.float32)
    raw[:, 8] = _safe_rate(df["aces"].to_numpy(dtype=np.float32), svc_g)
    raw[:, 9] = _safe_rate(df["first_serves_made"].to_numpy(dtype=np.float32), ts)
    raw[:, 10] = _safe_rate(df["first_serve_points_won"].to_numpy(dtype=np.float32), fsm)
    raw[:, 11] = _safe_rate(df["second_serve_points_won"].to_numpy(dtype=np.float32), ts - fsm)
    raw[:, 12] = _safe_rate(df["return_points_won"].to_numpy(dtype=np.float32), rpa)
    raw[:, 13] = df["double_faults"].to_numpy(dtype=np.float32)
    raw[:, 14] = df["break_points_saved"].to_numpy(dtype=np.float32)
    raw[:, 15] = df["break_points_faced"].to_numpy(dtype=np.float32)
    raw[:, 16] = np.where(np.isnan(gap), np.nan, np.log1p(gap)).astype(np.float32)
    raw[:, 17] = np.where(np.isnan(ts), np.nan, np.log1p(ts)).astype(np.float32)
    return raw


def build_context_tensor(features: pd.DataFrame) -> np.ndarray:
    """Select the reduced current context from directional feature rows."""
    missing = set(GRU_CONTEXT_NAMES) - set(features.columns)
    if missing:
        raise ValueError(f"context frame missing columns: {sorted(missing)}")
    return features[GRU_CONTEXT_NAMES].to_numpy(dtype=np.float32)
