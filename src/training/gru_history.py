"""Causal sequence-history preparation for the GRU discovery notebook.

Reads the needed columns from the DuckDB training snapshot's
``silver.player_matches`` table once, in a single globally-ordered query, and
builds an in-RAM ``float32`` history store keyed by ``(player_id, match_id)``.
Each entry holds the player's up-to-10 causal prior matches, left-padded. Missing
numeric values remain NaN until fit-band imputation. The store is shared
by all discovery trials; imputation is applied separately per fit band so it
stays fold-safe.

No per-target queries, no dataframe self-joins, no Postgres/network access at
history-build time, no rolling aggregates, and no dbt/schema/cache changes here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.db.snapshot import SNAPSHOT_PATH

# Fixed history geometry shared by the discovery model.
HISTORY_LEN = 10
N_RAW = 18  # raw timestep values retained from each player-perspective row
STORE_WIDTH = N_RAW  # missing values are fit-band imputed before batching

# Raw timestep feature order. Missing numeric values remain NaN until fit-band
# imputation; valid_mask still distinguishes padding from a real record.
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

# Context tensor: surface one-hot (three), indoor, best-of, tournament level,
# round, Elo difference, and age difference. Carpet is implied by the flags.
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

HISTORY_QUERY = """
SELECT
    match_id,
    match_date,
    match_num,
    surface,
    player_id,
    opponent_id,
    match_won,
    player_ranking,
    player_rank_points,
    opponent_ranking,
    opponent_rank_points,
    aces,
    double_faults,
    first_serves_made,
    total_serve_points,
    first_serve_points_won,
    second_serve_points_won,
    service_games,
    return_points_won,
    return_points_available,
    break_points_saved,
    break_points_faced
FROM silver.player_matches
ORDER BY match_date, match_num, match_id
"""


def read_player_match_history_df(
    snapshot: str | Path | duckdb.DuckDBPyConnection | pd.DataFrame = SNAPSHOT_PATH,
) -> pd.DataFrame:
    """Fetch all player-perspective rows once, globally ordered for causality.

    Sources the wide ``silver.player_matches`` table from the DuckDB training
    snapshot (the discovery source of truth). ``snapshot`` accepts a snapshot
    path, an open DuckDB connection, or a pre-built DataFrame already sourced
    from the snapshot; in every case the read is a single ordered query over the
    persisted table with no per-target lookups and no Postgres/network access.

    The returned frame is sorted by ``(match_date, match_num, match_id)`` so
    :func:`build_history_store` can apply causal padding, and ``match_date`` is
    normalized to ``datetime64`` for the gap-day math.
    """
    if isinstance(snapshot, pd.DataFrame):
        df = snapshot.copy()
    else:
        con = (
            duckdb.connect(str(snapshot), read_only=True)
            if not isinstance(snapshot, duckdb.DuckDBPyConnection)
            else snapshot
        )
        try:
            df = con.execute(HISTORY_QUERY).fetchdf()
        finally:
            if isinstance(snapshot, (str, Path)):
                con.close()
    df["match_date"] = pd.to_datetime(df["match_date"])
    return df


@dataclass
class HistoryStore:
    """In-RAM causal history store keyed by ``(player_id, match_id)``."""

    histories: np.ndarray  # [N, HISTORY_LEN, STORE_WIDTH] float32, raw part may hold NaN
    valid_mask: np.ndarray  # [N, HISTORY_LEN] bool, True where a real prior row exists
    index: Mapping[tuple[str, str], int]
    player_ids: np.ndarray  # [N] str, the perspective player per entry
    match_ids: np.ndarray  # [N] str

    def impute(self, fit_store_indices: np.ndarray) -> np.ndarray:
        """Return a finite store copy filled with fit-band statistics."""
        fill = compute_fill_stats(self, np.asarray(fit_store_indices))
        return apply_imputation(self, fill)

    def gather(
        self,
        imputed: np.ndarray,
        player_idx: np.ndarray,
        opponent_idx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Gather player/opponent sequences and valid-length masks for target rows."""
        return (
            imputed[player_idx],
            imputed[opponent_idx],
            self.valid_mask[player_idx],
            self.valid_mask[opponent_idx],
        )


def _safe_rate(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Element-wise num/den, NaN where den <= 0 or either side is null."""
    out = np.full(len(num), np.nan, dtype=np.float32)
    ok = (den > 0) & (~np.isnan(den)) & (~np.isnan(num))
    out[ok] = (num[ok] / den[ok]).astype(np.float32)
    return out


def _transform_rows(df: pd.DataFrame) -> np.ndarray:
    """Vectorize raw timestep values for every player-match row."""
    ts = df["total_serve_points"].to_numpy(dtype=np.float32)
    fsm = df["first_serves_made"].to_numpy(dtype=np.float32)
    svc_g = df["service_games"].to_numpy(dtype=np.float32)
    rpa = df["return_points_available"].to_numpy(dtype=np.float32)

    # Gap in days since this player's immediately preceding match (NaN if none).
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


def build_history_store(df: pd.DataFrame) -> HistoryStore:
    """Build the causal history store from an ordered player-match frame.

    ``df`` must be sorted by ``(match_date, match_num, match_id)``; use
    :func:`read_player_match_history_df` for the database-backed frame or pass a
    hermetic frame in tests. Each entry's history is the snapshot of that
    player's rolling buffer taken *before* the current row is appended, so the
    target's own row never enters either its own or its opponent's history.
    """
    required = {
        "match_id",
        "match_date",
        "match_num",
        "surface",
        "player_id",
        "opponent_id",
        "match_won",
        "player_ranking",
        "player_rank_points",
        "opponent_rank_points",
        "opponent_ranking",
        "aces",
        "double_faults",
        "first_serves_made",
        "total_serve_points",
        "first_serve_points_won",
        "second_serve_points_won",
        "service_games",
        "return_points_won",
        "return_points_available",
        "break_points_saved",
        "break_points_faced",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"history frame missing columns: {sorted(missing)}")

    work = df.reset_index(drop=True)
    raw = _transform_rows(work)

    n = len(work)
    histories = np.zeros((n, HISTORY_LEN, STORE_WIDTH), dtype=np.float32)
    valid_mask = np.zeros((n, HISTORY_LEN), dtype=bool)
    player_ids = work["player_id"].to_numpy()
    match_ids = work["match_id"].to_numpy()
    index: dict[tuple[str, str], int] = {}

    buffers: dict[str, list[int]] = {}
    for i in range(n):
        pid = player_ids[i]
        buf = buffers.setdefault(pid, [])
        count = len(buf)
        for pos, src in enumerate(buf):
            slot = HISTORY_LEN - count + pos
            histories[i, slot] = raw[src]
            valid_mask[i, slot] = True
        buf.append(i)
        if len(buf) > HISTORY_LEN:
            buf.pop(0)
        index[(pid, match_ids[i])] = i

    return HistoryStore(
        histories=histories,
        valid_mask=valid_mask,
        index=index,
        player_ids=player_ids,
        match_ids=match_ids,
    )


def map_split_indices(store: HistoryStore, info: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Map each split row to player and opponent history-store indices.

    Every directional split row must resolve to both a player and an opponent
    history; a missing key raises so causal coverage is provable.
    """
    pids = info["player_id"].to_numpy()
    oids = info["opponent_id"].to_numpy()
    mids = info["match_id"].to_numpy()

    def _lookup(ids: np.ndarray) -> np.ndarray:
        out = np.empty(len(ids), dtype=np.int64)
        for k, (pid, mid) in enumerate(zip(ids, mids, strict=True)):
            key = (pid, mid)
            if key not in store.index:
                raise KeyError(f"no history key {key} for split row {k}")
            out[k] = store.index[key]
        return out

    player_idx = _lookup(pids)
    opponent_idx = _lookup(oids)
    return player_idx, opponent_idx


def build_context_tensor(X: pd.DataFrame) -> np.ndarray:
    """Select the reduced current context from an existing ``X_*`` frame."""
    present = list(GRU_CONTEXT_NAMES)
    missing = set(present) - set(X.columns)
    if missing:
        raise ValueError(f"context frame missing columns: {sorted(missing)}")
    return X[present].to_numpy(dtype=np.float32)


def compute_fill_stats(store: HistoryStore, fit_store_indices: np.ndarray) -> np.ndarray:
    """Per-raw-column means over finite history entries of the fit targets."""
    h = store.histories[np.asarray(fit_store_indices)]
    v = store.valid_mask[np.asarray(fit_store_indices)]
    fill = np.zeros(N_RAW, dtype=np.float32)
    for j in range(N_RAW):
        present = v & np.isfinite(h[..., j])
        vals = h[..., j][present]
        fill[j] = np.nanmean(vals) if vals.size else 0.0
    return fill


def apply_imputation(store: HistoryStore, fill: np.ndarray) -> np.ndarray:
    """Return a finite store copy, filling missing raw values with ``fill``."""
    out = store.histories.copy()
    v = store.valid_mask
    for j in range(N_RAW):
        target = v & ~np.isfinite(out[..., j])
        out[..., j][target] = fill[j]
    return out
