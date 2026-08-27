"""Build runtime GRU (``nn``) inputs from PostgreSQL under the nn naming."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.constants import (
    NN_PREPROCESSING_ARTIFACT,
    SILVER_PLAYER_MATCHES,
)
from src.db.client import execute_df
from src.training import nn_history as gh

# Re-exported shared contract so Task 3/4 import a single source of truth.
HISTORY_LEN = gh.HISTORY_LEN
N_RAW = gh.N_RAW
GRU_RAW_NAMES = gh.GRU_RAW_NAMES
GRU_CONTEXT_NAMES = gh.GRU_CONTEXT_NAMES

# Raw player-perspective columns needed to reproduce the offline transform.
_HISTORY_COLS = [
    "match_id",
    "match_date",
    "match_num",
    "surface",
    "player_id",
    "opponent_id",
    "match_won",
    "player_ranking",
    "player_rank_points",
    "opponent_ranking",
    "opponent_rank_points",
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
]
_NUMERIC_COLS = [
    c
    for c in _HISTORY_COLS
    if c not in ("match_id", "match_date", "surface", "player_id", "opponent_id")
]

# Fetch one extra row beyond HISTORY_LEN so the oldest kept timestep still has a
# preceding match for its gap-day, mirroring the offline full-history shift.
_FETCH_WINDOW = HISTORY_LEN + 1

_PLAYER_HISTORY_BULK_SQL = f"""
SELECT req.player_id AS req_player_id, req.as_of_iso, pm.*
FROM unnest(%s::text[], %s::date[]) AS req(player_id, as_of_iso)
LEFT JOIN LATERAL (
    SELECT {", ".join(_HISTORY_COLS)}
    FROM {SILVER_PLAYER_MATCHES}
    WHERE player_id = req.player_id
      AND match_date < req.as_of_iso::date
    ORDER BY match_date DESC, match_num DESC, match_id DESC
    LIMIT {_FETCH_WINDOW}
) pm ON true
"""


@dataclass
class GRUPreprocessing:
    """Persisted final-model history fill/scaling stats consumed at serving.

    Produced by the training pipeline (see :func:`gru_preprocessing_from_store`)
    and loaded alongside the pinned ``nn_best`` model. The same stats are applied
    to valid history timesteps online and offline, so a request built for a
    historical fixture matches its training-time preparation.
    """

    fill_stats: np.ndarray  # [N_RAW] fill for missing valid values
    scale_mean: np.ndarray  # [N_RAW] standardization mean for valid values
    scale_scale: np.ndarray  # [N_RAW] standardization scale (>0) for valid values
    raw_names: list[str] = field(default_factory=lambda: list(GRU_RAW_NAMES))
    context_names: list[str] = field(default_factory=lambda: list(GRU_CONTEXT_NAMES))
    history_len: int = HISTORY_LEN
    n_raw: int = N_RAW
    context_mean: np.ndarray = field(
        default_factory=lambda: np.zeros(len(GRU_CONTEXT_NAMES), dtype=np.float32)
    )
    context_scale: np.ndarray = field(
        default_factory=lambda: np.ones(len(GRU_CONTEXT_NAMES), dtype=np.float32)
    )

    def __post_init__(self) -> None:
        for name, arr in (
            ("fill_stats", self.fill_stats),
            ("scale_mean", self.scale_mean),
            ("scale_scale", self.scale_scale),
        ):
            if arr.shape[0] != self.n_raw:
                raise ValueError(f"{name} must have length {self.n_raw}, got {arr.shape[0]}")
        if not np.all(self.scale_scale > 0):
            raise ValueError("scale_scale must be strictly positive")
        for name, arr in (
            ("context_mean", self.context_mean),
            ("context_scale", self.context_scale),
        ):
            if arr.shape[0] != len(self.context_names):
                raise ValueError(f"{name} must have length {len(self.context_names)}")
        if not np.all(self.context_scale > 0):
            raise ValueError("context_scale must be strictly positive")


@dataclass
class GRUInput:
    """Single-request GRU tensors."""

    player_hist: np.ndarray  # [HISTORY_LEN, N_RAW] float32
    opponent_hist: np.ndarray  # [HISTORY_LEN, N_RAW] float32
    player_len: int
    opponent_len: int
    player_mask: np.ndarray  # [HISTORY_LEN] bool
    opponent_mask: np.ndarray  # [HISTORY_LEN] bool
    context: np.ndarray  # [12] float32
    preprocessing: GRUPreprocessing


@dataclass
class GRUBatch:
    """Batched GRU tensors preserving request order."""

    player_hist: np.ndarray  # [B, HISTORY_LEN, N_RAW] float32
    opponent_hist: np.ndarray  # [B, HISTORY_LEN, N_RAW] float32
    player_len: np.ndarray  # [B] int
    opponent_len: np.ndarray  # [B] int
    player_mask: np.ndarray  # [B, HISTORY_LEN] bool
    opponent_mask: np.ndarray  # [B, HISTORY_LEN] bool
    context: np.ndarray  # [B, 12] float32
    preprocessing: GRUPreprocessing

    def to_single(self, i: int) -> GRUInput:
        """Return the ``i``-th request's tensors without reordering."""
        return GRUInput(
            player_hist=self.player_hist[i],
            opponent_hist=self.opponent_hist[i],
            player_len=int(self.player_len[i]),
            opponent_len=int(self.opponent_len[i]),
            player_mask=self.player_mask[i],
            opponent_mask=self.opponent_mask[i],
            context=self.context[i],
            preprocessing=self.preprocessing,
        )


def gru_preprocessing_from_store(
    store: gh.HistoryStore, fit_store_indices: np.ndarray
) -> GRUPreprocessing:
    """Build the served preprocessing artifact from an offline fit band."""
    fill = gh.compute_fill_stats(store, fit_store_indices)
    mean, scale = gh.compute_scale_stats(store, fit_store_indices)
    return GRUPreprocessing(fill_stats=fill, scale_mean=mean, scale_scale=scale)


def load_gru_preprocessing(path: str | Path) -> GRUPreprocessing:
    """Load and validate the served GRU preprocessing JSON artifact."""
    data = json.loads(Path(path).read_text())
    if data.get("artifact_name") != NN_PREPROCESSING_ARTIFACT:
        raise RuntimeError(f"{NN_PREPROCESSING_ARTIFACT} artifact_name mismatch")

    raw_names = [str(n) for n in data["raw_feature_names"]]
    context_names = [str(n) for n in data["context_feature_names"]]
    if raw_names != list(GRU_RAW_NAMES):
        raise RuntimeError(
            f"{NN_PREPROCESSING_ARTIFACT} raw_feature_names mismatch: "
            f"got {raw_names}, expected {list(GRU_RAW_NAMES)}"
        )
    if context_names != list(GRU_CONTEXT_NAMES):
        raise RuntimeError(
            f"{NN_PREPROCESSING_ARTIFACT} context_feature_names mismatch: "
            f"got {context_names}, expected {list(GRU_CONTEXT_NAMES)}"
        )
    history_len = int(data["history_len"])
    if history_len != HISTORY_LEN:
        raise RuntimeError(
            f"{NN_PREPROCESSING_ARTIFACT} history_len {history_len} != {HISTORY_LEN}"
        )
    n_raw = int(data["n_raw_features"])
    if n_raw != N_RAW:
        raise RuntimeError(f"{NN_PREPROCESSING_ARTIFACT} n_raw_features {n_raw} != {N_RAW}")
    n_context = int(data["n_context_features"])
    if n_context != len(GRU_CONTEXT_NAMES):
        raise RuntimeError(
            f"{NN_PREPROCESSING_ARTIFACT} n_context_features {n_context} "
            f"!= {len(GRU_CONTEXT_NAMES)}"
        )
    fill = np.asarray(data["fill"], dtype=np.float32)
    mean = np.asarray(data["history_mean"], dtype=np.float32)
    scale = np.asarray(data["history_scale"], dtype=np.float32)
    context_mean = np.asarray(data["context_mean"], dtype=np.float32)
    context_scale = np.asarray(data["context_scale"], dtype=np.float32)
    expected_hash = hashlib.sha256(
        json.dumps(
            {k: v for k, v in data.items() if k not in ("artifact_name", "content_hash")},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if data.get("content_hash") != expected_hash:
        raise RuntimeError(f"{NN_PREPROCESSING_ARTIFACT} content_hash mismatch")
    return GRUPreprocessing(
        fill_stats=fill,
        scale_mean=mean,
        scale_scale=scale,
        raw_names=raw_names,
        context_names=context_names,
        history_len=history_len,
        n_raw=n_raw,
        context_mean=context_mean,
        context_scale=context_scale,
    )


def _coerce_date(value: object) -> date:
    """Coerce an as-of value (date/datetime/str) to a plain date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError(f"cannot coerce {value!r} to a date")


def _apply_preprocessing(
    seq: np.ndarray, mask: np.ndarray, preproc: GRUPreprocessing
) -> np.ndarray:
    """Impute missing valid values, then standardize valid timesteps only.

    Zero-padded (invalid) slots are never touched, so lengths/masks stay valid
    and cold starts remain all-zero and finite.
    """
    out = seq.copy()
    for j in range(preproc.n_raw):
        col = out[:, j]
        valid = mask
        missing = valid & ~np.isfinite(col)
        if np.any(missing):
            col[missing] = preproc.fill_stats[j]
        finite_valid = valid & np.isfinite(col)
        if np.any(finite_valid):
            col[finite_valid] = (col[finite_valid] - preproc.scale_mean[j]) / preproc.scale_scale[j]
        out[:, j] = col
    return out


def _sequence_for_rows(
    group: pd.DataFrame | None, preproc: GRUPreprocessing
) -> tuple[np.ndarray, np.ndarray]:
    """Build a right-justified, padded, preprocessed sequence for one key."""
    seq = np.zeros((preproc.history_len, preproc.n_raw), dtype=np.float32)
    mask = np.zeros((preproc.history_len,), dtype=bool)
    if group is None or len(group) == 0:
        return seq, mask

    ordered = group.sort_values(["match_date", "match_num", "match_id"]).reset_index(drop=True)
    raw = gh._transform_rows(ordered)  # [n, N_RAW] chronological, gap via shift(1)
    kept = raw[-preproc.history_len :]  # last up to HISTORY_LEN rows
    start = preproc.history_len - kept.shape[0]
    seq[start:] = kept
    mask[start:] = True
    seq = _apply_preprocessing(seq, mask, preproc)
    return seq, mask


def _query_histories(
    keys: set[tuple[str, date]],
) -> dict[tuple[str, date], pd.DataFrame | None]:
    """One set-wise PostgreSQL read for all requested (player, as_of) pairs."""
    pids = [k[0] for k in keys]
    dates = [k[1] for k in keys]
    records = execute_df(_PLAYER_HISTORY_BULK_SQL, [pids, dates]).to_dict("records")

    by_key: dict[tuple[str, date], list[dict[Hashable, Any]]] = {k: [] for k in keys}
    for rec in records:
        key = (rec["req_player_id"], _coerce_date(rec["as_of_iso"]))
        if key not in by_key:
            continue
        if rec.get("player_id") is None:  # LEFT JOIN kept the key with no priors
            continue
        by_key[key].append(rec)

    out: dict[tuple[str, date], pd.DataFrame | None] = {}
    for key, recs in by_key.items():
        if not recs:
            out[key] = None
            continue
        g = pd.DataFrame(recs)[_HISTORY_COLS].copy()
        g["match_date"] = pd.to_datetime(g["match_date"])
        for c in _NUMERIC_COLS:
            g[c] = pd.to_numeric(g[c], errors="coerce")
        out[key] = g
    return out


def _build_gru_batch(
    features_df: pd.DataFrame,
    as_of_dates: Sequence[date],
    preproc: GRUPreprocessing,
    history_lookup: Mapping[tuple[str, date], pd.DataFrame | None] | None = None,
) -> GRUBatch:
    """Assemble batched GRU tensors from validated directional rows.

    ``history_lookup`` bypasses PostgreSQL for hermetic callers; production passes
    ``None`` to issue one set-wise query across every requested player/date.
    """
    missing_ctx = set(GRU_CONTEXT_NAMES) - set(features_df.columns)
    if missing_ctx:
        raise ValueError(f"inference row missing context columns: {sorted(missing_ctx)}")
    if "player_id" not in features_df.columns or "opponent_id" not in features_df.columns:
        raise ValueError("inference row must carry player_id and opponent_id")

    player_ids = features_df["player_id"].tolist()
    opponent_ids = features_df["opponent_id"].tolist()
    as_of = [_coerce_date(a) for a in as_of_dates]
    if len(player_ids) != len(as_of):
        raise ValueError("as_of_dates must be aligned with the inference rows")

    keys: set[tuple[str, date]] = set()
    for p, o, a in zip(player_ids, opponent_ids, as_of, strict=True):
        keys.add((p, a))
        keys.add((o, a))

    lookup = _query_histories(keys) if history_lookup is None else dict(history_lookup)
    seqs = {key: _sequence_for_rows(lookup.get(key), preproc) for key in keys}

    b = len(features_df)
    ph = np.zeros((b, preproc.history_len, preproc.n_raw), dtype=np.float32)
    oh = np.zeros((b, preproc.history_len, preproc.n_raw), dtype=np.float32)
    pm = np.zeros((b, preproc.history_len), dtype=bool)
    om = np.zeros((b, preproc.history_len), dtype=bool)
    pl = np.zeros((b,), dtype=np.int64)
    ol = np.zeros((b,), dtype=np.int64)

    for i, (p, o, a) in enumerate(zip(player_ids, opponent_ids, as_of, strict=True)):
        ps, pmsk = seqs[(p, a)]
        os_, omsk = seqs[(o, a)]
        ph[i] = ps
        pm[i] = pmsk
        pl[i] = int(pmsk.sum())
        oh[i] = os_
        om[i] = omsk
        ol[i] = int(omsk.sum())

    ctx = gh.build_context_tensor(features_df)
    ctx = ((ctx - preproc.context_mean) / preproc.context_scale).astype(np.float32)
    return GRUBatch(ph, oh, pl, ol, pm, om, ctx, preproc)


def build_gru_inputs_bulk(
    features_df: pd.DataFrame,
    *,
    as_of_dates: Sequence[date],
    preprocessing: GRUPreprocessing,
    history_lookup: Mapping[tuple[str, date], pd.DataFrame | None] | None = None,
) -> GRUBatch:
    """Batched runtime GRU inputs from already-built directional rows.

    Pass ``history_lookup`` to bypass PostgreSQL (hermetic tests); otherwise one
    set-wise query covers every requested ``(player_id, as_of_date)`` pair.
    """
    return _build_gru_batch(features_df, as_of_dates, preprocessing, history_lookup=history_lookup)


def build_gru_input_single(
    features: Mapping[str, Any],
    *,
    as_of_date: date,
    preprocessing: GRUPreprocessing,
) -> GRUInput:
    """Single-request runtime GRU inputs from one validated directional row.

    ``player_id``/``opponent_id`` are already carried by the directional row, so
    only the as-of boundary is supplied to bound the history window.
    """
    df = pd.DataFrame([dict(features)])
    return _build_gru_batch(df, [as_of_date], preprocessing).to_single(0)


def build_gru_request_inputs(
    requests: list[dict[str, Any]],
    preprocessing: GRUPreprocessing,
) -> GRUBatch:
    """High-level entry: build directional rows, then GRU tensors, in one call.

    ``requests`` are the same dicts accepted by
    :func:`src.features.inference.build_inference_features_bulk`. Player and
    opponent IDs are passed through without canonicalization.
    """
    from src.features.inference import build_inference_features_bulk

    df = build_inference_features_bulk(requests)
    as_of = [_as_of_from_request(r) for r in requests]
    return _build_gru_batch(df, as_of, preprocessing)


def _as_of_from_request(request: dict[str, Any]) -> date:
    """Extract the as-of date from a raw inference request dict."""
    a = request.get("as_of_date")
    if a is None:
        return date.today()
    return _coerce_date(a)


def simulate_player_histories(
    player_matches_df: pd.DataFrame,
    keys: set[tuple[str, date]],
) -> dict[tuple[str, date], pd.DataFrame | None]:
    """Hermetic stand-in for :func:`_query_histories` (no database).

    Applies the exact production filter — strictly-before ``match_date`` and the
    latest ``_FETCH_WINDOW`` rows per key — so offline/online equivalence tests
    run without PostgreSQL while exercising the real transform path.
    """
    df = player_matches_df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"])
    out: dict[tuple[str, date], pd.DataFrame | None] = {}
    for pid, as_of in keys:
        sub = df[(df["player_id"] == pid) & (df["match_date"].dt.date < as_of)]
        sub = sub.sort_values(["match_date", "match_num", "match_id"], ascending=False).head(
            _FETCH_WINDOW
        )
        sub = sub.sort_values(["match_date", "match_num", "match_id"]).reset_index(drop=True)
        out[(pid, as_of)] = sub if len(sub) else None
    return out
