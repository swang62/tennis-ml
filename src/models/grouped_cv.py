"""Grouped cross-validation helpers shared by the 02 tuning notebooks.

Gold holds two directional rows per physical match, keyed by
(match_id, player_id). Splitting must never separate a match's two
orientations, so every fold holds out whole matches and all 02 notebooks
derive their fold assignment from this module. GroupKFold is deterministic
for a fixed group order, so the persisted fold_assignment.parquet is
identical across the linear, GBDT, and NN runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# numpy arrays are not a typing.Sequence, so union them for the input params.
ArrayInput = np.ndarray | Sequence


def grouped_fold_indices(
    match_ids: ArrayInput, n_splits: int, _random_state: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Row-level (fit_idx, val_idx) pairs from GroupKFold on match_ids.

    X is the identity matrix ``arange(n)[:, None]``; groups are ``match_ids``.
    ``_random_state`` is accepted for signature parity with sklearn splitters
    but GroupKFold does not shuffle, which is what keeps every notebook's
    folds identical.
    """
    ids: np.ndarray = np.asarray(match_ids)
    n = len(ids)
    splits = GroupKFold(n_splits=n_splits).split(np.arange(n)[:, None], groups=ids)
    return [(fit_idx, val_idx) for fit_idx, val_idx in splits]


def fold_assignment_frame(
    match_ids: ArrayInput, player_ids: ArrayInput, n_splits: int, random_state: int
) -> pd.DataFrame:
    """One row per physical match: ``match_id`` and the fold holding it out.

    Deterministic for a fixed ``match_ids`` order (the same GroupKFold split
    as ``grouped_fold_indices``). ``player_ids`` only pins the per-match row
    order; it is not part of the returned frame.
    """
    ids: np.ndarray = np.asarray(match_ids)
    pids: np.ndarray = np.asarray(player_ids)
    match_to_fold: dict = {}
    for fold, (_fit_idx, val_idx) in enumerate(grouped_fold_indices(ids, n_splits, random_state)):
        for match in np.unique(ids[val_idx]):
            match_to_fold[match] = fold
    per_match = pd.DataFrame({"match_id": ids, "player_id": pids})
    per_match = per_match.sort_values(["match_id", "player_id"]).drop_duplicates("match_id")
    per_match["fold"] = per_match["match_id"].map(match_to_fold)
    return per_match[["match_id", "fold"]].reset_index(drop=True)


def assert_groups_intact(match_ids: ArrayInput, fit_idx: np.ndarray, val_idx: np.ndarray) -> None:
    """Raise ValueError if any match_id appears in both fit and validation rows."""
    ids: np.ndarray = np.asarray(match_ids)
    overlap = set(ids[fit_idx]) & set(ids[val_idx])
    if overlap:
        shown = sorted(overlap)[:5]
        raise ValueError(
            f"{len(overlap)} match_id(s) appear in both fit and validation folds: {shown}"
        )


def save_fold_assignment(frame: pd.DataFrame, path: str | Path) -> None:
    """Persist the fold assignment frame to parquet."""
    pd.DataFrame(frame).to_parquet(Path(path), index=False)


def load_fold_assignment(path: str | Path) -> pd.DataFrame:
    """Read a fold assignment frame persisted by ``save_fold_assignment``."""
    return pd.read_parquet(Path(path))


def persist_fold_assignment(
    match_ids: ArrayInput,
    player_ids: ArrayInput,
    n_splits: int,
    random_state: int,
    path: str | Path,
) -> pd.DataFrame:
    """Compute the fold frame, assert it matches any existing file, then save.

    Every 02 notebook calls this with the same inputs, so the persisted
    fold_assignment.parquet is identical across runs and a stale file cannot
    silently disagree with a fresh computation.
    """
    frame = fold_assignment_frame(match_ids, player_ids, n_splits, random_state)
    path = Path(path)
    if path.exists():
        pd.testing.assert_frame_equal(load_fold_assignment(path), frame)
    save_fold_assignment(frame, path)
    return frame
