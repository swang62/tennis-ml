"""Grouped cross-validation helpers shared by the split and 02 tuning notebooks.

Gold holds two directional rows per physical match, keyed by
(match_id, player_id). Splitting must never separate a match's two
orientations, so every fold holds out whole matches. GroupKFold is
deterministic for a fixed group order, so the fold_assignment.parquet
persisted by the 01 split notebook is identical across the linear, GBDT,
and NN runs.

Lifecycle: 01 creates/replaces fold_assignment.parquet exactly once per
training run, right after writing the new split artifacts; each 02 tuner
loads and validates that current-run assignment and never writes it.
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


def create_fold_assignment(
    match_ids: ArrayInput,
    player_ids: ArrayInput,
    n_splits: int,
    random_state: int,
    path: str | Path,
) -> pd.DataFrame:
    """Compute the current split's fold frame and persist it, replacing any
    prior run's assignment.

    Called exactly once per training run by the 01 split notebook after it
    writes the new split artifacts; the 02 tuners only load what this wrote.
    """
    frame = fold_assignment_frame(match_ids, player_ids, n_splits, random_state)
    save_fold_assignment(frame, path)
    return frame


def load_validated_fold_assignment(
    match_ids: ArrayInput,
    player_ids: ArrayInput,
    n_splits: int,
    random_state: int,
    path: str | Path,
) -> pd.DataFrame:
    """Load the current run's fold assignment and validate it against the
    split this notebook is consuming, before any tuning starts.

    The persisted frame must be exactly what GroupKFold computes for the
    current split: the same match IDs, fold count, canonical ordering (one
    row per match, sorted by match_id), and fold values. Anything else — a
    stale file from a previous snapshot, a different split, a different fold
    count, or a corrupted file — raises with a clear message instead of
    silently diverging.
    """
    path = Path(path)
    frame = load_fold_assignment(path)
    expected = fold_assignment_frame(match_ids, player_ids, n_splits, random_state)
    _validate_fold_assignment(frame, expected, path)
    return frame


def _validate_fold_assignment(frame: pd.DataFrame, expected: pd.DataFrame, path: Path) -> None:
    """Raise ValueError with a clear message on any stale/corrupt mismatch."""
    if list(frame.columns) != ["match_id", "fold"]:
        raise ValueError(
            f"fold assignment {path} has columns {list(frame.columns)}; "
            'expected ["match_id", "fold"]'
        )
    if frame["match_id"].duplicated().any():
        raise ValueError(f"fold assignment {path} contains duplicate match_ids")
    frame_ids = set(frame["match_id"])
    current_ids = set(expected["match_id"])
    if frame_ids != current_ids:
        missing = sorted(current_ids - frame_ids)[:5]
        extra = sorted(frame_ids - current_ids)[:5]
        raise ValueError(
            f"fold assignment {path} is stale or mismatched for the current split: "
            f"{len(current_ids - frame_ids)} current match_id(s) missing (e.g. {missing}) "
            f"and {len(frame_ids - current_ids)} unknown match_id(s) present (e.g. {extra})"
        )
    expected_folds = set(expected["fold"])
    if set(frame["fold"]) != expected_folds:
        raise ValueError(
            f"fold assignment {path} has folds {sorted(set(frame['fold']))}; "
            f"expected {sorted(expected_folds)} for the current split"
        )
    if list(frame["match_id"]) != list(expected["match_id"]):
        raise ValueError(
            f"fold assignment {path} is not in the canonical ordering of the current "
            "split (one row per match, sorted by match_id)"
        )
    try:
        pd.testing.assert_frame_equal(frame, expected, check_dtype=False)
    except AssertionError as exc:
        raise ValueError(
            f"fold assignment {path} does not match the current split's computed folds"
        ) from exc
