"""Time-forward, match-group-safe, label-stratified cross-validation folds.

Gold holds two directional rows per physical match — one per player — keyed
by ``match_id``, with ``match_won`` the label relative to each row's player
side. A complete match group therefore contributes exactly one positive and
one negative row; the helpers below validate that invariant and exploit it
so every validation fold is exactly label-balanced.

Folds are chronological bands of ``match_date``: the train band's distinct
dates are split into ``n_splits + 1`` contiguous, non-overlapping bands,
fold ``k`` validates band ``k + 1``, and training for that fold is every
match from dates strictly before the band — so ``max(fit dates) <
min(validation dates)`` holds for every fold, no date straddles a fold
boundary, and complete match groups (both orientations share one
``match_date``) always move together. The earliest band is a grow-in
window with no earlier data: it is never a validation fold and only feeds
training for later folds.

``grouped_fold_indices`` and the persisted assignment are deterministic
functions of ``(match_ids, match_dates, n_splits)``; ``random_state`` is
accepted for signature parity with sklearn splitters and never used.

Lifecycle: the 01 split notebook creates/replaces ``fold_assignment.parquet``
exactly once per run; every 02 tuner loads and validates the current
assignment and never writes it. The persisted frame carries one row per
physical match — ``match_id``, ``match_date``, ``fold`` (0 = grow-in band,
never validated; 1..n_splits = the validation fold holding that match out)
— so a stale file from a previous snapshot fails the load validation.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

# numpy arrays are not a typing.Sequence, so union them for the input params.
ArrayInput = np.ndarray | Sequence


def _forward_bands(match_dates: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split the sorted unique match dates into ``n_splits + 1`` contiguous bands.

    Returns inclusive ``(lo, hi)`` date pairs, one per band. Boundaries fall
    between distinct dates, so no date ever straddles a fold.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    unique_dates = np.sort(np.unique(match_dates))
    if len(unique_dates) < n_splits + 1:
        raise ValueError(
            f"cannot form {n_splits} time-forward folds from {len(unique_dates)} "
            "distinct match dates (need at least n_splits + 1)"
        )
    bands = np.array_split(unique_dates, n_splits + 1)
    return [(band[0], band[-1]) for band in bands]


def _validate_label_invariant(match_ids: ArrayInput, labels: ArrayInput) -> None:
    """Every match_id must hold exactly two directional rows, one positive
    and one negative — the precondition that makes folds exactly balanced."""
    ids = np.asarray(match_ids)
    labs = np.asarray(labels)
    if len(ids) != len(labs):
        raise ValueError(f"labels length {len(labs)} != match_ids length {len(ids)}")
    counts = (
        pd.DataFrame({"match_id": ids, "label": labs})
        .groupby("match_id")
        .agg(n_rows=("label", "size"), n_positive=("label", "sum"))
    )
    bad = counts[(counts["n_rows"] != 2) | (counts["n_positive"] != 1)]
    if not bad.empty:
        shown = [str(i) for i in bad.index[:5]]
        raise ValueError(
            f"label stratification requires exactly one positive and one negative "
            f"orientation per match_id; {len(bad)} match_id(s) violate it, e.g. {shown}"
        )


def _validate_fold_balance(labels: np.ndarray, val_idx: np.ndarray, fold: int) -> None:
    """A validation fold must hold both classes in equal count."""
    val_labels = labels[val_idx]
    n_positive = int(np.sum(val_labels == 1))
    n_negative = int(np.sum(val_labels == 0))
    if n_positive == 0 or n_negative == 0 or n_positive != n_negative:
        raise ValueError(
            f"validation fold {fold} has {n_positive} positive and {n_negative} negative "
            "rows; every fold must hold both classes in equal count "
            "(one per directional match)"
        )


def grouped_fold_indices(
    match_ids: ArrayInput,
    match_dates: ArrayInput,
    n_splits: int,
    _random_state: int,
    labels: ArrayInput | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Row-level (fit_idx, val_idx) pairs for time-forward, match-safe folds.

    Fold ``k`` validates the ``(k+1)``-th date band; fitting rows are every
    match from dates strictly before that band, so no fit date is ever equal
    to or later than a validation date and complete match groups always move
    together (both orientations share one ``match_date``). When ``labels`` is
    given, the directional-label invariant is validated once and each fold's
    class balance is checked before any fit can run.
    """
    ids: np.ndarray = np.asarray(match_ids)
    dates: np.ndarray = np.asarray(match_dates)
    if labels is not None:
        _validate_label_invariant(ids, labels)
    bands = _forward_bands(dates, n_splits)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_splits):
        band_lo, band_hi = bands[fold + 1]
        val_idx = np.flatnonzero((dates >= band_lo) & (dates <= band_hi))
        fit_idx = np.flatnonzero(dates < band_lo)
        if labels is not None:
            _validate_fold_balance(np.asarray(labels), val_idx, fold)
        splits.append((fit_idx, val_idx))
    return splits


def fold_assignment_frame(
    match_ids: ArrayInput,
    match_dates: ArrayInput,
    n_splits: int,
    _random_state: int,
    labels: ArrayInput | None = None,
) -> pd.DataFrame:
    """One row per physical match: ``match_id``, ``match_date``, and ``fold``.

    ``fold`` is the date-band index: 0 is the grow-in band that is never a
    validation fold; folds 1..n_splits hold out the matches of validation
    bands 1..n_splits, which are exactly the folds ``grouped_fold_indices``
    returns. Deterministic for fixed ``(match_ids, match_dates, n_splits)``.
    """
    ids: np.ndarray = np.asarray(match_ids)
    dates: np.ndarray = np.asarray(match_dates)
    if labels is not None:
        _validate_label_invariant(ids, labels)
    bands = _forward_bands(dates, n_splits)
    match_to_fold: dict = {}
    for fold, (lo, hi) in enumerate(bands):
        for match in np.unique(ids[(dates >= lo) & (dates <= hi)]):
            match_to_fold[match] = fold
    per_match = pd.DataFrame({"match_id": ids, "match_date": dates})
    per_match = per_match.drop_duplicates("match_id").sort_values("match_id")
    per_match["fold"] = per_match["match_id"].map(match_to_fold)
    return per_match[["match_id", "match_date", "fold"]].reset_index(drop=True)


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
    match_dates: ArrayInput,
    n_splits: int,
    random_state: int,
    labels: ArrayInput,
    path: str | Path,
) -> pd.DataFrame:
    """Compute the current split's fold frame, validate the directional-label
    invariant, and persist it, replacing any prior run's assignment.

    Called exactly once per training run by the 01 split notebook after it
    writes the new split artifacts; the 02 tuners only load what this wrote.
    ``labels`` is required because a time-forward assignment that cannot be
    exactly balanced per fold must fail here, not inside a tuner.
    """
    frame = fold_assignment_frame(match_ids, match_dates, n_splits, random_state, labels)
    save_fold_assignment(frame, path)
    return frame


def load_validated_fold_assignment(
    match_ids: ArrayInput,
    match_dates: ArrayInput,
    n_splits: int,
    random_state: int,
    path: str | Path,
) -> pd.DataFrame:
    """Load the current run's fold assignment and validate it against the
    split this notebook is consuming, before any tuning starts.

    The persisted frame must be exactly what the time-forward fold bands
    compute for the current split: the same match IDs and dates, fold count,
    canonical ordering (one row per match, sorted by match_id), and fold
    values. Anything else — a stale file from a previous snapshot, a
    different split, a different fold count, or a corrupted file — raises
    with a clear message instead of silently diverging.
    """
    path = Path(path)
    frame = load_fold_assignment(path)
    expected = fold_assignment_frame(match_ids, match_dates, n_splits, random_state)
    _validate_fold_assignment(frame, expected, path)
    return frame


def _validate_fold_assignment(frame: pd.DataFrame, expected: pd.DataFrame, path: Path) -> None:
    """Raise ValueError with a clear message on any stale/corrupt mismatch."""
    if list(frame.columns) != ["match_id", "match_date", "fold"]:
        raise ValueError(
            f"fold assignment {path} has columns {list(frame.columns)}; "
            'expected ["match_id", "match_date", "fold"]'
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
    if not (frame["match_date"].to_numpy() == expected["match_date"].to_numpy()).all():
        raise ValueError(
            f"fold assignment {path} carries match dates that differ from the "
            "current split's date bands"
        )
    try:
        pd.testing.assert_frame_equal(frame, expected, check_dtype=False)
    except AssertionError as exc:
        raise ValueError(
            f"fold assignment {path} does not match the current split's computed folds"
        ) from exc
