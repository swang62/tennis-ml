"""Hermetic tests for the grouped fold helpers shared by the 02 notebooks.

Gold holds two directional rows per physical match; these helpers must
never split a match across a fold and must be deterministic so linear,
GBDT, and NN all use the same persisted fold map.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.grouped_cv import (
    assert_groups_intact,
    fold_assignment_frame,
    grouped_fold_indices,
    load_fold_assignment,
    persist_fold_assignment,
    save_fold_assignment,
)

# Two orientations of three physical matches, ordered like the gold artifacts.
MATCH_IDS = np.array(["m1", "m1", "m2", "m2", "m3", "m3", "m4", "m4"])
PLAYER_IDS = np.array(["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"])


def test_folds_never_split_a_match():
    splits = grouped_fold_indices(MATCH_IDS, 2, 42)
    assert len(splits) == 2
    for fit_idx, val_idx in splits:
        assert_groups_intact(MATCH_IDS, fit_idx, val_idx)  # raises on overlap
    # Every row lands in exactly one validation fold.
    val_rows = np.concatenate([val_idx for _, val_idx in splits])
    assert sorted(val_rows.tolist()) == list(range(len(MATCH_IDS)))


def test_fold_assignment_matches_split_indices():
    frame = fold_assignment_frame(MATCH_IDS, PLAYER_IDS, 2, 42)
    assert list(frame.columns) == ["match_id", "fold"]
    assert len(frame) == 4  # one row per physical match
    splits = grouped_fold_indices(MATCH_IDS, 2, 42)
    for fold, (_, val_idx) in enumerate(splits):
        held_out = set(MATCH_IDS[val_idx])
        assert set(frame.loc[frame["fold"] == fold, "match_id"]) == held_out


def test_fold_assignment_is_deterministic():
    a = fold_assignment_frame(MATCH_IDS, PLAYER_IDS, 3, 7)
    b = fold_assignment_frame(MATCH_IDS, PLAYER_IDS, 3, 7)
    pd.testing.assert_frame_equal(a, b)


def test_assert_groups_intact_raises_on_overlap():
    with pytest.raises(ValueError, match="match_id"):
        assert_groups_intact(MATCH_IDS, np.array([0, 1]), np.array([1, 2]))


def test_persist_roundtrip_and_staleness_guard(tmp_path):
    path = tmp_path / "fold_assignment.parquet"
    frame = persist_fold_assignment(MATCH_IDS, PLAYER_IDS, 2, 42, path)
    assert path.exists()
    pd.testing.assert_frame_equal(load_fold_assignment(path), frame)
    # A conflicting pre-existing file must fail loudly, not silently diverge.
    save_fold_assignment(frame.assign(fold=0), path)
    with pytest.raises(AssertionError):
        persist_fold_assignment(MATCH_IDS, PLAYER_IDS, 2, 42, path)
