"""Hermetic tests for the grouped fold helpers shared by the split and 02 notebooks.

Gold holds two directional rows per physical match; these helpers must
never split a match across a fold and must be deterministic so linear,
GBDT, and NN all use the same persisted fold map. Lifecycle: the 01 split
notebook creates/replaces fold_assignment.parquet once per run; every 02
tuner loads and validates the current assignment and never writes it.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.grouped_cv import (
    assert_groups_intact,
    create_fold_assignment,
    fold_assignment_frame,
    grouped_fold_indices,
    load_fold_assignment,
    load_validated_fold_assignment,
    save_fold_assignment,
)

# Two orientations of four physical matches, ordered like the gold artifacts.
MATCH_IDS = np.array(["m1", "m1", "m2", "m2", "m3", "m3", "m4", "m4"])
PLAYER_IDS = np.array(["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"])
# A second split, five physical matches: the snapshot grew (240 -> 245 style).
NEW_MATCH_IDS = np.array(["m1", "m1", "m2", "m2", "m3", "m3", "m4", "m4", "m5", "m5"])
NEW_PLAYER_IDS = np.array(["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"])


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


def test_create_replaces_stale_assignment_from_previous_snapshot(tmp_path):
    """A new split replaces the prior run's assignment once, without failing."""
    path = tmp_path / "fold_assignment.parquet"
    create_fold_assignment(MATCH_IDS, PLAYER_IDS, 2, 42, path)  # old snapshot
    stale = load_fold_assignment(path)
    assert len(stale) == 4

    # Snapshot grew: 01 re-creates the assignment, and a 02 tuner must then
    # load it successfully instead of asserting it against the stale file.
    frame = create_fold_assignment(NEW_MATCH_IDS, NEW_PLAYER_IDS, 2, 42, path)
    assert len(frame) == 5
    assert not stale.equals(frame)
    loaded = load_validated_fold_assignment(NEW_MATCH_IDS, NEW_PLAYER_IDS, 2, 42, path)
    pd.testing.assert_frame_equal(loaded, frame)


def test_same_run_reuse_across_tuners(tmp_path):
    """All three 02 tuners load one identical assignment for the current split."""
    path = tmp_path / "fold_assignment.parquet"
    create_fold_assignment(MATCH_IDS, PLAYER_IDS, 3, 42, path)
    first = load_validated_fold_assignment(MATCH_IDS, PLAYER_IDS, 3, 42, path)
    second = load_validated_fold_assignment(MATCH_IDS, PLAYER_IDS, 3, 42, path)
    pd.testing.assert_frame_equal(first, second)


def test_stale_assignment_from_prior_snapshot_is_rejected(tmp_path):
    path = tmp_path / "fold_assignment.parquet"
    create_fold_assignment(MATCH_IDS, PLAYER_IDS, 2, 42, path)
    with pytest.raises(ValueError, match="stale or mismatched"):
        load_validated_fold_assignment(NEW_MATCH_IDS, NEW_PLAYER_IDS, 2, 42, path)


def test_fold_count_mismatch_is_rejected(tmp_path):
    path = tmp_path / "fold_assignment.parquet"
    create_fold_assignment(MATCH_IDS, PLAYER_IDS, 3, 42, path)
    with pytest.raises(ValueError, match="folds"):
        load_validated_fold_assignment(MATCH_IDS, PLAYER_IDS, 2, 42, path)


def test_corrupted_fold_values_are_rejected(tmp_path):
    path = tmp_path / "fold_assignment.parquet"
    create_fold_assignment(MATCH_IDS, PLAYER_IDS, 2, 42, path)
    # Same matches, folds, and order, but permuted fold values.
    frame = load_fold_assignment(path)
    frame = frame.assign(fold=frame["fold"].map({0: 1, 1: 0}))
    save_fold_assignment(frame, path)
    with pytest.raises(ValueError, match="computed folds"):
        load_validated_fold_assignment(MATCH_IDS, PLAYER_IDS, 2, 42, path)


def test_reordered_assignment_is_rejected(tmp_path):
    path = tmp_path / "fold_assignment.parquet"
    create_fold_assignment(MATCH_IDS, PLAYER_IDS, 2, 42, path)
    # Same rows, wrong canonical ordering.
    save_fold_assignment(load_fold_assignment(path).iloc[::-1].reset_index(drop=True), path)
    with pytest.raises(ValueError, match="canonical ordering"):
        load_validated_fold_assignment(MATCH_IDS, PLAYER_IDS, 2, 42, path)


def test_02_notebook_never_writes_the_assignment(tmp_path):
    """02 tuners only load; the persisted file is untouched by validation."""
    path = tmp_path / "fold_assignment.parquet"
    create_fold_assignment(MATCH_IDS, PLAYER_IDS, 2, 42, path)
    before = path.read_bytes()
    load_validated_fold_assignment(MATCH_IDS, PLAYER_IDS, 2, 42, path)
    assert path.read_bytes() == before
