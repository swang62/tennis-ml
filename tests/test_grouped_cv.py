"""Hermetic tests for time-forward grouped folds."""

import numpy as np
import pandas as pd
import pytest

from src.training.grouped_cv import (
    assert_groups_intact,
    create_fold_assignment,
    fold_assignment_frame,
    grouped_fold_indices,
    load_fold_assignment,
    load_validated_fold_assignment,
    save_fold_assignment,
)


def make_rows(n_dates=6, matches_per_date=2):
    """Two directional rows per physical match (labels ``[1, 0]``), with
    ``matches_per_date`` matches on each of ``n_dates`` consecutive dates,
    ordered like the gold artifacts."""
    dates = pd.date_range("2024-01-01", periods=n_dates)
    match_ids, match_dates, labels = [], [], []
    for date in dates:
        for i in range(matches_per_date):
            match_id = f"m_{date.date()}_{i}"
            match_ids += [match_id, match_id]
            match_dates += [date, date]
            labels += [1, 0]
    return np.array(match_ids), np.array(match_dates), np.array(labels)


def test_folds_never_split_a_match():
    ids, dates, _ = make_rows()
    splits = grouped_fold_indices(ids, dates, 2, 42)
    assert len(splits) == 2
    for fit_idx, val_idx in splits:
        assert_groups_intact(ids, fit_idx, val_idx)  # raises on overlap
    # No match is validated twice; every non-grow-in match is validated once.
    val_sets = [set(ids[val_idx]) for _, val_idx in splits]
    assert not set.intersection(*val_sets)
    val_matches = set().union(*val_sets)
    # The earliest date band is grow-in training data: never a validation fold.
    grow_in = {f"m_2024-01-01_{i}" for i in range(2)} | {f"m_2024-01-02_{i}" for i in range(2)}
    assert grow_in.isdisjoint(val_matches)
    assert val_matches | grow_in == set(np.unique(ids))


def test_no_fit_date_is_at_or_after_validation_date():
    ids, dates, _ = make_rows()
    for fit_idx, val_idx in grouped_fold_indices(ids, dates, 3, 42):
        fit_dates = np.unique(dates[fit_idx])
        val_dates = np.unique(dates[val_idx])
        assert fit_dates.max() < val_dates.min()  # strict temporal direction
        # Same-date integrity: a date is never split across fit/validation.
        assert set(fit_dates.tolist()).isdisjoint(set(val_dates.tolist()))


def test_fit_sets_grow_cumulatively():
    ids, dates, _ = make_rows(n_dates=10, matches_per_date=2)
    splits = grouped_fold_indices(ids, dates, 3, 42)
    assert len(splits) == 3
    previous_fit_dates: set = set()
    for fit_idx, _val_idx in splits:
        fit_dates = set(dates[fit_idx].tolist())
        assert previous_fit_dates.issubset(fit_dates)
        assert len(fit_dates) > len(previous_fit_dates)  # strictly growing
        previous_fit_dates = fit_dates


def test_each_validation_fold_is_balanced():
    ids, dates, labels = make_rows()
    splits = grouped_fold_indices(ids, dates, 3, 42, labels)
    assert len(splits) == 3
    for _fold, (_fit_idx, val_idx) in enumerate(splits):
        val_labels = labels[val_idx]
        assert (val_labels == 1).any() and (val_labels == 0).any()
        assert int(np.sum(val_labels == 1)) == int(np.sum(val_labels == 0))


def test_label_invariant_violation_raises():
    ids, dates, labels = make_rows()
    bad = labels.copy()
    bad[0] = 0  # first match now has two negative orientations
    with pytest.raises(ValueError, match="exactly one positive and one negative"):
        grouped_fold_indices(ids, dates, 2, 42, bad)
    with pytest.raises(ValueError, match="exactly one positive and one negative"):
        fold_assignment_frame(ids, dates, 2, 42, bad)
    with pytest.raises(ValueError, match="exactly one positive and one negative"):
        create_fold_assignment(ids, dates, 2, 42, bad, "/tmp/unused.parquet")


def test_insufficient_distinct_dates_raises():
    ids, dates, _ = make_rows(n_dates=3, matches_per_date=1)
    with pytest.raises(ValueError, match="distinct match dates"):
        grouped_fold_indices(ids, dates, 3, 42)


def test_fold_assignment_matches_split_indices():
    ids, dates, labels = make_rows()
    frame = fold_assignment_frame(ids, dates, 2, 42, labels)
    assert list(frame.columns) == ["match_id", "match_date", "fold"]
    assert len(frame) == len(np.unique(ids))  # one row per physical match
    # Validation fold k (band k+1) holds out exactly the matches whose
    # persisted fold is k + 1.
    splits = grouped_fold_indices(ids, dates, 2, 42)
    for k, (_fit_idx, val_idx) in enumerate(splits):
        held_out = set(ids[val_idx])
        assert set(frame.loc[frame["fold"] == k + 1, "match_id"]) == held_out
    # Grow-in band (fold 0) is everything strictly before the first
    # validation band, and fold dates advance strictly in fold order.
    validated_min = frame.loc[frame["fold"] > 0, "match_date"].min()
    assert (frame.loc[frame["fold"] == 0, "match_date"] < validated_min).all()
    min_date_by_fold = frame.groupby("fold")["match_date"].min()
    max_date_by_fold = frame.groupby("fold")["match_date"].max()
    for fold in range(1, 3):
        assert max_date_by_fold[fold - 1] < min_date_by_fold[fold]


def test_fold_assignment_is_deterministic():
    ids, dates, _ = make_rows(n_dates=8, matches_per_date=3)
    a = fold_assignment_frame(ids, dates, 3, 7)
    b = fold_assignment_frame(ids, dates, 3, 7)
    pd.testing.assert_frame_equal(a, b)


def test_assert_groups_intact_raises_on_overlap():
    with pytest.raises(ValueError, match="match_id"):
        assert_groups_intact(np.array(["m1", "m1"]), np.array([0, 1]), np.array([1]))


def test_create_replaces_stale_assignment_from_previous_snapshot(tmp_path):
    """A new split replaces the prior run's assignment once, without failing."""
    path = tmp_path / "fold_assignment.parquet"
    ids, dates, labels = make_rows(n_dates=6)
    create_fold_assignment(ids, dates, 2, 42, labels, path)  # old snapshot
    stale = load_fold_assignment(path)
    assert len(stale) == 12

    # A grown snapshot regenerates the assignment and replaces the stale file.
    new_ids, new_dates, new_labels = make_rows(n_dates=9)
    frame = create_fold_assignment(new_ids, new_dates, 2, 42, new_labels, path)
    assert len(frame) == 18
    assert not stale.equals(frame)
    loaded = load_validated_fold_assignment(new_ids, new_dates, 2, 42, path)
    pd.testing.assert_frame_equal(loaded, frame)


def test_same_run_reuse_across_tuners(tmp_path):
    """All three 02 tuners load one identical assignment for the current split."""
    path = tmp_path / "fold_assignment.parquet"
    ids, dates, labels = make_rows(n_dates=6)
    create_fold_assignment(ids, dates, 3, 42, labels, path)
    first = load_validated_fold_assignment(ids, dates, 3, 42, path)
    second = load_validated_fold_assignment(ids, dates, 3, 42, path)
    pd.testing.assert_frame_equal(first, second)


def test_stale_assignment_from_prior_snapshot_is_rejected(tmp_path):
    path = tmp_path / "fold_assignment.parquet"
    ids, dates, labels = make_rows(n_dates=6)
    create_fold_assignment(ids, dates, 2, 42, labels, path)
    new_ids, new_dates, _ = make_rows(n_dates=9)
    with pytest.raises(ValueError, match="stale or mismatched"):
        load_validated_fold_assignment(new_ids, new_dates, 2, 42, path)


def test_changed_match_dates_are_rejected(tmp_path):
    """The persisted frame carries match_date; a snapshot whose date bands
    moved must be detected even when every match_id is unchanged."""
    path = tmp_path / "fold_assignment.parquet"
    ids, dates, labels = make_rows(n_dates=6)
    create_fold_assignment(ids, dates, 2, 42, labels, path)
    shifted = dates + pd.Timedelta(days=7)  # same match ids, later dates
    with pytest.raises(ValueError, match="match dates"):
        load_validated_fold_assignment(ids, shifted, 2, 42, path)


def test_fold_count_mismatch_is_rejected(tmp_path):
    path = tmp_path / "fold_assignment.parquet"
    ids, dates, labels = make_rows(n_dates=8)
    create_fold_assignment(ids, dates, 3, 42, labels, path)
    with pytest.raises(ValueError, match="folds"):
        load_validated_fold_assignment(ids, dates, 2, 42, path)


def test_corrupted_fold_values_are_rejected(tmp_path):
    path = tmp_path / "fold_assignment.parquet"
    ids, dates, labels = make_rows(n_dates=6)
    create_fold_assignment(ids, dates, 2, 42, labels, path)
    # Same matches, dates, and fold set, but permuted fold values.
    frame = load_fold_assignment(path)
    frame = frame.assign(fold=frame["fold"].map({0: 1, 1: 2, 2: 0}))
    save_fold_assignment(frame, path)
    with pytest.raises(ValueError, match="computed folds"):
        load_validated_fold_assignment(ids, dates, 2, 42, path)


def test_reordered_assignment_is_rejected(tmp_path):
    path = tmp_path / "fold_assignment.parquet"
    ids, dates, labels = make_rows(n_dates=6)
    create_fold_assignment(ids, dates, 2, 42, labels, path)
    # Same rows, wrong canonical ordering.
    save_fold_assignment(load_fold_assignment(path).iloc[::-1].reset_index(drop=True), path)
    with pytest.raises(ValueError, match="canonical ordering"):
        load_validated_fold_assignment(ids, dates, 2, 42, path)


def test_02_notebook_never_writes_the_assignment(tmp_path):
    """02 tuners only load; the persisted file is untouched by validation."""
    path = tmp_path / "fold_assignment.parquet"
    ids, dates, labels = make_rows(n_dates=6)
    create_fold_assignment(ids, dates, 2, 42, labels, path)
    before = path.read_bytes()
    load_validated_fold_assignment(ids, dates, 2, 42, path)
    assert path.read_bytes() == before
