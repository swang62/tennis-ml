"""Hermetic behavior tests for the incremental Elo materializer.

The materializer is tested against an in-memory repository (a fake at the
database boundary), so no live PostgreSQL is required. The fake honors
commit/rollback so transaction-atomicity and fail-closed behavior are real.

Progress is owned solely by the shared ``bronze.etl_state`` timestamp watermark.
These tests pin that contract: Elo reads the timestamp watermark to select new
matches and to fail closed on historical corrections, and never advances progress
itself, so a rerun after a later-phase failure rates no match twice.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime

import pytest

from src.features import elo
from src.features.elo import (
    EloHistoryChanged,
    EloRunResult,
    MatchEvent,
    SnapshotRow,
    elo_source_hash,
    expected_score,
    k_factor,
    materialize_elo,
    regress_rating,
)

BASE = datetime(2024, 1, 1, 9, 0, 0)


def _ev(match_id, d, match_num, surface, winner, p1, p2, ingested_at=BASE):
    """Build a MatchEvent; winner is always p1 (matches the bronze CHECK)."""
    return MatchEvent(
        match_id=match_id,
        match_date=date.fromisoformat(d),
        match_num=match_num,
        surface=surface,
        winner_id=winner,
        player1_id=p1,
        player2_id=p2,
        ingested_at=ingested_at,
    )


def _snap_row(
    player_id,
    match_id,
    d,
    match_num,
    surface,
    winner,
    p1,
    p2,
    post_elo,
    ingested_at=BASE,
    source_hash=None,
):
    """A persisted snapshot row (as a dict) with the fields the fake reads."""
    if source_hash is None:
        source_hash = elo_source_hash(
            _ev(match_id, d, match_num, surface, winner, p1, p2, ingested_at)
        )
    return {
        "player_id": player_id,
        "match_id": match_id,
        "match_date": date.fromisoformat(d),
        "match_num": match_num,
        "surface": surface,
        "pre_elo": 0.0,
        "post_elo": post_elo,
        "prior_overall_matches": 0,
        "source_hash": source_hash,
    }


class MemoryEloRepo:
    """In-memory EloRepo honoring begin/commit/rollback like psycopg."""

    def __init__(self, events, snapshots=None, watermark=None):
        self._events = list(events)
        self._snapshots = list(snapshots) if snapshots else []
        self._watermark = watermark
        self._pending: list[SnapshotRow] = []
        self._in_tx = False
        self.committed = 0
        self.begin_called = False
        self._fail_after: int | None = None  # batch insert raises after this many rows
        self._insert_calls = 0

    # --- snapshot tuple key for causal ordering ---
    @staticmethod
    def _key_ev(e):
        return (e.match_date, e.match_num, e.match_id)

    def _all_snapshots(self):
        return self._snapshots + [asdict(x) for x in self._pending]

    def _snap_match_ids(self):
        return {s["match_id"] for s in self._snapshots}

    # --- EloRepo interface ---
    def get_watermark(self):
        return self._watermark

    def count_events_through(self, watermark):
        return sum(1 for e in self._events if e.ingested_at <= watermark)

    def count_snapshots_through(self, watermark):
        snapped = self._snap_match_ids()
        return sum(1 for e in self._events if e.ingested_at <= watermark and e.match_id in snapped)

    def count_mismatched_history(self, watermark):
        snapped = self._snap_match_ids()
        stored: dict[str, set] = {}
        for s in self._snapshots:
            stored.setdefault(s["match_id"], set()).add(s.get("source_hash"))
        mismatches = 0
        for e in self._events:
            if e.ingested_at > watermark:
                continue
            if e.match_id not in snapped:
                mismatches += 1
                continue
            if elo_source_hash(e) not in stored.get(e.match_id, set()):
                mismatches += 1
        return mismatches

    def snapshot_events(self, watermark):
        snapped = self._snap_match_ids()
        if watermark is None:
            sel = [e for e in self._events if e.match_id not in snapped]
        else:
            sel = [
                e for e in self._events if e.ingested_at > watermark and e.match_id not in snapped
            ]
        sel.sort(key=self._key_ev)
        return list(sel)

    def get_prior_overall(self, player_id):
        cands = [s for s in self._all_snapshots() if s["player_id"] == player_id]
        if not cands:
            return None
        cands.sort(key=lambda s: (s["match_date"], s["match_num"], s["match_id"]), reverse=True)
        s = cands[0]
        return (s["post_elo"], s["prior_overall_matches"], s["match_date"])

    def get_prior_overall_many(self, player_ids):
        result: dict = {}
        for player_id in player_ids:
            row = self.get_prior_overall(player_id)
            if row is not None:
                result[player_id] = row
        return result

    def insert_snapshots(self, rows: list[SnapshotRow]):
        assert self._in_tx, "insert outside transaction"
        self._insert_calls += len(rows)
        if self._fail_after is not None and self._insert_calls > self._fail_after:
            raise RuntimeError("simulated DB write failure")
        self._pending.extend(rows)

    def begin(self):
        self._in_tx = True
        self._pending = []
        self._insert_calls = 0
        self.begin_called = True

    def commit(self):
        self._snapshots.extend(dict(asdict(r)) for r in self._pending)
        self._in_tx = False
        self._pending = []
        self.committed += 1

    def rollback(self):
        self._pending = []
        self._in_tx = False


# --------------------------------------------------------------------------- #
# Pure math
# --------------------------------------------------------------------------- #


def test_expected_score_is_symmetric():
    assert expected_score(1500, 1500) == pytest.approx(0.5)
    a = expected_score(1600, 1400)
    b = expected_score(1400, 1600)
    assert a == pytest.approx(1 - b)
    assert a > 0.5


def test_k_factor_bounds():
    assert k_factor(0) == pytest.approx(62.0)
    assert k_factor(5) == pytest.approx(62.0)
    large = k_factor(10_000)
    assert 43.0 <= large <= 43.1


def test_regress_within_grace_is_identity():
    assert regress_rating(1800.0, None) == pytest.approx(1800.0)
    assert regress_rating(1800.0, 90) == pytest.approx(1800.0)
    assert regress_rating(1800.0, 50) == pytest.approx(1800.0)


def test_regress_pulls_partially_after_layoff():
    out = regress_rating(1800.0, 160)
    assert out == pytest.approx(1500.0 + 300.0 * (0.99**10), rel=1e-6)
    assert 1500.0 < out < 1800.0


def test_regress_capped_at_fifty_percent():
    out = regress_rating(1800.0, 100_000)
    assert out == pytest.approx(1650.0)


# --------------------------------------------------------------------------- #
# Materialization behavior (unified timestamp watermark)
# --------------------------------------------------------------------------- #


def test_first_match_uses_defaults_and_moves_rating():
    repo = MemoryEloRepo(events=[_ev("m1", "2024-01-01", 1, "hard", "A", "A", "B")])
    result = materialize_elo(repo=repo)

    assert result == EloRunResult(processed=1, snapshots=2, watermark=None)
    assert repo.committed == 1
    snaps = {s["player_id"]: s for s in repo._snapshots}
    assert set(snaps) == {"A", "B"}

    assert snaps["A"]["pre_elo"] == pytest.approx(1500.0)
    assert snaps["B"]["pre_elo"] == pytest.approx(1500.0)
    assert snaps["A"]["post_elo"] > 1500.0
    assert snaps["B"]["post_elo"] < 1500.0
    assert snaps["A"]["k_overall"] == pytest.approx(62.0)
    assert snaps["B"]["k_overall"] == pytest.approx(62.0)
    assert len(repo._snapshots) == 2


def test_same_day_ordering_second_match_sees_first_update():
    repo = MemoryEloRepo(
        events=[
            _ev("m1", "2024-01-01", 1, "hard", "A", "A", "B"),
            _ev("m2", "2024-01-01", 2, "hard", "A", "A", "C"),
        ]
    )
    materialize_elo(repo=repo)

    snaps = sorted(repo._snapshots, key=lambda s: s["match_id"])
    a_first = next(s for s in snaps if s["match_id"] == "m1" and s["player_id"] == "A")
    a_second = next(s for s in snaps if s["match_id"] == "m2" and s["player_id"] == "A")
    assert a_second["pre_elo"] == pytest.approx(a_first["post_elo"])


def test_no_op_when_no_new_matches_without_watermark():
    wm = None
    repo = MemoryEloRepo(
        events=[_ev("m1", "2024-01-01", 1, "hard", "A", "A", "B")],
        snapshots=[
            _snap_row("A", "m1", "2024-01-01", 1, "hard", "A", "A", "B", 1510.0),
            _snap_row("B", "m1", "2024-01-01", 1, "hard", "A", "A", "B", 1490.0),
        ],
        watermark=wm,
    )
    result = materialize_elo(repo=repo)
    assert result == EloRunResult(processed=0, snapshots=0, watermark=wm)
    assert repo.committed == 0
    assert repo.begin_called is False


def test_no_op_when_no_new_matches_past_timestamp_watermark():
    # Match fully processed before the watermark; a rerun must be a no-op.
    wm = datetime(2024, 1, 2, 0, 0, 0)
    repo = MemoryEloRepo(
        events=[_ev("m1", "2024-01-01", 1, "hard", "A", "A", "B", ingested_at=BASE)],
        snapshots=[
            _snap_row("A", "m1", "2024-01-01", 1, "hard", "A", "A", "B", 1510.0, ingested_at=BASE),
            _snap_row("B", "m1", "2024-01-01", 1, "hard", "A", "A", "B", 1490.0, ingested_at=BASE),
        ],
        watermark=wm,
    )
    result = materialize_elo(repo=repo)
    assert result == EloRunResult(processed=0, snapshots=0, watermark=wm)
    assert repo.committed == 0


def test_incremental_selects_only_matches_after_watermark():
    # Watermark covers m1; m2 arrived later (ingested_at after watermark).
    wm = BASE
    repo = MemoryEloRepo(
        events=[
            _ev("m1", "2024-01-01", 1, "hard", "A", "A", "B", ingested_at=BASE),
            _ev(
                "m2",
                "2024-01-02",
                1,
                "hard",
                "A",
                "A",
                "C",
                ingested_at=datetime(2024, 1, 1, 11, 0, 0),
            ),
        ],
        snapshots=[
            _snap_row("A", "m1", "2024-01-01", 1, "hard", "A", "A", "B", 1510.0, ingested_at=BASE),
            _snap_row("B", "m1", "2024-01-01", 1, "hard", "A", "A", "B", 1490.0, ingested_at=BASE),
        ],
        watermark=wm,
    )
    result = materialize_elo(repo=repo)
    assert result == EloRunResult(processed=1, snapshots=2, watermark=wm)
    assert {s["match_id"] for s in repo._snapshots} == {"m1", "m2"}


def test_fail_closed_on_historical_insert_before_watermark():
    # Watermark says m1 processed; m0 inserted before it with no snapshot.
    wm = datetime(2024, 1, 1, 10, 0, 0)
    repo = MemoryEloRepo(
        events=[
            _ev(
                "m0",
                "2023-12-01",
                1,
                "hard",
                "A",
                "A",
                "B",
                ingested_at=datetime(2023, 12, 1, 9, 0, 0),
            ),
            _ev("m1", "2024-01-01", 1, "hard", "A", "A", "B", ingested_at=BASE),
        ],
        snapshots=[
            _snap_row("A", "m1", "2024-01-01", 1, "hard", "A", "A", "B", 1510.0, ingested_at=BASE),
            _snap_row("B", "m1", "2024-01-01", 1, "hard", "A", "A", "B", 1490.0, ingested_at=BASE),
        ],
        watermark=wm,
    )
    with pytest.raises(EloHistoryChanged):
        materialize_elo(repo=repo)

    assert repo.committed == 0
    assert {s["match_id"] for s in repo._snapshots} == {"m1"}


def test_fail_closed_on_changed_processed_match():
    # m1 was processed on hard; the source now reports clay (content change).
    wm = datetime(2024, 1, 2, 0, 0, 0)
    original_hash = elo_source_hash(
        _ev("m1", "2024-01-01", 1, "hard", "A", "A", "B", ingested_at=BASE)
    )
    repo = MemoryEloRepo(
        events=[_ev("m1", "2024-01-01", 1, "clay", "A", "A", "B", ingested_at=BASE)],
        snapshots=[
            _snap_row(
                "A",
                "m1",
                "2024-01-01",
                1,
                "hard",
                "A",
                "A",
                "B",
                1510.0,
                ingested_at=BASE,
                source_hash=original_hash,
            ),
            _snap_row(
                "B",
                "m1",
                "2024-01-01",
                1,
                "hard",
                "A",
                "A",
                "B",
                1490.0,
                ingested_at=BASE,
                source_hash=original_hash,
            ),
        ],
        watermark=wm,
    )
    with pytest.raises(EloHistoryChanged):
        materialize_elo(repo=repo)

    assert repo.committed == 0
    assert {s["match_id"] for s in repo._snapshots} == {"m1"}


def test_rollback_on_failure_leaves_state_unchanged():
    repo = MemoryEloRepo(
        events=[
            _ev("m1", "2024-01-01", 1, "hard", "A", "A", "B"),
            _ev("m2", "2024-01-02", 1, "hard", "A", "A", "C"),
        ]
    )
    repo._fail_after = 2
    with pytest.raises(RuntimeError):
        materialize_elo(repo=repo)

    assert repo.committed == 0
    assert repo._snapshots == []


def test_rerun_is_noop_after_processing_without_watermark_advance():
    repo = MemoryEloRepo(
        events=[
            _ev("m1", "2024-01-01", 1, "hard", "A", "A", "B"),
            _ev("m2", "2024-01-02", 1, "hard", "A", "A", "C"),
        ]
    )
    result = materialize_elo(repo=repo)
    assert result.watermark is None
    # a second run must not rate anything twice (Elo never advances progress)
    again = materialize_elo(repo=repo)
    assert again.processed == 0
    assert len([s for s in repo._snapshots if s["match_id"] == "m1"]) == 2
    assert len([s for s in repo._snapshots if s["match_id"] == "m2"]) == 2


def test_rerun_after_final_gold_failure_does_not_duplicate_elo():
    # Simulate one incremental ETL run: base dbt + Elo succeed, but the final
    # gold phase fails, so ETL never advances the shared watermark. The Elo
    # snapshots for the new match already exist. The next ETL run re-reads the
    # old watermark, must not re-rate m2, and must still let final gold run.
    wm = BASE  # covers the previously processed match m1 only
    repo = MemoryEloRepo(
        events=[
            _ev("m1", "2024-01-01", 1, "hard", "A", "A", "B", ingested_at=BASE),
            _ev(
                "m2",
                "2024-01-02",
                1,
                "hard",
                "A",
                "A",
                "C",
                ingested_at=datetime(2024, 1, 1, 11, 0, 0),
            ),
        ],
        snapshots=[
            _snap_row("A", "m1", "2024-01-01", 1, "hard", "A", "A", "B", 1510.0, ingested_at=BASE),
            _snap_row("B", "m1", "2024-01-01", 1, "hard", "A", "A", "B", 1490.0, ingested_at=BASE),
        ],
        watermark=wm,
    )
    # Run 1: Elo processes m2 (commits snapshots); gold then fails -> watermark stays.
    run1 = materialize_elo(repo=repo)
    assert run1 == EloRunResult(processed=1, snapshots=2, watermark=wm)

    # Run 2 (after gold failure): Elo reads the unchanged watermark, must be a
    # no-op because m2 already has snapshots, and must not duplicate them.
    run2 = materialize_elo(repo=repo)
    assert run2 == EloRunResult(processed=0, snapshots=0, watermark=wm)

    # Exactly two snapshots for m2 (no double rating); historical m1 untouched.
    assert len([s for s in repo._snapshots if s["match_id"] == "m2"]) == 2
    assert len([s for s in repo._snapshots if s["match_id"] == "m1"]) == 2


def test_ordering_respects_match_num_then_match_id():
    repo = MemoryEloRepo(
        events=[
            _ev("ma", "2024-01-01", 3, "hard", "A", "A", "B"),
            _ev("mb", "2024-01-01", 1, "hard", "A", "A", "C"),
        ]
    )
    materialize_elo(repo=repo)
    mb_a = next(s for s in repo._snapshots if s["match_id"] == "mb" and s["player_id"] == "A")
    ma_a = next(s for s in repo._snapshots if s["match_id"] == "ma" and s["player_id"] == "A")
    assert ma_a["pre_elo"] == pytest.approx(mb_a["post_elo"])


def test_match_id_is_deterministic_tiebreaker_when_date_and_match_num_equal():
    # Two different-tournament matches share the SAME date and match_num; only the
    # globally-unique match_id (bronze PK) disambiguates causal order. Inserted in
    # reverse match_id order to prove the tie-break is match_id, not insertion order.
    repo = MemoryEloRepo(
        events=[
            _ev("mb", "2024-01-01", 5, "hard", "A", "A", "B"),
            _ev("ma", "2024-01-01", 5, "hard", "A", "A", "C"),
        ]
    )
    materialize_elo(repo=repo)
    ma_a = next(s for s in repo._snapshots if s["match_id"] == "ma" and s["player_id"] == "A")
    mb_a = next(s for s in repo._snapshots if s["match_id"] == "mb" and s["player_id"] == "A")
    # Smaller match_id "ma" processes first; "mb" sees "ma"'s post-rating.
    assert mb_a["pre_elo"] == pytest.approx(ma_a["post_elo"])


def test_deterministic_elo_under_shuffled_input():
    # Same-day matches with ascending match_num; causal order must be identical
    # regardless of the order rows arrive from the repository (no nondeterminism).
    import random

    base_events = [_ev(f"m{i}", "2024-01-01", i, "hard", "A", "A", f"P{i}") for i in range(1, 8)]
    reference = MemoryEloRepo(events=list(base_events))
    materialize_elo(repo=reference)
    ref = sorted(
        (s["match_id"], round(s["post_elo"], 6))
        for s in reference._snapshots
        if s["player_id"] == "A"
    )

    for seed in range(10):
        rng = random.Random(seed)
        shuffled = list(base_events)
        rng.shuffle(shuffled)
        repo = MemoryEloRepo(events=shuffled)
        materialize_elo(repo=repo)
        got = sorted(
            (s["match_id"], round(s["post_elo"], 6))
            for s in repo._snapshots
            if s["player_id"] == "A"
        )
        assert got == ref
