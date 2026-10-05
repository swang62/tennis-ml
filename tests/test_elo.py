"""Hermetic behavior tests for the incremental Elo materializer.

The materializer is tested against an in-memory repository (a fake at the
database boundary), so no live PostgreSQL is required. The fake honors
commit/rollback so transaction-atomicity and fail-closed behavior are real.

These tests pin that contract: unrated matches are appended, and any change to
already-rated history (rewritten, removed, or back-dated matches) deletes and
re-rates everything from the earliest affected date, atomically.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime

import pytest

from src.features import elo
from src.features.elo import (
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

    def __init__(self, events, snapshots=None):
        self._events = list(events)
        self._snapshots = list(snapshots) if snapshots else []
        self._pending: list[SnapshotRow] = []
        self._pending_delete_from: date | None = None
        self._replay_from: date | None = None
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
    def earliest_stale_date(self):
        stale = []
        by_id = {e.match_id: e for e in self._events}
        rated: dict[str, list] = {}
        for s in self._snapshots:
            rated.setdefault(s["match_id"], []).append(s)
        for match_id, rows in rated.items():
            first = min(r["match_date"] for r in rows)
            event = by_id.get(match_id)
            if event is None:
                stale.append(first)
            elif elo_source_hash(event) not in {r["source_hash"] for r in rows}:
                stale.append(min(first, event.match_date))
        for e in self._events:
            if e.match_id in rated:
                continue
            players = (e.player1_id, e.player2_id)
            if any(
                s["player_id"] in players
                and (s["match_date"], s["match_num"], s["match_id"]) > self._key_ev(e)
                for s in self._snapshots
            ):
                stale.append(e.match_date)
        return min(stale, default=None)

    def delete_snapshots_from(self, day):
        assert self._in_tx, "delete outside transaction"
        self._pending_delete_from = day

    def snapshot_events(self, replay_from):
        self._replay_from = replay_from
        snapped = self._snap_match_ids()
        sel = [
            e
            for e in self._events
            if e.match_id not in snapped
            or (replay_from is not None and e.match_date >= replay_from)
        ]
        sel.sort(key=self._key_ev)
        return list(sel)

    def get_prior_overall(self, player_id):
        cands = [
            s
            for s in self._all_snapshots()
            if s["player_id"] == player_id
            and (self._replay_from is None or s["match_date"] < self._replay_from)
        ]
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
        self._pending_delete_from = None
        self._insert_calls = 0
        self.begin_called = True

    def commit(self):
        if self._pending_delete_from is not None:
            self._snapshots = [
                s for s in self._snapshots if s["match_date"] < self._pending_delete_from
            ]
        self._snapshots.extend(dict(asdict(r)) for r in self._pending)
        self._in_tx = False
        self._pending = []
        self.committed += 1

    def rollback(self):
        self._pending = []
        self._pending_delete_from = None
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
# Materialization behavior
# --------------------------------------------------------------------------- #


def _rated(match_id, d, match_num, surface, p1, p2, post_a=1510.0, post_b=1490.0):
    """Both participants' persisted snapshot rows for a match won by p1."""
    return [
        _snap_row(p1, match_id, d, match_num, surface, p1, p1, p2, post_a),
        _snap_row(p2, match_id, d, match_num, surface, p1, p1, p2, post_b),
    ]


def _by_player_match(snapshots):
    return {(s["match_id"], s["player_id"]): round(s["pre_elo"], 6) for s in snapshots}


def test_first_match_uses_defaults_and_moves_rating():
    repo = MemoryEloRepo(events=[_ev("m1", "2024-01-01", 1, "hard", "A", "A", "B")])
    result = materialize_elo(repo=repo)

    assert result == EloRunResult(processed=1, snapshots=2, replay_from=None)
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


def test_no_op_when_every_match_is_already_rated():
    repo = MemoryEloRepo(
        events=[_ev("m1", "2024-01-01", 1, "hard", "A", "A", "B")],
        snapshots=_rated("m1", "2024-01-01", 1, "hard", "A", "B"),
    )
    result = materialize_elo(repo=repo)

    assert result == EloRunResult(processed=0, snapshots=0, replay_from=None)
    assert repo.committed == 0
    assert repo.begin_called is False


def test_new_later_match_is_appended_without_replay():
    repo = MemoryEloRepo(
        events=[
            _ev("m1", "2024-01-01", 1, "hard", "A", "A", "B"),
            _ev("m2", "2024-01-02", 1, "hard", "A", "A", "C"),
        ],
        snapshots=_rated("m1", "2024-01-01", 1, "hard", "A", "B"),
    )
    result = materialize_elo(repo=repo)

    assert result == EloRunResult(processed=1, snapshots=2, replay_from=None)
    assert {s["match_id"] for s in repo._snapshots} == {"m1", "m2"}


def test_historical_insert_replays_later_ratings_to_match_a_fresh_rebuild():
    # m0 arrives after m1 was rated, but is dated earlier: m1's ratings are stale.
    events = [
        _ev("m0", "2023-12-01", 1, "hard", "A", "A", "B"),
        _ev("m1", "2024-01-01", 1, "hard", "A", "A", "B"),
    ]
    repo = MemoryEloRepo(
        events=events, snapshots=_rated("m1", "2024-01-01", 1, "hard", "A", "B", 1500.0, 1500.0)
    )
    result = materialize_elo(repo=repo)

    fresh = MemoryEloRepo(events=events)
    materialize_elo(repo=fresh)
    assert result.replay_from == date(2023, 12, 1)
    assert _by_player_match(repo._snapshots) == _by_player_match(fresh._snapshots)
    assert len(repo._snapshots) == 4


def test_rewritten_match_is_re_rated_with_its_new_content():
    # m1 was rated on hard; the source now reports clay.
    stale = _rated("m1", "2024-01-01", 1, "hard", "A", "B")
    changed = _ev("m1", "2024-01-01", 1, "clay", "A", "A", "B")
    repo = MemoryEloRepo(events=[changed], snapshots=stale)

    result = materialize_elo(repo=repo)

    assert result.replay_from == date(2024, 1, 1)
    assert len(repo._snapshots) == 2
    assert {s["source_hash"] for s in repo._snapshots} == {elo_source_hash(changed)}
    assert {s["surface"] for s in repo._snapshots} == {"clay"}


def test_removed_match_is_dropped_and_later_matches_re_rated_without_it():
    repo = MemoryEloRepo(
        events=[_ev("m2", "2024-01-02", 1, "hard", "A", "A", "C")],
        snapshots=_rated("m1", "2024-01-01", 1, "hard", "A", "B"),
    )
    result = materialize_elo(repo=repo)

    assert result.replay_from == date(2024, 1, 1)
    assert {s["match_id"] for s in repo._snapshots} == {"m2"}
    a_snapshot = next(s for s in repo._snapshots if s["player_id"] == "A")
    assert a_snapshot["pre_elo"] == pytest.approx(1500.0)


def test_failed_replay_leaves_existing_snapshots_untouched():
    stale = _rated("m1", "2024-01-01", 1, "hard", "A", "B")
    repo = MemoryEloRepo(
        events=[_ev("m1", "2024-01-01", 1, "clay", "A", "A", "B")], snapshots=list(stale)
    )
    repo._fail_after = 1

    with pytest.raises(RuntimeError):
        materialize_elo(repo=repo)

    assert repo.committed == 0
    assert repo._snapshots == stale


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


def test_rerun_rates_nothing_twice():
    repo = MemoryEloRepo(
        events=[
            _ev("m1", "2024-01-01", 1, "hard", "A", "A", "B"),
            _ev("m2", "2024-01-02", 1, "hard", "A", "A", "C"),
        ]
    )
    materialize_elo(repo=repo)

    again = materialize_elo(repo=repo)

    assert again == EloRunResult(processed=0, snapshots=0, replay_from=None)
    assert len([s for s in repo._snapshots if s["match_id"] == "m1"]) == 2
    assert len([s for s in repo._snapshots if s["match_id"] == "m2"]) == 2


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
