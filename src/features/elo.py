"""Atomic, append-only per-player Elo materialization.

Reads ``bronze.match_events`` in strict causal order and writes two
``silver.elo_snapshots`` rows per physical match (one per participant), carrying
both global and current-surface ratings.

Progress is owned solely by ``bronze.etl_state`` (its ``source_watermark``
TIMESTAMPTZ). ETL advances that watermark only after base dbt, this Elo phase,
and the final ``gold.match_features`` build all succeed. This materializer reads
the shared watermark to select new matches and to fail closed on historical
corrections; it never advances progress itself, so a rerun after a later-phase
failure rebuilds or reuses the snapshots it already wrote without rating a match
twice.

The core is pure (the per-match math) and the database boundary is a small
repository protocol, so behavior is testable without a live database. The real
repository uses the project's pooled psycopg connection.
"""

from __future__ import annotations

import hashlib
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol, cast

import psycopg
from psycopg.rows import tuple_row

from src.constants import (
    BRONZE_ETL_STATE,
    BRONZE_MATCHES_TABLE,
    ELO_DEFAULT_RATING,
    ELO_INACTIVITY_GRACE_DAYS,
    ELO_INACTIVITY_REGRESS_CAP,
    ELO_INACTIVITY_REGRESS_PER_7D,
    ELO_K_BASE,
    ELO_K_DIVISOR,
    ELO_K_MIN,
    SILVER_ELO_SNAPSHOTS,
)
from src.db.client import connection

# ETL records the shared watermark under this pipeline key; Elo reads the same row.
ETL_PIPELINE = "dbt"


class EloHistoryChanged(RuntimeError):
    """Raised when a source match exists at or before the shared ETL watermark."""


@dataclass(frozen=True)
class MatchEvent:
    """A causal match key plus the fields needed to rate it."""

    match_id: str
    match_date: date
    match_num: int
    surface: str
    winner_id: str
    player1_id: str
    player2_id: str
    ingested_at: datetime  # bronze insert timestamp; drives the timestamp watermark


@dataclass
class SnapshotRow:
    """One participant's pre/post Elo for a single physical match."""

    player_id: str
    match_id: str
    match_date: date
    match_num: int
    surface: str
    pre_elo: float
    post_elo: float
    pre_elo_surface: float
    post_elo_surface: float
    prior_overall_matches: int
    prior_surface_matches: int
    k_overall: float
    k_surface: float
    source_hash: str


@dataclass
class EloRunResult:
    """What a materialization run did."""

    processed: int
    snapshots: int
    watermark: datetime | None  # shared etl_state timestamp the run read (diagnostics)


@dataclass
class _PriorState:
    """A participant's reconstructed state entering a match."""

    rating: float
    count_in: int  # prior matches before the current match
    last_date: date | None


# --------------------------------------------------------------------------- #
# Pure Elo math
# --------------------------------------------------------------------------- #


def expected_score(rating_a: float, rating_b: float) -> float:
    """Logistic expected win probability of ``a`` over ``b``."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def k_factor(prior_matches: int) -> float:
    """Adaptive K: min(ELO_K_MIN, ELO_K_BASE + ELO_K_DIVISOR / (prior + 1))."""
    return min(ELO_K_MIN, ELO_K_BASE + ELO_K_DIVISOR / (prior_matches + 1))


def regress_rating(rating: float, gap_days: int | None) -> float:
    """Pull a stale rating toward 1500 after the inactivity grace period.

    No regression through ``ELO_INACTIVITY_GRACE_DAYS``; then 1% of the
    remaining distance to 1500 per further 7 days, capped at a total 50%
    regression of the original distance.
    """
    if gap_days is None or gap_days <= ELO_INACTIVITY_GRACE_DAYS:
        return rating
    excess = gap_days - ELO_INACTIVITY_GRACE_DAYS
    periods = excess // 7
    if periods <= 0:
        return rating
    factor = (1.0 - ELO_INACTIVITY_REGRESS_PER_7D) ** periods
    factor = max(factor, 1.0 - ELO_INACTIVITY_REGRESS_CAP)
    return ELO_DEFAULT_RATING + (rating - ELO_DEFAULT_RATING) * factor


def elo_source_hash(event: MatchEvent) -> str:
    """Stable sha256 of the source match content that drives Elo.

    Used to detect a historical change (not just an insertion) at or before the
    ETL watermark during validation, before any snapshot is mutated.
    """
    payload = "|".join(
        [
            event.match_id,
            event.match_date.isoformat(),
            str(event.match_num),
            event.surface,
            event.winner_id,
            event.player1_id,
            event.player2_id,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _gap_days(current: date, last: date | None) -> int | None:
    if last is None:
        return None
    return (current - last).days


def _update(
    pre_a: float, k_a: float, pre_b: float, k_b: float, score_a: int
) -> tuple[float, float]:
    """Post-ratings for the two participants of one orientation."""
    expected_a = expected_score(pre_a, pre_b)
    post_a = pre_a + k_a * (score_a - expected_a)
    post_b = pre_b + k_b * ((1 - score_a) - (1 - expected_a))
    return post_a, post_b


def _prior_overall(repo: EloRepo, player_id: str) -> _PriorState:
    """Participant's global state entering a match: rating, count-in, last date."""
    row = repo.get_prior_overall(player_id)
    if row is None:
        return _PriorState(rating=ELO_DEFAULT_RATING, count_in=0, last_date=None)
    post_elo, prior_matches_stored, last_date = row
    return _PriorState(rating=post_elo, count_in=prior_matches_stored + 1, last_date=last_date)


def _prior_surface(repo: EloRepo, player_id: str, surface: str) -> _PriorState:
    row = repo.get_prior_surface(player_id, surface)
    if row is None:
        return _PriorState(rating=ELO_DEFAULT_RATING, count_in=0, last_date=None)
    post_elo_surface, prior_surface_matches_stored, last_date = row
    return _PriorState(
        rating=post_elo_surface, count_in=prior_surface_matches_stored + 1, last_date=last_date
    )


def _process_event(repo: EloRepo, event: MatchEvent) -> list[SnapshotRow]:
    """Compute both participants' snapshot rows for one match."""
    p1, p2, surface = event.player1_id, event.player2_id, event.surface

    p1o = _prior_overall(repo, p1)
    p2o = _prior_overall(repo, p2)
    p1s = _prior_surface(repo, p1, surface)
    p2s = _prior_surface(repo, p2, surface)

    source_hash = elo_source_hash(event)

    p1_pre_o = regress_rating(p1o.rating, _gap_days(event.match_date, p1o.last_date))
    p2_pre_o = regress_rating(p2o.rating, _gap_days(event.match_date, p2o.last_date))
    p1_pre_s = regress_rating(p1s.rating, _gap_days(event.match_date, p1s.last_date))
    p2_pre_s = regress_rating(p2s.rating, _gap_days(event.match_date, p2s.last_date))

    k1_o = k_factor(p1o.count_in)
    k2_o = k_factor(p2o.count_in)
    k1_s = k_factor(p1s.count_in)
    k2_s = k_factor(p2s.count_in)

    score1 = 1 if event.winner_id == p1 else 0
    post1_o, post2_o = _update(p1_pre_o, k1_o, p2_pre_o, k2_o, score1)
    post1_s, post2_s = _update(p1_pre_s, k1_s, p2_pre_s, k2_s, score1)

    snap1 = SnapshotRow(
        player_id=p1,
        match_id=event.match_id,
        match_date=event.match_date,
        match_num=event.match_num,
        surface=surface,
        pre_elo=float(p1_pre_o),
        post_elo=float(post1_o),
        pre_elo_surface=float(p1_pre_s),
        post_elo_surface=float(post1_s),
        prior_overall_matches=int(p1o.count_in),
        prior_surface_matches=int(p1s.count_in),
        k_overall=float(k1_o),
        k_surface=float(k1_s),
        source_hash=source_hash,
    )
    snap2 = SnapshotRow(
        player_id=p2,
        match_id=event.match_id,
        match_date=event.match_date,
        match_num=event.match_num,
        surface=surface,
        pre_elo=float(p2_pre_o),
        post_elo=float(post2_o),
        pre_elo_surface=float(p2_pre_s),
        post_elo_surface=float(post2_s),
        prior_overall_matches=int(p2o.count_in),
        prior_surface_matches=int(p2s.count_in),
        k_overall=float(k2_o),
        k_surface=float(k2_s),
        source_hash=source_hash,
    )
    return [snap1, snap2]


def _run(repo: EloRepo) -> EloRunResult:
    # Shared progress watermark (TIMESTAMPTZ) from bronze.etl_state. Elo reads it
    # but never advances it: ETL owns final advancement after every phase succeeds.
    watermark = repo.get_watermark()

    # Fail closed before any mutation: every source match at/before the shared
    # watermark must already be snapshotted with matching content. A historical
    # insert or change slips in with an old ingested_at (<= watermark) and is
    # caught here, since etl_state stores only a timestamp.
    if watermark is not None:
        if repo.count_events_through(watermark) != repo.count_snapshots_through(watermark):
            raise EloHistoryChanged(
                "source match introduced at/before the shared ETL watermark "
                f"{watermark}; rebuild Elo explicitly"
            )
        if repo.count_mismatched_history(watermark) > 0:
            raise EloHistoryChanged(
                "source match content changed at/before the shared ETL watermark "
                f"{watermark}; rebuild Elo explicitly"
            )

    # Only matches not yet covered by the shared watermark and lacking a snapshot
    # are eligible. The snapshot PK also guards against rating a match twice on a
    # rerun after a later-phase failure.
    events = repo.fetch_new_events(watermark)
    if not events:
        return EloRunResult(processed=0, snapshots=0, watermark=watermark)

    # Strict causal order: ascending match_date, then match_num, with match_id as
    # the deterministic tie-breaker. match_num is per-tournament, so (match_date,
    # match_num) is NOT globally unique; match_id (the bronze PK) breaks ties
    # deterministically. Sorting here makes the invariant real at the processing
    # layer regardless of how the repository returns rows.
    events.sort(key=lambda e: (e.match_date, e.match_num, e.match_id))

    repo.begin()
    try:
        for event in events:
            for row in _process_event(repo, event):
                repo.insert_snapshot(row)
        repo.commit()
    except Exception:
        repo.rollback()
        raise

    return EloRunResult(processed=len(events), snapshots=len(events) * 2, watermark=watermark)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def materialize_elo(repo: EloRepo | None = None) -> EloRunResult:
    """Materialize Elo snapshots for all unprocessed matches.

    With no ``repo`` the project's PostgreSQL connection is used. Pass a
    repository (e.g. a hermetic fake) to test the logic without a database.

    This does not advance pipeline progress; ETL advances bronze.etl_state only
    after base dbt, Elo, and gold.match_features all succeed.
    """
    if repo is None:
        repo = PsycopgEloRepo()
        try:
            return _run(repo)
        finally:
            repo.close()
    return _run(repo)


# --------------------------------------------------------------------------- #
# Repository boundary
# --------------------------------------------------------------------------- #


class EloRepo(Protocol):
    """Database boundary the materializer depends on."""

    def get_watermark(self) -> datetime | None: ...
    def count_events_through(self, watermark: datetime) -> int: ...
    def count_snapshots_through(self, watermark: datetime) -> int: ...
    def count_mismatched_history(self, watermark: datetime) -> int: ...
    def fetch_new_events(self, watermark: datetime | None) -> list[MatchEvent]: ...
    def get_prior_overall(self, player_id: str) -> tuple[float, int, date] | None: ...
    def get_prior_surface(self, player_id: str, surface: str) -> tuple[float, int, date] | None: ...
    def insert_snapshot(self, row: SnapshotRow) -> None: ...
    def begin(self) -> None: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...


class PsycopgEloRepo:
    """psycopg-backed EloRepo using the project's pooled connection."""

    def __init__(self) -> None:
        self._checkout = connection()
        self._conn = self._checkout.__enter__()
        self._prev_autocommit = self._conn.autocommit
        self._conn.autocommit = True
        self._cur = self._conn.cursor(row_factory=tuple_row)
        self._tx: AbstractContextManager[psycopg.Transaction] | None = None

    def close(self) -> None:
        if self._tx is not None:
            self._tx.__exit__(None, None, None)
            self._tx = None
        self._conn.autocommit = self._prev_autocommit
        self._checkout.__exit__(None, None, None)

    def get_watermark(self) -> datetime | None:
        self._cur.execute(
            f"SELECT source_watermark FROM {BRONZE_ETL_STATE} WHERE pipeline = %s",
            (ETL_PIPELINE,),
        )
        row = self._cur.fetchone()
        if row is None or row[0] is None:
            return None
        return row[0]

    def count_events_through(self, watermark: datetime) -> int:
        self._cur.execute(
            f"SELECT COUNT(*) FROM {BRONZE_MATCHES_TABLE} WHERE ingested_at <= %s",
            (watermark,),
        )
        row = self._cur.fetchone()
        if row is None:
            return 0
        return int(row[0])

    def count_snapshots_through(self, watermark: datetime) -> int:
        self._cur.execute(
            f"SELECT COUNT(DISTINCT match_id) FROM {SILVER_ELO_SNAPSHOTS} "
            f"WHERE match_id IN "
            f"(SELECT match_id FROM {BRONZE_MATCHES_TABLE} WHERE ingested_at <= %s)",
            (watermark,),
        )
        row = self._cur.fetchone()
        if row is None:
            return 0
        return int(row[0])

    def count_mismatched_history(self, watermark: datetime) -> int:
        self._cur.execute(
            f"SELECT match_id, match_date, match_num, surface, winner_id, "
            f"player1_id, player2_id FROM {BRONZE_MATCHES_TABLE} "
            f"WHERE ingested_at <= %s",
            (watermark,),
        )
        sources = self._cur.fetchall()
        if not sources:
            return 0
        ids = [r[0] for r in sources]
        self._cur.execute(
            f"SELECT DISTINCT match_id, source_hash FROM {SILVER_ELO_SNAPSHOTS} "
            f"WHERE match_id = ANY(%s)",
            (ids,),
        )
        stored: dict[str, set[str]] = {}
        for match_id, source_hash in self._cur.fetchall():
            stored.setdefault(match_id, set()).add(source_hash)

        mismatches = 0
        for r in sources:
            event = MatchEvent(
                match_id=r[0],
                match_date=r[1],
                match_num=int(r[2]),
                surface=r[3],
                winner_id=r[4],
                player1_id=r[5],
                player2_id=r[6],
                ingested_at=watermark,
            )
            hashes = stored.get(event.match_id)
            if not hashes or elo_source_hash(event) not in hashes:
                mismatches += 1
        return mismatches

    def fetch_new_events(self, watermark: datetime | None) -> list[MatchEvent]:
        base = (
            f"SELECT match_id, match_date, match_num, surface, winner_id, "
            f"player1_id, player2_id, ingested_at FROM {BRONZE_MATCHES_TABLE} "
        )
        if watermark is None:
            self._cur.execute(
                base + f"WHERE match_id NOT IN "
                f"(SELECT DISTINCT match_id FROM {SILVER_ELO_SNAPSHOTS}) "
                f"ORDER BY match_date, match_num, match_id"  # match_id breaks ties (per-tournament match_num)
            )
        else:
            self._cur.execute(
                base + f"WHERE ingested_at > %s "
                f"AND match_id NOT IN "
                f"(SELECT DISTINCT match_id FROM {SILVER_ELO_SNAPSHOTS}) "
                f"ORDER BY match_date, match_num, match_id",  # match_id breaks ties (per-tournament match_num)
                (watermark,),
            )
        return [
            MatchEvent(
                match_id=r[0],
                match_date=r[1],
                match_num=int(r[2]),
                surface=r[3],
                winner_id=r[4],
                player1_id=r[5],
                player2_id=r[6],
                ingested_at=r[7],
            )
            for r in self._cur.fetchall()
        ]

    def get_prior_overall(self, player_id: str) -> tuple[float, int, date] | None:
        self._cur.execute(
            f"SELECT post_elo, prior_overall_matches, match_date "
            f"FROM {SILVER_ELO_SNAPSHOTS} "
            f"WHERE player_id = %s "
            f"ORDER BY match_date DESC, match_num DESC, match_id DESC LIMIT 1",
            (player_id,),
        )
        row = self._cur.fetchone()
        if row is None:
            return None
        return (float(row[0]), int(row[1]), row[2])

    def get_prior_surface(self, player_id: str, surface: str) -> tuple[float, int, date] | None:
        self._cur.execute(
            f"SELECT post_elo_surface, prior_surface_matches, match_date "
            f"FROM {SILVER_ELO_SNAPSHOTS} "
            f"WHERE player_id = %s AND surface = %s "
            f"ORDER BY match_date DESC, match_num DESC, match_id DESC LIMIT 1",
            (player_id, surface),
        )
        row = self._cur.fetchone()
        if row is None:
            return None
        return (float(row[0]), int(row[1]), row[2])

    def insert_snapshot(self, row: SnapshotRow) -> None:
        self._cur.execute(
            f"INSERT INTO {SILVER_ELO_SNAPSHOTS} "
            f"(player_id, match_id, match_date, match_num, surface, "
            f" pre_elo, post_elo, pre_elo_surface, post_elo_surface, "
            f" prior_overall_matches, prior_surface_matches, k_overall, k_surface, "
            f" source_hash) "
            f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                row.player_id,
                row.match_id,
                row.match_date,
                row.match_num,
                row.surface,
                row.pre_elo,
                row.post_elo,
                row.pre_elo_surface,
                row.post_elo_surface,
                row.prior_overall_matches,
                row.prior_surface_matches,
                row.k_overall,
                row.k_surface,
                row.source_hash,
            ),
        )

    def begin(self) -> None:
        self._conn.autocommit = False
        tx = cast("AbstractContextManager[psycopg.Transaction]", self._conn.transaction())
        tx.__enter__()
        self._tx = tx

    def commit(self) -> None:
        if self._tx is not None:
            self._tx.__exit__(None, None, None)
            self._tx = None
        self._conn.autocommit = True

    def rollback(self) -> None:
        if self._tx is not None:
            self._tx.__exit__(RuntimeError, RuntimeError("elo rollback"), None)
            self._tx = None
        self._conn.autocommit = True
