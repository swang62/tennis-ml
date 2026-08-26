"""Atomic, append-only per-player Elo materialization.

Reads ``bronze.match_events`` in strict causal order and writes two
``silver.elo_snapshots`` rows per physical match (one per participant), carrying
both global and current-surface ratings.

Progress is owned solely by ``bronze.etl_state`` (its ``source_watermark``
TIMESTAMPTZ). ETL advances that watermark only after base dbt, this Elo phase,
and the final ``gold.match_features`` build all succeed. This materializer reads
the shared watermark to select new matches and to fail closed on historical
corrections; it never advances progress itself, so a rerun after a later-phase
failure reuses the snapshots it already wrote without rating a match twice.

The core is pure (the per-match math) and the database boundary is a small
repository protocol, so behavior is testable without a live database. The real
repository uses the project's pooled psycopg connection.
"""

from __future__ import annotations

import hashlib
import os
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol, cast

import duckdb
import psycopg
from psycopg.rows import tuple_row
from tqdm.auto import tqdm

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
    ROOT,
    SILVER_ELO_SNAPSHOTS,
    get_database_url,
)
from src.db.client import connection
from src.features.elo_math import regress_rating

# ETL records the shared watermark under this pipeline key; Elo reads the same row.
ETL_PIPELINE = "dbt"
ELO_MATCH_BATCH_SIZE = 25_000
ELO_SNAPSHOT_PATH = ROOT / "data" / "elo_snapshot.duckdb"


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
    prior_overall_matches: int
    k_overall: float
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


def _state_from_overall_row(row: tuple[float, int, date]) -> _PriorState:
    post_elo, prior_matches_stored, last_date = row
    return _PriorState(rating=post_elo, count_in=prior_matches_stored + 1, last_date=last_date)


def _process_event(
    event: MatchEvent,
    overall: dict[str, _PriorState],
) -> list[SnapshotRow]:
    """Compute both participants' snapshot rows for one match.

    ``overall`` is the authoritative in-memory state map, preloaded by ``_run``.
    Every participant must already have an entry (cold state defaults fill in
    players with no history), so this never touches the repository.
    """
    p1, p2, surface = event.player1_id, event.player2_id, event.surface

    p1o = overall[p1]
    p2o = overall[p2]
    source_hash = elo_source_hash(event)

    p1_pre_o = regress_rating(p1o.rating, _gap_days(event.match_date, p1o.last_date))
    p2_pre_o = regress_rating(p2o.rating, _gap_days(event.match_date, p2o.last_date))
    k1_o = k_factor(p1o.count_in)
    k2_o = k_factor(p2o.count_in)

    score1 = 1 if event.winner_id == p1 else 0
    post1_o, post2_o = _update(p1_pre_o, k1_o, p2_pre_o, k2_o, score1)
    overall[p1] = _PriorState(post1_o, p1o.count_in + 1, event.match_date)
    overall[p2] = _PriorState(post2_o, p2o.count_in + 1, event.match_date)

    snap1 = SnapshotRow(
        player_id=p1,
        match_id=event.match_id,
        match_date=event.match_date,
        match_num=event.match_num,
        surface=surface,
        pre_elo=float(p1_pre_o),
        post_elo=float(post1_o),
        prior_overall_matches=int(p1o.count_in),
        k_overall=float(k1_o),
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
        prior_overall_matches=int(p2o.count_in),
        k_overall=float(k2_o),
        source_hash=source_hash,
    )
    return [snap1, snap2]


def _run(repo: EloRepo) -> EloRunResult:
    # Shared progress watermark (TIMESTAMPTZ) from bronze.etl_state. Elo reads it
    # but never advances it: ETL owns final advancement after every phase succeeds.
    watermark = repo.get_watermark()

    if watermark is not None:
        # Fail closed before any mutation: every source match at/before the shared
        # watermark must already be snapshotted with matching content. A historical
        # insert or change slips in with an old ingested_at (<= watermark) and is
        # caught here, since etl_state stores only a timestamp. Runs against the
        # local snapshot built above (same data, no per-run Postgres scans).
        if repo.count_events_through(watermark) != repo.count_snapshots_through(watermark):
            raise EloHistoryChanged(
                "source match introduced at/before the shared ETL watermark "
                f"{watermark}; run seed --reset before rebuilding"
            )
        if repo.count_mismatched_history(watermark) > 0:
            raise EloHistoryChanged(
                "source match content changed at/before the shared ETL watermark "
                f"{watermark}; run seed --reset before rebuilding"
            )

    # Copy the two Elo source tables into local DuckDB once, then run every read
    # (validation, selection, prior-state) against that local snapshot so the
    # rating loop never touches Postgres.
    print("ELO SNAPSHOT: generating snapshot of matches")
    snapshot_started = time.perf_counter()
    events = repo.snapshot_events(watermark)
    print(
        f"ELO SNAPSHOT: captured {len(events)} matches in "
        f"{time.perf_counter() - snapshot_started:.1f}s"
    )

    if not events:
        return EloRunResult(processed=0, snapshots=0, watermark=watermark)

    # Strict causal order: ascending match_date, then match_num, with match_id as
    # the deterministic tie-breaker. match_num is per-tournament, so (match_date,
    # match_num) is NOT globally unique; match_id (the bronze PK) breaks ties
    # deterministically. Sorting here makes the invariant real at the processing
    # layer regardless of how the repository returns rows.
    events.sort(key=lambda e: (e.match_date, e.match_num, e.match_id))

    overall: dict[str, _PriorState] = {}
    player_ids = {event.player1_id for event in events} | {event.player2_id for event in events}
    preloaded = repo.get_prior_overall_many(player_ids)
    # Pre-seed every participant so per-event processing reads only this in-memory
    # map (never the repository). Players with no stored history get cold state.
    overall.update(
        {player_id: _state_from_overall_row(row) for player_id, row in preloaded.items()}
    )
    for player_id in player_ids:
        overall.setdefault(player_id, _PriorState(ELO_DEFAULT_RATING, 0, None))
    n_batches = (len(events) + ELO_MATCH_BATCH_SIZE - 1) // ELO_MATCH_BATCH_SIZE
    rating_started = time.perf_counter()
    with tqdm(total=len(events), unit="match", desc="ELO RATING") as bar:
        for batch_idx, start in enumerate(range(0, len(events), ELO_MATCH_BATCH_SIZE), start=1):
            batch_started = time.perf_counter()
            batch = events[start : start + ELO_MATCH_BATCH_SIZE]
            repo.begin()
            try:
                snapshots: list[SnapshotRow] = []
                for event in batch:
                    snapshots.extend(_process_event(event, overall))
                repo.insert_snapshots(snapshots)
                repo.commit()
            except Exception:
                repo.rollback()
                raise
            bar.update(len(batch))
            bar.set_postfix_str(f"batch {batch_idx}/{n_batches}")
            print(
                f"ELO BATCH: {batch_idx}/{n_batches} ({len(batch)} matches, "
                f"{time.perf_counter() - batch_started:.2f}s)"
            )
    print(f"ELO RATING: rated {len(events)} matches in {time.perf_counter() - rating_started:.1f}s")

    return EloRunResult(processed=len(events), snapshots=len(events) * 2, watermark=watermark)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def materialize_elo(repo: EloRepo | None = None) -> EloRunResult:
    """Materialize Elo snapshots for all unprocessed matches.

    With no ``repo`` the project's PostgreSQL connection is used. Pass a
    repository (e.g. a hermetic fake) to test the logic without a database.

    It preserves existing snapshots and processes only matches without snapshots.

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
    def snapshot_events(self, watermark: datetime | None) -> list[MatchEvent]: ...
    def get_prior_overall_many(
        self, player_ids: set[str]
    ) -> dict[str, tuple[float, int, date]]: ...
    def insert_snapshots(self, rows: list[SnapshotRow]) -> None: ...
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
        self._snapshot_con: duckdb.DuckDBPyConnection | None = None

    def close(self) -> None:
        if self._snapshot_con is not None:
            self._snapshot_con.close()
            self._snapshot_con = None
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
        # Runs against the local DuckDB snapshot (set by snapshot_events), which
        # is an exact copy of bronze.match_events for this run.
        con = self._snapshot_con
        if con is not None:
            row = con.execute(
                "SELECT COUNT(*) FROM bronze_match_events WHERE ingested_at <= ?",
                (watermark,),
            ).fetchone()
            return int(row[0]) if row else 0
        self._cur.execute(
            f"SELECT COUNT(*) FROM {BRONZE_MATCHES_TABLE} WHERE ingested_at <= %s",
            (watermark,),
        )
        row = self._cur.fetchone()
        if row is None:
            return 0
        return int(row[0])

    def count_snapshots_through(self, watermark: datetime) -> int:
        con = self._snapshot_con
        if con is not None:
            row = con.execute(
                "SELECT COUNT(DISTINCT match_id) FROM silver_elo_snapshots "
                "WHERE match_id IN (SELECT match_id FROM bronze_match_events "
                "WHERE ingested_at <= ?)",
                (watermark,),
            ).fetchone()
            return int(row[0]) if row else 0
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
        con = self._snapshot_con
        if con is not None:
            # Compute full source hashes locally from the snapshot and compare to
            # the stored source_hash for every snapped match through the watermark.
            sources = con.execute(
                "SELECT match_id, match_date, match_num, surface, winner_id, "
                "player1_id, player2_id FROM bronze_match_events WHERE ingested_at <= ?",
                (watermark,),
            ).fetchall()
            if not sources:
                return 0
            ids = [r[0] for r in sources]
            stored_rows = con.execute(
                "SELECT DISTINCT match_id, source_hash FROM silver_elo_snapshots "
                "WHERE match_id IN (SELECT UNNEST(?))",
                (ids,),
            ).fetchall()
            snap_stored: dict[str, set[str | None]] = {}
            for match_id, source_hash in stored_rows:
                snap_stored.setdefault(str(match_id), set()).add(source_hash)
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
                hashes = snap_stored.get(event.match_id)
                if not hashes or elo_source_hash(event) not in hashes:
                    mismatches += 1
            return mismatches
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
        stored: dict[str, set[str | None]] = {}
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

    def snapshot_events(self, watermark: datetime | None) -> list[MatchEvent]:
        """Copy the Elo work set and required prior state to DuckDB."""
        if watermark is not None:
            self._cur.execute(
                f"""
                SELECT EXISTS (
                    SELECT 1
                    FROM {BRONZE_MATCHES_TABLE} m
                    WHERE m.ingested_at > %s
                      AND NOT EXISTS (
                          SELECT 1
                          FROM {SILVER_ELO_SNAPSHOTS} e
                          WHERE e.match_id = m.match_id
                      )
                )
                """,
                (watermark,),
            )
            row = self._cur.fetchone()
            if row is None or not row[0]:
                return []

        ELO_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = ELO_SNAPSHOT_PATH.with_name(f".{ELO_SNAPSHOT_PATH.name}.{os.getpid()}.tmp")
        scope = "full source" if watermark is None else "incremental work set"
        print(f"ELO SNAPSHOT: writing {scope} to {ELO_SNAPSHOT_PATH}")
        con = duckdb.connect(str(tmp_path))
        try:
            pg_url = get_database_url().replace("'", "''")
            con.execute(f"ATTACH '{pg_url}' AS pg (TYPE postgres)")
            con.execute("BEGIN TRANSACTION")
            if watermark is None:
                con.execute(
                    'CREATE TABLE bronze_match_events AS SELECT * FROM pg."bronze"."match_events"'
                )
                con.execute(
                    'CREATE TABLE silver_elo_snapshots AS SELECT * FROM pg."silver"."elo_snapshots"'
                )
            else:
                con.execute(
                    "CREATE TABLE bronze_match_events AS "
                    'SELECT * FROM pg."bronze"."match_events" m '
                    "WHERE m.ingested_at > ? "
                    "AND NOT EXISTS ( "
                    'SELECT 1 FROM pg."silver"."elo_snapshots" e '
                    "WHERE e.match_id = m.match_id "
                    ")",
                    (watermark,),
                )
                con.execute(
                    "CREATE TABLE silver_elo_snapshots AS "
                    'SELECT e.* FROM pg."silver"."elo_snapshots" e '
                    "WHERE e.player_id IN ( "
                    "SELECT player1_id FROM bronze_match_events "
                    "UNION "
                    "SELECT player2_id FROM bronze_match_events "
                    ")"
                )
            con.execute("COMMIT")
            if self._snapshot_con is not None:
                self._snapshot_con.close()
            con.close()
            con = None
            os.replace(tmp_path, ELO_SNAPSHOT_PATH)
            self._snapshot_con = duckdb.connect(str(ELO_SNAPSHOT_PATH), read_only=True)
            rows = self._snapshot_con.execute(
                """
                SELECT m.match_id, m.match_date, m.match_num, m.surface, m.winner_id,
                       m.player1_id, m.player2_id, m.ingested_at
                FROM bronze_match_events m
                WHERE (? IS NULL OR m.ingested_at > ?)
                  AND NOT EXISTS (
                      SELECT 1 FROM silver_elo_snapshots e
                      WHERE e.match_id = m.match_id
                  )
                ORDER BY m.match_date, m.match_num, m.match_id
                """,
                (watermark, watermark),
            ).fetchall()
        finally:
            if con is not None:
                con.close()
            if tmp_path.exists():
                tmp_path.unlink()
        return [
            MatchEvent(
                match_id=row[0],
                match_date=row[1],
                match_num=int(row[2]),
                surface=row[3],
                winner_id=row[4],
                player1_id=row[5],
                player2_id=row[6],
                ingested_at=row[7],
            )
            for row in rows
        ]

    def get_prior_overall_many(self, player_ids: set[str]) -> dict[str, tuple[float, int, date]]:
        if not player_ids:
            return {}
        if self._snapshot_con is not None:
            rows = self._snapshot_con.execute(
                """
                SELECT DISTINCT ON (player_id) player_id, post_elo,
                       prior_overall_matches, match_date
                FROM silver_elo_snapshots
                WHERE player_id IN (SELECT UNNEST(?))
                ORDER BY player_id, match_date DESC, match_num DESC, match_id DESC
                """,
                (list(player_ids),),
            ).fetchall()
            return {row[0]: (float(row[1]), int(row[2]), row[3]) for row in rows}
        self._cur.execute(
            f"SELECT DISTINCT ON (player_id) player_id, post_elo, prior_overall_matches, match_date "
            f"FROM {SILVER_ELO_SNAPSHOTS} WHERE player_id = ANY(%s) "
            f"ORDER BY player_id, match_date DESC, match_num DESC, match_id DESC",
            (list(player_ids),),
        )
        return {row[0]: (float(row[1]), int(row[2]), row[3]) for row in self._cur.fetchall()}

    def insert_snapshots(self, rows: list[SnapshotRow]) -> None:
        # COPY streams the whole batch in one protocol operation (fastest bulk
        # insert for a flat table like this); the PK rejects any duplicate
        # (player_id, match_id) before a transaction commit.
        with (
            self._conn.cursor() as cur,
            cur.copy(
                f"COPY {SILVER_ELO_SNAPSHOTS} "
                f"(player_id, match_id, match_date, match_num, surface, "
                f" pre_elo, post_elo, prior_overall_matches, k_overall, source_hash) "
                f"FROM STDIN"
            ) as copy,
        ):
            for row in rows:
                copy.write_row(
                    (
                        row.player_id,
                        row.match_id,
                        row.match_date,
                        row.match_num,
                        row.surface,
                        row.pre_elo,
                        row.post_elo,
                        row.prior_overall_matches,
                        row.k_overall,
                        row.source_hash,
                    )
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
