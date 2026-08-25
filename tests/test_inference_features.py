"""Hermetic inference-builder tests using an in-memory DuckDB fixture."""

import math
import re
from datetime import date, timedelta

import duckdb
import pandas as pd
import pytest

from src.constants import SILVER_ROLLING_FEATURES
from src.features import inference
from src.features.columns import (
    CONTEXT_COLS,
    DIFF_COLS,
    FEATURE_COLS,
    TOUR_AVERAGES_FALLBACK_COLS,
)
from src.features.elo import regress_rating
from src.features.inference import (
    build_inference_features,
    build_inference_features_bulk,
)

# All fixture matches precede this fixed as-of date.
AS_OF_AFTER_ALL_MATCHES = date(2026, 9, 1)

_DB: duckdb.DuckDBPyConnection | None = None

# Rewrite PostgreSQL's multi-argument unnest for DuckDB without changing query semantics.
_BULK_UNNEST_RE = re.compile(
    r"unnest\((?P<args>(?:\?::\w+\[\])(?:, \?::\w+\[\])*)\) AS (?P<alias>\w+)\((?P<names>[^)]+)\)"
)


def _translate_unnest(sql: str) -> str:
    def repl(match: re.Match[str]) -> str:
        columns = ", ".join(
            f"UNNEST({arg.upper()}) AS {name.strip()}"
            for arg, name in zip(
                match.group("args").split(", "),
                match.group("names").split(","),
                strict=True,
            )
        )
        return f"(SELECT {columns}) AS {match.group('alias')}"

    return _BULK_UNNEST_RE.sub(repl, sql)


def execute_df(sql: str, params: list[object] | None = None) -> pd.DataFrame:
    """Test stand-in for src.db.client.execute_df over the in-memory DuckDB."""
    assert _DB is not None, "the _duck_db_backed fixture must be active"
    return _DB.execute(_translate_unnest(sql.replace("%s", "?")), params or []).df()


# ── Fixture data ─────────────────────────────────────────────────────────────
# Mirrors the seeded players, snapshots, meeting, and tour-average singleton used below.

_S0AG = (
    2.0,
    11500.0,
    24.43,
    0.8,
    0.2,
    0.7,
    0.6,
    0.7,
    0.55,
    0.63,
    0.42,
    1.13513,
    0.05,
    0.4,
    3,
    3.0,
    20.0,
    0.8,
    0.8,
    0.8,
)
_Z355 = (
    4.0,
    4555.0,
    28.85,
    0.5,
    0.1,
    0.6,
    0.5,
    0.6,
    0.5,
    0.55,
    0.4,
    0.88888,
    0.03,
    0.3,
    1,
    5.0,
    25.0,
    0.6,
    0.6,
    0.6,
)
_MINOR = (
    1.0,
    12050.0,
    22.7,
    0.5,
    0.15,
    0.65,
    0.55,
    0.62,
    0.5,
    0.58,
    0.42,
    1.0,
    0.04,
    0.35,
    2,
    2.0,
    30.0,
    0.55,
    0.55,
    0.6,
)

_SNAP_COLS = (
    "latest_player_ranking",
    "latest_player_rank_points",
    "latest_player_age",
    "win_rate_10",
    "ace_rate_10",
    "first_serve_pct_10",
    "break_points_saved_pct_10",
    "first_serve_win_pct_10",
    "second_serve_win_pct_10",
    "serve_win_pct_10",
    "return_points_won_pct_10",
    "dominance",
    "df_rate_10",
    "aces_per_svc_game_10",
    "streak",
    "avg_player_rank_10",
    "avg_rank_faced_10",
    "clay_win_rate_10",
    "grass_win_rate_10",
    "hard_win_rate_10",
    "matches_10",
)


def _snap_rows() -> list[tuple[object, ...]]:
    """One snapshot per seeded player match, stats constant per player."""
    rows: list[tuple[object, ...]] = []
    dates: list[tuple[str, str, int]] = [
        ("S0AG", "s1", 1),
        ("S0AG", "s2", 2),
        ("S0AG", "s3", 3),
        ("S0AG", "s4", 4),
        ("S0AG", "s5", 5),
        ("S0AG", "s6", 6),
        ("Z355", "z1", 1),
        ("Z355", "z2", 2),
        ("Z355", "z3", 3),
        ("Z355", "z4", 4),
        ("A0E2", "a1", 1),
        ("F0FV", "f1", 1),
    ]
    snap_dates = {
        "s1": date(2026, 1, 20),
        "s2": date(2026, 2, 16),
        "s3": date(2026, 2, 18),
        "s4": date(2026, 3, 7),
        "s5": date(2026, 5, 20),
        "s6": date(2026, 7, 12),
        "z1": date(2026, 1, 4),
        "z2": date(2026, 1, 18),
        "z3": date(2026, 2, 25),
        "z4": date(2026, 7, 12),
        "a1": date(2026, 2, 1),
        "f1": date(2026, 4, 1),
    }
    for pid, match_id, num in dates:
        stats = _S0AG if pid == "S0AG" else _Z355 if pid == "Z355" else _MINOR
        rows.append((pid, match_id, snap_dates[match_id], num, "hard", *stats, min(num, 10)))
    return rows


# One Elo snapshot per relevant player at the parity match's completion; stats
# are constant. S0AG/Z355 share the pm-s6 same-day completed match; ELO_R is a
# single stale (early-year) match used to exercise inactivity regression.
_ELO_COLS = (
    "player_id",
    "match_id",
    "match_date",
    "match_num",
    "surface",
    "pre_elo",
    "post_elo",
    "prior_overall_matches",
    "k_overall",
    "source_hash",
)


def _elo_rows() -> list[tuple[object, ...]]:
    return [
        (
            "S0AG",
            "pm-s6",
            date(2026, 7, 12),
            6,
            "hard",
            0.0,
            1625.0,
            5,
            0.0,
            "hS0AG",
        ),
        (
            "Z355",
            "pm-s6",
            date(2026, 7, 12),
            6,
            "hard",
            0.0,
            1575.0,
            4,
            0.0,
            "hZ355",
        ),
        (
            "ELO_R",
            "elo-r1",
            date(2026, 1, 1),
            1,
            "hard",
            0.0,
            2000.0,
            0,
            0.0,
            "hR",
        ),
    ]


# Independent hand-computed expectation for both perspectives of the parity match.
_PARITY_GOLD = (
    # Match metadata precedes FEATURE_COLS.
    "pm-s6",
    date(2026, 7, 12),
    "S0AG",
    "Z355",
    "hard",
    # 20 matchup diffs (S0AG side minus Z355)
    -2.0,
    6945.0,
    -4.42,
    0.3,
    0.1,
    0.1,
    0.1,
    0.1,
    0.05,
    0.08,
    0.02,
    0.24625,
    0.02,
    0.1,
    0.0,
    -5.0,
    2.0,
    0.2,  # surface_form_diff (hard): 0.8 - 0.6
    0.0,  # days_since_last_match_diff: S0AG/Z355 share a same-day snapshot (0 rest days)
    50.0,  # elo_diff: 1625 - 1575 (latest completed overall Elo through as_of)
    0.0,  # player_elo_gradient_10: only a same-day completed Elo (excluded by strict <)
    0.0,  # opponent_elo_gradient_10
    # 6 absolute state values (matches_10 exposure pair)
    6.0,  # player_matches_10 (S0AG s6, inclusive as-of)
    4.0,  # opponent_matches_10 (Z355 z4, inclusive as-of)
    0.0,
    0.0,
    8.0,
    13.0,
    # 3 pair-level head-to-head (strictly before as_of excludes the match itself)
    0.0,
    (0 + 1) / (0 + 2) - 0.5,  # h2h_advantage: no prior meetings -> 0.0
    (0 + 1) / (0 + 2) - 0.5,  # h2h_surface_advantage: 0.0
    # 7 context values (is_clay, is_grass, is_hard, is_indoor, best_of,
    # tournament_level, round_encoded)
    0.0,
    0.0,
    1.0,
    0.0,
    3.0,  # best_of: gold COALESCE(best_of, 3) default
    0.0,
    0.0,
)

_PARITY_GOLD_BA = (
    # match_id, match_date, player_id, opponent_id, surface
    "pm-s6",
    date(2026, 7, 12),
    "Z355",
    "S0AG",
    "hard",
    # 20 matchup diffs (Z355 side minus S0AG = negated)
    2.0,
    -6945.0,
    4.42,
    -0.3,
    -0.1,
    -0.1,
    -0.1,
    -0.1,
    -0.05,
    -0.08,
    -0.02,
    -0.24625,  # dominance_diff (lifetime): 0.88888 - 1.13513
    -0.02,
    -0.1,
    0.0,
    5.0,
    -2.0,
    -0.2,  # surface_form_diff (hard): 0.6 - 0.8
    0.0,  # days_since_last_match_diff: Z355/S0AG share a same-day snapshot (0 rest days)
    -50.0,  # elo_diff: 1575 - 1625
    0.0,  # player_elo_gradient_10: Z355 (same-day Elo excluded)
    0.0,  # opponent_elo_gradient_10: S0AG (same-day Elo excluded)
    # 6 absolute state values (Z355 first; matches_10 pair exchanged)
    4.0,  # player_matches_10 (Z355 z4, inclusive as-of)
    6.0,  # opponent_matches_10 (S0AG s6, inclusive as-of)
    0.0,
    0.0,
    13.0,
    8.0,
    # 3 pair-level head-to-head (strictly before as_of excludes the match itself)
    0.0,
    (0 + 1) / (0 + 2) - 0.5,  # h2h_advantage: no prior meetings -> 0.0
    (0 + 1) / (0 + 2) - 0.5,  # h2h_surface_advantage: 0.0
    # 7 context values
    0.0,
    0.0,
    1.0,
    0.0,
    3.0,  # best_of: gold COALESCE(best_of, 3) default
    0.0,
    0.0,
)


def _seed(con: duckdb.DuckDBPyConnection) -> None:
    """Create the PostgreSQL-shaped tables and insert the fixture data."""
    con.execute("CREATE SCHEMA silver")
    con.execute("CREATE SCHEMA bronze")
    con.execute("CREATE SCHEMA gold")

    snap_ddl = ", ".join(f'"{c}" DOUBLE' for c in _SNAP_COLS)
    con.execute(
        f"""
        CREATE TABLE silver.rolling_features (
            player_id VARCHAR, match_id VARCHAR, snapshot_date DATE,
            player_match_number INTEGER, surface VARCHAR, {snap_ddl}
        )
        """
    )
    con.execute(
        """
        CREATE TABLE bronze.player_profiles (
            player_id VARCHAR, height DOUBLE, handedness VARCHAR, turned_pro INTEGER
        )
        """
    )
    # Bronze holds one row per physical match; the H2H queries select on
    # match_id/winner_id/surface and filter on player1_id/player2_id/match_date.
    con.execute(
        """
        CREATE TABLE bronze.match_events (
            match_id VARCHAR, match_date DATE,
            player1_id VARCHAR, player2_id VARCHAR, winner_id VARCHAR,
            surface VARCHAR, match_num INTEGER
        )
        """
    )
    fallback_ddl = ", ".join(f'"{c}" DOUBLE' for c in TOUR_AVERAGES_FALLBACK_COLS)
    con.execute(
        f"""
        CREATE TABLE gold.tour_averages (
            singleton_id INTEGER, pool_as_of_date DATE,
            snapshot_pool_rows INTEGER, snapshot_pool_players INTEGER,
            profile_rows INTEGER, player_match_rows INTEGER, {fallback_ddl}
        )
        """
    )
    gold_cols = ", ".join(f'"{c}" DOUBLE' for c in FEATURE_COLS)
    con.execute(
        f"""
        CREATE TABLE gold.match_features (
            match_id VARCHAR, match_date DATE, player_id VARCHAR, opponent_id VARCHAR,
            surface VARCHAR, {gold_cols}
        )
        """
    )

    con.executemany(
        f"INSERT INTO silver.rolling_features VALUES ({', '.join(['?'] * 26)})",
        _snap_rows(),
    )
    # Mirror the production silver.elo_snapshots table the inference Elo SQL
    # reads; post_elo drives the as-of Elo ratings.
    con.execute(
        """
        CREATE TABLE silver.elo_snapshots (
            player_id VARCHAR, match_id VARCHAR, match_date DATE, match_num INTEGER,
            surface VARCHAR, pre_elo DOUBLE, post_elo DOUBLE,
            prior_overall_matches INTEGER, k_overall DOUBLE, source_hash VARCHAR
        )
        """
    )
    con.executemany(
        f"INSERT INTO silver.elo_snapshots VALUES ({', '.join(['?'] * len(_ELO_COLS))})",
        _elo_rows(),
    )
    # Mirror the production silver.player_matches table the inference SQL joins
    # on (player_id, match_id); match_num drives the latest-snapshot ordering.
    con.execute(
        """
        CREATE TABLE silver.player_matches (
            player_id VARCHAR, match_id VARCHAR, match_num INTEGER
        )
        """
    )
    con.executemany(
        "INSERT INTO silver.player_matches VALUES (?, ?, ?)",
        [(r[0], r[1], r[3]) for r in _snap_rows()],
    )
    # Seed one bronze-style hard-court meeting; other matches are not H2H fixtures.
    con.executemany(
        "INSERT INTO bronze.match_events VALUES (?, ?, ?, ?, ?, ?, ?)",
        [("pm-s6", date(2026, 7, 12), "S0AG", "Z355", "S0AG", "hard", 1)],
    )
    con.executemany(
        "INSERT INTO bronze.player_profiles VALUES (?, ?, ?, ?)",
        [
            ("S0AG", 191.0, "R", 2018),
            ("Z355", 198.0, "R", 2013),
        ],
    )
    # Pool aggregates over the 12 snapshots: hard_win_rate_10 -> 0.7,
    # win_rate_10 -> 0.65, median streak -> 2.5.
    singleton = {
        "singleton_id": 1,
        "pool_as_of_date": date(2026, 8, 9),
        "snapshot_pool_rows": 12,
        "snapshot_pool_players": 4,
        "profile_rows": 2,
        "player_match_rows": 13,
        "latest_player_ranking": 25.0,
        "latest_player_rank_points": 1000.0,
        "latest_player_age": 26.0,
        "streak": 2.5,
        "win_rate_10": 0.65,
        "ace_rate_10": 0.15,
        "first_serve_pct_10": 0.65,
        "break_points_saved_pct_10": 0.55,
        "first_serve_win_pct_10": 0.62,
        "second_serve_win_pct_10": 0.5,
        "serve_win_pct_10": 0.58,
        "return_points_won_pct_10": 0.42,
        "dominance": 1.0,  # pool lifetime dominance mean: TRUNC(0.42 / (1 - 0.58), 5)
        "df_rate_10": 0.04,
        "aces_per_svc_game_10": 0.35,
        "avg_player_rank_10": 15.0,
        "avg_rank_faced_10": 30.0,
        "clay_win_rate_10": 0.55,
        "grass_win_rate_10": 0.5,
        "hard_win_rate_10": 0.7,
        "matches_30d_default": 3.0,
        "rate_default": 0.5,
        "left_handed_rate": 0.12,
        "avg_years_pro": 10.0,
    }
    con.execute(
        f"INSERT INTO gold.tour_averages VALUES ({', '.join(['?'] * len(singleton))})",
        list(singleton.values()),
    )
    con.executemany(
        f"INSERT INTO gold.match_features VALUES ({', '.join(['?'] * (len(FEATURE_COLS) + 5))})",
        [_PARITY_GOLD, _PARITY_GOLD_BA],
    )


@pytest.fixture(autouse=True)
def _duck_db_backed(monkeypatch):
    """Route every DB call made by the module to a fresh in-memory DuckDB."""
    global _DB
    con = duckdb.connect()
    try:
        _seed(con)
        _DB = con
        monkeypatch.setattr("src.features.inference.execute_df", execute_df)
        monkeypatch.setattr("src.features.tour_averages.execute_df", execute_df)
        yield
    finally:
        _DB = None
        con.close()


def _insert_prior_meetings(
    pair_a: str, pair_b: str, meetings: list[tuple[str, str, int, str]]
) -> None:
    """Insert meetings as bronze rows with the winner on player1_id."""
    rows = []
    for match_id, date_iso, a_won, surface in meetings:
        winner, loser = (pair_a, pair_b) if a_won else (pair_b, pair_a)
        rows.append((match_id, date.fromisoformat(date_iso), winner, loser, winner, surface))
    assert _DB is not None
    _DB.executemany(
        "INSERT INTO bronze.match_events "
        "(match_id, match_date, player1_id, player2_id, winner_id, surface) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )


def _assert_mirror(row_ab, row_ba):
    """Assert the mirror relationship for swapped player ids."""
    # Ids swap to requested order.
    assert row_ab["player_id"] == row_ba["opponent_id"]
    assert row_ab["opponent_id"] == row_ba["player_id"]
    # Signed diffs negate.
    for col in DIFF_COLS:
        assert row_ab[col] == -row_ba[col], col
    # h2h advantages negate (within float tolerance; division rounding differs
    # by ~1 ulp per orientation); exposure is invariant.
    assert row_ab["h2h_advantage"] == pytest.approx(-row_ba["h2h_advantage"])
    assert row_ab["h2h_surface_advantage"] == pytest.approx(-row_ba["h2h_surface_advantage"])
    assert row_ab["h2h_exposure"] == row_ba["h2h_exposure"]
    # Paired features exchange.
    for pc in [c for c in FEATURE_COLS if c.startswith("player_")]:
        oc = "opponent_" + pc[len("player_") :]
        assert row_ab[pc] == row_ba[oc], pc
        assert row_ab[oc] == row_ba[pc], oc
    # Invariant context.
    for col in CONTEXT_COLS:
        assert row_ab[col] == row_ba[col], col


def test_output_schema_contract():
    """Exact column order [*FEATURE_COLS, "player_id", "opponent_id"], one row."""
    out = build_inference_features("S0AG", "Z355", "clay", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    assert out.columns.tolist() == [*FEATURE_COLS, "player_id", "opponent_id"]
    assert len(out.columns) == len(FEATURE_COLS) + 2  # features + 2 ids
    assert len(out) == 1
    assert out["player_id"].dtype == object
    assert out["opponent_id"].dtype == object


@pytest.mark.parametrize("surface", ["clay", "grass", "hard", "carpet"])
def test_two_known_players_each_surface(surface):
    """Known-vs-known row: valid one-hot, requested ids, finite features."""
    out = build_inference_features("S0AG", "Z355", surface, as_of_date=AS_OF_AFTER_ALL_MATCHES)
    row = out.iloc[0]
    assert row["player_id"] == "S0AG"  # requested order preserved
    assert row["opponent_id"] == "Z355"
    expected_one_hots = {"is_clay": 0, "is_grass": 0, "is_hard": 0}
    if surface != "carpet":
        expected_one_hots[f"is_{surface}"] = 1
    assert row["is_clay"] == expected_one_hots["is_clay"]
    assert row["is_grass"] == expected_one_hots["is_grass"]
    assert row["is_hard"] == expected_one_hots["is_hard"]
    assert "is_carpet" not in out.columns  # carpet no longer has a model feature
    for col in FEATURE_COLS:
        assert math.isfinite(row[col]), f"{col} is not finite: {row[col]!r}"
    assert row["tournament_level"] == 0
    assert row["round_encoded"] == 0


@pytest.mark.parametrize("best_of", [1, 3, 5])
def test_best_of_scalar_bulk_parity(best_of):
    """Scalar and bulk builders emit byte-identical rows for each best_of value."""
    req = {
        "player_id": "S0AG",
        "opponent_id": "Z355",
        "surface": "hard",
        "as_of_date": AS_OF_AFTER_ALL_MATCHES,
        "best_of": best_of,
    }
    scalar = build_inference_features(**req)
    bulk = build_inference_features_bulk([req])
    pd.testing.assert_frame_equal(bulk, scalar, check_exact=True)
    assert scalar.iloc[0]["best_of"] == best_of


def test_reversed_ids_mirror():
    """Swapping ids produces the complementary feature row."""
    out_ab = build_inference_features("S0AG", "Z355", "hard", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    out_ba = build_inference_features("Z355", "S0AG", "hard", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    row_ab, row_ba = out_ab.iloc[0], out_ba.iloc[0]
    _assert_mirror(row_ab, row_ba)
    # No NaN/inf anywhere in either orientation.
    for out in (out_ab, out_ba):
        assert not out[FEATURE_COLS].isnull().to_numpy().any()
        for col in FEATURE_COLS:
            assert math.isfinite(out.iloc[0][col]), f"{col} not finite"


def test_known_players_profile_features():
    """Known profile features use requested order and as-of years_pro."""
    out = build_inference_features("S0AG", "Z355", "clay", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    row = out.iloc[0]
    assert row["player_is_left_handed"] == 0.0
    assert row["opponent_is_left_handed"] == 0.0
    assert row["player_years_pro"] == 8.0  # 2026 - 2018
    assert row["opponent_years_pro"] == 13.0  # 2026 - 2013
    assert "player_height" not in out.columns
    assert "height_diff" not in out.columns


def test_historical_as_of_excludes_later_snapshots():
    """Use the newest snapshot strictly before the as-of date."""
    out = build_inference_features("S0AG", "Z355", "hard", as_of_date=date(2026, 6, 30))
    # Cross-check the expected snapshot directly in the fixture table.
    snapshot = execute_df(
        f"SELECT player_match_number, win_rate_10 "
        f"FROM {SILVER_ROLLING_FEATURES} "
        "WHERE player_id = %s AND snapshot_date < %s::date "
        "ORDER BY player_match_number DESC LIMIT 1",
        ["S0AG", "2026-06-30"],
    ).iloc[0]
    assert snapshot["player_match_number"] == 5
    assert float(snapshot["win_rate_10"]) == 0.8
    # The inference row's per-side observed-matches equals that snapshot's window.
    assert out.loc[0, "player_matches_10"] == pytest.approx(5.0)


@pytest.mark.parametrize(
    "args",
    [("S0AG", "UNKNOWN_ID"), ("UNKNOWN_ID", "S0AG")],
    ids=["known-first", "reversed"],
)
def test_one_missing_player_imputed_no_nans(args):
    """One unknown player is pool-imputed; ids keep the requested order."""
    player_id, opponent_id = args
    out = build_inference_features(
        player_id, opponent_id, "hard", as_of_date=AS_OF_AFTER_ALL_MATCHES
    )
    row = out.iloc[0]
    assert row["player_id"] == player_id  # requested order, not sorted
    assert row["opponent_id"] == opponent_id
    for col in FEATURE_COLS:
        assert math.isfinite(row[col]), f"{col} is not finite: {row[col]!r}"
    # The known player's form differs from the pool default (diff exists).
    assert row["form_diff"] != 0
    assert row["streak_diff"] != 0  # known streak vs pool-mean streak differ
    # The unknown side is a cold start (no snapshot): zero observed matches, while
    # the known side carries its own window. matches_10 is the clearest signal.
    unknown_prefix = "player" if player_id == "UNKNOWN_ID" else "opponent"
    known_prefix = "opponent" if unknown_prefix == "player" else "player"
    assert row[f"{unknown_prefix}_matches_10"] == 0
    assert row[f"{known_prefix}_matches_10"] > 0
    # Unknown profile features come from the finite, bounded tour-average singleton.
    assert 0.0 <= row[f"{unknown_prefix}_is_left_handed"] <= 1.0, (
        f"left_handed_rate out of bounds: {row[f'{unknown_prefix}_is_left_handed']}"
    )
    assert 0.0 <= row[f"{unknown_prefix}_years_pro"] <= 50.0, (
        f"avg_years_pro out of bounds: {row[f'{unknown_prefix}_years_pro']}"
    )
    assert math.isfinite(row[f"{known_prefix}_years_pro"])


def test_both_unknowns_neutral_diffs():
    """Two unknown players receive identical defaults and neutral diffs."""
    out = build_inference_features("A0ZZ", "ZZ99", "hard", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    row = out.iloc[0]
    assert row["player_id"] == "A0ZZ"
    assert row["opponent_id"] == "ZZ99"
    for col in FEATURE_COLS:
        assert math.isfinite(row[col]), f"{col} is not finite: {row[col]!r}"
    for col in DIFF_COLS:
        assert row[col] == 0, f"{col} should be neutral for two unknowns: {row[col]!r}"
    assert [row["is_clay"], row["is_grass"], row["is_hard"]] == [0, 0, 1]


@pytest.mark.parametrize("surface", ["Clay", "", 0])
def test_invalid_surface_raises(surface):
    with pytest.raises((ValueError, TypeError)):
        build_inference_features("S0AG", "Z355", surface, as_of_date=AS_OF_AFTER_ALL_MATCHES)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tournament_level": 4, "tournament": "grand_slam"},
        {"round_encoded": 7, "round": "f"},
    ],
)
def test_tournament_string_and_int_conflict_raises(kwargs):
    """Passing both the int and the string alias for one context feature is ambiguous."""
    with pytest.raises(ValueError):
        build_inference_features(
            "S0AG",
            "Z355",
            "hard",
            as_of_date=AS_OF_AFTER_ALL_MATCHES,
            **kwargs,
        )


@pytest.mark.parametrize("kwargs", [{"tournament": 4}, {"round": 7}])
def test_tournament_string_non_string_raises(kwargs):
    """The string aliases reject non-string values instead of silently mapping to 0."""
    with pytest.raises(TypeError):
        build_inference_features(
            "S0AG",
            "Z355",
            "hard",
            as_of_date=AS_OF_AFTER_ALL_MATCHES,
            **kwargs,
        )


def test_null_handedness_falls_back_to_pool_rate(monkeypatch):
    """A profile with NULL handedness uses the pool left-handed rate, not a
    silent hardcoded 0 (parity with match_features.sql, which keeps non-L/R
    handedness NULL for train-time imputation)."""
    profile = pd.DataFrame(
        [
            {
                "player_id": "S0AG",
                "height": 191.0,
                "handedness": None,
                "turned_pro": 2018,
            }
        ]
    )
    monkeypatch.setattr("src.features.inference.execute_df", lambda _sql, _params=None: profile)
    agg = {"left_handed_rate": 0.08, "avg_years_pro": 8.0}
    values = inference._profile_values("S0AG", date(2026, 9, 1), agg)
    assert values["is_left_handed"] == pytest.approx(0.08)
    # Non-NULL cells are still read directly.
    assert values["years_pro"] == 8.0  # 2026 - 2018


def test_h2h_first_and_second_meeting_boundaries():
    """Same-day meetings are excluded; strictly-before priors accumulate."""
    a, b = "H2H_F", "H2H_G"
    _insert_prior_meetings(a, b, [("h2h-f1", "2026-05-20", 1, "clay")])
    # The day before the first meeting: no priors yet.
    out1 = build_inference_features(a, b, "clay", as_of_date=date(2026, 5, 19))
    row1 = out1.iloc[0]
    assert row1["h2h_exposure"] == 0
    assert row1["h2h_advantage"] == 0.0
    assert row1["h2h_surface_advantage"] == 0.0
    # A second meeting on 2026-06-20.
    _insert_prior_meetings(a, b, [("h2h-f2", "2026-06-20", 0, "clay")])
    # Same-day as the second meeting: it is excluded (strictly before as_of).
    out_same_day = build_inference_features(a, b, "clay", as_of_date=date(2026, 6, 20))
    row_sd = out_same_day.iloc[0]
    assert row_sd["h2h_exposure"] == 1  # only f1; f2 is same-day, excluded
    assert row_sd["h2h_advantage"] == pytest.approx((1 + 1) / (1 + 2) - 0.5)
    assert row_sd["h2h_surface_advantage"] == pytest.approx((1 + 1) / (1 + 2) - 0.5)
    # Strictly after both meetings: both priors count, A won f1, B won f2.
    out_after = build_inference_features(a, b, "clay", as_of_date=date(2026, 6, 21))
    row_after = out_after.iloc[0]
    assert row_after["h2h_exposure"] == 2
    assert row_after["h2h_advantage"] == pytest.approx((1 + 1) / (2 + 2) - 0.5)  # 0.0
    assert row_after["h2h_surface_advantage"] == pytest.approx((1 + 1) / (2 + 2) - 0.5)


def test_h2h_complete_history_uses_every_prior_meeting():
    """Every causally prior meeting contributes; no recency cap drops old results."""
    a, b = "H2H_C", "H2H_D"
    _insert_prior_meetings(
        a,
        b,
        [
            ("h2h-r1", "2026-01-05", 1, "hard"),  # oldest; A's only win, still counted
            ("h2h-r2", "2026-02-05", 0, "hard"),
            ("h2h-r3", "2026-03-05", 0, "hard"),
            ("h2h-r4", "2026-04-05", 0, "hard"),
            ("h2h-r5", "2026-05-05", 0, "hard"),
            ("h2h-r6", "2026-06-05", 0, "hard"),
        ],
    )
    out = build_inference_features(a, b, "hard", as_of_date=date(2026, 7, 1))
    row = out.iloc[0]
    # All six prior meetings count (no five-meeting cap).
    assert row["h2h_exposure"] == 6
    # A won exactly one of the six: advantage uses the complete history.
    assert row["h2h_advantage"] == pytest.approx((1 + 1) / (6 + 2) - 0.5)
    # All six window meetings are hard: surface advantage matches the overall.
    assert row["h2h_surface_advantage"] == pytest.approx((1 + 1) / (6 + 2) - 0.5)
    # Reversing sides preserves exposure and negates both advantages.
    row_ba = build_inference_features(b, a, "hard", as_of_date=date(2026, 7, 1))
    _assert_mirror(row, row_ba.iloc[0])


def test_h2h_surface_advantage_filters_complete_history():
    """Surface H2H advantage uses every prior meeting on the matching surface."""
    a, b = "H2H_L", "H2H_M"
    _insert_prior_meetings(
        a,
        b,
        [
            ("h2h-v1", "2026-01-05", 1, "clay"),  # oldest clay win, now counted
            ("h2h-v2", "2026-02-05", 1, "clay"),
            ("h2h-v3", "2026-03-05", 0, "clay"),
            ("h2h-v4", "2026-04-05", 0, "clay"),
            ("h2h-v5", "2026-05-05", 0, "hard"),
            ("h2h-v6", "2026-06-05", 1, "clay"),
        ],
    )
    out = build_inference_features(a, b, "clay", as_of_date=date(2026, 7, 1))
    row = out.iloc[0]
    # All six prior meetings count; A won three of them overall.
    assert row["h2h_exposure"] == 6
    assert row["h2h_advantage"] == pytest.approx((3 + 1) / (6 + 2) - 0.5)  # 0.0
    # Five of the six are clay; A won three clay meetings, including the oldest
    # v1 (no longer dropped by a recency cap), so the surface advantage counts it.
    assert row["h2h_surface_advantage"] == pytest.approx((3 + 1) / (5 + 2) - 0.5)
    # A recency-capped window would have excluded v1 and given a different value.
    assert row["h2h_surface_advantage"] != pytest.approx((2 + 1) / (4 + 2) - 0.5)
    row_ba = build_inference_features(b, a, "clay", as_of_date=date(2026, 7, 1))
    _assert_mirror(row, row_ba.iloc[0])


def test_h2h_complete_history_scalar_bulk_parity():
    """Scalar and bulk builders agree on complete (>5) prior-meeting H2H history."""
    a, b = "H2H_P", "H2H_Q"
    _insert_prior_meetings(
        a,
        b,
        [
            ("h2h-p1", "2026-01-05", 1, "hard"),
            ("h2h-p2", "2026-02-05", 1, "hard"),
            ("h2h-p3", "2026-03-05", 0, "hard"),
            ("h2h-p4", "2026-04-05", 0, "hard"),
            ("h2h-p5", "2026-05-05", 0, "hard"),
            ("h2h-p6", "2026-06-05", 1, "hard"),
            ("h2h-p7", "2026-07-05", 0, "clay"),
        ],
    )
    req = {
        "player_id": a,
        "opponent_id": b,
        "surface": "hard",
        "as_of_date": date(2026, 8, 1),
    }
    scalar = build_inference_features(**req)
    bulk = build_inference_features_bulk([req])
    pd.testing.assert_frame_equal(bulk, scalar, check_exact=True)
    # All seven priors count (six hard, one clay) in exposure and advantage.
    assert scalar.iloc[0]["h2h_exposure"] == 7
    assert scalar.iloc[0]["h2h_advantage"] == pytest.approx((3 + 1) / (7 + 2) - 0.5)
    assert scalar.iloc[0]["h2h_surface_advantage"] == pytest.approx((3 + 1) / (6 + 2) - 0.5)


# ── Train/inference parity ──

_PARITY_MATCH_SQL = """
SELECT mf.*
FROM gold.match_features mf
JOIN LATERAL (
    SELECT * FROM silver.rolling_features rf
    WHERE rf.player_id = mf.player_id
      AND rf.snapshot_date <= mf.match_date
    ORDER BY rf.player_match_number DESC
    LIMIT 1
) prp ON true
JOIN LATERAL (
    SELECT * FROM silver.rolling_features rf
    WHERE rf.player_id = mf.opponent_id
      AND rf.snapshot_date <= mf.match_date
    ORDER BY rf.player_match_number DESC
    LIMIT 1
) pro ON true
WHERE mf.player_id = %s
ORDER BY mf.match_date, mf.match_id
LIMIT 1
"""


def test_train_inference_parity_on_historical_match():
    """Gold and inference rows agree under inclusive snapshot semantics."""
    gold_ab = execute_df(_PARITY_MATCH_SQL, ["S0AG"]).iloc[0]
    out_ab = build_inference_features(
        str(gold_ab["player_id"]),
        str(gold_ab["opponent_id"]),
        str(gold_ab["surface"]),
        as_of_date=gold_ab["match_date"],
        tournament_level=int(gold_ab["tournament_level"]),
        round_encoded=int(gold_ab["round_encoded"]),
        best_of=int(gold_ab["best_of"]),
    )
    infer_ab = out_ab.iloc[0]
    assert out_ab.columns.tolist() == [*FEATURE_COLS, "player_id", "opponent_id"]
    compared = 0
    for col in FEATURE_COLS:
        gold_val = float(gold_ab[col])
        assert infer_ab[col] == pytest.approx(gold_val, abs=1e-6), col
        compared += 1
    # The fixture must actually compare the whole contract, so it cannot
    # silently degenerate to a handful of columns.
    assert compared == len(FEATURE_COLS)

    # The opposite perspective of the SAME physical match must also match its
    # gold row, and the two inference orientations must mirror each other.
    gold_ba = execute_df(_PARITY_MATCH_SQL, ["Z355"]).iloc[0]
    out_ba = build_inference_features(
        str(gold_ba["player_id"]),
        str(gold_ba["opponent_id"]),
        str(gold_ba["surface"]),
        as_of_date=gold_ba["match_date"],
        tournament_level=int(gold_ba["tournament_level"]),
        round_encoded=int(gold_ba["round_encoded"]),
        best_of=int(gold_ba["best_of"]),
    )
    infer_ba = out_ba.iloc[0]
    for col in FEATURE_COLS:
        assert infer_ba[col] == pytest.approx(float(gold_ba[col]), abs=1e-6), col
    _assert_mirror(infer_ab, infer_ba)


def test_new_contract_features_present_and_finite():
    """Contract feature diffs and exposure counts are present and finite."""
    out = build_inference_features("S0AG", "Z355", "hard", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    row = out.iloc[0]
    assert "return_points_won_pct_diff" in out.columns
    assert "player_matches_10" in out.columns
    assert "opponent_matches_10" in out.columns
    for col in (
        "return_points_won_pct_diff",
        "player_matches_10",
        "opponent_matches_10",
    ):
        assert math.isfinite(row[col]), f"{col} not finite: {row[col]!r}"
    # Latest snapshots at this as-of are S0AG s6 / Z355 z4.
    assert row["return_points_won_pct_diff"] == pytest.approx(0.42 - 0.40)
    assert row["player_matches_10"] == 6
    assert row["opponent_matches_10"] == 4


# ── Lifetime dominance uses the materialized unbounded-history ratio ──


def _lifetime_dominance(return_pts_won: float, serve_pts_won: float) -> float:
    """Reproduce the pipeline's (untruncated) lifetime Dominance Ratio formula."""
    return return_pts_won / (1.0 - serve_pts_won)


def test_dominance_diff_formula_and_finite():
    """dominance_diff uses the lifetime Dominance Ratio, not last-10 rates."""
    out = build_inference_features("S0AG", "Z355", "hard", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    row = out.iloc[0]
    expected = _lifetime_dominance(0.42, 0.63) - _lifetime_dominance(0.40, 0.55)
    assert math.isfinite(row["dominance_diff"]), "dominance_diff not finite"
    assert row["dominance_diff"] == pytest.approx(expected, abs=1e-5)
    # The stored snapshot value itself reproduces the formula (cross-check the
    # fixture materialization).
    snap = execute_df(
        "SELECT dominance FROM silver.rolling_features "
        "WHERE player_id = %s AND snapshot_date < %s::date "
        "ORDER BY player_match_number DESC LIMIT 1",
        ["S0AG", "2026-09-01"],
    ).iloc[0]
    assert float(snap["dominance"]) == pytest.approx(_lifetime_dominance(0.42, 0.63), abs=1e-5)


def test_form_diff_is_unweighted_prior_win_rate_diff():
    """form_diff is the simple difference of each side's prior trailing-10 win
    rate (revised form smoothing: no EWMA/weighted blend). It equals the latest
    strictly-before snapshot win_rate_10 gap, not a pooled or averaged value."""
    out = build_inference_features("S0AG", "Z355", "hard", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    row = out.iloc[0]

    def _latest_wr(pid: str) -> float:
        return float(
            execute_df(
                f"SELECT win_rate_10 FROM {SILVER_ROLLING_FEATURES} "
                "WHERE player_id = %s AND snapshot_date < %s::date "
                "ORDER BY player_match_number DESC LIMIT 1",
                [pid, AS_OF_AFTER_ALL_MATCHES.isoformat()],
            ).iloc[0]["win_rate_10"]
        )

    player_wr, opponent_wr = _latest_wr("S0AG"), _latest_wr("Z355")
    # Simple unweighted difference.
    assert row["form_diff"] == pytest.approx(player_wr - opponent_wr)
    # A weighted/pooled blend (near the mean) would be a different value, so the
    # contract explicitly rejects it.
    assert row["form_diff"] != pytest.approx((player_wr + opponent_wr) / 2.0)


def test_inference_gradient_excludes_same_day_post_elo():
    """Leakage guard: the inferred gradient equals the OLS slope over post_elo
    strictly before as_of. A same-day post_elo (e.g. the target match's outcome)
    is excluded, so it cannot enter the gradient history."""
    rows = [
        ("GL", f"gl{i}", date(2026, 1, 1) + timedelta(days=10 * i), i, 1500.0 + 10.0 * i)
        for i in range(8)
    ]
    _insert_elo_rows(rows)
    as_of = date(2026, 6, 1)
    # A same-day post_elo must be ignored by the strict < window.
    _insert_elo_rows([("GL", "gl-leak", as_of, 99, 9999.0)])
    out = build_inference_features("GL", "ZZZH", "hard", as_of_date=as_of)
    # Only the eight strictly-before snapshots (1500..1570) drive the slope of 10.
    assert out.iloc[0]["player_elo_gradient_10"] == pytest.approx(10.0)
    # Bulk path agrees (parity of the leakage guard).
    bulk = build_inference_features_bulk(
        [{"player_id": "GL", "opponent_id": "ZZZH", "surface": "hard", "as_of_date": as_of}]
    )
    assert bulk.iloc[0]["player_elo_gradient_10"] == pytest.approx(10.0)


# ── surface_form_diff and days_since_last_match_diff ──


def test_surface_form_and_days_since_diffs():
    """Surface form and days-since-last-match diffs match the feature contract."""
    # At 2026-07-12 both players have a same-day snapshot (0 rest days), so the
    # days-since diff is 0 - 0 = 0.0 (the 90-day cap leaves 0 unchanged).
    out = build_inference_features("S0AG", "Z355", "hard", as_of_date=date(2026, 7, 12))
    row = out.iloc[0]
    assert row["surface_form_diff"] == pytest.approx(0.8 - 0.6)
    assert row["days_since_last_match_diff"] == pytest.approx(0.0)
    # Surface-specific selection: clay and grass read their own carried rates
    # (same player rates for all surfaces in the fixture, so both give 0.2).
    for surface in ("clay", "grass"):
        diff = build_inference_features("S0AG", "Z355", surface, as_of_date=date(2026, 7, 12))
        assert diff.iloc[0]["surface_form_diff"] == pytest.approx(0.8 - 0.6), surface
    # Carpet has no per-surface rate: both sides use the neutral rate_default.
    carpet = build_inference_features("S0AG", "Z355", "carpet", as_of_date=date(2026, 7, 12))
    assert carpet.iloc[0]["surface_form_diff"] == 0.0
    # Cold start: unknown players fall back to the 90-day cap on both sides, so
    # the days diff stays neutral (LN(1+90) - LN(1+90) = 0).
    cold = build_inference_features("ZZZZ", "YYYY", "hard", as_of_date=date(2026, 9, 1))
    assert cold.iloc[0]["days_since_last_match_diff"] == 0.0


def test_days_since_caps_stale_and_missing_at_90():
    """Missing and stale rest ages cap at 90, not a tour-average fallback."""
    # Cold opponent caps to 90. A stale player (A0E2's only snapshot is
    # 2026-02-01) far in the future also caps to 90, so the diff stays neutral.
    stale = build_inference_features("A0E2", "ZZZZ", "hard", as_of_date=date(2026, 12, 1))
    assert stale.iloc[0]["days_since_last_match_diff"] == pytest.approx(0.0)
    # A fresh opponent keeps its real age, so the diff is
    # LN(1+age) - LN(1+90), which distinguishes the 90 cap from the prior 30-day cap.
    fresh_age = (date(2026, 3, 15) - date(2026, 2, 1)).days  # 42
    fresh = build_inference_features("A0E2", "ZZZZ", "hard", as_of_date=date(2026, 3, 15))
    assert fresh.iloc[0]["days_since_last_match_diff"] == pytest.approx(
        math.log(1.0 + fresh_age) - math.log(1.0 + 90.0)
    )


def test_days_since_diff_boundaries_and_swap_sign():
    """days_since_last_match_diff = LN(1+player) - LN(1+opponent); swap negates it."""
    # S0AG has a same-day snapshot at 2026-07-12 (0 rest days); a missing
    # opponent caps at 90. Player fresh (LN(1+0)) vs missing opponent (LN(1+90)).
    neg = build_inference_features("S0AG", "ZZZZ", "hard", as_of_date=date(2026, 7, 12))
    assert neg.iloc[0]["days_since_last_match_diff"] == pytest.approx(
        math.log(1.0 + 0.0) - math.log(1.0 + 90.0)
    )
    # Swap the players: missing player (LN(1+90)) vs fresh opponent (LN(1+0)).
    pos = build_inference_features("ZZZZ", "S0AG", "hard", as_of_date=date(2026, 7, 12))
    assert pos.iloc[0]["days_since_last_match_diff"] == pytest.approx(
        math.log(1.0 + 90.0) - math.log(1.0 + 0.0)
    )
    # Swapping players negates the diff exactly.
    assert pos.iloc[0]["days_since_last_match_diff"] == pytest.approx(
        -neg.iloc[0]["days_since_last_match_diff"]
    )


def test_is_indoor_1_when_indoor():
    out = build_inference_features(
        "S0AG", "Z355", "hard", is_indoor=1, as_of_date=AS_OF_AFTER_ALL_MATCHES
    )
    assert out["is_indoor"].iloc[0] == 1


# ── As-of Elo inference ──


def test_elo_diff_present_finite_and_mirror():
    """elo_diff is emitted, finite, and mirrors across swap."""
    out_ab = build_inference_features("S0AG", "Z355", "hard", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    row = out_ab.iloc[0]
    assert row["elo_diff"] == pytest.approx(50.0)  # 1625 - 1575
    assert math.isfinite(row["elo_diff"])
    out_ba = build_inference_features("Z355", "S0AG", "hard", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    assert out_ba.iloc[0]["elo_diff"] == pytest.approx(-50.0)


def test_elo_cold_start_zero_diff():
    """Two unknown players both default to 1500 Elo, giving neutral Elo diffs."""
    out = build_inference_features("A0ZZ", "ZZ99", "hard", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    row = out.iloc[0]
    assert row["elo_diff"] == 0
    for col in FEATURE_COLS:
        assert math.isfinite(row[col]), col


def test_elo_regression_from_stale_match():
    """A stale completed Elo is inactive-regressed toward 1500 by as_of_date."""
    as_of = date(2026, 9, 1)
    out = build_inference_features("ELO_R", "ZZZZ", "hard", as_of_date=as_of)
    row = out.iloc[0]
    gap = (as_of - date(2026, 1, 1)).days
    # ZZZZ has no Elo history, so it defaults to 1500; the diff equals the
    # regressed rating minus the 1500 default.
    assert row["elo_diff"] == pytest.approx(regress_rating(2000.0, gap) - 1500.0)


@pytest.mark.parametrize("surface", ["clay", "grass", "hard"])
def test_elo_scalar_bulk_parity(surface):
    """Scalar and bulk builders emit identical overall Elo diffs."""
    req = {
        "player_id": "S0AG",
        "opponent_id": "Z355",
        "surface": surface,
        "as_of_date": AS_OF_AFTER_ALL_MATCHES,
    }
    scalar = build_inference_features(**req)
    bulk = build_inference_features_bulk([req])
    assert scalar.iloc[0]["elo_diff"] == bulk.iloc[0]["elo_diff"]


# ── Elo-gradient (player_elo_gradient_10 / opponent_elo_gradient_10) ──


def _insert_elo_rows(rows):
    """Insert raw Elo snapshots: rows of (player_id, match_id, date, match_num, post_elo)."""
    assert _DB is not None
    _DB.executemany(
        "INSERT INTO silver.elo_snapshots "
        "(player_id, match_id, match_date, match_num, surface, pre_elo, "
        "post_elo, prior_overall_matches, k_overall, source_hash) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (pid, mid, d, num, "hard", 0.0, pe, 0, 0.0, f"h{pid}{num}")
            for pid, mid, d, num, pe in rows
        ],
    )


def test_gradient_zero_history_defaults_to_zero():
    """A player with no Elo history before as_of yields gradient exactly 0."""
    out = build_inference_features("ZZZG", "ZZZH", "hard", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    row = out.iloc[0]
    assert row["player_elo_gradient_10"] == 0.0
    assert row["opponent_elo_gradient_10"] == 0.0
    for col in FEATURE_COLS:
        assert math.isfinite(row[col]), col


def test_gradient_single_history_defaults_to_zero():
    """Fewer than two strictly-before snapshots -> gradient 0 (gold parity)."""
    _insert_elo_rows([("G1", "g1", date(2026, 3, 1), 1, 1600.0)])
    out = build_inference_features("G1", "ZZZH", "hard", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    assert out.iloc[0]["player_elo_gradient_10"] == 0.0


def test_gradient_partial_history_ols_slope():
    """Three strictly-before snapshots produce the OLS slope (b = 20)."""
    rows = [
        ("G2", f"g{i}", date(2026, 1, 1) + timedelta(days=30 * i), i, 1500.0 + 20.0 * i)
        for i in range(3)
    ]
    _insert_elo_rows(rows)
    out = build_inference_features("G2", "ZZZH", "hard", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    assert out.iloc[0]["player_elo_gradient_10"] == pytest.approx(20.0)


def test_gradient_full_history_uses_last_ten():
    """Twelve snapshots: only the most recent ten drive the slope (b = 10)."""
    rows = [
        ("G3", f"g{i}", date(2026, 1, 1) + timedelta(days=10 * i), i, 1500.0 + 10.0 * i)
        for i in range(12)
    ]
    _insert_elo_rows(rows)
    out = build_inference_features("G3", "ZZZH", "hard", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    # Last ten are i=2..11 -> 1520..1610, slope 10.
    assert out.iloc[0]["player_elo_gradient_10"] == pytest.approx(10.0)


def test_gradient_strict_date_boundary_excludes_same_day():
    """Same-day and future snapshots are excluded from the gradient window."""
    as_of = date(2026, 6, 1)
    rows = [
        ("G4", "ga", date(2026, 5, 25), 1, 1500.0),
        ("G4", "gb", date(2026, 5, 26), 2, 1530.0),
        ("G4", "gc", date(2026, 5, 27), 3, 1560.0),
        ("G4", "gd", date(2026, 6, 1), 4, 9999.0),  # same-day, excluded
        ("G4", "ge", date(2026, 6, 2), 5, 8888.0),  # future, excluded
    ]
    _insert_elo_rows(rows)
    out = build_inference_features("G4", "ZZZH", "hard", as_of_date=as_of)
    # Only the three strictly-before snapshots (slope 30) contribute.
    assert out.iloc[0]["player_elo_gradient_10"] == pytest.approx(30.0)


def test_gradient_scalar_bulk_parity():
    """Scalar and bulk builders emit identical gradient columns."""
    rows = [
        ("G5", f"g{i}", date(2026, 1, 1) + timedelta(days=10 * i), i, 1500.0 + 7.0 * i)
        for i in range(5)
    ]
    _insert_elo_rows(rows)
    req = {
        "player_id": "G5",
        "opponent_id": "ZZZH",
        "surface": "hard",
        "as_of_date": AS_OF_AFTER_ALL_MATCHES,
    }
    scalar = build_inference_features(**req)
    bulk = build_inference_features_bulk([req])
    assert scalar.iloc[0]["player_elo_gradient_10"] == bulk.iloc[0]["player_elo_gradient_10"]
    assert scalar.iloc[0]["opponent_elo_gradient_10"] == bulk.iloc[0]["opponent_elo_gradient_10"]


def test_gradient_swap_exchanges_raw_values():
    """Swap negates form/elo diffs but exchanges the raw gradient values."""
    _insert_elo_rows(
        [
            ("GA", f"ga{i}", date(2026, 1, 1) + timedelta(days=10 * i), i, 1500.0 + 5.0 * i)
            for i in range(4)
        ]
        + [
            ("GB", f"gb{i}", date(2026, 1, 1) + timedelta(days=10 * i), i, 1700.0 + 15.0 * i)
            for i in range(4)
        ]
    )
    out_ab = build_inference_features("GA", "GB", "hard", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    out_ba = build_inference_features("GB", "GA", "hard", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    row_ab, row_ba = out_ab.iloc[0], out_ba.iloc[0]
    # GA slope 5, GB slope 15; swap exchanges the raw per-player values.
    assert row_ab["player_elo_gradient_10"] == pytest.approx(5.0)
    assert row_ab["opponent_elo_gradient_10"] == pytest.approx(15.0)
    assert row_ba["player_elo_gradient_10"] == pytest.approx(15.0)
    assert row_ba["opponent_elo_gradient_10"] == pytest.approx(5.0)
    # Signed diffs still negate while gradients exchange.
    assert row_ab["elo_diff"] == pytest.approx(-row_ba["elo_diff"])
    assert row_ab["form_diff"] == pytest.approx(-row_ba["form_diff"])
