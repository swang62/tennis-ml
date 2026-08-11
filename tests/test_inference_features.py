"""Hermetic inference-builder tests against an in-memory DuckDB stand-in.

The seeded PostgreSQL used by earlier versions of this suite is replaced by a
per-test in-memory DuckDB holding the same deterministic fixture data. Every
SQL call the builder makes (latest snapshot, 30-day activity, profiles,
head-to-head, the gold.tour_averages singleton, and the direct cross-checks in
this file) executes against that DuckDB with `%s` params translated to `?`. No
live database, connection, DATABASE_URL, or seed is involved.
"""

import math
from datetime import date
from typing import cast, override

import duckdb
import pandas as pd
import pytest

from src.constants import SILVER_ROLLING_FEATURES
from src.features import inference
from src.features.columns import DIFF_COLS, FEATURE_COLS, TOUR_AVERAGES_FALLBACK_COLS
from src.features.inference import build_inference_features
from src.features.tour_averages import load_tour_averages

# All fixture matches are in 2026 (2026-01-04 .. 2026-07-12); a fixed as-of
# date after the last match exercises the full snapshot history deterministically.
AS_OF_AFTER_ALL_MATCHES = date(2026, 9, 1)

_DB: duckdb.DuckDBPyConnection | None = None


def execute_df(sql: str, params: list[object] | None = None) -> pd.DataFrame:
    """Test stand-in for src.db.client.execute_df over the in-memory DuckDB."""
    assert _DB is not None, "the _duck_db_backed fixture must be active"
    return _DB.execute(sql.replace("%s", "?"), params or []).df()


# ── Fixture data ─────────────────────────────────────────────────────────────
#
# Mirrors the deterministic seeded set the live suite used, narrowed to the
# players the tests reference:
#   S0AG (righty, turned pro 2018) and Z355 (righty, 2013) both have rolling
#   snapshots; A0E2 and F0FV have one snapshot each so their 30-day window is
#   exercised; the single S0AG-vs-Z355 meeting (2026-07-12) drives head-to-head
#   and the train/inference parity check; gold.tour_averages holds one
#   full-pool singleton row whose rate cells equal the pool aggregates, exactly
#   as dbt materializes them.

_S0AG = (
    2.0,
    11500.0,
    24.43,
    0.8,
    0.8,
    0.2,
    0.7,
    0.6,
    0.7,
    0.55,
    0.63,
    0.42,
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
    0.4,
    0.5,
    0.1,
    0.6,
    0.5,
    0.6,
    0.5,
    0.55,
    0.4,
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
    0.4,
    0.5,
    0.15,
    0.65,
    0.55,
    0.62,
    0.5,
    0.58,
    0.42,
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
    "weighted_form_10",
    "win_rate_10",
    "ace_rate_10",
    "first_serve_pct_10",
    "break_points_saved_pct_10",
    "first_serve_win_pct_10",
    "second_serve_win_pct_10",
    "serve_win_pct_10",
    "return_points_won_pct_10",
    "df_rate_10",
    "aces_per_svc_game_10",
    "streak",
    "avg_player_rank_10",
    "avg_rank_faced_10",
    "clay_win_rate_10",
    "grass_win_rate_10",
    "hard_win_rate_10",
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
        rows.append((pid, match_id, snap_dates[match_id], num, "hard", *stats))
    return rows


def _match_rows() -> list[tuple[object, ...]]:
    """Both player perspectives for seeded matches (match_number assigned)."""
    return [
        ("pm-s1", date(2026, 1, 20), "hard", "S0AG", "OPP1", 2.0, 20.0, 11500.0, 24.43, 1, 1),
        ("pm-s2", date(2026, 2, 16), "hard", "S0AG", "OPP2", 2.0, 20.0, 11500.0, 24.43, 1, 2),
        ("pm-s3", date(2026, 2, 18), "hard", "S0AG", "OPP3", 2.0, 20.0, 11500.0, 24.43, 1, 3),
        ("pm-s4", date(2026, 3, 7), "hard", "S0AG", "OPP4", 2.0, 20.0, 11500.0, 24.43, 1, 4),
        ("pm-s5", date(2026, 5, 20), "hard", "S0AG", "OPP5", 2.0, 20.0, 11500.0, 24.43, 1, 5),
        ("pm-s6", date(2026, 7, 12), "hard", "S0AG", "Z355", 2.0, 4.0, 11500.0, 24.43, 1, 6),
        ("pm-s6", date(2026, 7, 12), "hard", "Z355", "S0AG", 4.0, 2.0, 4555.0, 28.85, 0, 4),
        ("pm-z1", date(2026, 1, 4), "hard", "Z355", "OPP6", 4.0, 20.0, 4555.0, 28.85, 1, 1),
        ("pm-z2", date(2026, 1, 18), "hard", "Z355", "OPP7", 4.0, 20.0, 4555.0, 28.85, 1, 2),
        ("pm-z3", date(2026, 2, 25), "hard", "Z355", "OPP8", 4.0, 20.0, 4555.0, 28.85, 1, 3),
        ("pm-a1", date(2026, 3, 15), "hard", "A0E2", "X1", 1.0, 30.0, 12050.0, 22.7, 1, 1),
        ("pm-a2", date(2026, 3, 21), "hard", "A0E2", "F0FV", 1.0, 32.0, 12050.0, 22.7, 1, 2),
        ("pm-f1", date(2026, 3, 21), "hard", "F0FV", "A0E2", 32.0, 1.0, 1510.0, 19.4, 0, 1),
    ]


# Hand-computed gold row for the single parity match (S0AG vs Z355, hard,
# 2026-07-12): the independent expectation the inference builder must reproduce.
# Value order: match metadata, then FEATURE_COLS.
_PARITY_GOLD = (
    # match_id, match_date, player_id, opponent_id, surface
    "pm-s6",
    date(2026, 7, 12),
    "S0AG",
    "Z355",
    "hard",
    # 15 matchup diffs (canonical S0AG side minus Z355)
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
    0.1,
    0.0,
    -5.0,
    2.0,
    # 12 absolute state values
    0.8,
    0.4,
    53.0,
    137.0,
    0.0,
    0.0,
    0.8,
    0.6,
    0.0,
    0.0,
    8.0,
    13.0,
    # 2 head-to-head counts (no strictly-prior meetings)
    0.0,
    0.0,
    # 7 context values (is_clay, is_grass, is_hard, is_carpet, is_indoor,
    # tournament_level, round_encoded)
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
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
        CREATE TABLE silver.player_matches (
            match_id VARCHAR, match_date DATE, surface VARCHAR,
            player_id VARCHAR, opponent_id VARCHAR,
            player_ranking DOUBLE, opponent_ranking DOUBLE,
            player_rank_points DOUBLE, player_age DOUBLE,
            match_won INTEGER, player_match_number INTEGER
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
        f"INSERT INTO silver.rolling_features VALUES ({', '.join(['?'] * 25)})",
        _snap_rows(),
    )
    con.executemany(
        f"INSERT INTO silver.player_matches VALUES ({', '.join(['?'] * 11)})",
        _match_rows(),
    )
    con.executemany(
        "INSERT INTO bronze.player_profiles VALUES (?, ?, ?, ?)",
        [
            ("S0AG", 191.0, "R", 2018),
            ("Z355", 198.0, "R", 2013),
        ],
    )
    # Pool aggregates over the 12 snapshots: weighted_form_10 -> 0.6,
    # hard_win_rate_10 -> 0.7, win_rate_10 -> 0.65, median streak -> 2.5.
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
        "weighted_form_10": 0.6,
        "win_rate_10": 0.65,
        "ace_rate_10": 0.15,
        "first_serve_pct_10": 0.65,
        "break_points_saved_pct_10": 0.55,
        "first_serve_win_pct_10": 0.62,
        "second_serve_win_pct_10": 0.5,
        "serve_win_pct_10": 0.58,
        "return_points_won_pct_10": 0.42,
        "df_rate_10": 0.04,
        "aces_per_svc_game_10": 0.35,
        "avg_player_rank_10": 15.0,
        "avg_rank_faced_10": 30.0,
        "clay_win_rate_10": 0.55,
        "grass_win_rate_10": 0.5,
        "hard_win_rate_10": 0.7,
        "days_since_default": 14.0,
        "matches_30d_default": 3.0,
        "rate_default": 0.5,
        "left_handed_rate": 0.12,
        "avg_years_pro": 10.0,
    }
    con.execute(
        f"INSERT INTO gold.tour_averages VALUES ({', '.join(['?'] * len(singleton))})",
        list(singleton.values()),
    )
    con.execute(
        f"INSERT INTO gold.match_features VALUES ({', '.join(['?'] * 41)})",
        _PARITY_GOLD,
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


def _insert_prior_meetings(pair_a: str, pair_b: str, meetings: list[tuple[str, str, int]]) -> None:
    """Insert canonical prior meetings as two complementary player perspectives."""
    rows = []
    for match_id, date_iso, a_won in meetings:
        rows.append((match_id, date.fromisoformat(date_iso), pair_a, pair_b, a_won))
        rows.append((match_id, date.fromisoformat(date_iso), pair_b, pair_a, 1 - a_won))
    assert _DB is not None
    _DB.executemany(
        "INSERT INTO silver.player_matches "
        "(match_id, match_date, player_id, opponent_id, match_won) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )


def test_output_schema_contract():
    """Exact column order [*FEATURE_COLS, "player_id", "opponent_id"], one row."""
    out = build_inference_features("S0AG", "Z355", "clay", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    assert out.columns.tolist() == [*FEATURE_COLS, "player_id", "opponent_id"]
    assert len(out.columns) == 38  # 36 features + 2 ids
    assert len(out) == 1
    assert out["player_id"].dtype == object
    assert out["opponent_id"].dtype == object


@pytest.mark.parametrize("surface", ["clay", "grass", "hard", "carpet"])
def test_two_known_players_each_surface(surface):
    """Known-vs-known row: valid one-hot, canonical ids, finite features."""
    out = build_inference_features("S0AG", "Z355", surface, as_of_date=AS_OF_AFTER_ALL_MATCHES)
    row = out.iloc[0]
    assert row["player_id"] == "S0AG"  # 'S0AG' < 'Z355'
    assert row["opponent_id"] == "Z355"
    expected_one_hots = {"is_clay": 0, "is_grass": 0, "is_hard": 0, "is_carpet": 0}
    expected_one_hots[f"is_{surface}"] = 1
    assert row["is_clay"] == expected_one_hots["is_clay"]
    assert row["is_grass"] == expected_one_hots["is_grass"]
    assert row["is_hard"] == expected_one_hots["is_hard"]
    assert row["is_carpet"] == expected_one_hots["is_carpet"]
    assert sum(expected_one_hots.values()) == 1
    for col in FEATURE_COLS:
        assert math.isfinite(row[col]), f"{col} is not finite: {row[col]!r}"
    assert row["tournament_level"] == 0
    assert row["round_encoded"] == 0


def test_reversed_ids_canonical_identical():
    """Swapping the two ids must produce an identical canonical row."""
    row_ab = build_inference_features("S0AG", "Z355", "hard", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    row_ba = build_inference_features("Z355", "S0AG", "hard", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    pd.testing.assert_frame_equal(row_ab, row_ba, check_exact=True)


def test_repeated_identical_inputs_are_deterministic():
    """Identical requests produce identical canonical rows."""
    a = build_inference_features("S0AG", "Z355", "hard", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    b = build_inference_features("S0AG", "Z355", "hard", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    c = build_inference_features("S0AG", "Z355", "hard", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    pd.testing.assert_frame_equal(b, c, check_exact=True)


def test_known_players_profile_features():
    """Known profile features use canonical order and as-of years_pro."""
    out = build_inference_features("S0AG", "Z355", "clay", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    row = out.iloc[0]
    assert row["player_is_left_handed"] == 0.0
    assert row["opponent_is_left_handed"] == 0.0
    assert row["player_years_pro"] == 8.0  # 2026 - 2018
    assert row["opponent_years_pro"] == 13.0  # 2026 - 2013
    assert "player_height" not in out.columns
    assert "height_diff" not in out.columns


def test_years_pro_time_aware_and_cold_start():
    """years_pro derives from as_of_date, not a fixed snapshot of turned_pro.

    At an as-of date before any seeded match (empty snapshot pool), profile
    values still come from bronze.player_profiles and years_pro tracks the
    as-of year, while all rolling features fall back to their constants.
    """
    out = build_inference_features("S0AG", "Z355", "hard", as_of_date=date(2025, 6, 1))
    row = out.iloc[0]
    assert row["player_years_pro"] == 7.0  # 2025 - 2018
    assert row["opponent_years_pro"] == 12.0  # 2025 - 2013
    for col in FEATURE_COLS:
        assert math.isfinite(row[col]), f"{col} is not finite: {row[col]!r}"


def test_materialized_defaults_return_expected_values():
    """The materialized gold.tour_averages singleton drives cold-start imputation.

    Scalar inference performs no on-demand AVG/PERCENTILE queries: it reads the
    single full-pool singleton row regardless of the as-of date. Every default
    cell is finite, re-reads are deterministic, and a cold-start pair imputes
    exactly those materialized values on both sides (so every diff stays
    neutral).
    """
    defaults = cast(dict[str, float], load_tour_averages())
    assert set(TOUR_AVERAGES_FALLBACK_COLS).issubset(defaults.keys())
    for col in TOUR_AVERAGES_FALLBACK_COLS:
        assert not pd.isna(defaults[col]), f"{col} is NULL for the singleton row"
        assert math.isfinite(float(defaults[col])), f"{col} not finite for the singleton row"

    # Deterministic: re-reading the singleton returns identical values.
    assert defaults == load_tour_averages()

    # The builder imputes exactly these materialized defaults for two unknown
    # players (cold-start sides equal the defaults; every diff stays neutral).
    out = build_inference_features("ZZZZ", "YYYY", "hard", as_of_date=date(2026, 9, 1))
    row = out.iloc[0]
    for col in DIFF_COLS:
        assert row[col] == 0, f"{col} should be neutral for two unknowns: {row[col]!r}"
    assert row["player_weighted_form_10"] == pytest.approx(float(defaults["weighted_form_10"]))
    assert row["player_surface_win_rate_10"] == pytest.approx(float(defaults["hard_win_rate_10"]))
    # win_rate_10 / ace_rate_10 are only exposed as canonical-minus-opponent
    # diffs; two unknowns impute the same default on both sides, so the diffs
    # collapse to exactly 0 and lock the imputed default values.
    assert row["win_rate_diff"] == 0
    assert row["ace_rate_diff"] == 0
    assert row["player_days_since_last_match"] == int(defaults["days_since_default"])
    assert row["player_matches_30d"] == int(defaults["matches_30d_default"])
    assert row["player_is_left_handed"] == pytest.approx(float(defaults["left_handed_rate"]))
    assert row["player_years_pro"] == pytest.approx(float(defaults["avg_years_pro"]))


def test_historical_as_of_excludes_later_snapshots():
    """Use the newest snapshot strictly before the as-of date."""
    out = build_inference_features("S0AG", "Z355", "hard", as_of_date=date(2026, 6, 30))
    # Cross-check the expected snapshot directly in the fixture table.
    snapshot = execute_df(
        f"SELECT player_match_number, win_rate_10, weighted_form_10 "
        f"FROM {SILVER_ROLLING_FEATURES} "
        "WHERE player_id = %s AND snapshot_date < %s::date "
        "ORDER BY player_match_number DESC LIMIT 1",
        ["S0AG", "2026-06-30"],
    ).iloc[0]
    assert snapshot["player_match_number"] == 5
    assert float(snapshot["win_rate_10"]) == 0.8
    # The inference row's per-side weighted form equals that snapshot's value.
    assert out.loc[0, "player_weighted_form_10"] == pytest.approx(
        float(snapshot["weighted_form_10"])
    )


class _FixedTodayDate(date):
    """datetime.date subclass whose today() returns a fixed date."""

    @classmethod
    @override
    def today(cls) -> date:
        return date(2026, 9, 1)


def test_default_today_fecha(monkeypatch):
    """Default date.today matches an explicit as-of date."""
    out_explicit = build_inference_features("S0AG", "Z355", "clay", as_of_date=date(2026, 9, 1))
    monkeypatch.setattr("src.features.inference.date", _FixedTodayDate)
    out_default = build_inference_features("S0AG", "Z355", "clay")
    pd.testing.assert_frame_equal(out_default, out_explicit, check_exact=True)
    row = out_default.iloc[0]
    assert not out_default[FEATURE_COLS].isnull().to_numpy().any()
    for side in ("player", "opponent"):
        assert row[f"{side}_days_since_last_match"] >= 0
        assert row[f"{side}_matches_30d"] >= 0
        assert math.isfinite(row[f"{side}_matches_30d"])


@pytest.mark.parametrize(
    "args",
    [("S0AG", "UNKNOWN_ID"), ("UNKNOWN_ID", "S0AG")],
    ids=["known-first", "reversed"],
)
def test_one_missing_player_imputed_no_nans(args):
    """One unknown player is pool-imputed without changing canonical ids."""
    player_id, opponent_id = args
    out = build_inference_features(
        player_id, opponent_id, "hard", as_of_date=AS_OF_AFTER_ALL_MATCHES
    )
    row = out.iloc[0]
    assert row["player_id"] == "S0AG"
    assert row["opponent_id"] == "UNKNOWN_ID"
    for col in FEATURE_COLS:
        assert math.isfinite(row[col]), f"{col} is not finite: {row[col]!r}"
    # The known player's form differs from the pool default (diff exists).
    assert row["win_rate_diff"] != 0
    # Check exposed values that use the same pool as the builder.
    pool = execute_df(
        "SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY streak) AS streak, "
        "AVG(weighted_form_10) AS weighted_form_10, "
        "AVG(win_rate_10) AS win_rate_10, "
        "AVG(hard_win_rate_10) AS hard_win_rate_10 "
        f"FROM {SILVER_ROLLING_FEATURES} WHERE snapshot_date < %s::date",
        ["2026-09-01"],
    ).iloc[0]
    assert row["streak_diff"] != 0  # known streak vs pool-mean streak differ
    assert row["opponent_weighted_form_10"] == pytest.approx(float(pool["weighted_form_10"]))
    assert row["opponent_surface_win_rate_10"] == pytest.approx(float(pool["hard_win_rate_10"]))
    # Profile-derived features for the unknown player are pool-imputed from
    # the gold.tour_averages singleton. They must be finite, non-NaN, and
    # within valid ranges (the exact value shifts with data — the contract is
    # "plausible float," not a specific compute).
    assert 0.0 <= row["opponent_is_left_handed"] <= 1.0, (
        f"left_handed_rate out of bounds: {row['opponent_is_left_handed']}"
    )
    assert 0.0 <= row["opponent_years_pro"] <= 50.0, (
        f"avg_years_pro out of bounds: {row['opponent_years_pro']}"
    )
    assert math.isfinite(row["player_years_pro"])


def test_one_missing_player_reversed_identical():
    """Mirror of the canonical test with one unknown: both argument orders agree."""
    row_known_first = build_inference_features(
        "S0AG", "UNKNOWN_ID", "hard", as_of_date=AS_OF_AFTER_ALL_MATCHES
    )
    row_reversed = build_inference_features(
        "UNKNOWN_ID", "S0AG", "hard", as_of_date=AS_OF_AFTER_ALL_MATCHES
    )
    pd.testing.assert_frame_equal(row_known_first, row_reversed, check_exact=True)


def test_both_unknowns_neutral_diffs():
    """Two unknown players receive identical defaults, so all diffs are 0.

    as_of_date is passed explicitly (not the default) so the test stays
    deterministic; the neutral-diff property holds for any as-of date.
    """
    out = build_inference_features("A0ZZ", "ZZ99", "hard", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    row = out.iloc[0]
    assert row["player_id"] == "A0ZZ"
    assert row["opponent_id"] == "ZZ99"
    for col in FEATURE_COLS:
        assert math.isfinite(row[col]), f"{col} is not finite: {row[col]!r}"
    for col in DIFF_COLS:
        assert row[col] == 0, f"{col} should be neutral for two unknowns: {row[col]!r}"
    assert [row["is_clay"], row["is_grass"], row["is_hard"]] == [0, 0, 1]


@pytest.mark.parametrize("surface", ["Clay", "CLAY", "", None])
def test_invalid_surface_raises(surface):
    with pytest.raises((ValueError, TypeError)):
        build_inference_features("S0AG", "Z355", surface, as_of_date=AS_OF_AFTER_ALL_MATCHES)


@pytest.mark.parametrize("tournament_level", [-1, 5, True, "4", 4.0])
def test_invalid_tournament_level_raises(tournament_level):
    with pytest.raises((ValueError, TypeError)):
        build_inference_features(
            "S0AG",
            "Z355",
            "hard",
            as_of_date=AS_OF_AFTER_ALL_MATCHES,
            tournament_level=tournament_level,
        )


@pytest.mark.parametrize("round_encoded", [-1, 8, True, "7", 7.0])
def test_invalid_round_encoded_raises(round_encoded):
    with pytest.raises((ValueError, TypeError)):
        build_inference_features(
            "S0AG",
            "Z355",
            "hard",
            as_of_date=AS_OF_AFTER_ALL_MATCHES,
            round_encoded=round_encoded,
        )


@pytest.mark.parametrize(
    "tournament_level, expected",
    [
        (4, 4),
        (3, 3),
        (2, 2),
        (1, 1),
        (0, 0),
    ],
)
def test_valid_tournament_levels_accepted(tournament_level, expected):
    out = build_inference_features(
        "S0AG",
        "Z355",
        "hard",
        as_of_date=AS_OF_AFTER_ALL_MATCHES,
        tournament_level=tournament_level,
    )
    assert out.iloc[0]["tournament_level"] == expected


@pytest.mark.parametrize(
    "round_encoded, expected",
    [
        (7, 7),
        (6, 6),
        (5, 5),
        (4, 4),
        (3, 3),
        (2, 2),
        (1, 1),
        (0, 0),
    ],
)
def test_valid_round_encodings_accepted(round_encoded, expected):
    out = build_inference_features(
        "S0AG",
        "Z355",
        "hard",
        as_of_date=AS_OF_AFTER_ALL_MATCHES,
        round_encoded=round_encoded,
    )
    assert out.iloc[0]["round_encoded"] == expected


@pytest.mark.parametrize(
    "tournament, expected",
    [
        ("grand_slam", 4),
        ("masters", 3),
        ("atp_500", 2),
        ("atp_250", 1),
        ("davis_cup", 0),
        ("atp_finals", 0),
        ("olympics", 0),
        ("professional", 0),
    ],
)
def test_tournament_string_convenience_maps_to_codebook(tournament, expected):
    """String tournament names map to the SAME encodings as the dbt codebook."""
    out = build_inference_features(
        "S0AG",
        "Z355",
        "hard",
        as_of_date=AS_OF_AFTER_ALL_MATCHES,
        tournament=tournament,
    )
    assert out.iloc[0]["tournament_level"] == expected


@pytest.mark.parametrize(
    "round, expected",
    [
        ("r128", 1),
        ("r64", 2),
        ("r32", 3),
        ("r16", 4),
        ("qf", 5),
        ("sf", 6),
        ("f", 7),
    ],
)
def test_round_string_convenience_maps_to_codebook(round, expected):
    """String round names map to the SAME encodings as the dbt codebook."""
    out = build_inference_features(
        "S0AG",
        "Z355",
        "hard",
        as_of_date=AS_OF_AFTER_ALL_MATCHES,
        round=round,
    )
    assert out.iloc[0]["round_encoded"] == expected


@pytest.mark.parametrize("tournament", ["Roland Garros", "grand slam", "Grand_Slam", "random", ""])
def test_unknown_tournament_string_maps_to_zero(tournament):
    """Unknown tournament strings map to 0, matching the codebook's ELSE branch."""
    out = build_inference_features(
        "S0AG",
        "Z355",
        "hard",
        as_of_date=AS_OF_AFTER_ALL_MATCHES,
        tournament=tournament,
    )
    assert out.iloc[0]["tournament_level"] == 0


@pytest.mark.parametrize("round", ["final", "F", "QF", "unknown", ""])
def test_unknown_round_string_maps_to_zero(round):
    """Unknown round strings map to 0, matching the codebook's ELSE branch."""
    out = build_inference_features(
        "S0AG",
        "Z355",
        "hard",
        as_of_date=AS_OF_AFTER_ALL_MATCHES,
        round=round,
    )
    assert out.iloc[0]["round_encoded"] == 0


def test_tournament_string_equals_int_encoding():
    """String and int paths produce byte-identical rows for the same encoding."""
    out_str = build_inference_features(
        "S0AG",
        "Z355",
        "clay",
        as_of_date=AS_OF_AFTER_ALL_MATCHES,
        tournament="grand_slam",
        round="f",
    )
    out_int = build_inference_features(
        "S0AG",
        "Z355",
        "clay",
        as_of_date=AS_OF_AFTER_ALL_MATCHES,
        tournament_level=4,
        round_encoded=7,
    )
    pd.testing.assert_frame_equal(out_str, out_int, check_exact=True)


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


@pytest.mark.parametrize(
    "player_id, as_of, expected",
    [
        # F0FV's only prior seeded match is 2026-03-21 (vs A0E2); the
        # [2026-04-28, 2026-05-28) window contains none of his matches, so the
        # count is 0. The old ROWS-frame formulation returned 1 here (every
        # preceding match, regardless of date) — this case catches that bug.
        ("F0FV", date(2026, 5, 28), 0),
        # [2026-02-20, 2026-03-22): A0E2's 2026-03-15 and 2026-03-21 matches
        # are inside; the 2026-03-22 match itself is excluded (strict <).
        ("A0E2", date(2026, 3, 22), 2),
    ],
    ids=["30d-window-empty", "30d-window-two"],
)
def test_matches_30d_window_regression(player_id, as_of, expected):
    """Regression: matches_30d uses a real date window, not a ROWS frame."""
    out = build_inference_features(player_id, "UNKNOWN_PLAYER", "hard", as_of_date=as_of)
    row = out.iloc[0]
    assert row["player_id"] == player_id  # 'A0E2'/'F0FV' < 'UNKNOWN_PLAYER'
    assert row["opponent_id"] == "UNKNOWN_PLAYER"
    assert row["player_matches_30d"] == expected


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


# ── Head-to-head (perspective-explicit, last-5 recency) ──
#
# Seed data has no repeated pair, so H2H tests use isolated synthetic rows.


def test_h2h_zero_prior_meetings_neutral():
    """A pair that never met (UNKNOWN_ID has no silver rows at all) gets the
    locked neutral fallback: 0 counts for the canonical player side."""
    out = build_inference_features("S0AG", "UNKNOWN_ID", "hard", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    row = out.iloc[0]
    assert row["player_id"] == "S0AG"
    assert row["opponent_id"] == "UNKNOWN_ID"
    assert row["player_h2h_matches"] == 0
    assert row["player_h2h_wins"] == 0
    assert "player_h2h_win_rate" not in out.columns
    assert "opponent_h2h_matches" not in out.columns


def test_h2h_real_seeded_meeting():
    """A real seeded meeting: S0AG beat Z355 once (Hamburg, 2026-07-12), so
    after that date the pair has exactly 1 prior, won by the canonical S0AG
    side. Reverse the raw ids and the H2H values must not change."""
    out = build_inference_features("S0AG", "Z355", "hard", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    row = out.iloc[0]
    assert row["player_id"] == "S0AG"  # 'S0AG' < 'Z355'
    assert row["player_h2h_matches"] == 1
    assert row["player_h2h_wins"] == 1
    # Before that meeting (strictly-before): zero priors, neutral.
    out_before = build_inference_features("S0AG", "Z355", "hard", as_of_date=date(2026, 7, 12))
    row_before = out_before.iloc[0]
    assert row_before["player_h2h_matches"] == 0
    # Reversed raw ids: identical canonical row (H2H included).
    row_ba = build_inference_features("Z355", "S0AG", "hard", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    pd.testing.assert_frame_equal(out, row_ba, check_exact=True)


def test_h2h_first_and_second_meeting_boundaries():
    """First meeting (as-of on the meeting date) -> zero priors + neutral;
    second meeting -> exactly one prior, with correct wins on both sides."""
    a, b = "H2H_F", "H2H_G"
    _insert_prior_meetings(a, b, [("h2h-f1", "2026-05-20", 1)])
    # On the first meeting's own date: strictly-before excludes it.
    out1 = build_inference_features(a, b, "clay", as_of_date=date(2026, 5, 20))
    row1 = out1.iloc[0]
    assert row1["player_h2h_matches"] == 0
    assert row1["player_h2h_wins"] == 0
    # A second meeting: exactly one prior, A won it.
    _insert_prior_meetings(a, b, [("h2h-f2", "2026-06-20", 0)])
    out2 = build_inference_features(a, b, "clay", as_of_date=date(2026, 6, 20))
    row2 = out2.iloc[0]
    assert row2["player_h2h_matches"] == 1
    assert row2["player_h2h_wins"] == 1
    # After the second meeting: both priors, 1 win each side.
    out3 = build_inference_features(a, b, "clay", as_of_date=date(2026, 6, 21))
    row3 = out3.iloc[0]
    assert row3["player_h2h_matches"] == 2
    assert row3["player_h2h_wins"] == 1


def test_h2h_last5_recency_drops_oldest():
    """A 6th prior meeting drops the oldest from the window: only the 5 most
    recent meetings count. The dropped meeting is the pair's only A win."""
    a, b = "H2H_C", "H2H_D"
    _insert_prior_meetings(
        a,
        b,
        [
            ("h2h-r1", "2026-01-05", 1),  # oldest; A's only win -> must be dropped
            ("h2h-r2", "2026-02-05", 0),
            ("h2h-r3", "2026-03-05", 0),
            ("h2h-r4", "2026-04-05", 0),
            ("h2h-r5", "2026-05-05", 0),
            ("h2h-r6", "2026-06-05", 0),
        ],
    )
    out = build_inference_features(a, b, "hard", as_of_date=date(2026, 7, 1))
    row = out.iloc[0]
    assert row["player_h2h_matches"] == 5
    assert row["player_h2h_wins"] == 0  # the sole A win is the dropped oldest


def test_h2h_same_date_meetings_excluded():
    """Meetings on the as-of date itself are excluded (strictly-before rule);
    the next day both count."""
    a, b = "H2H_E", "H2H_I"
    _insert_prior_meetings(a, b, [("h2h-s1", "2026-03-10", 1), ("h2h-s2", "2026-03-10", 0)])
    out_same = build_inference_features(a, b, "hard", as_of_date=date(2026, 3, 10))
    row_same = out_same.iloc[0]
    assert row_same["player_h2h_matches"] == 0
    out_next = build_inference_features(a, b, "hard", as_of_date=date(2026, 3, 11))
    row_next = out_next.iloc[0]
    assert row_next["player_h2h_matches"] == 2
    assert row_next["player_h2h_wins"] == 1


def test_h2h_reversed_raw_ids_identical():
    """Reversing the raw input ids must produce the identical canonical row
    (H2H is computed after canonicalization from the same prior meetings)."""
    a, b = "H2H_J", "H2H_K"
    _insert_prior_meetings(a, b, [("h2h-rv1", "2026-04-10", 1), ("h2h-rv2", "2026-05-10", 0)])
    row_ab = build_inference_features(a, b, "hard", as_of_date=date(2026, 6, 1))
    row_ba = build_inference_features(b, a, "hard", as_of_date=date(2026, 6, 1))
    pd.testing.assert_frame_equal(row_ab, row_ba, check_exact=True)
    assert row_ab.iloc[0]["player_h2h_matches"] == 2
    assert row_ab.iloc[0]["player_h2h_wins"] == 1


# ── Train/inference parity (strongest train/serve agreement check) ──
#
# The gold row for the single parity match (S0AG vs Z355, hard, 2026-07-12) is
# a hand-computed fixture literal; the builder must reproduce it exactly.

_PARITY_MATCH_SQL = """
SELECT mf.*
FROM gold.match_features mf
JOIN silver.player_matches p
  ON p.match_id = mf.match_id AND p.player_id = mf.player_id
JOIN silver.player_matches o
  ON o.match_id = mf.match_id AND o.player_id = mf.opponent_id
JOIN silver.rolling_features prp
  ON prp.player_id = mf.player_id
 AND prp.player_match_number = p.player_match_number - 1
JOIN silver.rolling_features pro
  ON pro.player_id = mf.opponent_id
 AND pro.player_match_number = o.player_match_number - 1
WHERE p.player_match_number > 1
  AND o.player_match_number > 1
  AND prp.snapshot_date < mf.match_date
  AND pro.snapshot_date < mf.match_date
ORDER BY mf.match_date, mf.match_id
LIMIT 1
"""


def test_train_inference_parity_on_historical_match():
    """The gold row built from snapshot N-1 and an inference row built at that
    match's date (from only strictly-earlier data) must agree on EVERY
    FEATURE_COL within 1e-6. No column is skipped: the finalized contract
    gives gold and inference identical strictly-prior date semantics for
    ranking, rank points, age, and rolling state."""
    gold_row = execute_df(_PARITY_MATCH_SQL).iloc[0]
    out = build_inference_features(
        str(gold_row["player_id"]),
        str(gold_row["opponent_id"]),
        str(gold_row["surface"]),
        as_of_date=gold_row["match_date"],
        tournament_level=int(gold_row["tournament_level"]),
        round_encoded=int(gold_row["round_encoded"]),
    )
    infer_row = out.iloc[0]
    assert out.columns.tolist() == [*FEATURE_COLS, "player_id", "opponent_id"]
    compared = 0
    for col in FEATURE_COLS:
        gold_val = float(gold_row[col])
        assert infer_row[col] == pytest.approx(gold_val, abs=1e-6), col
        compared += 1
    # The fixture must actually compare the whole contract, so it cannot
    # silently degenerate to a handful of columns.
    assert compared == len(FEATURE_COLS)


# ── is_indoor context feature ────────────────────────────────────


def test_is_indoor_defaults_to_0_when_not_supplied():
    """Missing indoor => is_indoor = 0 (safe outdoor default)."""
    out = build_inference_features("S0AG", "Z355", "hard", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    assert out["is_indoor"].iloc[0] == 0


def test_is_indoor_1_when_indoor():
    out = build_inference_features(
        "S0AG", "Z355", "hard", is_indoor=1, as_of_date=AS_OF_AFTER_ALL_MATCHES
    )
    assert out["is_indoor"].iloc[0] == 1


def test_is_indoor_0_when_outdoor():
    out = build_inference_features(
        "S0AG", "Z355", "hard", is_indoor=0, as_of_date=AS_OF_AFTER_ALL_MATCHES
    )
    assert out["is_indoor"].iloc[0] == 0


def test_inference_row_has_no_nan_with_indoor():
    """Full inference row with indoor is NaN-free."""
    out = build_inference_features(
        "S0AG",
        "Z355",
        "hard",
        is_indoor=1,
        tournament="grand_slam",
        as_of_date=AS_OF_AFTER_ALL_MATCHES,
    )
    assert not out[FEATURE_COLS].isnull().to_numpy().any()
    assert "is_indoor" in out.columns


def test_cold_start_row_has_no_nan_with_indoor():
    """Unknown players + indoor => no NaN in the feature row."""
    out = build_inference_features(
        "ZZZZ", "YYYY", "clay", is_indoor=1, as_of_date=AS_OF_AFTER_ALL_MATCHES
    )
    assert not out[FEATURE_COLS].isnull().to_numpy().any()
    assert out["is_indoor"].iloc[0] == 1
