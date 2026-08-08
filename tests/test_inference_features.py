"""Integration tests for the PostgreSQL-backed ID inference builder."""

import math
from datetime import date
from typing import override

import pandas as pd
import pytest

from src.constants import PROFILES_TABLE, SILVER_ROLLING_FEATURES
from src.db.client import execute_df, get_conn
from src.features import inference
from src.features.columns import DIFF_COLS, FEATURE_COLS
from src.features.inference import build_inference_features

# All seeded matches are in 2026 (2026-03-15 .. 2026-07-15); a fixed as-of date
# after the last match exercises the full snapshot history deterministically.
AS_OF_AFTER_ALL_MATCHES = date(2026, 9, 1)


@pytest.fixture(autouse=True)
def _require_gold(gold_ready):
    """Skip the whole module until the dbt gold/silver layers exist (Task 4)."""


@pytest.fixture(autouse=True)
def _cleanup_h2h_rows(_require_gold):
    """Remove synthetic rows that would violate later dbt tests."""
    yield
    with get_conn().cursor() as cur:
        cur.execute(
            "DELETE FROM silver.player_matches "
            "WHERE player_id LIKE 'H2H_%' OR opponent_id LIKE 'H2H_%'"
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
    values still come from gold.player_profiles and years_pro tracks the
    as-of year, while all rolling features fall back to their constants.
    """
    out = build_inference_features("S0AG", "Z355", "hard", as_of_date=date(2025, 6, 1))
    row = out.iloc[0]
    assert row["player_years_pro"] == 7.0  # 2025 - 2018
    assert row["opponent_years_pro"] == 12.0  # 2025 - 2013
    for col in FEATURE_COLS:
        assert math.isfinite(row[col]), f"{col} is not finite: {row[col]!r}"


def test_pool_aggregates_return_expected_values():
    """Pool queries return finite aggregates used for cold-start imputation."""
    as_of = "2026-09-01"
    cases = [
        (
            inference._POOL_AGG_SQL,
            [as_of],
            [
                "latest_player_ranking",
                "latest_player_rank_points",
                "latest_player_age",
                "streak",
                "weighted_form_10",
                "win_rate_10",
                "ace_rate_10",
                "first_serve_pct_10",
                "break_points_saved_pct_10",
                "first_serve_win_pct_10",
                "second_serve_win_pct_10",
                "serve_win_pct_10",
                "df_rate_10",
                "aces_per_svc_game_10",
                "avg_player_rank_10",
                "avg_rank_faced_10",
                "clay_win_rate_10",
                "grass_win_rate_10",
                "hard_win_rate_10",
            ],
        ),
        (inference._POOL_COUNTS_SQL, [as_of], ["snapshot_pool_rows", "snapshot_pool_players"]),
        (inference._MEDIAN_DAYS_SQL, [as_of, as_of], ["median_days_since"]),
        (inference._MEDIAN_MATCHES_30D_SQL, [as_of, as_of, as_of], ["median_matches_30d"]),
        (inference._PROFILE_POOL_AGG_SQL, [2026], ["left_handed_rate", "avg_years_pro"]),
        (inference._PROFILE_COUNTS_SQL, [], ["profile_rows"]),
    ]
    for sql, params, expected_cols in cases:
        df = execute_df(sql, params)
        assert df.shape == (1, len(expected_cols)), sql[:60]
        assert df.columns.tolist() == expected_cols, sql[:60]
        for col in expected_cols:
            assert not pd.isna(df.iloc[0][col]), f"{col} is NULL for {sql[:60]}"
            assert math.isfinite(float(df.iloc[0][col])), f"{col} not finite for {sql[:60]}"

    # Deterministic: re-running a pool read returns identical aggregates.
    first = execute_df(inference._POOL_AGG_SQL, [as_of])
    pd.testing.assert_frame_equal(first, execute_df(inference._POOL_AGG_SQL, [as_of]))

    # The builder imputes exactly these aggregates for two unknown players
    # (cold-start sides equal the pool aggregates; every diff stays neutral).
    out = build_inference_features("ZZZZ", "YYYY", "hard", as_of_date=date(2026, 9, 1))
    row = out.iloc[0]
    agg = execute_df(inference._POOL_AGG_SQL, [as_of]).iloc[0]
    profile_agg = execute_df(inference._PROFILE_POOL_AGG_SQL, [2026]).iloc[0]
    median_days = float(
        execute_df(inference._MEDIAN_DAYS_SQL, [as_of, as_of]).iloc[0]["median_days_since"]
    )
    median_matches = float(
        execute_df(inference._MEDIAN_MATCHES_30D_SQL, [as_of, as_of, as_of]).iloc[0][
            "median_matches_30d"
        ]
    )
    assert row["player_weighted_form_10"] == pytest.approx(float(agg["weighted_form_10"]))
    assert row["player_surface_win_rate_10"] == pytest.approx(float(agg["hard_win_rate_10"]))
    # win_rate_10 / ace_rate_10 are only exposed as canonical-minus-opponent
    # diffs; two unknowns impute the same pool value on both sides, so the
    # diffs collapse to exactly 0 and lock the imputed pool values.
    assert row["win_rate_diff"] == 0
    assert row["ace_rate_diff"] == 0
    assert row["player_days_since_last_match"] == pytest.approx(round(median_days))
    assert row["player_matches_30d"] == pytest.approx(round(median_matches))
    assert row["player_is_left_handed"] == pytest.approx(float(profile_agg["left_handed_rate"]))
    assert row["player_years_pro"] == pytest.approx(float(profile_agg["avg_years_pro"]))


def test_historical_as_of_excludes_later_snapshots():
    """Use the newest snapshot strictly before the as-of date."""
    out = build_inference_features("S0AG", "Z355", "hard", as_of_date=date(2026, 6, 30))
    # Cross-check the expected snapshot directly in the gold table.
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
    # Profile-derived features for the unknown player come from the on-demand
    # aggregate over ALL profiles (mean left-handed rate / years-pro at the
    # as-of date), so they are finite and non-NaN.
    profile_pool = execute_df(
        "SELECT "
        "AVG(CASE WHEN handedness = 'L' THEN 1 ELSE 0 END) AS left_handed_rate, "
        "AVG(2026 - turned_pro) AS avg_years_pro "
        f"FROM {PROFILES_TABLE}",
    ).iloc[0]
    assert row["opponent_is_left_handed"] == pytest.approx(float(profile_pool["left_handed_rate"]))
    assert row["opponent_years_pro"] == pytest.approx(float(profile_pool["avg_years_pro"]))
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


def _insert_prior_meetings(pair_a: str, pair_b: str, meetings: list[tuple[str, str, int]]) -> None:
    """Insert canonical prior meetings as two complementary player perspectives."""
    rows = []
    match_ids = []
    for match_id, date_iso, a_won in meetings:
        match_ids.append(match_id)
        rows.append((match_id, date_iso, pair_a, pair_b, a_won))
        rows.append((match_id, date_iso, pair_b, pair_a, 1 - a_won))
    with get_conn().cursor() as cur:
        # Make reruns idempotent without disturbing other synthetic pairs.
        cur.executemany(
            "DELETE FROM silver.player_matches WHERE match_id = %s",
            [(mid,) for mid in match_ids],
        )
        cur.executemany(
            "INSERT INTO silver.player_matches "
            "(match_id, match_date, player_id, opponent_id, match_won) "
            "VALUES (%s, CAST(%s AS DATE), %s, %s, %s)",
            rows,
        )


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
# Gold and inference share N-1 snapshots. Match-day fields and gold NULLs are
# intentionally asymmetric: inference only knows prior values and imputes NULLs.
ASYM_MATCH_DAY_COLS = {
    "player_ranking",
    "opponent_ranking",
    "rank_diff",
    "player_rank_trend_10",
    "opponent_rank_trend_10",
    "rank_trend_diff",
    "rank_points_diff",
    "player_age",
    "opponent_age",
    "age_diff",
}

# Select a match whose both sides resolve to the same N-1 snapshots.
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
    match's date (from only strictly-earlier data) must agree on every
    non-NULL, non-asymmetric feature: floats within 1e-6, ints exactly."""
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
        gold_val = gold_row[col]
        if pd.isna(gold_val):
            # Gold keeps NULLs (no zero-fill); inference imputes pool means.
            continue
        if col in ASYM_MATCH_DAY_COLS:
            # Documented match-day vs last-observed asymmetry.
            continue
        infer_val = infer_row[col]
        if isinstance(gold_val, float):
            assert infer_val == pytest.approx(float(gold_val), abs=1e-6), col
        else:
            assert int(infer_val) == int(gold_val), col
        compared += 1
    # The fixture must actually compare most of the contract, so it cannot
    # silently degenerate to a handful of columns.
    assert compared >= 20


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
