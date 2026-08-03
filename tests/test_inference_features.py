"""Focused pytest tests for the ID-based inference feature builder.

Target: `src.features.inference.build_inference_features`.

Tests run against the seeded DuckDB at `data/tennis.duckdb` (built from
`infra/duckdb/init.sql` + `infra/duckdb/seed.py` + the dbt gold models). A
session-scoped autouse fixture rebuilds the gold tables once if they are
missing or empty; otherwise it skips so repeated runs stay fast. All
date-dependent tests pass an explicit `as_of_date` for determinism, except the
default-today test, which monkeypatches `date.today` to a fixed date.
"""

import math
import subprocess
from datetime import date
from typing import override

import duckdb
import pandas as pd
import pytest

from src.constants import GOLD_ROLLING_FEATURES, PROFILES_TABLE, ROOT
from src.db.client import execute_df
from src.features import inference
from src.features.columns import DIFF_COLS, FEATURE_COLS
from src.features.inference import build_inference_features
from src.utils import run_dbt_build

DB_PATH = ROOT / "data" / "tennis.duckdb"

# All seeded matches are in 2026 (2026-03-15 .. 2026-07-15); a fixed as-of date
# after the last match exercises the full snapshot history deterministically.
AS_OF_AFTER_ALL_MATCHES = date(2026, 9, 1)


def _gold_rolling_ready() -> bool:
    """True when the seeded gold rolling table already has rows."""
    if not DB_PATH.exists():
        return False
    conn = duckdb.connect(str(DB_PATH))
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM {GOLD_ROLLING_FEATURES}").fetchone()
    except Exception:
        return False
    finally:
        conn.close()
    return bool(row is not None and row[0] > 0)


@pytest.fixture(scope="session", autouse=True)
def seeded_gold_db():
    """Rebuild the gold tables once if missing/empty; skip otherwise.

    Runs the bootstrap steps behind `just db-reset` + `just db-seed` +
    `just db-dbt`: init the schemas, seed ~100 real matches via
    `infra/duckdb/seed.py` (raw ATP -> bronze + filtered ATP profiles), then
    `dbt build` the gold models via the shared `run_dbt_build()` helper. The
    seed's best-effort Wikipedia enrichment is skipped with `--offline` so
    tests never depend on live Wikipedia. Subprocess output is captured so a
    bootstrap failure reports the exact failing command.
    """
    if _gold_rolling_ready():
        yield
        return

    commands = [
        ["uv", "run", "python", "infra/duckdb/initialize_schemas.py", "init"],
        ["uv", "run", "python", "infra/duckdb/seed.py", "--offline"],
    ]
    for command in commands:
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"gold bootstrap command failed: {' '.join(command)}\n"
                f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
            )
    run_dbt_build()

    # The singleton connection may have opened the pre-rebuild DB file; drop
    # it so the next get_conn() reconnects to the freshly built one.
    import src.db.client as db_client

    db_client._conn = None
    yield


def test_output_schema_contract():
    """Exact column order [*FEATURE_COLS, "player_id", "opponent_id"], one row."""
    out = build_inference_features("S0AG", "Z355", "clay", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    assert out.columns.tolist() == [*FEATURE_COLS, "player_id", "opponent_id"]
    assert len(out.columns) == 56  # 54 features + 2 ids
    assert len(out) == 1
    assert out["player_id"].dtype == object
    assert out["opponent_id"].dtype == object


@pytest.mark.parametrize("surface", ["clay", "grass", "hard"])
def test_two_known_players_each_surface(surface):
    """Known-vs-known row: valid one-hot, canonical ids, finite features."""
    out = build_inference_features("S0AG", "Z355", surface, as_of_date=AS_OF_AFTER_ALL_MATCHES)
    row = out.iloc[0]
    assert row["player_id"] == "S0AG"  # 'S0AG' < 'Z355'
    assert row["opponent_id"] == "Z355"
    expected_one_hots = {"is_clay": 0, "is_grass": 0, "is_hard": 0}
    expected_one_hots[f"is_{surface}"] = 1
    assert row["is_clay"] == expected_one_hots["is_clay"]
    assert row["is_grass"] == expected_one_hots["is_grass"]
    assert row["is_hard"] == expected_one_hots["is_hard"]
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


def test_known_players_profile_features():
    """Profile-derived features for two known players, in canonical order.

    S0AG (Sinner): height 191, right-handed, turned pro 2018. Z355 (Zverev):
    height 198, right-handed, turned pro 2013. Canonical order puts S0AG on
    the player_* side; years_pro is years-pro AT the as-of date (2026 - year),
    not the raw turned_pro year.
    """
    out = build_inference_features("S0AG", "Z355", "clay", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    row = out.iloc[0]
    assert row["player_height"] == 191.0
    assert row["opponent_height"] == 198.0
    assert row["height_diff"] == -7.0
    assert row["player_is_left_handed"] == 0.0
    assert row["opponent_is_left_handed"] == 0.0
    assert row["handedness_diff"] == 0.0
    assert row["player_years_pro"] == 8.0  # 2026 - 2018
    assert row["opponent_years_pro"] == 13.0  # 2026 - 2013
    assert row["years_pro_diff"] == -5.0


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
    assert row["years_pro_diff"] == -5.0
    for col in FEATURE_COLS:
        assert math.isfinite(row[col]), f"{col} is not finite: {row[col]!r}"


def test_historical_as_of_excludes_later_snapshots():
    """Regression: the as-of lookup must use the newest snapshot strictly before
    the date, not the first or the overall latest.

    S0AG's seeded snapshot sequence: #1 = 2026-04-12 (Monte Carlo, won),
    #2..#3 won, #4 = 2026-05-28 (Roland Garros, loss), #5 = 2026-06-29
    (Wimbledon R128, won). At as_of 2026-06-30 the newest strictly-before
    snapshot is #5 with win_rate_10 = 4/5 = 0.8; snapshot #6 (2026-07-01) is
    one-match stale (5/6 = 0.833). A builder that used snapshot #1 (win_rate
    1.0) or the overall latest (0.9) fails loudly.
    """
    out = build_inference_features("S0AG", "Z355", "hard", as_of_date=date(2026, 6, 30))
    assert out.loc[0, "player_win_rate_10"] == 0.8
    # Cross-check the expected snapshot directly in the gold table.
    snapshot = execute_df(
        f"SELECT player_match_number, win_rate_10 FROM {GOLD_ROLLING_FEATURES} "
        "WHERE player_id = ? AND snapshot_date < ?::DATE "
        "ORDER BY player_match_number DESC LIMIT 1",
        ["S0AG", "2026-06-30"],
    ).iloc[0]
    assert snapshot["player_match_number"] == 5
    assert snapshot["win_rate_10"] == 0.8
    assert out.loc[0, "player_win_rate_10"] == snapshot["win_rate_10"]


class _FixedTodayDate(date):
    """datetime.date subclass whose today() returns a fixed date."""

    @classmethod
    @override
    def today(cls) -> date:
        return date(2026, 9, 1)


def test_default_today_fecha(monkeypatch):
    """Default as_of_date (date.today) builds the same row as an explicit date.

    `date.today` cannot be monkeypatched on the immutable datetime.date C type,
    so the module-level `date` name in src.features.inference is swapped for a
    subclass with a fixed `today()` (monkeypatch restores it afterwards). The
    explicit-date row is built BEFORE the patch so its isinstance checks see
    the real date class; the default branch (`as_of_date is None`) is the code
    path under test.
    """
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
    """One unknown player is imputed from the global pool; canonical ids hold.

    In the reversed order 'UNKNOWN_ID' would be the raw lower id only if it
    sorted below 'S0AG' — it does not, so the canonical lower id 'S0AG' still
    wins the player_* side and the unknown gets the opponent side (Bento's bio
    lookup then misses -> zero bio vector, preserved by id passthrough).
    """
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
    # The opponent side equals the on-demand global pool aggregates
    # (MEDIAN for ranking/streak, MEAN for rates), same SQL as the builder.
    pool = execute_df(
        "SELECT MEDIAN(latest_player_ranking) AS latest_player_ranking, "
        "MEDIAN(win_streak) AS win_streak, "
        "AVG(win_rate_10) AS win_rate_10, "
        "AVG(hard_win_rate_10) AS hard_win_rate_10 "
        f"FROM {GOLD_ROLLING_FEATURES} WHERE snapshot_date < ?::DATE",
        ["2026-09-01"],
    ).iloc[0]
    assert row["opponent_ranking"] == round(float(pool["latest_player_ranking"]))
    assert row["opponent_win_streak"] == int(float(pool["win_streak"]))
    assert row["opponent_win_rate_10"] == pytest.approx(pool["win_rate_10"])
    assert row["opponent_surface_win_rate_10"] == pytest.approx(pool["hard_win_rate_10"])
    # Profile-derived features for the unknown player come from the on-demand
    # aggregate over ALL profiles (mean height / left-handed rate / years-pro
    # at the as-of date), so they are finite and non-NaN.
    profile_pool = execute_df(
        "SELECT AVG(height) AS avg_height, "
        "AVG(CASE WHEN handedness = 'L' THEN 1 ELSE 0 END) AS left_handed_rate, "
        "AVG(2026 - turned_pro) AS avg_years_pro "
        f"FROM {PROFILES_TABLE}",
    ).iloc[0]
    assert row["opponent_height"] == pytest.approx(profile_pool["avg_height"])
    assert row["opponent_is_left_handed"] == pytest.approx(profile_pool["left_handed_rate"])
    assert row["opponent_years_pro"] == pytest.approx(profile_pool["avg_years_pro"])
    assert math.isfinite(row["height_diff"])
    assert math.isfinite(row["handedness_diff"])
    assert math.isfinite(row["years_pro_diff"])


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


@pytest.mark.parametrize("surface", ["Clay", "CLAY", "carpet", "", None])
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
    agg = {"avg_height": 185.0, "left_handed_rate": 0.08, "avg_years_pro": 8.0}
    values = inference._profile_values("S0AG", date(2026, 9, 1), agg)
    assert values["is_left_handed"] == pytest.approx(0.08)
    # Non-NULL cells are still read directly.
    assert values["height"] == 191.0
    assert values["years_pro"] == 8.0  # 2026 - 2018
