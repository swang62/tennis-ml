"""Focused pytest tests for the ID-based inference feature builder.

Target: `src.features.inference.build_inference_features`.

Tests run against the seeded DuckDB at `data/tennis.duckdb` (built from
`infra/duckdb/init.sql` + `infra/duckdb/seed.sql` + the dbt gold models). A
session-scoped autouse fixture rebuilds the gold tables once if they are
missing or empty; otherwise it skips so repeated runs stay fast. All
date-dependent tests pass an explicit `as_of_date` for determinism, except the
default-today test, which monkeypatches `date.today` to a fixed date.
"""

import math
import subprocess
from datetime import date

import duckdb
import pandas as pd
import pytest

from src.constants import ROOT
from src.db.client import execute_df
from src.features.inference import build_inference_features
from src.features.rolling import DIFF_COLS, FEATURE_COLS

DB_PATH = ROOT / "data" / "tennis.duckdb"

# All seeded matches are in 2026 (2026-01-13 .. 2026-08-27); a fixed as-of date
# after the last match exercises the full snapshot history deterministically.
AS_OF_AFTER_ALL_MATCHES = date(2026, 9, 1)


def _gold_rolling_ready() -> bool:
    """True when the seeded gold rolling table already has rows."""
    if not DB_PATH.exists():
        return False
    conn = duckdb.connect(str(DB_PATH))
    try:
        row = conn.execute("SELECT COUNT(*) FROM gold.player_rolling_features").fetchone()
    except Exception:
        return False
    finally:
        conn.close()
    return bool(row is not None and row[0] > 0)


@pytest.fixture(scope="session", autouse=True)
def seeded_gold_db():
    """Rebuild the gold tables once if missing/empty; skip otherwise.

    Runs the bootstrap steps behind `just db-reset` + `just db-seed` +
    `just db-dbt`: init the schemas, seed the 20-match bronze fixture, then
    `dbt build` the gold models. Subprocess output is captured so a bootstrap
    failure reports the exact failing command.
    """
    if _gold_rolling_ready():
        yield
        return

    commands = [
        ["uv", "run", "python", "infra/duckdb/run_init.py", "init"],
        ["uv", "run", "python", "infra/duckdb/run_init.py", "seed"],
        ["uv", "run", "dbt", "build", "--project-dir", "dbt", "--profiles-dir", "dbt"],
    ]
    for command in commands:
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"gold bootstrap command failed: {' '.join(command)}\n"
                f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
            )

    # The singleton connection may have opened the pre-rebuild DB file; drop
    # it so the next get_conn() reconnects to the freshly built one.
    import src.db.client as db_client

    db_client._conn = None
    yield


def test_output_schema_contract():
    """Exact column order [*FEATURE_COLS, "player_id", "opponent_id"], one row."""
    out = build_inference_features("S0AG", "Z355", "clay", as_of_date=AS_OF_AFTER_ALL_MATCHES)
    assert out.columns.tolist() == [*FEATURE_COLS, "player_id", "opponent_id"]
    assert len(out.columns) == 47  # 45 features + 2 ids
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


def test_historical_as_of_excludes_later_snapshots():
    """Regression: the as-of lookup must use snapshot #1, not #2 or the latest.

    S0AG's snapshot sequence: #1 = 2026-01-13 (2026-ao-r1-002), #2 = 2026-01-15
    (2026-ao-r1-005). At as_of 2026-01-14 only snapshot #1 is strictly before
    the date; a builder that used snapshot #2 (or the latest snapshot) is
    one-match stale. S0AG won its first match, so snapshot #1 win_rate_10 is
    1.0. This test fails loudly if the builder ever drifts to snapshot #2.
    """
    out = build_inference_features("S0AG", "Z355", "hard", as_of_date=date(2026, 1, 14))
    assert out.loc[0, "player_win_rate_10"] == 1.0
    # Cross-check the expected snapshot directly in the gold table.
    snapshot = execute_df(
        "SELECT player_match_number, win_rate_10 FROM gold.player_rolling_features "
        "WHERE player_id = ? AND snapshot_date < ?::DATE "
        "ORDER BY player_match_number DESC LIMIT 1",
        ["S0AG", "2026-01-14"],
    ).iloc[0]
    assert snapshot["player_match_number"] == 1
    assert snapshot["win_rate_10"] == 1.0
    assert out.loc[0, "player_win_rate_10"] == snapshot["win_rate_10"]


class _FixedTodayDate(date):
    """datetime.date subclass whose today() returns a fixed date."""

    @classmethod
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
        "FROM gold.player_rolling_features WHERE snapshot_date < ?::DATE",
        ["2026-09-01"],
    ).iloc[0]
    assert row["opponent_ranking"] == round(float(pool["latest_player_ranking"]))
    assert row["opponent_win_streak"] == int(float(pool["win_streak"]))
    assert row["opponent_win_rate_10"] == pytest.approx(pool["win_rate_10"])
    assert row["opponent_surface_win_rate_10"] == pytest.approx(pool["hard_win_rate_10"])


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


@pytest.mark.parametrize("tournament_level", [-1, 5, 9, "4"])
def test_invalid_tournament_level_raises(tournament_level):
    with pytest.raises((ValueError, TypeError)):
        build_inference_features(
            "S0AG",
            "Z355",
            "hard",
            as_of_date=AS_OF_AFTER_ALL_MATCHES,
            tournament_level=tournament_level,
        )


@pytest.mark.parametrize("round_encoded", [-1, 8, 9, "7"])
def test_invalid_round_encoded_raises(round_encoded):
    with pytest.raises((ValueError, TypeError)):
        build_inference_features(
            "S0AG",
            "Z355",
            "hard",
            as_of_date=AS_OF_AFTER_ALL_MATCHES,
            round_encoded=round_encoded,
        )


@pytest.mark.parametrize("tournament_level", list(range(0, 5)))
def test_valid_tournament_levels_accepted(tournament_level):
    out = build_inference_features(
        "S0AG",
        "Z355",
        "hard",
        as_of_date=AS_OF_AFTER_ALL_MATCHES,
        tournament_level=tournament_level,
    )
    assert out.iloc[0]["tournament_level"] == tournament_level


@pytest.mark.parametrize("round_encoded", list(range(0, 8)))
def test_valid_round_encodings_accepted(round_encoded):
    out = build_inference_features(
        "S0AG",
        "Z355",
        "hard",
        as_of_date=AS_OF_AFTER_ALL_MATCHES,
        round_encoded=round_encoded,
    )
    assert out.iloc[0]["round_encoded"] == round_encoded


@pytest.mark.parametrize(
    "as_of, expected",
    [
        # A0E2's last prior match is 2026-01-15, 54 days before 2026-03-10, so
        # the [2026-02-08, 2026-03-10) window contains zero A0E2 matches. The
        # old ROWS-frame formulation returned 3 here (every preceding AO match,
        # regardless of date) — this regression case catches that exact bug.
        (date(2026, 3, 10), 0),
        # [2025-12-16, 2026-01-15): A0E2's 2026-01-13 and 2026-01-14 matches
        # are inside; the 2026-01-15 match itself is excluded (strict <).
        (date(2026, 1, 15), 2),
    ],
    ids=["30d-window-empty", "30d-window-two"],
)
def test_matches_30d_window_regression(as_of, expected):
    """Regression: matches_30d uses a real date window, not a ROWS frame."""
    out = build_inference_features("A0E2", "UNKNOWN_PLAYER", "hard", as_of_date=as_of)
    row = out.iloc[0]
    assert row["player_id"] == "A0E2"  # 'A0E2' < 'UNKNOWN_PLAYER'
    assert row["opponent_id"] == "UNKNOWN_PLAYER"
    assert row["player_matches_30d"] == expected
