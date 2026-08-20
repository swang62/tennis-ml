"""Hermetic guard for the H2H no-current-match contract.

No live database: reads the gold model and its singular leakage test from
disk and pins the semantics statically (the same approach as
test_dbt_incremental.py). Regression this guards against: h2h_exposure was
changed from a recent-5 count to a LIFETIME strictly-prior count while the
test still re-derived it capped at 5, and the advantage comparison used a
1e-6 tolerance against a value the model TRUNC()s to 5 decimals.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = (ROOT / "dbt/models/gold/match_features.sql").read_text()
TEST = (ROOT / "dbt/tests/gold/match_features_h2h_no_current_match.sql").read_text()


def test_model_sources_exposure_from_unbounded_lifetime_count():
    # h2h_exposure is the LIFETIME unordered-pair count: fed by the separate
    # uncapped pair_lifetime CTE, never by the bounded recent-5 window.
    assert "pair_lifetime AS" in MODEL
    assert "lifetime_exposure" in MODEL
    assert "COALESCE(l.lifetime_exposure, 0) AS h2h_exposure" in MODEL
    pair_lifetime = MODEL[MODEL.index("pair_lifetime AS") :]
    pair_lifetime = pair_lifetime[: pair_lifetime.index("-- Directional H2H")]
    assert "LIMIT" not in pair_lifetime  # uncapped
    assert "meeting.match_date < current_match.match_date" in pair_lifetime


def test_model_advantages_use_bounded_strictly_prior_window():
    # The recent-5 advantages stay bounded and every meeting is strictly prior.
    assert "LIMIT 5" in MODEL
    assert "WHERE rn <= 5" in MODEL
    assert MODEL.count("meeting.match_date < current_match.match_date") >= 3


def test_model_truncates_advantage_to_five_decimals():
    assert "TRUNC(((COALESCE(h.wins_for_player, 0) + 1.0)" in MODEL
    assert "::NUMERIC, 5)" in MODEL


def test_test_re_derives_exposure_uncapped():
    # exposure must be the LIFETIME count (no recent-5 cap); the recent-5 cap
    # applies only to the advantage window.
    assert "COUNT(*) AS exp_exposure" in TEST
    assert "COUNT(*) FILTER (WHERE rn <= 5) AS exp_recent_meetings" in TEST
    derived_block = TEST[TEST.index("derived AS") :]
    derived_block = derived_block[: derived_block.index("GROUP BY")]
    from_to_group = derived_block[derived_block.index("FROM prior_meeting_rows") :]
    assert "WHERE" not in from_to_group  # exposure count is not row-filtered


def test_test_advantage_uses_recent_window_and_mirrors_trunc():
    # The advantage denominator is the recent-5 meeting count, and the derived
    # value is TRUNC'd identically to the model's stored column.
    assert "exp_recent_meetings" in TEST
    assert "::NUMERIC, 5)::DOUBLE PRECISION) > 1e-6" in TEST
