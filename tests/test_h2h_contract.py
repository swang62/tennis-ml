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


def test_model_sources_exposure_from_bounded_recent_five():
    # h2h_exposure is the count of the FIVE most recent unordered-pair
    # meetings: fed by the bounded recent-5 pair_meetings window. There is no
    # lifetime/unbounded count anywhere in the model.
    assert "pair_lifetime" not in MODEL
    assert "lifetime_exposure" not in MODEL
    assert "COALESCE(h.recent_meetings, 0) AS h2h_exposure" in MODEL


def test_model_advantages_use_bounded_strictly_prior_window():
    # The recent-5 advantages stay bounded and every meeting is strictly prior.
    assert "LIMIT 5" in MODEL
    assert "WHERE rn <= 5" in MODEL
    assert MODEL.count("meeting.match_date < current_match.match_date") >= 2


def test_model_does_not_truncate_advantages():
    # No truncation at the output boundary; advantages are stored at full
    # arithmetic precision.
    assert "TRUNC" not in MODEL


def test_test_re_derives_exposure_capped_at_five():
    # exposure must be the recent-5 count (identical to the advantage window).
    assert "COUNT(*) FILTER (WHERE rn <= 5) AS exp_recent_meetings" in TEST
    assert "exp_exposure" not in TEST
    assert "mf.h2h_exposure IS DISTINCT FROM COALESCE(d.exp_recent_meetings, 0)" in TEST


def test_test_advantage_uses_recent_window_and_no_trunc():
    # The advantage denominator is the recent-5 meeting count, and the derived
    # value is compared at full precision (the model no longer truncates).
    assert "exp_recent_meetings" in TEST
    assert "TRUNC" not in TEST
    assert "::DOUBLE PRECISION)) > 1e-6" in TEST
