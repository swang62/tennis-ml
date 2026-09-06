"""Hermetic tests for scrape parsing and date math."""

import re
from datetime import date

import pytest

import src.flows.rankings as scrape
import src.utils.scrape as scrape_mod

# Mirror the real #dateWeek-filter DOM so mocks exercise date parsing, not a boolean stub.
_FILTER_OPTION_RE = re.compile(r'<option[^>]*value="([^"]*)"[^>]*>(.*?)</option>', re.S)


def _week_in_filter_real(filter_html: str, wanted: str, latest: str) -> bool:
    """Mirror ``_week_in_filter``: match on option value OR text, dash or dot format."""
    wanted_n = re.sub(r"[^0-9]", "", wanted)
    latest_n = re.sub(r"[^0-9]", "", latest)
    for value, text in _FILTER_OPTION_RE.findall(filter_html):
        fields = [value, text.strip()]
        norms = [re.sub(r"[^0-9]", "", f) for f in fields if f]
        if wanted_n in norms:
            return True
        if "Current Week" in fields and wanted_n == latest_n:
            return True
    return False


# Realistic #dateWeek-filter markup mirroring atptour.com: the latest week is
# "Current Week", older weeks carry their date as the option value.
_REAL_FILTER_HTML = (
    '<select id="dateWeek-filter">'
    '<option value="Current Week">2026.08.10</option>'
    '<option value="2026-08-03">2026.08.03</option>'
    '<option value="2026-07-27">2026.07.27</option>'
    "</select>"
)


class _FrozenToday(date):
    """date subclass with a fixed today(), for deterministic date-math tests."""

    _today: date

    @classmethod
    def today(cls):
        return cls._today


# ── Date math ────────────────────────────────────────────────────


def test_latest_completed_monday():
    assert scrape.latest_completed_monday(date(2026, 1, 5)) == date(2026, 1, 5)
    assert scrape.latest_completed_monday(date(2026, 1, 6)) == date(2026, 1, 5)
    assert scrape.latest_completed_monday(date(2026, 1, 11)) == date(2026, 1, 5)


def test_ranking_mondays_after():
    wm = date(2026, 1, 5)
    assert scrape.ranking_mondays_after(wm, date(2026, 1, 26)) == [
        date(2026, 1, 12),
        date(2026, 1, 19),
        date(2026, 1, 26),
    ]


def test_ranking_mondays_after_empty_when_current():
    wm = date(2026, 1, 19)
    assert scrape.ranking_mondays_after(wm, date(2026, 1, 19)) == []
    assert scrape.ranking_mondays_after(wm, date(2026, 1, 5)) == []


def test_missing_weeks_default_is_only_previous_week(monkeypatch):
    # Without end_date, scan from the watermark through the latest completed Monday.
    monkeypatch.setattr(scrape, "stored_ranking_mondays", lambda: {date(2026, 1, 5)})
    _FrozenToday._today = date(2026, 1, 20)
    monkeypatch.setattr(scrape, "date", _FrozenToday)
    watermark, weeks = scrape.missing_ranking_mondays.fn()
    assert watermark == date(2026, 1, 5)
    assert weeks == [date(2026, 1, 12), date(2026, 1, 19)]


def test_missing_weeks_includes_current_monday(monkeypatch):
    monkeypatch.setattr(scrape, "stored_ranking_mondays", lambda: {date(2026, 1, 5)})
    _FrozenToday._today = date(2026, 1, 19)
    monkeypatch.setattr(scrape, "date", _FrozenToday)

    _watermark, weeks = scrape.missing_ranking_mondays.fn()

    assert weeks == [date(2026, 1, 12), date(2026, 1, 19)]


def test_missing_weeks_resume_across_year_boundary(monkeypatch):
    monkeypatch.setattr(scrape, "stored_ranking_mondays", lambda: {date(2025, 12, 29)})
    _FrozenToday._today = date(2026, 1, 21)
    monkeypatch.setattr(scrape, "date", _FrozenToday)

    watermark, weeks = scrape.missing_ranking_mondays.fn()

    assert watermark == date(2025, 12, 29)
    assert weeks == [date(2026, 1, 5), date(2026, 1, 12), date(2026, 1, 19)]


def test_missing_weeks_backfill_to_end_date(monkeypatch):
    # Scan from the watermark; stored weeks are skipped by presence alone.
    monkeypatch.setattr(scrape, "stored_ranking_mondays", lambda: {date(2026, 1, 5)})
    _FrozenToday._today = date(2026, 2, 3)
    monkeypatch.setattr(scrape, "date", _FrozenToday)
    watermark, weeks = scrape.missing_ranking_mondays.fn(end_date=date(2026, 1, 26))
    assert watermark == date(2026, 1, 5)
    assert weeks == [date(2026, 1, 12), date(2026, 1, 19), date(2026, 1, 26)]


def test_missing_weeks_stored_week_skipped_presence_only(monkeypatch):
    # Stored weeks are skipped by presence alone; absent weeks are fetched.
    monkeypatch.setattr(
        scrape,
        "stored_ranking_mondays",
        lambda: {date(2026, 1, 5), date(2026, 1, 12)},
    )
    _FrozenToday._today = date(2026, 2, 3)
    monkeypatch.setattr(scrape, "date", _FrozenToday)
    watermark, weeks = scrape.missing_ranking_mondays.fn(end_date=date(2026, 1, 26))
    assert watermark == date(2026, 1, 12)
    assert weeks == [date(2026, 1, 19), date(2026, 1, 26)]


def test_missing_weeks_quits_when_end_date_at_watermark(monkeypatch):
    monkeypatch.setattr(scrape, "stored_ranking_mondays", lambda: {date(2026, 1, 5)})
    _FrozenToday._today = date(2026, 2, 3)
    monkeypatch.setattr(scrape, "date", _FrozenToday)
    _watermark, weeks = scrape.missing_ranking_mondays.fn(end_date=date(2026, 1, 5))
    assert weeks == []


def test_missing_weeks_explicit_start_snaps_and_clamps_to_watermark(monkeypatch):
    # Explicit starts snap to Mondays and cannot reach weeks before the watermark.
    monkeypatch.setattr(scrape, "stored_ranking_mondays", lambda: {date(2026, 1, 19)})
    _FrozenToday._today = date(2026, 2, 3)
    monkeypatch.setattr(scrape, "date", _FrozenToday)
    watermark, weeks = scrape.missing_ranking_mondays.fn(
        start_date=date(2026, 1, 6), end_date=date(2026, 2, 2)
    )
    assert watermark == date(2026, 1, 19)
    assert weeks == [date(2026, 1, 26), date(2026, 2, 2)]


def test_missing_weeks_no_week_before_stored_max_ever_fetched(monkeypatch):
    # A historical range before the watermark produces no weeks.
    monkeypatch.setattr(scrape, "stored_ranking_mondays", lambda: {date(2026, 1, 26)})
    _FrozenToday._today = date(2026, 2, 3)
    monkeypatch.setattr(scrape, "date", _FrozenToday)
    watermark, weeks = scrape.missing_ranking_mondays.fn(
        start_date=date(2026, 1, 5), end_date=date(2026, 1, 19)
    )
    assert watermark == date(2026, 1, 26)
    assert weeks == []


def test_missing_weeks_explicit_historical_start_resumes_after_watermark(monkeypatch):
    # Starts before the current year still resume after the stored watermark.
    monkeypatch.setattr(scrape, "stored_ranking_mondays", lambda: {date(2026, 1, 5)})
    _FrozenToday._today = date(2026, 2, 3)
    monkeypatch.setattr(scrape, "date", _FrozenToday)
    watermark, weeks = scrape.missing_ranking_mondays.fn(
        start_date=date(2025, 1, 6), end_date=date(2026, 1, 12)
    )
    assert watermark == date(2026, 1, 5)
    assert weeks == [date(2026, 1, 12)]


def test_missing_weeks_end_date_after_latest_completed_monday_clamped(monkeypatch):
    # Clamp end_date to the latest completed Monday to exclude future weeks.
    monkeypatch.setattr(scrape, "stored_ranking_mondays", lambda: {date(2026, 1, 5)})
    _FrozenToday._today = date(2026, 2, 3)
    monkeypatch.setattr(scrape, "date", _FrozenToday)
    _watermark, weeks = scrape.missing_ranking_mondays.fn(end_date=date(2026, 2, 9))
    assert weeks == [
        date(2026, 1, 12),
        date(2026, 1, 19),
        date(2026, 1, 26),
        date(2026, 2, 2),
    ]


def test_missing_weeks_end_only_presence_based(monkeypatch):
    # end_date-only: an absent week is fetched even though the old gate would
    # have called it complete (>= 100 rows); only stored weeks are skipped.
    monkeypatch.setattr(scrape, "stored_ranking_mondays", lambda: {date(2026, 1, 5)})
    _FrozenToday._today = date(2026, 2, 3)
    monkeypatch.setattr(scrape, "date", _FrozenToday)
    watermark, weeks = scrape.missing_ranking_mondays.fn(end_date=date(2026, 1, 19))
    assert watermark == date(2026, 1, 5)
    assert weeks == [date(2026, 1, 12), date(2026, 1, 19)]


def test_missing_weeks_start_only_presence_based(monkeypatch):
    # start_date-only (upper bound = current or most recent completed Monday): the stored
    # week is skipped, absent weeks are fetched.
    monkeypatch.setattr(scrape, "stored_ranking_mondays", lambda: {date(2026, 1, 5)})
    _FrozenToday._today = date(2026, 1, 26)
    monkeypatch.setattr(scrape, "date", _FrozenToday)
    watermark, weeks = scrape.missing_ranking_mondays.fn(start_date=date(2026, 1, 12))
    assert watermark == date(2026, 1, 5)
    assert weeks == [date(2026, 1, 12), date(2026, 1, 19), date(2026, 1, 26)]


def test_force_backfill_includes_stored_and_missing_weeks(monkeypatch):
    monkeypatch.setattr(
        scrape,
        "stored_ranking_mondays",
        lambda: {date(2026, 5, 4), date(2026, 5, 18)},
    )
    _FrozenToday._today = date(2026, 5, 20)
    monkeypatch.setattr(scrape, "date", _FrozenToday)

    watermark, weeks = scrape.missing_ranking_mondays.fn(
        start_date=date(2026, 5, 1), end_date=date(2026, 5, 18), force=True
    )

    assert watermark is None
    assert weeks == [date(2026, 5, 4), date(2026, 5, 11), date(2026, 5, 18)]


# ── Identity translation ─────────────────────────────────────────


def test_translate_keeps_live_atp_ids_without_legacy_map():
    rows = [
        {"player_id": "S0AG", "rank": 1, "points": 1000, "name": "J. Sinner"},
        {"player_id": "ZZ99", "rank": 2, "points": 900, "name": "Nobody"},
        {"player_id": "A0E2", "rank": 3, "points": 800, "name": "C. Alcaraz"},
        {"player_id": "S0AG", "rank": 201, "points": 1, "name": "Dup"},
    ]
    frame, skipped = scrape.translate_rank_rows(rows)
    assert frame["player_id"].tolist() == ["S0AG", "ZZ99", "A0E2"]
    assert len(skipped) == 1


# ── Backfill failure guard ───────────────────────────────────────


def test_backfill_that_cannot_access_the_site_fails():
    """A backfill with no parseable weeks fails instead of reporting no changes."""
    weeks = [date(2026, 1, 12), date(2026, 1, 19)]
    with pytest.raises(RuntimeError, match="could not access or parse"):
        scrape._fail_if_no_data_found(False, weeks)


def test_backfill_that_found_data_does_not_fail():
    # Even a week that parsed but wrote 0 rows (already present) found data.
    scrape._fail_if_no_data_found(True, [date(2026, 1, 12)])


# ── Player-profile discovery (shared by rankings and matches) ──────


def _overview_html(profile_id="XQ999", ioc="ITA", **overrides):
    body = "background\n"
    if "ids" not in overrides:
        body += f'\n<a href="/en/players/test-slug/{profile_id.lower()}/overview">overview</a>'
    if "dob" not in overrides:
        body += "\nAge 30 (1996/04/12)"
    if "weight" not in overrides:
        body += "\nWeight 185lb (84kg)"
    if "height" not in overrides:
        body += "\nHeight (183cm)"
    if "pro" not in overrides:
        body += "\nTurned pro 2018"
    if "plays" not in overrides:
        body += "\nPlays Right-handed, Two-handed Backhand"
    if ioc:  # no flag sprite -> IOC resolves to the UNK sentinel
        body += f'\n<use href="/images/flags.svg#flag-{ioc.lower()}">'
    for key, value in overrides.items():
        body += f"\n{key}: {value}"
    return body


class _FakePage:
    """Shared browser page double that serves canned HTML."""

    def __init__(self, html=""):
        self._html = html

    def goto(self, _url, **_kwargs):
        pass

    def wait_for_function(self, _script, **_kwargs):
        pass

    def content(self):
        return self._html


def test_parse_player_overview_valid_page():
    """A valid overview yields a full candidate, id uppercased and IOC parsed."""
    parsed, reason = scrape_mod.parse_player_overview(
        _overview_html(), "xq999", {"player": "Test Player"}
    )
    assert reason == ""
    assert parsed is not None
    assert parsed["id"] == "XQ999"
    assert parsed["player"] == "Test Player"
    assert parsed["birthdate"] == "19960412"
    assert parsed["weight"] == "84"
    assert parsed["height"] == "183"
    assert parsed["turnedpro"] == "2018"
    assert parsed["hand"] == "R"
    assert parsed["backhand"] == "2H"
    assert parsed["ioc"] == "ITA"


def test_parse_player_overview_extracts_rendered_bio_fields():
    html = """
    <title>Max Alcala Gurri | Overview | ATP Tour | Tennis</title>
    <a href="/en/players/max-alcala-gurri/a0ea/overview">x</a>
    <span>Age</span><span>23</span><span>(2002/09/11)</span>
    <span>Weight</span><span>149 lbs (68kg)</span>
    <span>Height</span><span>5'10\" (178cm)</span>
    <span>Birthplace</span><span>Barcelona</span>
    <span>Plays</span><span>Right-Handed, Two-Handed Backhand</span>
    <use href="/images/flags.svg#flag-esp">
    """

    parsed, reason = scrape_mod.parse_player_overview(html, "A0EA", {"player": "M. Alcala Gurri"})

    assert reason == ""
    assert parsed == {
        "id": "A0EA",
        "player": "Max Alcala Gurri",
        "atpname": "Max Alcala Gurri",
        "slug": "",
        "birthdate": "20020911",
        "weight": "68",
        "height": "178",
        "turnedpro": "",
        "hand": "R",
        "backhand": "2H",
        "birthplace": "Barcelona",
        "coaches": "",
        "ioc": "ESP",
    }


def test_parse_player_overview_does_not_take_footer_year_for_missing_turned_pro():
    parsed, reason = scrape_mod.parse_player_overview(
        _overview_html(pro="") + "\nTurned pro Follow player\n© Copyright 1994 - 2026 ATP Tour",
        "XQ999",
        {"player": "Test Player"},
    )

    assert reason == ""
    assert parsed is not None
    assert parsed["turnedpro"] == ""


def test_parse_player_overview_missing_display_name():
    parsed, reason = scrape_mod.parse_player_overview(_overview_html(), "XQ999", {"player": ""})
    assert parsed is None
    assert "missing display name" in reason


def test_parse_player_overview_profile_id_mismatch():
    """The page must render the player we navigated to: an embedded id that
    disagrees with the link id is a mismatch, never a discovery."""
    parsed, reason = scrape_mod.parse_player_overview(
        _overview_html(ids="\n<a href='/en/players/other/AAAAAA/overview'>x</a>"),
        "XQ999",
        {"player": "Test Player"},
    )
    assert parsed is None
    assert "does not match link id XQ999" in reason


def test_parse_player_overview_unknown_ioc_becomes_unk():
    """No flag sprite -> IOC resolves to the UNK sentinel, not a failure."""
    parsed, reason = scrape_mod.parse_player_overview(
        _overview_html(ioc=""), "XQ999", {"player": "Test Player"}
    )
    assert reason == ""
    assert parsed is not None
    assert parsed["ioc"] == "UNK"
