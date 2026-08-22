"""Fast, hermetic tests for the scrape flow parser and date math.

No Prefect server, no MLflow, no browser, no external fixture files — just pure logic.
"""

import re
from datetime import date

import pytest

import src.flows.rankings as scrape
import src.utils.scrape as scrape_mod

# Real ATP #dateWeek-filter structure (observed on atptour.com): each option's
# value is the week date (YYYY-MM-DD) or the literal "Current Week" for the
# latest week; the visible text is YYYY.MM.DD. These helpers let hermetic
# mocks answer ``_week_in_filter`` with the same logic the real page runs, so
# the fixtures reflect the actual DOM instead of a stubbed boolean.
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
    assert scrape.latest_completed_monday(date(2026, 1, 5)) == date(2025, 12, 29)
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
    # Watermark 2026-01-05; no end_date -> stop at the most recent completed
    # Monday (2026-01-19, today frozen to 2026-01-20), so the scan covers the
    # watermark..previous-week window only (watermark + 7 days default start).
    monkeypatch.setattr(scrape, "stored_ranking_mondays", lambda: {date(2026, 1, 5)})
    _FrozenToday._today = date(2026, 1, 20)
    monkeypatch.setattr(scrape, "date", _FrozenToday)
    watermark, weeks = scrape.missing_ranking_mondays.fn()
    assert watermark == date(2026, 1, 5)
    assert weeks == [date(2026, 1, 12), date(2026, 1, 19)]


def test_missing_weeks_no_args_does_not_use_completeness(monkeypatch):
    # The count-based completeness gate is gone entirely: presence is the only
    # completeness test for every path, so a no-arg run cannot refetch a stored
    # week. Asserting the old helpers no longer exist proves the semantics.
    assert not hasattr(scrape, "stored_ranking_monday_counts")
    assert not hasattr(scrape, "COMPLETE_RANKING_MIN_ROWS")
    monkeypatch.setattr(scrape, "stored_ranking_mondays", lambda: {date(2026, 1, 5)})
    _FrozenToday._today = date(2026, 1, 20)
    monkeypatch.setattr(scrape, "date", _FrozenToday)
    watermark, weeks = scrape.missing_ranking_mondays.fn()
    assert watermark == date(2026, 1, 5)
    assert weeks == [date(2026, 1, 12), date(2026, 1, 19)]


def test_missing_weeks_backfill_to_end_date(monkeypatch):
    # end_date-only: scan starts watermark + 7 days; stored weeks (any row
    # count) are never re-scraped, absent weeks are fetched.
    monkeypatch.setattr(scrape, "stored_ranking_mondays", lambda: {date(2026, 1, 5)})
    _FrozenToday._today = date(2026, 2, 3)
    monkeypatch.setattr(scrape, "date", _FrozenToday)
    watermark, weeks = scrape.missing_ranking_mondays.fn(end_date=date(2026, 1, 26))
    assert watermark == date(2026, 1, 5)
    assert weeks == [date(2026, 1, 12), date(2026, 1, 19), date(2026, 1, 26)]


def test_missing_weeks_stored_week_skipped_presence_only(monkeypatch):
    # A stored week is complete by presence alone — row counts are irrelevant
    # (the old >=100 refetch gate is gone). Stored 2026-01-05/12 are never
    # re-scraped; absent 2026-01-19/26 are fetched.
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
    # start_date 2026-01-06 (Tue) snaps forward to 2026-01-12, then the
    # watermark floor (2026-01-12 + 7 days = 2026-01-26) clamps it further —
    # an explicit start can never reach weeks before the stored max.
    monkeypatch.setattr(scrape, "stored_ranking_mondays", lambda: {date(2026, 1, 19)})
    _FrozenToday._today = date(2026, 2, 3)
    monkeypatch.setattr(scrape, "date", _FrozenToday)
    watermark, weeks = scrape.missing_ranking_mondays.fn(
        start_date=date(2026, 1, 6), end_date=date(2026, 2, 2)
    )
    assert watermark == date(2026, 1, 19)
    assert weeks == [date(2026, 1, 26), date(2026, 2, 2)]


def test_missing_weeks_no_week_before_stored_max_ever_fetched(monkeypatch):
    # Watermark 2026-01-26; an explicit historical range before it fetches
    # nothing — the scan floor is watermark + 7 days.
    monkeypatch.setattr(scrape, "stored_ranking_mondays", lambda: {date(2026, 1, 26)})
    _FrozenToday._today = date(2026, 2, 3)
    monkeypatch.setattr(scrape, "date", _FrozenToday)
    watermark, weeks = scrape.missing_ranking_mondays.fn(
        start_date=date(2026, 1, 5), end_date=date(2026, 1, 19)
    )
    assert watermark == date(2026, 1, 26)
    assert weeks == []


def test_missing_weeks_explicit_historical_start_clamped_to_current_year(monkeypatch):
    # An explicit start before Jan 1 of the current year is clamped to this
    # year's floor: nothing from 2025 is ever scheduled.
    monkeypatch.setattr(scrape, "stored_ranking_mondays", lambda: {date(2026, 1, 5)})
    _FrozenToday._today = date(2026, 2, 3)
    monkeypatch.setattr(scrape, "date", _FrozenToday)
    watermark, weeks = scrape.missing_ranking_mondays.fn(
        start_date=date(2025, 1, 6), end_date=date(2026, 1, 12)
    )
    assert watermark == date(2026, 1, 5)
    assert weeks == [date(2026, 1, 12)]


def test_missing_weeks_end_date_after_latest_completed_monday_clamped(monkeypatch):
    # An end_date beyond the most recent completed Monday is clamped down
    # (today frozen to 2026-02-03 -> latest completed Monday 2026-02-02), so a
    # future/unpublished week is never scheduled.
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
    # start_date-only (upper bound = most recent completed Monday): the stored
    # week is skipped, absent weeks are fetched.
    monkeypatch.setattr(scrape, "stored_ranking_mondays", lambda: {date(2026, 1, 5)})
    _FrozenToday._today = date(2026, 1, 26)
    monkeypatch.setattr(scrape, "date", _FrozenToday)
    watermark, weeks = scrape.missing_ranking_mondays.fn(start_date=date(2026, 1, 12))
    assert watermark == date(2026, 1, 5)
    assert weeks == [date(2026, 1, 12), date(2026, 1, 19)]


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


def test_fetch_week_skips_on_failure(monkeypatch, capsys):
    def fail_fetch(*_):
        raise TimeoutError("selector timed out")

    monkeypatch.setattr(scrape, "_fetch_week_html", fail_fetch)

    assert scrape.fetch_and_upsert_week(None, date(2026, 1, 5)) is None
    assert "Week 2026-01-05: skipped (could not load or parse)" in capsys.readouterr().out


def test_fetch_week_navigates_the_shared_page(monkeypatch):
    monkeypatch.setattr(scrape, "_jitter", lambda: None)
    calls: list[str] = []

    class Page:
        _filter = (
            '<select id="dateWeek-filter">'
            '<option value="2026-01-05">2026.01.05</option>'
            '<option value="2025-12-29">2025.12.29</option>'
            "</select>"
        )

        def goto(self, _url, **_kwargs):
            calls.append("goto")

        def evaluate(self, _js, arg):
            calls.append("evaluate")
            # arg is [wanted, latest]; answer against the real filter structure.
            return _week_in_filter_real(self._filter, arg[0], arg[1])

        def wait_for_selector(self, _selector, **_kwargs):
            calls.append("wait")

        def content(self):
            # Must contain a player link: the wait loop only returns once
            # rendered rows (player links) are present in the captured HTML.
            return '<a href="/en/players/jannik-sinner/s0ag/overview">J. Sinner</a>'

    assert scrape._fetch_week_html(
        Page(), "https://example.test/rankings", date(2026, 1, 5)
    ).endswith("</a>")
    assert calls == ["goto", "wait", "evaluate", "wait"]


def test_fetch_week_missing_option_after_filter_appears_skips_immediately(monkeypatch):
    """Filter element appears but lacks the week -> reject immediately.

    The option check runs exactly once; a never-published week is rejected at
    once instead of polling the filter for the rest of the verification budget,
    and the row-render wait is never reached.
    """
    monkeypatch.setattr(scrape, "_jitter", lambda: None)
    checks = 0
    row_waits = 0

    class Page:
        # 2026-01-12 is absent (no Current Week option either) -> never published.
        _filter = (
            '<select id="dateWeek-filter">'
            '<option value="2026-01-19">2026.01.19</option>'
            '<option value="2026-01-05">2026.01.05</option>'
            "</select>"
        )

        def goto(self, _url, **_kwargs):
            pass

        def wait_for_selector(self, selector, **_kwargs):
            nonlocal row_waits
            if selector == scrape.RANKINGS_TABLE_SELECTOR:
                row_waits += 1

        def evaluate(self, _js, arg):
            nonlocal checks
            checks += 1
            return _week_in_filter_real(self._filter, arg[0], arg[1])

        def content(self):
            return ""

    with pytest.raises(scrape.RankingsParseError, match="never published"):
        scrape._fetch_week_html(Page(), "https://example.test/rankings", date(2026, 1, 12))
    assert checks == 1
    assert row_waits == 0


def test_fetch_week_filter_absent_until_timeout_raises_unavailable(monkeypatch):
    """Filter element never appears within the budget -> verification error.

    Distinct from a missing week: the page is unverifiable (blocked or stuck
    on a widget), not proof that the week was never published.
    """
    monkeypatch.setattr(scrape, "_jitter", lambda: None)

    class Page:
        def goto(self, _url, **_kwargs):
            pass

        def wait_for_selector(self, _selector, **_kwargs):
            raise TimeoutError("selector timed out")

    with pytest.raises(scrape.RankingsParseError, match="unavailable"):
        scrape._fetch_week_html(Page(), "https://example.test/rankings", date(2026, 1, 12))


def test_fetch_week_filter_with_option_proceeds(monkeypatch):
    """Filter appears with the week as an option -> scrape proceeds."""
    monkeypatch.setattr(scrape, "_jitter", lambda: None)
    checks = 0

    class Page:
        _filter = (
            '<select id="dateWeek-filter"><option value="2026-01-05">2026.01.05</option></select>'
        )

        def goto(self, _url, **_kwargs):
            pass

        def wait_for_selector(self, _selector, **_kwargs):
            pass

        def evaluate(self, _js, arg):
            nonlocal checks
            checks += 1
            return _week_in_filter_real(self._filter, arg[0], arg[1])

        def content(self):
            return '<a href="/en/players/x/y/overview">x</a>'

    html = scrape._fetch_week_html(Page(), "https://example.test/rankings", date(2026, 1, 5))
    assert html.endswith("</a>")
    assert checks == 1


def test_fetch_week_existing_filter_incurs_no_fixed_delay(monkeypatch, capsys):
    """A normal page with the week already in the filter proceeds on the first check."""
    monkeypatch.setattr(scrape, "_jitter", lambda: None)
    checks = 0

    class Page:
        _filter = (
            '<select id="dateWeek-filter"><option value="2026-01-05">2026.01.05</option></select>'
        )

        def goto(self, _url, **_kwargs):
            pass

        def evaluate(self, _js, arg):
            nonlocal checks
            checks += 1
            return _week_in_filter_real(self._filter, arg[0], arg[1])

        def wait_for_selector(self, _selector, **_kwargs):
            pass

        def content(self):
            return '<a href="/en/players/x/y/overview">x</a>'

    html = scrape._fetch_week_html(Page(), "https://example.test/rankings", date(2026, 1, 5))
    assert html.endswith("</a>")
    assert checks == 1
    assert "waiting..." not in capsys.readouterr().out


def test_fetch_week_polls_until_rows_render_or_deadline(monkeypatch):
    """No player links in the HTML -> keep polling, don't return early.

    Regression: the old loop returned as soon as the page was not a Cloudflare
    challenge, so a page whose rows were still rendering was captured empty and
    the week wrongly skipped.
    """
    monkeypatch.setattr(scrape, "_jitter", lambda: None)
    monkeypatch.setattr(scrape, "CHALLENGE_RESOLVE_BUDGET_S", 1)
    waits = 0

    class Page:
        _filter = (
            '<select id="dateWeek-filter"><option value="2026-01-05">2026.01.05</option></select>'
        )

        def goto(self, _url, **_kwargs):
            pass

        def evaluate(self, _js, arg):
            return _week_in_filter_real(self._filter, arg[0], arg[1])

        def wait_for_selector(self, selector, **_kwargs):
            nonlocal waits
            if selector == scrape.RANKINGS_TABLE_SELECTOR:
                waits += 1

        def content(self):
            return "<table></table>" if waits < 3 else '<a href="/en/players/x/y/overview">x</a>'

    html = scrape._fetch_week_html(Page(), "https://example.test/rankings", date(2026, 1, 5))
    assert "<table></table>" not in html
    assert waits >= 3


def test_week_in_filter_matches_real_dom():
    """_week_in_filter matches on option value OR text, dash OR dot format.

    Drives the real function with a fake page whose evaluate mirrors the
    production JS logic (the same _week_in_filter_real the other mocks use),
    against realistic #dateWeek-filter markup observed on atptour.com.
    """

    class Page:
        def __init__(self, filter_html):
            self._filter = filter_html

        def evaluate(self, _js, arg):
            return _week_in_filter_real(self._filter, arg[0], arg[1])

    # Latest week is "Current Week" (date only in its text); older weeks carry
    # the date as the value, with dot-formatted visible text.
    real_filter = (
        '<select id="dateWeek-filter">'
        '<option value="Current Week">2026.08.10</option>'
        '<option value="2026-08-03">2026.08.03</option>'
        '<option value="2026-07-27">2026.07.27</option>'
        "</select>"
    )
    assert scrape._week_in_filter(Page(real_filter), date(2026, 8, 10)) is True
    assert scrape._week_in_filter(Page(real_filter), date(2026, 8, 3)) is True
    # A never-published week is absent.
    assert scrape._week_in_filter(Page(real_filter), date(2026, 8, 20)) is False

    # Flexibility: a filter that uses dash-form text and dot-form values still
    # matches, and one keyed only by text (no value date) still matches.
    dash_text = (
        '<select id="dateWeek-filter"><option value="2026.08.17">2026-08-17</option></select>'
    )
    assert scrape._week_in_filter(Page(dash_text), date(2026, 8, 17)) is True


# ── HTML parser ──────────────────────────────────────────────────

_MEGA_TABLE_HTML = """<table class="mega-table"><tbody>
<tr>
  <td class="rank bold heavy" width="10%" colspan="2">1</td>
  <td class="player bold heavy" width="60%" colspan="12">
    <ul class="player-stats">
      <li class="rank"></li>
      <li class="name"><a href="/en/players/jannik-sinner/s0ag/overview"><span class="lastName">J. Sinner</span></a></li>
    </ul>
  </td>
  <td class="points center bold extrabold" width="15%" colspan="3">12,030</td>
</tr>
<tr>
  <td class="rank bold heavy" width="10%" colspan="2">2</td>
  <td class="player bold heavy" width="60%" colspan="12">
    <ul class="player-stats">
      <li class="rank"></li>
      <li class="name"><a href="/en/players/carlos-alcaraz/a0e2/overview"><span class="lastName">C. Alcaraz</span></a></li>
    </ul>
  </td>
  <td class="points center bold extrabold" width="15%" colspan="3">1,000</td>
</tr>
</tbody></table>"""


def test_extract_rankings_parses_table():
    rows = scrape.extract_rankings_from_html(_MEGA_TABLE_HTML)
    assert len(rows) == 2
    assert rows[0] == {
        "rank": 1,
        "points": 12030,
        "name": "J. Sinner",
        "player_id": "S0AG",
        "slug": "jannik-sinner",
    }
    assert rows[1] == {
        "rank": 2,
        "points": 1000,
        "name": "C. Alcaraz",
        "player_id": "A0E2",
        "slug": "carlos-alcaraz",
    }


def test_extract_rankings_raises_on_challenge_page():
    html = "<html><body>Performing security verification</body></html>"
    with pytest.raises(scrape.RankingsParseError, match="no rankings table"):
        scrape.extract_rankings_from_html(html)


def test_extract_rankings_raises_on_missing_rank_cell():
    html = """<table><tr><td class="player-cell">
<a href="/en/players/carlos-alcaraz/a0e2/overview">Carlos Alcaraz</a>
</td></tr></table>"""
    with pytest.raises(scrape.RankingsParseError, match="missing rank cell"):
        scrape.extract_rankings_from_html(html)


def test_extract_rankings_skips_header_rows():
    html = """<table class="mega-table"><thead><tr><th>Rank</th></tr></thead><tbody>
<tr><td class="rank">1</td><td class="player"><a href="/en/players/novak-djokovic/d643/overview">N. Djokovic</a></td></tr>
</tbody></table>"""
    rows = scrape.extract_rankings_from_html(html)
    assert len(rows) == 1
    assert rows[0]["player_id"] == "D643"


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
    """A backfill that could not access/parse the site for any week fails.

    Rankings post weekly, so finding no parseable page for every expected week
    means the site was blocked or the markup changed — not a legitimate
    "nothing new" result (that path returns before the guard via the
    empty-``weeks`` early return).
    """
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
    """Shared browser page double: serves canned HTML and records navigation."""

    def __init__(self, html=""):
        self._html = html
        self.gotos: list[str] = []

    def goto(self, url, **_kwargs):
        self.gotos.append(url)

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


def test_fetch_overview_html_new_player_uses_slug(monkeypatch):
    """Discovery navigates the shared page to the candidate's profile URL."""
    monkeypatch.setattr(scrape_mod, "_jitter", lambda: None)
    page = _FakePage(_overview_html())

    html, err = scrape_mod._fetch_overview_html(page, "test-slug", "XQ999")

    assert err == ""
    assert html  # the canned page body
    assert len(page.gotos) == 1
    assert "test-slug" in page.gotos[0] and "XQ999" in page.gotos[0]


def test_fetch_overview_waits_for_player_details(monkeypatch):
    monkeypatch.setattr(scrape_mod, "_jitter", lambda: None)
    calls = []

    class Page(_FakePage):
        def wait_for_function(self, script, **kwargs):
            calls.append((script, kwargs["arg"]))

    html, err = scrape_mod._fetch_overview_html(Page(_overview_html()), "test-slug", "XQ999")

    assert err == ""
    assert html
    assert calls[0][1] == "XQ999"
    assert "Age" in calls[0][0]


def test_discover_players_existing_player_never_navigates(monkeypatch):
    """A player already in bronze is skipped: no navigation, no write."""
    monkeypatch.setattr(scrape_mod, "_known_profile_ids", lambda: {"XQ999"})
    monkeypatch.setattr(scrape_mod, "persist_atp_player", lambda *_, **__: 0)
    page = _FakePage(_overview_html())

    result = scrape_mod.discover_players(
        page,
        [{"id": "XQ999", "slug": "test-slug", "player": "Test Player"}],
        canonical={},
        profiles={},
    )

    assert result == {"known": 1, "discovered": 0, "failed": []}
    assert page.gotos == []  # never navigated


def test_discover_players_valid_new_player(monkeypatch):
    """A DB-missing player is fetched once, validated, persisted, and made
    resolvable for the rest of the run."""
    monkeypatch.setattr(scrape_mod, "_known_profile_ids", lambda: set())
    persisted = []
    monkeypatch.setattr(scrape_mod, "persist_atp_player", lambda *a, **_k: persisted.append(a) or 0)
    page = _FakePage(_overview_html("XQ999"))

    canonical, profiles = {}, {}
    result = scrape_mod.discover_players(
        page,
        [{"id": "XQ999", "slug": "test-slug", "player": "Test Player"}],
        canonical=canonical,
        profiles=profiles,
    )

    assert result["discovered"] == 1
    assert result["failed"] == []
    assert len(page.gotos) == 1  # exactly one overview navigation
    assert len(persisted) == 1  # exactly one persistence attempt
    assert persisted[0][0]["id"] == "XQ999"  # uppercased canonical id
    # The new identity is resolvable for the rest of this run.
    assert canonical["XQ999"] == "Test Player"
    assert profiles["XQ999"]["ioc"] == "ITA"


def test_discover_players_repeated_player_discovered_once(monkeypatch):
    """A player appearing many times on pages is fetched exactly once."""
    monkeypatch.setattr(scrape_mod, "_known_profile_ids", lambda: set())
    persisted = []
    monkeypatch.setattr(scrape_mod, "persist_atp_player", lambda *a, **_k: persisted.append(a) or 0)
    page = _FakePage(_overview_html("XQ999"))
    canonical, profiles = {}, {}

    result = scrape_mod.discover_players(
        page,
        [
            {"id": "XQ999", "slug": "test-slug", "player": "Test Player"},
            {"id": "xq999", "slug": "test-slug", "player": "Test Player"},
        ],
        canonical=canonical,
        profiles=profiles,
    )

    assert result["discovered"] == 1
    assert len(persisted) == 1  # one lookup, one persistence
    assert len(page.gotos) == 1  # one navigation


def test_discover_players_missing_identity_fails_without_write(monkeypatch):
    """A candidate with no display name fails structurally before any
    navigation or persistence."""
    monkeypatch.setattr(scrape_mod, "_known_profile_ids", lambda: set())
    monkeypatch.setattr(scrape_mod, "persist_atp_player", lambda *_, **__: 0)
    page = _FakePage(_overview_html("XQ999"))
    canonical, profiles = {}, {}

    result = scrape_mod.discover_players(
        page,
        [{"id": "XQ999", "slug": "test-slug", "player": ""}],
        canonical=canonical,
        profiles=profiles,
    )

    assert result["discovered"] == 0
    assert result["failed"] == [
        {"id": "XQ999", "player": "", "reason": "missing id or display name"}
    ]
    assert page.gotos == []  # never navigated
    assert canonical == {} and profiles == {}


def test_discover_players_failed_discovery_writes_nothing(monkeypatch):
    """A navigation failure is per-player: nothing is persisted and the failure
    is reported, not raised."""
    monkeypatch.setattr(scrape_mod, "_known_profile_ids", lambda: set())
    monkeypatch.setattr(scrape_mod, "persist_atp_player", lambda *_, **__: 0)
    monkeypatch.setattr(
        scrape_mod,
        "_fetch_overview_html",
        lambda _page, _slug, _pid: ("", "navigation failed (TimeoutError: timed out)"),
    )
    page = _FakePage(_overview_html("XQ999"))
    canonical, profiles = {}, {}

    result = scrape_mod.discover_players(
        page,
        [{"id": "XQ999", "slug": "test-slug", "player": "Test Player"}],
        canonical=canonical,
        profiles=profiles,
    )

    assert result["discovered"] == 0
    assert len(result["failed"]) == 1
    assert "navigation failed" in result["failed"][0]["reason"]
    # Nothing became resolvable and nothing was written.
    assert canonical == {} and profiles == {}


def _ranked_row(rank, player_id, slug, name, points):
    return (
        f'<tr><td class="rank">{rank}</td>'
        f'<td class="player"><a href="/en/players/{slug}/{player_id.lower()}/overview">'
        f"{name}</a></td>"
        f'<td class="points">{points}</td></tr>'
    )


def test_fetch_and_upsert_week_includes_newly_discovered_player(monkeypatch):
    """A newly discovered top-200 player is stored as a ranking row the same
    week: discovery refreshes rank_map so the identity filter keeps it."""
    week = date(2026, 1, 5)
    html = (
        "<table><tbody>"
        + _ranked_row(1, "S0AG", "jannik-sinner", "J. Sinner", 12030)
        + _ranked_row(5, "XQ999", "test-slug", "T. Test", 4500)
        + "</tbody></table>"
    )
    monkeypatch.setattr(scrape, "_fetch_week_html", lambda _p, _url, _w: html)
    monkeypatch.setattr(scrape_mod, "_known_profile_ids", lambda: {"S0AG"})  # XQ999 missing
    monkeypatch.setattr(scrape_mod, "persist_atp_player", lambda *_, **__: 0)
    copied = []
    monkeypatch.setattr(scrape, "_copy_df_into", lambda *a, **k: copied.append((a, k)) or 1)
    page = _FakePage(_overview_html("XQ999"))

    written = scrape.fetch_and_upsert_week(
        page,
        week,
        canonical={"S0AG": "Jannik Sinner"},
        profiles={},
    )

    assert written == 2  # the known player plus the newly discovered one
    frame = copied[0][0][1]  # (table, df, ...) positional args to _copy_df_into
    assert sorted(frame["player_id"].tolist()) == ["S0AG", "XQ999"]
    assert frame[frame["player_id"] == "XQ999"]["rank"].iloc[0] == 5


def test_fetch_and_upsert_week_appends_live_atp_ids_to_current_csv(monkeypatch, tmp_path):
    week = date(2026, 1, 5)
    html = (
        "<table><tbody>"
        + _ranked_row(1, "S0AG", "jannik-sinner", "J. Sinner", 12030)
        + "</tbody></table>"
    )
    current = tmp_path / "atp_rankings_current.csv"
    current.write_text("ranking_date,rank,player,points\n")
    monkeypatch.setattr(scrape, "CURRENT_RANKINGS_CSV", current)
    monkeypatch.setattr(scrape, "_fetch_week_html", lambda _p, _url, _w: html)
    monkeypatch.setattr(scrape_mod, "_known_profile_ids", lambda: {"S0AG"})
    monkeypatch.setattr(scrape, "_copy_df_into", lambda *_args, **_kwargs: 1)

    scrape.fetch_and_upsert_week(_FakePage(), week, canonical={}, profiles={})
    scrape.fetch_and_upsert_week(_FakePage(), week, canonical={}, profiles={})

    assert current.read_text().splitlines() == [
        "ranking_date,rank,player,points",
        "20260105,1,S0AG,12030",
    ]
