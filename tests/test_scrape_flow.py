"""Fast, hermetic tests for the scrape flow parser and date math.

No Prefect server, no MLflow, no browser, no external fixture files — just pure logic.
"""

from datetime import date

import pytest

import src.flows.scrape as scrape


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
    # watermark..previous-week window only.
    monkeypatch.setattr(scrape, "stored_ranking_mondays", lambda: {date(2026, 1, 5)})
    _FrozenToday._today = date(2026, 1, 20)
    monkeypatch.setattr(scrape, "date", _FrozenToday)
    watermark, weeks = scrape.missing_ranking_mondays.fn()
    assert watermark == date(2026, 1, 5)
    assert weeks == [date(2026, 1, 12), date(2026, 1, 19)]


def test_missing_weeks_backfill_to_end_date(monkeypatch):
    monkeypatch.setattr(scrape, "stored_ranking_mondays", lambda: {date(2026, 1, 5)})
    watermark, weeks = scrape.missing_ranking_mondays.fn(end_date=date(2026, 1, 26))
    assert watermark == date(2026, 1, 5)
    assert weeks == [date(2026, 1, 12), date(2026, 1, 19), date(2026, 1, 26)]


def test_missing_weeks_backfill_skips_stored_weeks(monkeypatch):
    # 2026-01-12 already stored -> skipped even though it is in range.
    monkeypatch.setattr(
        scrape,
        "stored_ranking_mondays",
        lambda: {date(2026, 1, 5), date(2026, 1, 12)},
    )
    watermark, weeks = scrape.missing_ranking_mondays.fn(end_date=date(2026, 1, 26))
    assert watermark == date(2026, 1, 12)
    assert weeks == [date(2026, 1, 19), date(2026, 1, 26)]


def test_missing_weeks_quits_when_end_date_at_watermark(monkeypatch):
    monkeypatch.setattr(scrape, "stored_ranking_mondays", lambda: {date(2026, 1, 5)})
    _watermark, weeks = scrape.missing_ranking_mondays.fn(end_date=date(2026, 1, 5))
    assert weeks == []


def test_missing_weeks_explicit_range_snaps_start_and_skips_stored(monkeypatch):
    # start_date 2026-01-06 (Tue) snaps forward to 2026-01-12; stored 2026-01-19
    # is skipped; end_date 2026-02-02 is inclusive.
    monkeypatch.setattr(scrape, "stored_ranking_mondays", lambda: {date(2026, 1, 19)})
    watermark, weeks = scrape.missing_ranking_mondays.fn(
        start_date=date(2026, 1, 6), end_date=date(2026, 2, 2)
    )
    assert watermark == date(2026, 1, 19)
    assert weeks == [date(2026, 1, 12), date(2026, 1, 26), date(2026, 2, 2)]


def test_missing_weeks_explicit_range_backfills_before_watermark(monkeypatch):
    # Watermark 2026-01-26; an explicit historical range before it still yields
    # the missing Mondays in that window (interior-gap backfill).
    monkeypatch.setattr(scrape, "stored_ranking_mondays", lambda: {date(2026, 1, 26)})
    watermark, weeks = scrape.missing_ranking_mondays.fn(
        start_date=date(2026, 1, 5), end_date=date(2026, 1, 19)
    )
    assert watermark == date(2026, 1, 26)
    assert weeks == [date(2026, 1, 5), date(2026, 1, 12), date(2026, 1, 19)]


def test_fetch_week_skips_on_failure(monkeypatch, capsys):
    def fail_fetch(*_):
        raise TimeoutError("selector timed out")

    monkeypatch.setattr(scrape, "_fetch_week_html", fail_fetch)

    assert scrape.fetch_and_upsert_week(None, date(2026, 1, 5), {}) is None
    assert "Week 2026-01-05: skipped (could not load or parse)" in capsys.readouterr().out


def test_fetch_week_navigates_the_shared_page(monkeypatch):
    monkeypatch.setattr(scrape, "_jitter", lambda: None)
    calls: list[str] = []

    class Page:
        def goto(self, _url, **_kwargs):
            calls.append("goto")

        def evaluate(self, _js, wanted):
            calls.append("evaluate")
            # The week is present in the filter.
            return wanted in ("2026.01.05",)

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
        def goto(self, _url, **_kwargs):
            pass

        def wait_for_selector(self, selector, **_kwargs):
            nonlocal row_waits
            if selector == scrape.RANKINGS_TABLE_SELECTOR:
                row_waits += 1

        def evaluate(self, _js, wanted):
            nonlocal checks
            checks += 1
            return wanted in ("2026.01.05",)

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
        def goto(self, _url, **_kwargs):
            pass

        def wait_for_selector(self, _selector, **_kwargs):
            pass

        def evaluate(self, _js, _wanted):
            nonlocal checks
            checks += 1
            return True

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
        def goto(self, _url, **_kwargs):
            pass

        def evaluate(self, _js, _wanted):
            nonlocal checks
            checks += 1
            return True

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
        def goto(self, _url, **_kwargs):
            pass

        def evaluate(self, _js, wanted):
            return wanted in ("2026.01.05",)

        def wait_for_selector(self, selector, **_kwargs):
            nonlocal waits
            if selector == scrape.RANKINGS_TABLE_SELECTOR:
                waits += 1

        def content(self):
            return "<table></table>" if waits < 3 else '<a href="/en/players/x/y/overview">x</a>'

    html = scrape._fetch_week_html(Page(), "https://example.test/rankings", date(2026, 1, 5))
    assert "<table></table>" not in html
    assert waits >= 3


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
    assert rows[0] == {"rank": 1, "points": 12030, "name": "J. Sinner", "player_id": "S0AG"}
    assert rows[1] == {"rank": 2, "points": 1000, "name": "C. Alcaraz", "player_id": "A0E2"}


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


def test_translate_keeps_mapped_drops_unmapped():
    rank_map = {"S0AG": "S0AG", "A0E2": "A0E2"}
    rows = [
        {"player_id": "S0AG", "rank": 1, "points": 1000, "name": "J. Sinner"},
        {"player_id": "ZZ99", "rank": 2, "points": 900, "name": "Nobody"},
        {"player_id": "A0E2", "rank": 3, "points": 800, "name": "C. Alcaraz"},
        {"player_id": "S0AG", "rank": 201, "points": 1, "name": "Dup"},
    ]
    frame, skipped = scrape.translate_rank_rows(rows, rank_map)
    assert frame["player_id"].tolist() == ["S0AG", "A0E2"]
    assert len(skipped) == 2


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
