"""Fast, hermetic tests for the rankings catch-up flow parser and date math.

No Prefect server, no MLflow, no browser, no external fixture files — just pure logic.
"""

from datetime import date

import pytest

import src.flows.scrape as scrape

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
    ]


def test_ranking_mondays_after_empty_when_current():
    wm = date(2026, 1, 19)
    assert scrape.ranking_mondays_after(wm, date(2026, 1, 19)) == []
    assert scrape.ranking_mondays_after(wm, date(2026, 1, 5)) == []


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
