"""Hermetic tests for Hawkeye response parsing (no browser, no network).

Regression: the live August run returned every Hawkeye payload wrapped in an
HTML shell (<html>...<pre>{...}</pre>...); the fetcher classified the shell as
a Cloudflare challenge before the embedded-JSON recovery could run, so all
matches were skipped. These tests pin the recovery and the rejection boundary.
"""

import pytest

import src.flows.matches as matches
from src.flows import rankings

_RAW = (
    '{"Tournament":{"EventType":"1000"},'
    '"Match":{"WinningPlayerId":"S0S1","Winner":"S0S1",'
    '"PlayerTeam1":{"PlayerId":"S0S1"},"PlayerTeam2":{"PlayerId":"N0AE"},'
    '"PlayerTeam":{"Sets":[]},"OpponentTeam":{"Sets":[]}}}'
)


class _Page:
    def __init__(self, body: str) -> None:
        self._body = body

    def goto(self, *_args, **_kwargs) -> None:
        pass

    def content(self) -> str:
        return self._body


def _fetch(body: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(rankings, "_jitter", lambda: None)
    return matches.fetch_hawkeye_match(_Page(body), 2026, "421", "ms001")


def test_html_wrapped_json_recovers_payload(monkeypatch):
    # The live shape: raw JSON inside an <html>/<pre> shell.
    payload, reason = _fetch(
        f"<html><head><meta charset='utf-8'></head><body><pre>{_RAW}</pre></body></html>",
        monkeypatch,
    )
    assert reason == ""
    assert payload is not None
    assert payload["Match"]["Winner"] == "S0S1"
    assert payload["Match"]["OpponentTeam"] == {"Sets": []}


def test_raw_json_body_is_accepted(monkeypatch):
    payload, reason = _fetch(_RAW, monkeypatch)
    assert reason == ""
    assert payload is not None
    assert payload["Match"]["Winner"] == "S0S1"


def test_challenge_html_without_json_is_rejected(monkeypatch):
    body = "<html><body>Just a moment... enable JavaScript and cookies to continue</body></html>"
    payload, reason = _fetch(body, monkeypatch)
    assert payload is None
    assert "challenge" in reason


def test_build_match_id_is_date_free_and_stable():
    """Same edition year + opaque tournament id + sequence -> one canonical id."""
    assert matches.build_match_id(2026, "418", 26) == "2026-418-026"
    assert matches.build_match_id(2026, "418", 26) == matches.build_match_id(2026, "418", 26)


def test_build_match_id_prefixes_only_a_missing_edition_year():
    """The date-derived year is prepended once, and only when the tournament id
    does not already repeat that same year at its start."""
    assert matches.build_match_id(2026, "2026-418", 26) == "2026-418-026"
    assert matches.build_match_id(2026, "2026", 26) == "2026-026"
    assert matches.build_match_id(2026, "1987", 26) == "2026-1987-026"
    assert matches.build_match_id(2026, "1987-foo", 26) == "2026-1987-foo-026"


def test_build_match_id_keeps_opaque_dashed_tournament_ids_distinct():
    """Tournament ids are embedded verbatim, so dashed ids cannot collide:
    ``2026-418`` and ``2026-41-8`` must stay distinguishable."""
    assert matches.build_match_id(2026, "418", 1) != matches.build_match_id(2026, "41-8", 1)
    assert matches.build_match_id(2026, "1967-southern-pro", 1) == "2026-1967-southern-pro-001"
