"""Hermetic tests for Hawkeye response parsing and rejection."""

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


class _SequencePage(_Page):
    def __init__(self, bodies: list[str]) -> None:
        super().__init__(bodies[0])
        self._bodies = iter(bodies)
        self.goto_count = 0

    def goto(self, *_args, **_kwargs) -> None:
        self.goto_count += 1
        self._body = next(self._bodies)


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


def test_build_match_id_reduces_date_like_year_to_edition_year():
    """A YYYYMMDD year at the scrape boundary is reduced to its four-digit
    edition year, never embedded as an id prefix."""
    assert matches.build_match_id(19670220, "1967-southern-pro", 1) == "1967-southern-pro-001"


def test_indoor_tournament_mapping_requires_the_matching_tier():
    assert matches.indoor_for_tournament("Paris Masters", "masters") == 1
    assert matches.indoor_for_tournament("Dallas", "atp_500") == 1
    assert matches.indoor_for_tournament("Dallas", "atp_250") == 1
    assert matches.indoor_for_tournament("Dallas", "masters") == 0
    assert matches.indoor_for_tournament("ATP Finals", "atp_finals") == 1
    assert matches.indoor_for_tournament("ATP Tour Finals", "atp_finals") == 1
    assert matches.indoor_for_tournament("Cincinnati Open", "masters") == 0


# Minimal inline Hawkeye payload: only the fields the mapper and the
# assertions below read. Side A (PlayerTeam1) is the winner, so its seed lands
# in the winner slot; draw_size/best_of/minutes fill their raw columns; absent
# entry values stay None. Used in place of the deleted JSON fixture.
_HAWKEYE_MS001 = {
    "Tournament": {"Singles": 96},
    "Match": {
        "WinningPlayerId": "S0S1",
        "NumberOfSets": 3,
        "MatchTime": "01:27:28",
        "PlayerTeam1": {
            "PlayerId": "S0S1",
            "SeedPlayerTeam": "5",
            "EntryStatusPlayerTeam": None,
        },
        "PlayerTeam2": {
            "PlayerId": "N0AE",
            "SeedPlayerTeam": "28",
            "EntryStatusPlayerTeam": None,
        },
        "PlayerTeam": {
            "Sets": [
                {
                    "SetScore": None,
                    "Stats": {
                        "ServiceStats": {
                            "Aces": {"Number": 7},
                            "DoubleFaults": {"Number": 0},
                            "FirstServe": {"Dividend": 37, "Divisor": 54},
                            "FirstServePointsWon": {"Dividend": 32},
                            "SecondServePointsWon": {"Dividend": 13},
                            "ServiceGamesPlayed": {"Number": 10},
                            "BreakPointsSaved": {"Dividend": 0, "Divisor": 0},
                        }
                    },
                }
            ]
        },
        "OpponentTeam": {
            "Sets": [
                {
                    "SetScore": None,
                    "Stats": {
                        "ServiceStats": {
                            "Aces": {"Number": 8},
                            "DoubleFaults": {"Number": 1},
                            "FirstServe": {"Dividend": 40, "Divisor": 64},
                            "FirstServePointsWon": {"Dividend": 32},
                            "SecondServePointsWon": {"Dividend": 12},
                            "ServiceGamesPlayed": {"Number": 11},
                            "BreakPointsSaved": {"Dividend": 2, "Divisor": 4},
                        }
                    },
                }
            ]
        },
    },
}


def test_hawkeye_to_bronze_stamps_source_metadata_winner_first():
    """Inline ms001: side A (PlayerTeam1) is the winner, so its seed/entry
    land in the winner slot; payload draw_size/best_of/minutes fill their raw
    columns; absent entry values stay None, never fabricated."""
    row = matches.hawkeye_to_bronze(
        _HAWKEYE_MS001,
        {
            "match_id": "2026-421-001",
            "match_date": "2026-08-10",
            "player1_id": "S0S1",
            "player2_id": "N0AE",
            "winner_id": "S0S1",
            "tournament": "masters",
            "round": "f",
            "player1_name": "Ben Shelton",
            "player2_name": "Brandon Nakashima",
        },
    )
    assert row is not None

    assert row["winner_seed"] == "5"  # PlayerTeam1 seed, winner side
    assert row["loser_seed"] == "28"  # PlayerTeam2 seed
    assert row["winner_entry"] is None and row["loser_entry"] is None
    assert row["draw_size"] == 96  # Tournament.Singles
    assert row["best_of"] == 3  # Match.NumberOfSets
    assert row["minutes"] == 87  # MatchTime 01:27:28
    assert row["player1_name"] == "Ben Shelton"
    assert row["player2_name"] == "Brandon Nakashima"
    assert row["winner_id"] == "S0S1"
    assert row["is_indoor"] == 0


def test_hawkeye_to_bronze_uses_normalized_match_number():
    row = matches.hawkeye_to_bronze(
        _HAWKEYE_MS001,
        {
            "match_id": "ms098",
            "match_num": 1,
            "match_date": "2026-08-02",
            "player1_id": "S0S1",
            "player2_id": "N0AE",
            "winner_id": "S0S1",
            "round": "r128",
        },
    )

    assert row is not None
    assert row["match_num"] == 1


def test_assign_match_numbers_does_not_use_atp_ms_ids():
    rows = [
        {"match_id": "ms098", "round": "r128", "match_date": "2026-08-02"},
        {"match_id": "ms048", "round": "r64", "match_date": "2026-08-04"},
        {"match_id": "ms110", "round": "r128", "match_date": "2026-08-02"},
        {"match_id": "ms001", "round": "f", "match_date": "2026-08-13"},
    ]

    numbered = matches.assign_match_numbers(rows)

    assert [row["match_num"] for row in numbered] == [1, 2, 3, 4]
    assert [row["match_id"] for row in numbered] == ["ms098", "ms110", "ms048", "ms001"]


def test_assign_match_numbers_preserves_gaps_in_a_96_draw():
    rows = [
        *(
            {"round": "r128", "match_date": "2026-08-02", "match_id": f"ms{i:03d}"}
            for i in range(32)
        ),
        *(
            {"round": "r64", "match_date": "2026-08-04", "match_id": f"ms{i:03d}"}
            for i in range(31)
        ),
        *(
            {"round": "r32", "match_date": "2026-08-05", "match_id": f"ms{i:03d}"}
            for i in range(16)
        ),
        *({"round": "r16", "match_date": "2026-08-09", "match_id": f"ms{i:03d}"} for i in range(8)),
        *({"round": "qf", "match_date": "2026-08-11", "match_id": f"ms{i:03d}"} for i in range(4)),
        *({"round": "sf", "match_date": "2026-08-12", "match_id": f"ms{i:03d}"} for i in range(2)),
        {"round": "f", "match_date": "2026-08-13", "match_id": "ms001"},
    ]

    numbered = matches.assign_match_numbers(rows, draw_size=96)

    assert [row["match_num"] for row in numbered if row["round"] == "r64"] == list(range(33, 64))
    assert numbered[-1]["match_num"] == 95


def test_assign_match_numbers_accounts_for_byes_in_alternate_draw_sizes():
    rows = [
        *({"round": "r32", "match_date": "2026-08-02"} for _ in range(12)),
        *({"round": "r16", "match_date": "2026-08-04"} for _ in range(8)),
        *({"round": "qf", "match_date": "2026-08-05"} for _ in range(4)),
        *({"round": "sf", "match_date": "2026-08-09"} for _ in range(2)),
        {"round": "f", "match_date": "2026-08-11"},
    ]

    numbered = matches.assign_match_numbers(rows, draw_size=28)

    assert [row["match_num"] for row in numbered if row["round"] == "r16"] == list(range(13, 21))
    assert numbered[-1]["match_num"] == 27


def test_bronze_id_uses_normalized_match_number_not_atp_ms_id():
    match_id, reason, _ = matches._bronze_match_id(
        {
            "match_id": "ms098",
            "match_num": 1,
            "match_date": "2026-08-02",
            "player1_id": "S0S1",
            "player2_id": "N0AE",
        },
        {},
        2026,
        "421",
    )

    assert reason == ""
    assert match_id == "2026-421-001"


# ── Discovery → resolution: match-level skip propagation ────────────


def test_resolve_discovered_matches_keeps_live_atp_ids_without_legacy_map():
    rows = [
        {
            "match_id": "ms001",
            "player1_id": "S0S1",
            "player1_name": "Ben Shelton",
            "player2_id": "N0AE",
            "player2_name": "Brandon Nakashima",
            "winner_id": "S0S1",
        },
        {
            "match_id": "ms002",
            "player1_id": "AAAAAA",
            "player1_name": "Nobody",
            "player2_id": "S0S1",
            "player2_name": "Ben Shelton",
            "winner_id": "S0S1",
        },
    ]
    resolved, skipped = matches.resolve_discovered_matches(rows, "masters")

    assert len(resolved) == 2
    assert resolved[0]["match_id"] == "ms001"
    assert resolved[0]["player1_id"] == "S0S1"
    assert resolved[0]["winner_id"] == "S0S1"
    assert resolved[0]["tournament"] == "masters"
    assert skipped == []
