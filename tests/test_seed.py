"""Pure-logic tests for infra/duckdb/seed.py's select_matches.

No raw CSV, no DuckDB: exercises only the deterministic match-selection
logic (top players by latest rank, recent-matches trim, dedupe, ordering).
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

SEED_PATH = Path(__file__).resolve().parents[1] / "infra" / "duckdb" / "seed.py"
_spec = spec_from_file_location("seed", SEED_PATH)
assert _spec is not None and _spec.loader is not None
seed = module_from_spec(_spec)
_spec.loader.exec_module(seed)

TOP_PLAYERS = seed.TOP_PLAYERS
RECENT = seed.RECENT
select_matches = seed.select_matches


def _match(winner_id, loser_id, winner_rank, loser_rank, tourney_id, tourney_date, match_num):
    return {
        "winner_id": winner_id,
        "loser_id": loser_id,
        "winner_rank": winner_rank,
        "loser_rank": loser_rank,
        "tourney_id": tourney_id,
        "tourney_date": tourney_date,
        "match_num": match_num,
    }


def test_select_matches_dedupes_and_orders_by_date():
    m1 = _match("a", "b", 1, 2, "T1", "20260101", 1)  # in both a's and b's histories
    m2 = _match("b", "c", 2, 3, "T2", "20260102", 1)
    m3 = _match("a", "c", 1, 3, "T3", "20260103", 1)

    selected = select_matches([m3, m1, m2])

    # m1 is shared between two selected players but appears exactly once;
    # output is re-sorted by (tourney_date, tourney_id, match_num).
    assert selected == [m1, m2, m3]


def test_select_matches_sorts_by_tourney_id_and_match_num():
    matches = [
        _match("a", "opp", 1, 100, "TB", "20260202", 2),
        _match("b", "opp", 2, 100, "TA", "20260202", 5),
        _match("c", "opp", 3, 100, "TA", "20260202", 3),
    ]

    selected = select_matches(list(reversed(matches)))

    assert [(m["tourney_id"], m["match_num"]) for m in selected] == [
        ("TA", 3),
        ("TA", 5),
        ("TB", 2),
    ]


def test_select_matches_tie_break_prefers_lower_player_id():
    # TOP_PLAYERS+1 players all tied at the same latest rank: the highest
    # player_id loses the top cut.
    matches = [
        _match(f"p{i:02d}", "zzz", 5, 999, f"T{i}", f"202601{i + 1:02d}", 1)
        for i in range(TOP_PLAYERS + 1)
    ]

    selected = select_matches(matches)

    selected_ids = {m["tourney_id"] for m in selected}
    assert selected_ids == {f"T{i}" for i in range(TOP_PLAYERS)}
    assert f"T{TOP_PLAYERS}" not in selected_ids


def test_select_matches_uses_latest_rank_not_earliest():
    matches = [
        _match("up", "zzz", 999, 999, "TU1", "20260101", 1),  # bad early rank
        _match("up", "zzz", 5, 999, "TU2", "20260102", 1),  # good latest rank
        _match("down", "zzz", 5, 999, "TD1", "20260103", 1),  # good early rank
        _match("down", "zzz", 999, 999, "TD2", "20260104", 1),  # bad latest rank
        _match("mid", "zzz", 5, 999, "TM", "20260105", 1),
        *[_match(f"f{i:02d}", "zzz", 4, 999, f"TF{i}", f"202601{i + 6:02d}", 1) for i in range(8)],
        _match("zzz", "nobody", 999, 500, "TZ", "20260114", 1),
    ]

    selected = select_matches(matches)

    tourneys = [m["tourney_id"] for m in selected]
    assert tourneys == ["TU1", "TU2", "TM", *[f"TF{i}" for i in range(8)]]
    assert {"TD1", "TD2", "TZ"}.isdisjoint(tourneys)


def test_select_matches_trims_to_most_recent_recent():
    matches = [
        _match("big", "opp", 1, 500, f"T{d}", f"202601{d:02d}", 1) for d in range(1, 2 * RECENT + 2)
    ]

    selected = select_matches(matches)

    assert len(selected) == RECENT
    assert [m["tourney_date"] for m in selected] == [
        f"202601{d:02d}" for d in range(RECENT + 2, 2 * RECENT + 2)
    ]


def test_select_matches_empty_input():
    assert select_matches([]) == []
