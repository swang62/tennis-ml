"""Seed selection and idempotent-write tests."""

from pathlib import Path

import pandas as pd
import pytest

from src.db import seed


@pytest.fixture(autouse=True)
def _skip_session_cleanup(monkeypatch):
    monkeypatch.setattr(seed, "clear_active_sessions", lambda: ([], []))


TOP_PLAYERS = seed.TOP_PLAYERS
RECENT = seed.RECENT
select_matches = seed.select_matches
discover_atp_csvs = seed.discover_atp_csvs
load_all_raw_atp_rows = seed.load_all_raw_atp_rows
parse_args = seed.parse_args
latest_official_ranks = seed.latest_official_ranks


def _ranking_frame(dates, ids, ranks):
    """Build a load_ranking_rows-shaped frame (datetimes + Int64 rank/points)."""
    return pd.DataFrame(
        {
            "ranking_date": pd.to_datetime(dates, format="%Y%m%d"),
            "player_id": list(ids),
            "rank": pd.array(ranks, dtype="Int64"),
            "points": pd.array([0] * len(ranks), dtype="Int64"),
        }
    )


def _patch_rankings(monkeypatch, frame, canonical_ids, rank_map):
    """Pin every file/network dependency behind latest_official_ranks."""
    monkeypatch.setattr(seed, "discover_ranking_csvs", lambda: [Path("dummy.csv")])
    monkeypatch.setattr(seed, "load_ranking_rows", lambda _paths, **_kwargs: frame)
    monkeypatch.setattr(
        seed, "canonical_players", lambda _path=None: dict.fromkeys(canonical_ids, "")
    )
    monkeypatch.setattr(seed, "load_ranking_player_map", lambda **_kwargs: rank_map)


def test_latest_official_ranks_direct_atp_ids_resolve(monkeypatch):
    """A ranking row whose player id is already a canonical ATP id resolves
    directly; the reviewed map is not needed for those rows."""
    frame = _ranking_frame(["20260105", "20260105"], ["A0E2", "Z355"], [1, 2])
    # Empty legacy map: would NOT resolve these ids on its own.
    _patch_rankings(monkeypatch, frame, canonical_ids={"A0E2", "Z355"}, rank_map={})

    assert latest_official_ranks() == {"A0E2": 1, "Z355": 2}


def test_latest_official_ranks_legacy_mapped_ids_resolve(monkeypatch):
    """Source ids that are not canonical ATP ids still resolve through the
    reviewed map (validation preserved via the canonical_ids passthrough)."""
    frame = _ranking_frame(["20260105", "20260105"], ["12345", "67890"], [3, 4])
    _patch_rankings(
        monkeypatch,
        frame,
        canonical_ids={"A0E2", "Z355"},
        rank_map={"12345": "A0E2", "67890": "Z355"},
    )

    assert latest_official_ranks() == {"A0E2": 3, "Z355": 4}


def test_latest_official_ranks_mixes_direct_and_mapped(monkeypatch):
    """Direct ATP ids and legacy-mapped source ids both resolve; an unresolved
    row is dropped without discarding the rows that did resolve."""
    frame = _ranking_frame(
        ["20260105", "20260105", "20260105"],
        ["A0E2", "67890", "99999"],
        [1, 4, 5],
    )
    _patch_rankings(
        monkeypatch,
        frame,
        canonical_ids={"A0E2", "Z355"},
        rank_map={"67890": "Z355"},
    )

    assert latest_official_ranks() == {"A0E2": 1, "Z355": 4}


def test_latest_official_ranks_excludes_unresolved_ids(monkeypatch):
    """Rows that are neither canonical ATP ids nor present in the reviewed map
    are excluded from the selection."""
    frame = _ranking_frame(["20260105", "20260105"], ["00001", "00002"], [8, 9])
    _patch_rankings(
        monkeypatch,
        frame,
        canonical_ids={"A0E2", "Z355"},
        rank_map={"00099": "Z355"},
    )

    assert latest_official_ranks() == {}


def test_main_clears_active_sessions_before_seeding(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(seed, "clear_active_sessions", lambda: ([1], [2]))
    monkeypatch.setattr(seed, "main_all", lambda **_kwargs: calls.append("seed"))

    seed.main(["--all"])

    assert calls == ["seed"]


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

    # Shared matches dedupe before chronological sorting.
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


def test_select_matches_uses_official_ranks_and_ignores_zero_match_ranks():
    matches = [
        _match("djokovic", "opponent", 0, 0, "DJ", "20260101", 1),
        _match("unknown", "opponent", 0, 0, "UNK", "20260102", 1),
    ]

    selected = select_matches(matches, official_ranks={"djokovic": 8})

    assert selected == [matches[0]]


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


def test_select_matches_trims_prior_year_to_most_recent_recent():
    # Prior-year (non-default-year) matches stay bounded at the RECENT tail.
    matches = [
        _match("big", "opp", 1, 500, f"T{d}", f"202501{d:02d}", 1) for d in range(1, 2 * RECENT + 2)
    ]

    selected = select_matches(matches)

    assert len(selected) == RECENT
    assert [m["tourney_date"] for m in selected] == [
        f"202501{d:02d}" for d in range(RECENT + 2, 2 * RECENT + 2)
    ]


def test_select_matches_includes_full_default_year_history_for_top_player():
    matches = [
        _match("big", "opp", 1, 500, f"T{d}", f"202601{d:02d}", 1) for d in range(1, 2 * RECENT + 2)
    ]

    selected = select_matches(matches)

    assert len(selected) == 2 * RECENT + 1
    assert [m["tourney_date"] for m in selected] == [
        f"202601{d:02d}" for d in range(1, 2 * RECENT + 2)
    ]


def test_select_matches_mixed_years_full_default_year_bounded_prior_years():
    prior = [
        _match("big", "opp", 1, 500, f"O{d}", f"202501{d:02d}", 1) for d in range(1, RECENT + 2)
    ]
    default_year = [
        _match("big", "opp", 1, 500, f"N{d}", f"202601{d:02d}", 1) for d in range(1, 2 * RECENT + 1)
    ]

    selected = select_matches(prior + default_year)

    # Every default-year match plus only the RECENT prior-year tail, in
    # chronological order across years (2025 before 2026).
    assert [(m["tourney_id"], m["tourney_date"]) for m in selected] == [
        *[(f"O{d}", f"202501{d:02d}") for d in range(2, RECENT + 2)],
        *[(f"N{d}", f"202601{d:02d}") for d in range(1, 2 * RECENT + 1)],
    ]


def test_select_matches_non_top_players_enter_only_through_top_player_matches():
    # "side" (rank 500) and "other"/"outsider" (ranks 998/999) all fall outside
    # the top-10 cut; only their matches against top players enter the selection.
    fillers = [_match(f"f{i}", "opp", 2, 999, f"TF{i}", "20260101", 1) for i in range(9)]
    matches = [
        *fillers,
        _match("big", "side", 1, 500, "T1", "20260101", 1),
        _match("side", "big", 500, 1, "T2", "20260102", 1),
        _match("outsider", "other", 999, 998, "T3", "20260103", 1),
    ]

    selected = select_matches(matches)

    assert {m["tourney_id"] for m in selected} == {"T1", "T2", *[f"TF{i}" for i in range(9)]}


def test_select_matches_empty_input():
    assert select_matches([]) == []


def test_discover_atp_csvs_includes_regular_excludes_challenger_and_non_csv(tmp_path):
    import csv

    def write(name):
        with open(tmp_path / name, "w") as f:
            csv.writer(f).writerow(["header"])

    write("2025.csv")
    write("2025_challenger.csv")
    write("2026.csv")
    write("2026_challenger.csv")
    write(".DS_Store")
    write("notes.txt")

    found = discover_atp_csvs(tmp_path)

    # Regular tour CSVs are discovered; Challenger-named and non-CSV files are not.
    assert found == [
        tmp_path / "2025.csv",
        tmp_path / "2026.csv",
    ]


def test_discover_atp_csvs_empty_dir(tmp_path):
    assert discover_atp_csvs(tmp_path) == []


def test_load_all_raw_atp_rows_sorts_chronologically(tmp_path):
    import csv

    columns = [
        "tourney_id",
        "tourney_date",
        "match_num",
        "winner_id",
        "loser_id",
        "winner_rank",
        "winner_rank_points",
        "winner_age",
        "loser_rank",
        "loser_rank_points",
        "loser_age",
        "tourney_level",
        "tourney_name",
        "round",
        "surface",
        "score",
        "indoor",
        "best_of",
        "w_ace",
        "w_df",
        "w_svpt",
        "w_1stIn",
        "w_1stWon",
        "w_2ndWon",
        "w_SvGms",
        "w_bpSaved",
        "w_bpFaced",
        "l_ace",
        "l_df",
        "l_svpt",
        "l_1stIn",
        "l_1stWon",
        "l_2ndWon",
        "l_SvGms",
        "l_bpSaved",
        "l_bpFaced",
    ]

    def write(name, rows):
        with open(tmp_path / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, 0) for c in columns})

    # Deliberately out of order across files: later date in the first file,
    # earlier date in the second. The merged result must be chronological.
    write(
        "2026.csv",
        [
            {
                "tourney_id": "T2",
                "tourney_date": "20260102",
                "match_num": 1,
                "winner_id": "b",
                "loser_id": "zzz",
                "winner_rank": 1,
                "loser_rank": 2,
            },
        ],
    )
    write(
        "2025.csv",
        [
            {
                "tourney_id": "T1",
                "tourney_date": "20260101",
                "match_num": 1,
                "winner_id": "a",
                "loser_id": "zzz",
                "winner_rank": 1,
                "loser_rank": 2,
            },
        ],
    )

    rows = load_all_raw_atp_rows([tmp_path / "2026.csv", tmp_path / "2025.csv"])

    assert [m["tourney_id"] for m in rows] == ["T1", "T2"]


def test_parse_args_enrich_is_independent_of_all():
    """--enrich is not mutually exclusive with --all; alone it keeps the miniset."""
    assert parse_args(["--enrich"]).enrich is True
    both = parse_args(["--all", "--enrich"])
    assert both.all is True
    assert both.enrich is True


def test_parse_args_reset_is_independent_of_all_and_enrich():
    """--reset (formerly --force) parses standalone and with --all/--enrich."""
    assert parse_args([]).reset is False
    assert parse_args(["--reset"]).reset is True
    combo = parse_args(["--all", "--enrich", "--reset"])
    assert (combo.all, combo.enrich, combo.reset) == (True, True, True)


def test_main_dispatches_without_network(monkeypatch):
    """Flag combos dispatch offline; --enrich is the only network gate."""
    calls = []

    monkeypatch.setattr(
        seed,
        "main_default",
        lambda enrich=False, reset=False: calls.append(("default", enrich, reset)),
    )
    monkeypatch.setattr(
        seed, "main_all", lambda enrich=False, reset=False: calls.append(("all", enrich, reset))
    )

    seed.main([])
    assert calls == [("default", False, False)]

    calls.clear()
    seed.main(["--all"])
    assert calls == [("all", False, False)]

    calls.clear()
    seed.main(["--enrich"])
    assert calls == [("default", True, False)]

    calls.clear()
    seed.main(["--all", "--enrich"])
    assert calls == [("all", True, False)]

    calls.clear()
    seed.main(["--reset"])
    assert calls == [("default", False, True)]

    calls.clear()
    seed.main(["--all", "--reset", "--enrich"])
    assert calls == [("all", True, True)]


def test_seed_rankings_and_enrichment_imports_only_seeded_players(monkeypatch):
    """Default/--all stay offline: rankings import only, never enrichment."""
    calls = []
    monkeypatch.setattr(seed, "ingest_rankings", lambda **kwargs: calls.append(kwargs) or {})
    monkeypatch.setattr(
        seed, "enrich_players", lambda _ids, _force=False: calls.append("enrich") or 0
    )

    seed.seed_rankings_and_enrichment(["A0E2", "Z355"], enrich=False, reset=False)

    # The seed scopes the rankings import (and its IOC fallback) to exactly its
    # match-corpus player set; the fallback is not suppressed.
    assert calls == [{"player_ids": {"A0E2", "Z355"}, "force": False, "match_rows": None}]


def test_seed_rankings_and_enrichment_enriches_only_when_flag(monkeypatch):
    """--enrich adds idempotent Wikipedia enrichment for the exact seeded set."""
    calls = []
    monkeypatch.setattr(seed, "ingest_rankings", lambda **_: calls.append("rankings") or {})
    monkeypatch.setattr(
        seed, "enrich_players", lambda ids, force=False: calls.append(("enrich", ids, force)) or 2
    )

    seed.seed_rankings_and_enrichment(["A0E2"], enrich=True, reset=False)

    assert calls == ["rankings", ("enrich", ["A0E2"], False)]


def test_seed_rankings_and_enrichment_reset_propagates(monkeypatch):
    """--reset reaches both rankings and enrichment."""
    calls = []
    monkeypatch.setattr(seed, "ingest_rankings", lambda **kwargs: calls.append(kwargs) or {})
    monkeypatch.setattr(seed, "clear_rankings", lambda: calls.append("clear-rankings"))
    monkeypatch.setattr(
        seed, "enrich_players", lambda ids, force=False: calls.append(("enrich", ids, force)) or 2
    )

    seed.seed_rankings_and_enrichment(["A0E2"], enrich=True, reset=True)

    assert calls == [
        "clear-rankings",
        {"player_ids": {"A0E2"}, "force": True, "match_rows": None},
        ("enrich", ["A0E2"], True),
    ]


def test_seed_rankings_and_enrichment_skips_empty_corpus(monkeypatch):
    calls = []
    monkeypatch.setattr(seed, "ingest_rankings", lambda **_: calls.append(1))
    monkeypatch.setattr(seed, "enrich_players", lambda _ids, force=False: calls.append(2) or force)

    seed.seed_rankings_and_enrichment([], enrich=True, reset=False)

    assert calls == []


def test_seed_enrich_output_is_the_batch_summary_only(monkeypatch, capsys):
    """--enrich seed output shows the batch summary, not success-only wording."""
    monkeypatch.setattr(seed, "ingest_rankings", lambda **_: None)

    def fake_enrich(_ids, force=False):  # noqa: ARG001
        print(
            "Enrichment summary: 1 attempted, 1 already enriched, 0 no name, 0 enriched, 0 failed"
        )
        return 0

    monkeypatch.setattr(seed, "enrich_players", fake_enrich)

    seed.seed_rankings_and_enrichment(["A0E2"], enrich=True, reset=False)

    out = capsys.readouterr().out
    assert (
        "Enrichment summary: 1 attempted, 1 already enriched, 0 no name, 0 enriched, 0 failed"
        in out
    )
    assert "seeded player profiles" not in out  # old success-only line is gone
