"""Hermetic tests for bronze insert-or-force-replace behavior."""

from datetime import date

import src.flows.matches as matches
from src.features.columns import BRONZE_COLUMNS, BRONZE_COLUMNS_INT

_BASE = {
    "match_id": "2026-418-026",
    "match_date": date(2026, 1, 5),
    "match_num": 26,
    "player1_id": "W1",
    "player2_id": "L1",
    "tournament": "grand_slam",
    "tournament_name": "Test Open",
    "round": "sf",
    "surface": "hard",
    "score": "6-4 7-6",
    "is_indoor": 0,
    "player1_ranking": 10,
    "player2_ranking": 20,
    "player1_rank_points": 9000,
    "player2_rank_points": 3000,
    "player1_age": 24.41,
    "player2_age": 28.75,
    "winner_id": "W1",
}


def _stored_row(**overrides):
    """Stored bronze row with every column; player1_aces holds the 0 sentinel."""
    stats = dict.fromkeys(BRONZE_COLUMNS_INT, 0)
    row = {
        **dict.fromkeys(BRONZE_COLUMNS, None),
        **stats,
        **_BASE,
        "player1_aces": 0,
        "player2_aces": 4,
        "score": "6-4 7-6",
        **overrides,
    }
    return row


def _candidate_row(**overrides):
    return _stored_row(**{"player1_aces": 12, "score": "6-4 7-6 6-3", **overrides})


def _capture_writes(monkeypatch, affected=None):
    calls = []

    def fake_copy(table, df, *, conflict_col, update_cols):
        calls.append(
            {"table": table, "df": df, "conflict_col": conflict_col, "update_cols": update_cols}
        )
        return len(df) if affected is None else affected

    monkeypatch.setattr(matches, "_copy_df_into", fake_copy)
    return calls


def test_all_valid_rows_are_inserted_in_one_do_nothing_write(monkeypatch):
    calls = _capture_writes(monkeypatch)
    rows = [_candidate_row(match_id=f"2026-418-0{n}") for n in (26, 27, 28)]

    outcome = matches.upsert_bronze_matches(rows, known_ids={})

    assert len(calls) == 1
    assert list(calls[0]["df"]["match_id"]) == ["2026-418-026", "2026-418-027", "2026-418-028"]
    assert calls[0]["conflict_col"] == "match_id"
    assert calls[0]["update_cols"] is None  # a stored row is never overwritten without force
    assert outcome["inserted"] == 3
    assert outcome["skipped"] == []


def test_invalid_rows_are_skipped_with_a_reason_and_the_rest_still_written(monkeypatch):
    calls = _capture_writes(monkeypatch)
    good = _candidate_row(match_id="2026-418-026")
    bad = _candidate_row(match_id="2026-418-027", match_date=None)

    outcome = matches.upsert_bronze_matches([good, bad], known_ids={})

    assert list(calls[0]["df"]["match_id"]) == ["2026-418-026"]
    assert outcome["inserted"] == 1
    assert [match_id for match_id, _ in outcome["skipped"]] == ["2026-418-027"]
    assert "match_date" in outcome["skipped"][0][1]


def test_nothing_is_written_when_no_row_is_valid(monkeypatch):
    calls = _capture_writes(monkeypatch)

    outcome = matches.upsert_bronze_matches([_candidate_row(match_num=None)], known_ids={})

    assert calls == []
    assert outcome["inserted"] == 0
    assert len(outcome["skipped"]) == 1


def test_rows_the_database_already_holds_count_as_noop(monkeypatch):
    _capture_writes(monkeypatch, affected=1)
    rows = [_candidate_row(match_id="2026-418-026"), _candidate_row(match_id="2026-418-027")]

    outcome = matches.upsert_bronze_matches(rows, known_ids={})

    assert outcome["inserted"] == 1
    assert outcome["noop"] == 1


def test_force_replaces_every_non_key_column_and_counts_stored_rows_as_updated(monkeypatch):
    calls = _capture_writes(monkeypatch)
    stored = _candidate_row(match_id="2026-418-026", player1_aces=12, best_of=5)
    fresh = _candidate_row(match_id="2026-418-027")

    outcome = matches.upsert_bronze_matches(
        [stored, fresh], known_ids={"2026-418-026": object()}, force=True
    )

    assert calls[0]["update_cols"] == [c for c in BRONZE_COLUMNS if c != "match_id"]
    written = calls[0]["df"].iloc[0].to_dict()
    assert set(written) == set(BRONZE_COLUMNS)
    assert written["player1_aces"] == 12
    assert written["best_of"] == 5
    assert outcome["updated"] == 1
    assert outcome["inserted"] == 1


def test_parse_args_force_defaults_false_and_flag_sets_true():
    assert matches.parse_args([]).force is False
    assert matches.parse_args(["--force"]).force is True


def test_main_threads_force_into_the_flow(monkeypatch):
    captured = {}
    monkeypatch.setattr(matches, "matches_flow", lambda **kwargs: captured.update(kwargs))

    matches.main(["--force"])

    assert captured["force"] is True


def test_validate_new_bronze_row_rejects_null_match_date():
    # match_date is a causal-order key and must be non-null at the scrape boundary.
    reason = matches.validate_new_bronze_row(_candidate_row(match_date=None))

    assert reason is not None
    assert "match_date" in reason


def test_validate_new_bronze_row_rejects_null_match_num():
    # match_num is a causal-order key and must be non-null at the scrape boundary.
    reason = matches.validate_new_bronze_row(_candidate_row(match_num=None))

    assert reason is not None
    assert "match_num" in reason
