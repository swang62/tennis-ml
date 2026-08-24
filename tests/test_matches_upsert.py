"""Hermetic tests for bronze insert-or-force-replace behavior."""

from datetime import date

import pandas as pd

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


def _query_for(existing_row):
    def query(sql, params):  # noqa: ARG001
        if existing_row is None:
            return pd.DataFrame(columns=BRONZE_COLUMNS)
        return pd.DataFrame([existing_row], columns=BRONZE_COLUMNS)

    return query


def test_existing_match_is_skipped_by_default_with_no_write(monkeypatch):
    calls = []
    monkeypatch.setattr(
        matches, "_copy_df_into", lambda *args, **kwargs: calls.append((args, kwargs)) or 0
    )

    record = matches.upsert_bronze_match(_candidate_row(), query=_query_for(_stored_row()))

    assert record["action"] == "noop"
    assert record["reason"] is None
    assert record["update_cols"] == []
    assert record["rows_affected"] == 0
    assert calls == []  # existing row: no write at all, no selective stat fills


def test_force_replaces_existing_row_across_every_non_key_column(monkeypatch):
    captured = {}

    def fake_copy(table, df, *, conflict_col, update_cols):
        captured.update(table=table, df=df, conflict_col=conflict_col, update_cols=update_cols)
        return 1

    monkeypatch.setattr(matches, "_copy_df_into", fake_copy)

    record = matches.upsert_bronze_match(
        _candidate_row(), force=True, query=_query_for(_stored_row())
    )

    assert record["action"] == "updated"
    assert record["reason"] is None
    assert record["rows_affected"] == 1
    assert record["update_cols"] == [c for c in BRONZE_COLUMNS if c != "match_id"]
    assert captured["conflict_col"] == "match_id"
    assert captured["update_cols"] == record["update_cols"]
    assert captured["table"] == matches.BRONZE_MATCHES_TABLE
    written = captured["df"].iloc[0].to_dict()
    assert set(written) == set(BRONZE_COLUMNS)
    assert written["player1_aces"] == 12  # candidate value wins, no sentinel logic
    assert written["score"] == "6-4 7-6 6-3"  # non-stat columns replaced too


def test_force_update_replaces_best_of(monkeypatch):
    written: dict[str, object] = {}

    def fake_copy(table, df, *, conflict_col, update_cols):  # noqa: ARG001
        written.update(df.iloc[0].to_dict())
        return 1

    monkeypatch.setattr(matches, "_copy_df_into", fake_copy)

    stored = _stored_row(best_of=3)
    candidate = _candidate_row(best_of=5)
    record = matches.upsert_bronze_match(candidate, force=True, query=_query_for(stored))

    assert record["action"] == "updated"
    assert "best_of" in written
    assert written["best_of"] == 5  # best_of is part of the force-replace update set


def test_repeated_force_runs_converge_to_the_candidate_row(monkeypatch):
    written = {}

    def fake_copy(table, df, *, conflict_col, update_cols):  # noqa: ARG001
        written.update(df.iloc[0].to_dict())
        return 1

    monkeypatch.setattr(matches, "_copy_df_into", fake_copy)

    candidate = _candidate_row(player1_aces=12, score="6-4 6-4")
    matches.upsert_bronze_match(candidate, force=True, query=_query_for(_stored_row()))
    first = dict(written)
    matches.upsert_bronze_match(candidate, force=True, query=_query_for(first))
    second = dict(written)

    assert second == first
    assert second["player1_aces"] == 12
    assert second["score"] == "6-4 6-4"


def test_new_match_insert_still_uses_on_conflict_do_nothing(monkeypatch):
    captured = {}

    def fake_copy(table, df, *, conflict_col, update_cols):  # noqa: ARG001
        captured.update(conflict_col=conflict_col, update_cols=update_cols)
        return 1

    monkeypatch.setattr(matches, "_copy_df_into", fake_copy)

    record = matches.upsert_bronze_match(_stored_row(), query=_query_for(None))

    assert record["action"] == "inserted"
    assert record["update_cols"] == []
    assert captured == {"conflict_col": "match_id", "update_cols": None}


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
