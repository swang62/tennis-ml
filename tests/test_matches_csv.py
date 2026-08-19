"""Hermetic tests for the Sackmann CSV sink shared with the bronze upsert."""

import csv

import src.flows.matches as matches


def _raw_csv_row(tourney_id: str, match_num: str, tourney_date: str = "20260105") -> dict[str, str]:
    """Minimal Sackmann-format row; id derivation only needs the three fields."""
    return {"tourney_id": tourney_id, "match_num": match_num, "tourney_date": tourney_date}


def _bronze_row() -> dict[str, object]:
    return {
        "match_id": "2026-418-026",
        "match_date": "2026-01-05",
        "tournament": "grand_slam",
        "tournament_name": "Test Open",
        "surface": "hard",
        "round": "sf",
        "is_indoor": 0,
        "winner_id": "W1",
        "player2_id": "L1",
        "player1_ranking": 10,
        "player2_ranking": 20,
        "player1_rank_points": 9000,
        "player2_rank_points": 3000,
        "player1_age": 24.41,
        "player2_age": 28.75,
        "score": "6-4 7-6",
        "player1_aces": 5,
        "player2_aces": 4,
        "player1_first_serves_made": 40,
        "player2_first_serves_made": 35,
    }


def test_csv_row_match_id_derives_year_from_row_date():
    """The date-derived year is prepended once, and only when the tourney_id
    does not already repeat that same year at its start."""
    assert matches._csv_row_match_id(_raw_csv_row("2026-418", "26")) == "2026-418-026"
    assert matches._csv_row_match_id(_raw_csv_row("1987", "26")) == "2026-1987-026"
    assert matches._csv_row_match_id(_raw_csv_row("1987-foo", "26")) == "2026-1987-foo-026"


def test_bronze_row_to_raw_match_round_trips_and_keeps_column_order():
    raw = matches.bronze_row_to_raw_match(_bronze_row())

    assert list(raw) == matches.RAW_MATCH_COLUMNS  # exact Sackmann header order
    assert raw["tourney_id"] == "2026-418"
    assert raw["match_num"] == "26"
    assert raw["tourney_date"] == "20260105"
    assert raw["winner_id"] == "W1"
    assert raw["loser_id"] == "L1"
    assert raw["surface"] == "Hard"
    assert raw["tourney_level"] == "G"
    assert raw["indoor"] == "O"
    assert raw["round"] == "SF"
    assert raw["winner_rank"] == "10"
    assert raw["loser_rank"] == "20"
    assert raw["winner_age"] == "24.410"
    assert raw["w_ace"] == "5"
    assert raw["l_ace"] == "4"
    assert raw["w_1stIn"] == "40"
    assert raw["l_1stIn"] == "35"


def test_append_raw_match_rows_creates_header_and_appends_once(tmp_path):
    path = tmp_path / "2026.csv"
    a = _raw_csv_row("2026-418", "001")
    b = _raw_csv_row("1987", "001")

    appended, ids = matches.append_raw_match_rows([a, b, a], path)

    assert appended == 2  # the in-batch repeat of a is deduped
    assert ids == {"2026-418-001", "2026-1987-001"}
    rows = list(csv.DictReader(path.open()))
    assert [r["tourney_id"] for r in rows] == ["2026-418", "1987"]
    assert rows[0]["tourney_date"] == "20260105"
    assert len(rows[0]) == len(matches.RAW_MATCH_COLUMNS)  # header column order preserved


def test_append_raw_match_rows_dedupes_against_existing_file(tmp_path):
    path = tmp_path / "2026.csv"
    a = _raw_csv_row("2026-418", "001")
    b = _raw_csv_row("2026-419", "001")

    matches.append_raw_match_rows([a], path)
    # Existing set loaded from the file: a is known, b is new.
    appended, ids = matches.append_raw_match_rows(
        [a, b], path, existing=matches.load_csv_match_ids(path)
    )
    assert appended == 1
    assert ids == {"2026-418-001", "2026-419-001"}
    appended_twice, _ids = matches.append_raw_match_rows([a, b], path, existing=ids)
    assert appended_twice == 0
    rows = list(csv.DictReader(path.open()))
    assert [r["tourney_id"] for r in rows] == ["2026-418", "2026-419"]
