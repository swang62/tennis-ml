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


def test_csv_row_match_id_normalizes_date_like_explicit_year():
    """An explicitly passed YYYYMMDD year is reduced to its edition year, so the
    CSV dedup set can never carry a date-prefixed id."""
    assert (
        matches._csv_row_match_id(_raw_csv_row("1967-southern-pro", "1", "19670220"), 19670220)
        == "1967-southern-pro-001"
    )


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


def test_raw_match_row_fills_profile_fields_in_winner_loser_columns():
    raw = matches.bronze_row_to_raw_match(
        _bronze_row(),
        {
            "W1": {"display_name": "Won One", "hand": "L", "height": "190", "ioc": "FRA"},
            "L1": {"display_name": "Lost One", "hand": "R", "height": "188", "ioc": "GBR"},
        },
    )

    assert list(raw) == matches.RAW_MATCH_COLUMNS  # exact Sackmann header order
    assert raw["winner_name"] == "Won One"
    assert raw["winner_hand"] == "L"
    assert raw["winner_ht"] == "190"
    assert raw["winner_ioc"] == "FRA"
    assert raw["loser_name"] == "Lost One"
    assert raw["loser_hand"] == "R"
    assert raw["loser_ht"] == "188"
    assert raw["loser_ioc"] == "GBR"


def test_raw_match_row_fills_source_metadata_columns():
    row = _bronze_row()
    row.update(
        {
            "winner_seed": "5",
            "loser_seed": "28",
            "winner_entry": "WC",
            "loser_entry": "Q",
            "draw_size": 96,
            "best_of": 3,
            "minutes": 87,
        }
    )

    raw = matches.bronze_row_to_raw_match(row)

    assert list(raw) == matches.RAW_MATCH_COLUMNS
    assert raw["winner_seed"] == "5"
    assert raw["winner_entry"] == "WC"
    assert raw["loser_seed"] == "28"
    assert raw["loser_entry"] == "Q"
    assert raw["draw_size"] == "96"
    assert raw["best_of"] == "3"
    assert raw["minutes"] == "87"


def test_bronze_row_to_raw_match_writes_normalized_best_of():
    """The raw CSV carries the normalized best_of verbatim, never the raw source."""
    row = _bronze_row()
    row["best_of"] = 5

    raw = matches.bronze_row_to_raw_match(row)

    assert raw["best_of"] == "5"


def test_raw_match_row_leaves_unknown_metadata_blank_and_prefers_profile_name():
    row = _bronze_row()
    row["player1_name"] = "Ben Shelton"
    row["player2_name"] = "Brandon Nakashima"

    raw = matches.bronze_row_to_raw_match(
        row, {"W1": {"display_name": "Profile Name", "hand": "", "height": "", "ioc": ""}}
    )

    assert list(raw) == matches.RAW_MATCH_COLUMNS
    for column in (
        "winner_seed",
        "winner_entry",
        "loser_seed",
        "loser_entry",
        "draw_size",
        "best_of",
        "minutes",
    ):
        assert raw[column] == ""  # absent source fields are never fabricated
    assert raw["winner_name"] == "Profile Name"  # profile display name wins
    assert raw["loser_name"] == "Brandon Nakashima"  # page name fallback
    assert raw["winner_hand"] == "" and raw["loser_ht"] == ""  # no profile values


def test_append_raw_match_rows_dedupes_metadata_rows(tmp_path):
    path = tmp_path / "2026.csv"
    a = matches.bronze_row_to_raw_match({**_bronze_row(), "winner_seed": "5", "draw_size": 96})
    b = matches.bronze_row_to_raw_match({**_bronze_row(), "match_id": "2026-419-001"})

    matches.append_raw_match_rows([a, b], path)
    # Metadata-bearing rows dedupe by the same canonical id as plain rows.
    appended, ids = matches.append_raw_match_rows(
        [a, b], path, existing=matches.load_csv_match_ids(path)
    )
    assert appended == 0
    assert ids == {"2026-418-026", "2026-419-001"}

    rows = list(csv.DictReader(path.open()))
    assert rows[0]["winner_seed"] == "5"
    assert rows[0]["draw_size"] == "96"
