from typing import cast

import pandas as pd

from src.features.validate import (
    IngestionCheckReport,
    _is_missing,
    run_ingestion_checks,
    validate_bronze_row,
)


def _valid_row() -> dict[str, object]:
    return {
        "match_id": "2026-test-001",
        "match_date": "2026-01-01",
        "player1_id": "A001",
        "player2_id": "B001",
        "tournament": "grand_slam",
        "round": "f",
        "surface": "clay",
        "player1_ranking": 1,
        "player2_ranking": 2,
        "player1_wins_last_10": 8,
        "player1_matches_last_10": 10,
        "player1_aces": 5,
        "player1_double_faults": 1,
        "player1_first_serves_made": 20,
        "player1_total_serve_points": 30,
        "player1_first_serve_points_won": 18,
        "player1_second_serve_points_won": 8,
        "player1_service_games": 12,
        "player1_break_points_saved": 3,
        "player1_break_points_faced": 4,
        "player2_wins_last_10": 6,
        "player2_matches_last_10": 10,
        "player2_aces": 3,
        "player2_double_faults": 2,
        "player2_first_serves_made": 18,
        "player2_total_serve_points": 30,
        "player2_first_serve_points_won": 16,
        "player2_second_serve_points_won": 7,
        "player2_service_games": 11,
        "player2_break_points_saved": 1,
        "player2_break_points_faced": 2,
        "player1_rank_points": 9000,
        "player2_rank_points": 3000,
        "player1_age": 24.41,
        "player2_age": 28.75,
        "match_num": 1,
        "winner_id": "A001",
    }


def test_validate_bronze_row_accepts_valid_row():
    assert validate_bronze_row(_valid_row()) == []


def test_validate_bronze_row_flags_invalid_bounds_and_identity_errors():
    row = _valid_row()
    row["player1_break_points_saved"] = -9
    row["player2_first_serves_made"] = 20001
    row["player1_rank_points"] = 20001
    row["player2_age"] = 100.5
    row["winner_id"] = "B001"

    issues = validate_bronze_row(row)

    assert "player1_break_points_saved must be non-negative: -9" in issues
    assert "player2_first_serves_made outside INTEGER 0..20000: 20001" in issues
    assert "player1_rank_points outside INTEGER 0..20000: 20001" in issues
    assert "player2_age outside 0..100: 100.5" in issues
    assert "winner_id must equal player1_id" in issues


def test_run_ingestion_checks_ignores_duplicate_match_ids_and_match_won_column():
    row = _valid_row()
    other = _valid_row() | {"match_won": 2}
    df = pd.DataFrame([row, other])

    result: IngestionCheckReport = run_ingestion_checks(df)

    assert result["passed"] is True
    assert result["results"] == []
    assert result["input_rows"] == 2
    assert result["valid_rows"] == 2
    assert result["dropped_rows"] == 0
    pd.testing.assert_frame_equal(result["valid_df"], df)


def test_run_ingestion_checks_drops_invalid_rows_and_keeps_valid_rows():
    valid = _valid_row()
    invalid = _valid_row() | {
        "match_id": "2026-test-002",
        "player1_break_points_saved": -9,
    }

    result: IngestionCheckReport = run_ingestion_checks(pd.DataFrame([valid, invalid]))

    assert result["passed"] is False
    assert result["input_rows"] == 2
    assert result["valid_rows"] == 1
    assert result["dropped_rows"] == 1
    assert any(
        "player1_break_points_saved must be non-negative: -9" in issue
        for issue in result["results"]
    )
    assert cast(pd.DataFrame, result["valid_df"]).to_dict(orient="records") == [valid]


def test_run_ingestion_checks_does_not_mutate_input_df():
    df = pd.DataFrame([_valid_row()])
    df_copy = df.copy(deep=True)

    run_ingestion_checks(df)

    pd.testing.assert_frame_equal(df, df_copy)


def test_validate_bronze_row_flags_blank_required_string_columns():
    row = _valid_row()
    row["surface"] = "  "

    issues = validate_bronze_row(row)

    assert "surface is blank" in issues


def test_validate_bronze_row_accepts_non_draw_rounds():
    # Non-draw rounds are valid and encode to 0 downstream.
    for round_value in ("rr", "0"):
        row = _valid_row()
        row["round"] = round_value

        issues = validate_bronze_row(row)

        assert issues == []


def test_validate_bronze_row_flags_equal_player_ids():
    row = _valid_row()
    row["player2_id"] = "A001"

    issues = validate_bronze_row(row)

    assert "player1_id equals player2_id" in issues


def test_is_missing():
    assert _is_missing(float("nan")) is True
    assert _is_missing(None) is True
    assert _is_missing(pd.NaT) is True
    assert _is_missing(0) is False
    assert _is_missing("") is False
    assert _is_missing(3) is False
    assert _is_missing("abc") is False


def test_run_ingestion_checks_accepts_extended_and_inconsistent_match_counts():
    valid = _valid_row()
    extended = _valid_row() | {
        "match_id": "2026-test-002",
        "player1_first_serves_made": 361,
        "player1_total_serve_points": 20001,
        "player2_break_points_saved": 7,
        "player2_break_points_faced": 6,
    }

    result: IngestionCheckReport = run_ingestion_checks(pd.DataFrame([valid, extended]))

    assert result["passed"] is True
    assert result["input_rows"] == 2
    assert result["valid_rows"] == 2
    assert result["dropped_rows"] == 0
    kept = cast(pd.DataFrame, result["valid_df"]).to_dict(orient="records")
    assert kept[0] == valid


def test_validate_bronze_row_accepts_null_is_indoor():
    """is_indoor is nullable: unknown indoor status is not an error."""
    row = _valid_row()
    row["is_indoor"] = None

    issues = validate_bronze_row(row)

    assert not any("is_indoor" in issue for issue in issues)


def test_validate_bronze_row_accepts_valid_is_indoor_values():
    for val in (0, 1):
        row = _valid_row()
        row["is_indoor"] = val
        assert validate_bronze_row(row) == [], f"is_indoor={val} should be valid"


def test_validate_bronze_row_accepts_valid_best_of_values():
    for val in (1, 3, 5):
        row = _valid_row()
        row["best_of"] = val
        assert validate_bronze_row(row) == [], f"best_of={val} should be valid"


def test_validate_bronze_row_accepts_null_best_of():
    """best_of is nullable: legacy/seed rows without it are valid."""
    row = _valid_row()
    row["best_of"] = None

    assert validate_bronze_row(row) == []


def test_validate_bronze_row_flags_invalid_best_of():
    row = _valid_row()
    row["best_of"] = 2

    issues = validate_bronze_row(row)

    assert "best_of must be 1, 3, or 5: 2" in issues
