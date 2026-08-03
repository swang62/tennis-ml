from typing import cast

import pandas as pd

from src.features.validate import IngestionCheckReport, run_ingestion_checks, validate_bronze_row


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
        "player1_break_points_won": 3,
        "player1_break_points_total": 4,
        "player2_wins_last_10": 6,
        "player2_matches_last_10": 10,
        "player2_aces": 3,
        "player2_double_faults": 2,
        "player2_first_serves_made": 18,
        "player2_total_serve_points": 30,
        "player2_break_points_won": 1,
        "player2_break_points_total": 2,
        "winner_id": "A001",
    }


def test_validate_bronze_row_accepts_valid_row():
    assert validate_bronze_row(_valid_row()) == []


def test_validate_bronze_row_flags_row_level_semantic_errors():
    row = _valid_row()
    row["player1_break_points_won"] = -9
    row["player2_first_serves_made"] = 31
    row["winner_id"] = "B001"

    issues = validate_bronze_row(row)

    assert "player1_break_points_won outside UTINYINT 0..255: -9" in issues
    assert "player2_first_serves_made exceeds player2_total_serve_points" in issues
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
        "player1_break_points_won": -9,
    }

    result: IngestionCheckReport = run_ingestion_checks(pd.DataFrame([valid, invalid]))

    assert result["passed"] is False
    assert result["input_rows"] == 2
    assert result["valid_rows"] == 1
    assert result["dropped_rows"] == 1
    assert any(
        "player1_break_points_won outside UTINYINT 0..255: -9" in issue
        for issue in result["results"]
    )
    assert cast(pd.DataFrame, result["valid_df"]).to_dict(orient="records") == [valid]
