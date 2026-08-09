"""Lightweight bronze-row validation before insertion."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypedDict

import pandas as pd

from src.features.columns import (
    _REQUIRED_STRING_COLUMNS,
    BRONZE_COLUMNS,
    BRONZE_COLUMNS_FLOAT,
    BRONZE_COLUMNS_INT,
    BRONZE_COLUMNS_INT32,
)

# Unknown indoor status is valid at ingest.
BRONZE_COLUMNS_NULLABLE: tuple[str, ...] = ("is_indoor", "tournament_name")


class IngestionCheckReport(TypedDict):
    passed: bool
    results: list[str]
    valid_df: pd.DataFrame
    input_rows: int
    valid_rows: int
    dropped_rows: int


def _is_missing(value: Any) -> bool:
    return bool(pd.isna(value))


def _as_number(value: Any) -> int | float | None:
    if _is_missing(value):
        return None
    return value if isinstance(value, int | float) else None


def validate_bronze_row(row: Mapping[str, Any]) -> list[str]:
    """Return row-level bronze issues not already delegated to SQL constraints."""
    issues: list[str] = []

    for column in BRONZE_COLUMNS:
        if column in BRONZE_COLUMNS_NULLABLE:
            continue
        value = row.get(column)
        if _is_missing(value):
            issues.append(f"{column} is null")

    for column in _REQUIRED_STRING_COLUMNS:
        value = row.get(column)
        if not isinstance(value, str) or not value.strip():
            issues.append(f"{column} is blank")

    for column in BRONZE_COLUMNS_INT:
        value = _as_number(row.get(column))
        if value is None:
            continue
        if value < 0 or value > 255:
            issues.append(f"{column} outside SMALLINT 0..255: {value}")

    for column in BRONZE_COLUMNS_INT32:
        value = _as_number(row.get(column))
        if value is None:
            continue
        if value < 0 or value > 20000:
            issues.append(f"{column} outside INTEGER 0..20000: {value}")

    for column in BRONZE_COLUMNS_FLOAT:
        value = _as_number(row.get(column))
        if value is None:
            continue
        if value < 0 or value > 100:
            issues.append(f"{column} outside 0..100: {value}")

    if row.get("player1_id") == row.get("player2_id"):
        issues.append("player1_id equals player2_id")
    if row.get("winner_id") != row.get("player1_id"):
        issues.append("winner_id must equal player1_id")

    for side in ("player1", "player2"):
        wins = _as_number(row.get(f"{side}_wins_last_10"))
        matches = _as_number(row.get(f"{side}_matches_last_10"))
        if wins is not None and matches is not None and wins > matches:
            issues.append(f"{side}_wins_last_10 exceeds {side}_matches_last_10")

        break_points_saved = _as_number(row.get(f"{side}_break_points_saved"))
        break_points_faced = _as_number(row.get(f"{side}_break_points_faced"))
        if (
            break_points_saved is not None
            and break_points_faced is not None
            and break_points_saved > break_points_faced
        ):
            issues.append(f"{side}_break_points_saved exceeds {side}_break_points_faced")

        first_serves_made = _as_number(row.get(f"{side}_first_serves_made"))
        total_serve_points = _as_number(row.get(f"{side}_total_serve_points"))
        if (
            first_serves_made is not None
            and total_serve_points is not None
            and first_serves_made > total_serve_points
        ):
            issues.append(f"{side}_first_serves_made exceeds {side}_total_serve_points")

    return issues


def run_ingestion_checks(df: pd.DataFrame) -> IngestionCheckReport:
    """Validate bronze rows and return the valid subset plus a drop report."""
    valid_rows: list[dict[str, Any]] = []
    dropped_rows: list[str] = []
    invalid_row_count = 0

    records = ({str(k): v for k, v in r.items()} for r in df.to_dict(orient="records"))
    for row_index, row in enumerate(records):
        row_issues = validate_bronze_row(row)
        if row_issues:
            invalid_row_count += 1
            dropped_rows.extend(
                f"row {row_index} ({row.get('match_id', '<missing>')}): {issue}"
                for issue in row_issues
            )
            continue
        valid_rows.append(row)

    valid_df = pd.DataFrame(valid_rows, columns=df.columns)
    return {
        "passed": invalid_row_count == 0,
        "results": dropped_rows,
        "valid_df": valid_df,
        "input_rows": len(df),
        "valid_rows": len(valid_df),
        "dropped_rows": invalid_row_count,
    }
