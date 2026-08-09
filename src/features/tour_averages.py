"""Load and validate the gold.tour_averages singleton for inference and profiles."""

from src.constants import TOUR_AVERAGES_TABLE
from src.db.client import execute_df, first_row_dict

_TOUR_AVERAGES_SQL = f"SELECT * FROM {TOUR_AVERAGES_TABLE}"
_VALIDATE_ONCE_SQL = f"SELECT singleton_id FROM {TOUR_AVERAGES_TABLE}"


def load_tour_averages() -> dict[str, object]:
    """Return the tour_averages singleton row, validated.

    Raises RuntimeError if the table is absent, has != 1 row,
    or singleton_id != 1.
    """
    df = execute_df(_TOUR_AVERAGES_SQL)
    if df.empty:
        raise RuntimeError(f"{TOUR_AVERAGES_TABLE} is empty: run dbt build first")
    if len(df) != 1:
        raise RuntimeError(f"{TOUR_AVERAGES_TABLE} has {len(df)} rows; expected exactly 1")
    row = first_row_dict(df)
    if row.get("singleton_id") != 1:
        raise RuntimeError(
            f"{TOUR_AVERAGES_TABLE} singleton_id = {row.get('singleton_id')}; expected 1"
        )
    return row
