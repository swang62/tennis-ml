"""Load and validate the gold.tour_averages singleton for inference and profiles."""

from src.constants import GOLD_TOUR_AVERAGES_TABLE
from src.db.client import execute_df, first_row_dict

_TOUR_AVERAGES_SQL = f"SELECT * FROM {GOLD_TOUR_AVERAGES_TABLE}"
_VALIDATE_ONCE_SQL = f"SELECT singleton_id FROM {GOLD_TOUR_AVERAGES_TABLE}"


def load_tour_averages() -> dict[str, object]:
    """Return the sole tour_averages row with singleton_id=1; otherwise raise RuntimeError."""
    df = execute_df(_TOUR_AVERAGES_SQL)
    if df.empty:
        raise RuntimeError(f"{GOLD_TOUR_AVERAGES_TABLE} is empty: run dbt build first")
    if len(df) != 1:
        raise RuntimeError(f"{GOLD_TOUR_AVERAGES_TABLE} has {len(df)} rows; expected exactly 1")
    row = first_row_dict(df)
    if row.get("singleton_id") != 1:
        raise RuntimeError(
            f"{GOLD_TOUR_AVERAGES_TABLE} singleton_id = {row.get('singleton_id')}; expected 1"
        )
    return row
