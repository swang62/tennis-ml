"""Pure Elo calculations safe to import from serving."""

from src.constants import (
    ELO_DEFAULT_RATING,
    ELO_INACTIVITY_GRACE_DAYS,
    ELO_INACTIVITY_REGRESS_CAP,
    ELO_INACTIVITY_REGRESS_PER_7D,
)


def regress_rating(rating: float, gap_days: int | None) -> float:
    """Pull a stale rating toward 1500 after the inactivity grace period."""
    if gap_days is None or gap_days <= ELO_INACTIVITY_GRACE_DAYS:
        return rating
    excess = gap_days - ELO_INACTIVITY_GRACE_DAYS
    periods = excess // 7
    if periods <= 0:
        return rating
    factor = (1.0 - ELO_INACTIVITY_REGRESS_PER_7D) ** periods
    factor = max(factor, 1.0 - ELO_INACTIVITY_REGRESS_CAP)
    return ELO_DEFAULT_RATING + (rating - ELO_DEFAULT_RATING) * factor
