"""Canonical player-directory query and normalization."""

from __future__ import annotations

import math
from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pandas as pd

from src.constants import BRONZE_PROFILES_TABLE, GOLD_PROFILES_TABLE
from src.utils.countries import resolve_ioc, valid_ioc

# Keep every bronze profile for directory and similarity builds; gold supplies
# the materialized rank and match count. Extra profile fields feed similarity.
PLAYERS_SQL = f"""
SELECT bp.player_id, bp.display_name, bp.ioc, bp.backhand, bp.handedness, bp.summary,
       bp.height, bp.birthdate, bp.turned_pro,
       COALESCE(gp.match_count, 0) AS matches_played,
       gp.current_rank
FROM {BRONZE_PROFILES_TABLE} bp
LEFT JOIN {GOLD_PROFILES_TABLE} gp ON gp.player_id = bp.player_id
ORDER BY gp.current_rank NULLS LAST, bp.display_name, bp.player_id
"""


def _json_safe(value: object) -> object:
    """Convert database and pandas scalars to JSON-safe values."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (float, Decimal, np.floating)):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, (np.integer, np.bool_)):
        return int(value)
    return value


def directory_players(df: pd.DataFrame) -> list[dict[str, object]]:
    """Map raw rows to the deploy artifact shape, preserving SQL order."""
    players: list[dict[str, object]] = []
    for row in df.to_dict("records"):
        record = {str(k): _json_safe(v) for k, v in row.items()}
        ioc = valid_ioc(record.get("ioc"))
        iso2, _ = resolve_ioc(ioc)
        players.append(
            {
                "player_id": record["player_id"],
                "display_name": record["display_name"],
                "matches_played": record["matches_played"],
                "current_rank": record["current_rank"],
                "ioc": ioc,
                "iso2": iso2,
            }
        )
    return players
