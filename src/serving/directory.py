"""Canonical player-directory query and normalization.

Single source of truth for the directory data contract used by the deploy-time
static directory artifact written under web/public/. The raw SQL and the IOC ->
(iso2, country) mapping live here only, so the artifact can never drift from
the source. The database's actual latest match date (``MAX(match_date)``) is
part of the contract: the artifact carries it, never the deployment wall-clock
time.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pandas as pd

from src.constants import BRONZE_PROFILES_TABLE, BRONZE_TABLE, PROFILES_TABLE
from src.countries import resolve_ioc, valid_ioc

# Directory read: bronze metadata (name/IOC) joined to the dbt-derived gold
# aggregates. current_rank is the player's latest official weekly rank
# (bronze.rankings), falling back to match-time rank from the most recent
# match when no ranking row exists — both materialized by dbt in gold.
PLAYERS_SQL = f"""
SELECT bp.player_id, bp.display_name, bp.ioc,
       gp.match_count AS matches_played,
       gp.latest_rank_points,
       gp.current_rank
FROM {BRONZE_PROFILES_TABLE} bp
LEFT JOIN {PROFILES_TABLE} gp ON gp.player_id = bp.player_id
ORDER BY gp.current_rank NULLS LAST, bp.display_name, bp.player_id
"""

# The actual database latest match date, queried from bronze at deploy time and
# baked into the static web directory artifact ("last updated" footer). Always
# the database value; MAX over an empty table yields NULL, never deploy time.
LATEST_MATCH_DATE_SQL = f"""
SELECT MAX(match_date) AS latest_match_date
FROM {BRONZE_TABLE}
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
    """Map raw directory rows to the deploy artifact's player shape.

    Each entry carries JSON-safe values plus the resolved iso2/country_name
    for the player's normalized IOC, preserving the SQL row order.
    """
    players: list[dict[str, object]] = []
    for row in df.to_dict("records"):
        record = {str(k): _json_safe(v) for k, v in row.items()}
        ioc = valid_ioc(record.get("ioc"))
        iso2, country_name = resolve_ioc(ioc)
        players.append({**record, "ioc": ioc, "iso2": iso2, "country_name": country_name})
    return players


def latest_match_date(df: pd.DataFrame) -> object:
    """Extract the single MAX(match_date) row as a JSON-safe ISO string.

    None when the match table is empty (MAX over no rows). The value is the
    database's latest match date, never the deployment time.
    """
    if df.empty:
        return None
    return _json_safe(df.iloc[0]["latest_match_date"])
