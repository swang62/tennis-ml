"""Canonical player-directory query and normalization.

Single source of truth for the directory data contract used by the deploy-time
static directory artifact written under web/public/. The raw SQL and the IOC ->
(iso2, country) mapping live here only, so the artifact can never drift from
the source.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pandas as pd

from src.constants import BRONZE_PROFILES_TABLE, GOLD_PROFILES_TABLE
from src.countries import resolve_ioc, valid_ioc
from src.models.similarity import DEFAULT_CLUSTER_LABELS, DEFAULT_CLUSTERS

# Directory read: bronze metadata (name/IOC) joined to the dbt-derived gold
# aggregates. current_rank is the player's latest official weekly rank
# (bronze.rankings), falling back to match-time rank from the most recent
# match when no ranking row exists — both materialized by dbt in gold.
PLAYERS_SQL = f"""
SELECT bp.player_id, bp.display_name, bp.ioc,
       gp.match_count AS matches_played,
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


def _load_cluster_labels() -> dict[str, str]:
    """Map player_id -> archetype label from the clustering artifacts, if present.

    The cluster-assignment parquet and description JSON are built by the
    playstyle-cluster exploration notebook. Absent artifacts (e.g. before the
    first clustering run) yield an empty map so the directory never breaks.
    """
    if not DEFAULT_CLUSTERS.exists() or not DEFAULT_CLUSTER_LABELS.exists():
        return {}
    try:
        labels = json.loads(DEFAULT_CLUSTER_LABELS.read_text())
        assigns = pd.read_parquet(DEFAULT_CLUSTERS)
    except (FileNotFoundError, ValueError, KeyError):
        return {}
    return {
        str(row["player_id"]): labels.get(str(row["cluster_id"]), f"cluster_{row['cluster_id']}")
        for _, row in assigns.iterrows()
    }


def directory_players(df: pd.DataFrame) -> list[dict[str, object]]:
    """Map raw directory rows to the deploy artifact's player shape.

    Each entry carries only the static-picker fields, preserving SQL row order.
    cluster_label is None until the clustering artifacts exist (the frontend
    renders "-" for it).
    """
    cluster_labels = _load_cluster_labels()
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
                "cluster_label": cluster_labels.get(str(record["player_id"])),
            }
        )
    return players
