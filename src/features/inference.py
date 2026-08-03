"""ID-based inference feature builder backed by DuckDB gold tables.

Builds ONE finalized canonical `FEATURE_COLS` row for a match between two
players as of a given date, mirroring `gold.match_features` semantics:

* Canonicalization: the lexicographically lower player id is the `player_*`
  side and the higher is `opponent_*`, so `(X, Y)` and `(Y, X)` produce
  identical rows (same as `match_features.sql`'s `WHERE p.player_id < o.player_id`).
* Rolling form comes from each player's newest
  `gold.rolling_features` snapshot STRICTLY BEFORE the as-of date
  (no dedicated latest table/view; the newest snapshot is immediately usable).
* `days_since_last_match` (as-of date minus that snapshot's date) and
  `matches_30d` (count of `silver.player_matches` rows in
  `[as_of - 30 days, as_of)`) are computed at the as-of date.
* Missing players (no eligible snapshot) and NULL snapshot cells are imputed
  from on-demand global aggregates over the eligible snapshot pool: MEDIAN
  for ranking/streak-related values, MEAN for other numerics. When BOTH
  players are missing, both sides receive the same imputed values, so every
  pairwise differential is 0 (neutral).
* Profile-derived identity (height, is_left_handed, years_pro-at-as-of-date)
  comes from `gold.player_profiles`; players without a profile and NULL cells
  are imputed from on-demand aggregates over all profiles (mean height, mean
  left-handed rate, mean years-pro at the as-of date), so the same neutral
  differentials hold for two profile-less players.
* If the eligible pool is empty, constant fallbacks are used:
  ranking=100, rates=0.0, streak=0, days_since_last_match=365, matches_30d=0,
  height=183, left-handed rate=0.0, years_pro=8.

Player ids and dates NEVER appear inside SQL strings: every one flows through
`?` placeholders and a params list.
"""

from __future__ import annotations

import builtins
from datetime import date, datetime
from numbers import Real
from time import perf_counter
from typing import cast

import pandas as pd

from src.constants import (
    GOLD_ROLLING_FEATURES,
    PROFILES_TABLE,
    SILVER_PLAYER_MATCHES,
)
from src.db.client import execute_df, first_row_dict
from src.features.columns import (
    FEATURE_COLS,
    OPPONENT_COLS,
    PLAYER_COLS,
)

VALID_SURFACES = {"clay", "grass", "hard"}
VALID_TOURNAMENT_LEVELS = {0, 1, 2, 3, 4}
VALID_ROUND_ENCODINGS = {0, 1, 2, 3, 4, 5, 6, 7}

# Optional convenience layer: human-readable tournament/round strings mapped to
# the SAME integer encodings as the CASE expressions in
# dbt/models/gold/match_features.sql. Unknown strings map to 0 (that codebook's
# ELSE branch), so stored encodings are unchanged (finals stays 7, unknown 0).
_TOURNAMENT_LEVELS = {
    "grand_slam": 4,
    "masters": 3,
    "atp_500": 2,
    "atp_250": 1,
}
_ROUND_ENCODINGS = {
    "r128": 1,
    "r64": 2,
    "r32": 3,
    "r16": 4,
    "qf": 5,
    "sf": 6,
    "f": 7,
}

# Constant fallbacks when the imputation pool is empty (no eligible snapshots
# anywhere before the as-of date). Mirrors the documented cold-start fallback
# days_since_last_match=365 from gold.match_features.
_DEFAULT_RANKING = 100
_DEFAULT_RATE = 0.0
_DEFAULT_STREAK = 0
_DEFAULT_DAYS_SINCE = 365
_DEFAULT_MATCHES_30D = 0
_DEFAULT_HEIGHT = 183.0
_DEFAULT_LEFT_HANDED_RATE = 0.0
_DEFAULT_YEARS_PRO = 8.0

_SURFACE_TO_SNAPSHOT_COL = {
    "clay": "clay_win_rate_10",
    "grass": "grass_win_rate_10",
    "hard": "hard_win_rate_10",
}

# Latest snapshot per player strictly before the as-of date.
_LATEST_SNAPSHOT_SQL = f"""
SELECT * FROM {GOLD_ROLLING_FEATURES}
WHERE player_id = ?
  AND snapshot_date < ?::DATE
ORDER BY player_match_number DESC
LIMIT 1
"""

# 30-day pre-match activity count, same window semantics as player_matches'
# matches_30d_before ([as_of - 30 days, as_of), strictly).
_MATCHES_30D_SQL = f"""
SELECT COUNT(*) AS n
FROM {SILVER_PLAYER_MATCHES}
WHERE player_id = ?
  AND match_date >= ?::DATE - INTERVAL '30 days'
  AND match_date < ?::DATE
"""

# On-demand global imputation pool over all snapshots strictly before the
# as-of date. One row; MEDIAN for ranking/streak-related, MEAN for others.
_POOL_AGG_SQL = f"""
SELECT
    MEDIAN(latest_player_ranking) AS latest_player_ranking,
    MEDIAN(win_streak)            AS win_streak,
    AVG(win_rate_5)               AS win_rate_5,
    AVG(win_rate_10)              AS win_rate_10,
    AVG(win_rate_20)              AS win_rate_20,
    AVG(ace_rate_5)               AS ace_rate_5,
    AVG(ace_rate_10)              AS ace_rate_10,
    AVG(first_serve_pct_5)        AS first_serve_pct_5,
    AVG(first_serve_pct_10)       AS first_serve_pct_10,
    AVG(break_pct_5)              AS break_pct_5,
    AVG(break_pct_10)             AS break_pct_10,
    AVG(avg_player_rank_10)       AS avg_player_rank_10,
    AVG(avg_player_rank_20)       AS avg_player_rank_20,
    AVG(clay_win_rate_10)         AS clay_win_rate_10,
    AVG(grass_win_rate_10)        AS grass_win_rate_10,
    AVG(hard_win_rate_10)         AS hard_win_rate_10
FROM {GOLD_ROLLING_FEATURES}
WHERE snapshot_date < ?::DATE
"""

# MEDIAN of per-player days_since_last_match at the as-of date, over players
# that have an eligible snapshot. One row; NULL when the pool is empty.
_MEDIAN_DAYS_SQL = f"""
SELECT MEDIAN(days_since) AS median_days_since
FROM (
    SELECT DATEDIFF('day', MAX(snapshot_date), ?::DATE) AS days_since
    FROM {GOLD_ROLLING_FEATURES}
    WHERE snapshot_date < ?::DATE
    GROUP BY player_id
)
"""

# MEDIAN of per-player matches_30d at the as-of date, over players that have
# an eligible snapshot (players with zero window matches contribute 0).
_MEDIAN_MATCHES_30D_SQL = f"""
SELECT MEDIAN(matches_30d) AS median_matches_30d
FROM (
    SELECT pr.player_id, COUNT(pm.match_id) AS matches_30d
    FROM (
        SELECT DISTINCT player_id
        FROM {GOLD_ROLLING_FEATURES}
        WHERE snapshot_date < ?::DATE
    ) pr
    LEFT JOIN {SILVER_PLAYER_MATCHES} pm
        ON pm.player_id = pr.player_id
       AND pm.match_date >= ?::DATE - INTERVAL '30 days'
       AND pm.match_date < ?::DATE
    GROUP BY pr.player_id
)
"""

# Per-player static identity from gold.player_profiles (one row per player).
_PROFILE_SQL = f"""
SELECT player_id, height, handedness, turned_pro
FROM {PROFILES_TABLE}
WHERE player_id = ?
"""

# On-demand profile imputation pool over ALL profiles (identity is static, so
# no as-of window applies beyond the years_pro computation): mean height, mean
# left-handed rate, and mean years-pro at the as-of date. One row; NULL when
# no profiles exist.
_PROFILE_POOL_AGG_SQL = f"""
SELECT
    AVG(height) AS avg_height,
    AVG(CASE WHEN handedness = 'L' THEN 1 ELSE 0 END) AS left_handed_rate,
    AVG(?::INTEGER - turned_pro) AS avg_years_pro
FROM {PROFILES_TABLE}
"""

_POOL_COUNTS_SQL = f"""
SELECT
    COUNT(*) AS snapshot_pool_rows,
    COUNT(DISTINCT player_id) AS snapshot_pool_players
FROM {GOLD_ROLLING_FEATURES}
WHERE snapshot_date < ?::DATE
"""

_PROFILE_COUNTS_SQL = f"""
SELECT COUNT(*) AS profile_rows
FROM {PROFILES_TABLE}
"""


def _to_date(value: object) -> date:
    """Coerce a DuckDB DATE cell (Timestamp/date/datetime/str) to a plain date."""
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError(f"cannot coerce {value!r} to a date")


def _agg_or(agg: dict[str, float], col: str, default: float) -> float:
    """Return the pool aggregate for `col`, or `default` if NULL/missing."""
    value = agg.get(col)
    if value is None or pd.isna(value):
        return default
    return float(value)


def _side_values(
    row: dict[str, object] | None,
    as_of_date: date,
    surface: str,
    agg: dict[str, float],
    median_days_since: float,
    median_matches_30d: float,
) -> dict[str, int | float]:
    """Build one canonical side's 16 values, imputing NULLs and cold starts.

    Keys: "ranking" plus the GOLD_ROLLING_COLS names. `row` is the side's
    latest eligible snapshot (None on cold start).
    """

    def cell(snapshot_col: str, default: float) -> float:
        if row is not None:
            value = row.get(snapshot_col)
            # NaN self-compares unequal, so `value == value` is the NaN test.
            if value is not None and isinstance(value, Real) and value == value:
                return float(value)
        return _agg_or(agg, snapshot_col, default)

    ranking = round(cell("latest_player_ranking", _DEFAULT_RANKING))
    avg_rank_10 = cell("avg_player_rank_10", _DEFAULT_RANKING)
    avg_rank_20 = cell("avg_player_rank_20", _DEFAULT_RANKING)

    if row is not None:
        days_since = int((as_of_date - _to_date(row["snapshot_date"])).days)
        matches_30d = int(
            execute_df(
                _MATCHES_30D_SQL,
                [row["player_id"], as_of_date.isoformat(), as_of_date.isoformat()],
            ).iloc[0]["n"]
        )
    else:
        days_since = round(median_days_since)
        matches_30d = round(median_matches_30d)

    return {
        "ranking": ranking,
        "win_rate_5": cell("win_rate_5", _DEFAULT_RATE),
        "win_rate_10": cell("win_rate_10", _DEFAULT_RATE),
        "win_rate_20": cell("win_rate_20", _DEFAULT_RATE),
        "ace_rate_5": cell("ace_rate_5", _DEFAULT_RATE),
        "ace_rate_10": cell("ace_rate_10", _DEFAULT_RATE),
        "first_serve_pct_5": cell("first_serve_pct_5", _DEFAULT_RATE),
        "first_serve_pct_10": cell("first_serve_pct_10", _DEFAULT_RATE),
        "break_pct_5": cell("break_pct_5", _DEFAULT_RATE),
        "break_pct_10": cell("break_pct_10", _DEFAULT_RATE),
        "rank_trend_10": avg_rank_10 - ranking,
        "rank_trend_20": avg_rank_20 - ranking,
        "win_streak": int(cell("win_streak", _DEFAULT_STREAK)),
        "days_since_last_match": days_since,
        "matches_30d": matches_30d,
        "surface_win_rate_10": cell(_SURFACE_TO_SNAPSHOT_COL[surface], _DEFAULT_RATE),
    }


def _profile_values(pid: str, as_of_date: date, agg: dict[str, float]) -> dict[str, float]:
    """Fetch one side's profile-derived values from gold.player_profiles.

    Keys: "height", "is_left_handed", "years_pro". A missing profile (or a
    NULL height/turned_pro/handedness cell) falls back to the on-demand pool
    aggregates, so two unknown players get identical defaults (every profile
    differential collapses to 0) and the row stays NaN-free. years_pro is
    time-aware: as-of year minus turned_pro, not the raw year.
    """
    df = execute_df(_PROFILE_SQL, [pid])
    if df.empty:
        return {
            "height": _agg_or(agg, "avg_height", _DEFAULT_HEIGHT),
            "is_left_handed": _agg_or(agg, "left_handed_rate", _DEFAULT_LEFT_HANDED_RATE),
            "years_pro": _agg_or(agg, "avg_years_pro", _DEFAULT_YEARS_PRO),
        }
    row = df.iloc[0]
    height = row["height"]
    height = float(height) if not pd.isna(height) else _agg_or(agg, "avg_height", _DEFAULT_HEIGHT)
    turned_pro = row["turned_pro"]
    years_pro = (
        float(as_of_date.year - int(turned_pro))
        if not pd.isna(turned_pro)
        else _agg_or(agg, "avg_years_pro", _DEFAULT_YEARS_PRO)
    )
    # Missing/unknown handedness (NULL or any value other than L/R) uses the
    # pool left-handed rate, mirroring the SQL side where non-L/R handedness
    # stays NULL for train-time imputation. Never hardcode 0 here.
    handedness = row["handedness"]
    is_left_handed = (
        float(handedness == "L")
        if isinstance(handedness, str) and handedness in ("L", "R")
        else _agg_or(agg, "left_handed_rate", _DEFAULT_LEFT_HANDED_RATE)
    )
    return {
        "height": height,
        "is_left_handed": is_left_handed,
        "years_pro": years_pro,
    }


def _build_inference_features_with_meta(
    player_id: str,
    opponent_id: str,
    surface: str,
    *,
    as_of_date: date | None = None,
    tournament_level: int = 0,
    round_encoded: int = 0,
    tournament: str | None = None,
    round: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build ONE finalized canonical inference row.

    Returns a 1-row DataFrame with columns exactly:
        [*FEATURE_COLS, "player_id", "opponent_id"]
    (in that order). The canonical lower-id player is always the player_*
    side, matching gold.match_features' orientation. No NaNs in any cell.

    `tournament_level` and `round_encoded` are the integer context features
    (default 0), validated to the same value sets as the CASE expressions in
    dbt/models/gold/match_features.sql.

    Optional `tournament`/`round` string aliases are a convenience layer that
    maps through `_TOURNAMENT_LEVELS`/`_ROUND_ENCODINGS` to those SAME integer
    encodings; unknown strings map to 0 (the codebook's ELSE branch). Pass
    either the int or the string for each, never both.
    """
    started_at = perf_counter()

    # ── Boundary validation ──
    if surface not in VALID_SURFACES:
        raise ValueError(f"surface must be one of {sorted(VALID_SURFACES)}, got {surface!r}")
    if tournament is not None:
        if tournament_level != 0:
            raise ValueError("pass either tournament_level (int) or tournament (str), not both")
        if not isinstance(tournament, str):
            raise TypeError(f"tournament must be a string, got {tournament!r}")
        tournament_level = _TOURNAMENT_LEVELS.get(tournament, 0)
    if round is not None:
        if round_encoded != 0:
            raise ValueError("pass either round_encoded (int) or round (str), not both")
        if not isinstance(round, str):
            raise TypeError(f"round must be a string, got {round!r}")
        round_encoded = _ROUND_ENCODINGS.get(round, 0)
    if isinstance(tournament_level, bool) or (
        not isinstance(tournament_level, int) or tournament_level not in VALID_TOURNAMENT_LEVELS
    ):
        raise ValueError(
            f"tournament_level must be an int in {sorted(VALID_TOURNAMENT_LEVELS)}, "
            f"got {tournament_level!r}"
        )
    if isinstance(round_encoded, bool) or (
        not isinstance(round_encoded, int) or round_encoded not in VALID_ROUND_ENCODINGS
    ):
        raise ValueError(
            f"round_encoded must be an int in {sorted(VALID_ROUND_ENCODINGS)}, "
            f"got {round_encoded!r}"
        )
    if not isinstance(player_id, str) or not player_id.strip():
        raise ValueError(f"player_id must be a non-empty string, got {player_id!r}")
    if not isinstance(opponent_id, str) or not opponent_id.strip():
        raise ValueError(f"opponent_id must be a non-empty string, got {opponent_id!r}")
    if as_of_date is None:
        as_of_date = date.today()
    elif isinstance(as_of_date, datetime):
        # A datetime is accepted and truncated to its date component.
        as_of_date = as_of_date.date()
    elif not isinstance(as_of_date, date):
        raise TypeError(f"as_of_date must be a datetime.date (or datetime), got {as_of_date!r}")

    # ── Canonicalization: lower id is the player_* side ──
    lower_id, higher_id = sorted([player_id.strip(), opponent_id.strip()])

    # ── On-demand global imputation pool (one set of aggregates) ──
    as_of_iso = as_of_date.isoformat()
    agg = first_row_dict(execute_df(_POOL_AGG_SQL, [as_of_iso]))
    pool_counts = first_row_dict(execute_df(_POOL_COUNTS_SQL, [as_of_iso]))
    median_days = _agg_or(
        first_row_dict(execute_df(_MEDIAN_DAYS_SQL, [as_of_iso, as_of_iso])),
        "median_days_since",
        _DEFAULT_DAYS_SINCE,
    )
    median_matches = _agg_or(
        first_row_dict(execute_df(_MEDIAN_MATCHES_30D_SQL, [as_of_iso, as_of_iso, as_of_iso])),
        "median_matches_30d",
        _DEFAULT_MATCHES_30D,
    )
    profile_agg = first_row_dict(execute_df(_PROFILE_POOL_AGG_SQL, [as_of_date.year]))
    profile_counts = first_row_dict(execute_df(_PROFILE_COUNTS_SQL))

    def _latest_snapshot(pid: str) -> dict[str, object] | None:
        df = execute_df(_LATEST_SNAPSHOT_SQL, [pid, as_of_iso])
        if df.empty:
            return None
        return first_row_dict(df)

    player_snapshot = _latest_snapshot(lower_id)
    opponent_snapshot = _latest_snapshot(higher_id)

    player_side = _side_values(
        player_snapshot, as_of_date, surface, agg, median_days, median_matches
    )
    opponent_side = _side_values(
        opponent_snapshot, as_of_date, surface, agg, median_days, median_matches
    )
    player_side.update(_profile_values(lower_id, as_of_date, profile_agg))
    opponent_side.update(_profile_values(higher_id, as_of_date, profile_agg))

    # ── Assemble the canonical row in FEATURE_COLS order ──
    row: dict[str, int | float | str] = {}
    for col in PLAYER_COLS:
        row[col] = player_side[
            "ranking" if col == "player_ranking" else col.removeprefix("player_")
        ]
    for col in OPPONENT_COLS:
        row[col] = opponent_side[
            "ranking" if col == "opponent_ranking" else col.removeprefix("opponent_")
        ]

    # Differentials, computed after imputation (canonical side minus opponent)
    row["rank_diff"] = player_side["ranking"] - opponent_side["ranking"]
    row["win_rate_diff"] = player_side["win_rate_10"] - opponent_side["win_rate_10"]
    row["ace_rate_diff"] = player_side["ace_rate_10"] - opponent_side["ace_rate_10"]
    row["break_pct_diff"] = player_side["break_pct_10"] - opponent_side["break_pct_10"]
    row["win_streak_diff"] = player_side["win_streak"] - opponent_side["win_streak"]
    row["matches_30d_diff"] = player_side["matches_30d"] - opponent_side["matches_30d"]
    row["surface_win_rate_diff"] = (
        player_side["surface_win_rate_10"] - opponent_side["surface_win_rate_10"]
    )
    row["rank_trend_diff"] = player_side["rank_trend_10"] - opponent_side["rank_trend_10"]
    row["height_diff"] = player_side["height"] - opponent_side["height"]
    row["handedness_diff"] = player_side["is_left_handed"] - opponent_side["is_left_handed"]
    row["years_pro_diff"] = player_side["years_pro"] - opponent_side["years_pro"]

    # Context
    row["is_clay"] = int(surface == "clay")
    row["is_grass"] = int(surface == "grass")
    row["is_hard"] = int(surface == "hard")
    row["tournament_level"] = tournament_level
    row["round_encoded"] = round_encoded

    # Preserve the canonical ids, not the raw input order.
    row["player_id"] = lower_id
    row["opponent_id"] = higher_id

    final_cols = [*FEATURE_COLS, "player_id", "opponent_id"]
    out = pd.DataFrame({col: [row[col]] for col in final_cols})
    assert not out.isnull().to_numpy().any(), "inference row contains NaN"

    meta: dict[str, object] = {
        "raw_player_id": player_id.strip(),
        "raw_opponent_id": opponent_id.strip(),
        "canonical_player_id": lower_id,
        "canonical_opponent_id": higher_id,
        "surface": surface,
        "as_of_date": as_of_iso,
        "tournament_level": tournament_level,
        "round_encoded": round_encoded,
        "feature_count": len(FEATURE_COLS),
        "snapshot_pool_rows": int(float(pool_counts["snapshot_pool_rows"] or 0)),
        "snapshot_pool_players": int(float(pool_counts["snapshot_pool_players"] or 0)),
        "profile_rows": int(float(profile_counts["profile_rows"] or 0)),
        "median_days_since": float(median_days),
        "median_matches_30d": float(median_matches),
        "player_snapshot_found": player_snapshot is not None,
        "opponent_snapshot_found": opponent_snapshot is not None,
        "player_snapshot_date": None
        if player_snapshot is None
        else _to_date(player_snapshot["snapshot_date"]).isoformat(),
        "opponent_snapshot_date": None
        if opponent_snapshot is None
        else _to_date(opponent_snapshot["snapshot_date"]).isoformat(),
        "player_rolling_match_number": None
        if player_snapshot is None
        else int(float(cast(Real, player_snapshot["player_match_number"]))),
        "opponent_rolling_match_number": None
        if opponent_snapshot is None
        else int(float(cast(Real, opponent_snapshot["player_match_number"]))),
        "player_matches_30d": int(player_side["matches_30d"]),
        "opponent_matches_30d": int(opponent_side["matches_30d"]),
        "player_days_since_last_match": int(player_side["days_since_last_match"]),
        "opponent_days_since_last_match": int(opponent_side["days_since_last_match"]),
        "player_profile_height": float(player_side["height"]),
        "opponent_profile_height": float(opponent_side["height"]),
        "build_ms": builtins.round((perf_counter() - started_at) * 1000, 3),
    }
    return out, meta


def build_inference_features(
    player_id: str,
    opponent_id: str,
    surface: str,
    *,
    as_of_date: date | None = None,
    tournament_level: int = 0,
    round_encoded: int = 0,
    tournament: str | None = None,
    round: str | None = None,
) -> pd.DataFrame:
    out, _meta = _build_inference_features_with_meta(
        player_id,
        opponent_id,
        surface,
        as_of_date=as_of_date,
        tournament_level=tournament_level,
        round_encoded=round_encoded,
        tournament=tournament,
        round=round,
    )
    return out
