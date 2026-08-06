"""ID-based inference feature builder backed by PostgreSQL gold tables.

Builds ONE finalized canonical `FEATURE_COLS` row for a match between two
players as of a given date, mirroring `gold.match_features` semantics:

* Canonicalization: the lexicographically lower player id is the `player_*`
  side and the higher is `opponent_*`, so `(X, Y)` and `(Y, X)` produce
  identical rows (same as `match_features.sql`'s canonical-side collapse).
* Rolling form comes from each player's newest
  `silver.rolling_features` snapshot STRICTLY BEFORE the as-of date
  (no dedicated latest table/view; the newest snapshot is immediately usable).
* `days_since_last_match` (as-of date minus that snapshot's date) and
  `matches_30d` (count of `silver.player_matches` rows in
  `[as_of - 30 days, as_of)`) are computed at the as-of date.
* Head-to-head: the most recent 5 prior meetings between the canonical pair
  (match_date STRICTLY BEFORE the as-of date, deduped to distinct match_ids)
  come from ONE parameterized query. `player_h2h_*` is the canonical
  player side's perspective (matches + wins); zero prior meetings -> counts 0.
* Missing players (no eligible snapshot) and NULL snapshot cells are imputed
  from on-demand global aggregates over the eligible snapshot pool: MEDIAN
  for ranking/streak-related values, MEAN for other numerics. When BOTH
  players are missing, both sides receive the same imputed values, so every
  pairwise differential is 0 (neutral).
* Profile-derived identity (is_left_handed, years_pro-at-as-of-date) comes
  from `gold.player_profiles`; players without a profile and NULL cells are
  imputed from on-demand aggregates over all profiles (mean left-handed rate,
  mean years-pro at the as-of date), so the same neutral differentials hold
  for two profile-less players.
* If the eligible pool is empty, constant fallbacks are used:
  ranking=100, rank_points=500, age=26, rates=0.0, streak=0,
  days_since_last_match=365, matches_30d=0, left-handed rate=0.0,
  years_pro=8.

Player ids and dates NEVER appear inside SQL strings: every one flows through
`%s` placeholders and a params list.
"""

from __future__ import annotations

import builtins
from datetime import date, datetime
from decimal import Decimal
from numbers import Real
from time import perf_counter
from typing import cast

import pandas as pd

from src.constants import (
    PROFILES_TABLE,
    SILVER_PLAYER_MATCHES,
    SILVER_ROLLING_FEATURES,
)
from src.db.client import analytical_df, execute_df, first_row_dict
from src.features.columns import FEATURE_COLS

VALID_SURFACES = {"clay", "grass", "hard", "carpet"}
VALID_TOURNAMENT_LEVELS = {0, 1, 2, 3, 4}
VALID_ROUND_ENCODINGS = {0, 1, 2, 3, 4, 5, 6, 7}

# Optional convenience layer: human-readable tournament/round strings mapped to
# the SAME integer encodings as the CASE expressions in
# dbt/models/gold/match_features.sql. Unknown strings map to 0 (that codebook's
# ELSE branch), so stored encodings are unchanged.
_TOURNAMENT_LEVELS = {
    "grand_slam": 4,
    "masters": 3,
    "atp_500": 2,
    "atp_250": 1,
    "davis_cup": 0,
    "atp_finals": 0,
    "olympics": 0,
    "professional": 0,
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
_DEFAULT_RANK_POINTS = 500.0
_DEFAULT_AGE = 26.0
_DEFAULT_RATE = 0.0
_DEFAULT_STREAK = 0
_DEFAULT_DAYS_SINCE = 365
_DEFAULT_MATCHES_30D = 0
_DEFAULT_LEFT_HANDED_RATE = 0.0
_DEFAULT_YEARS_PRO = 8.0

_SURFACE_TO_SNAPSHOT_COL = {
    "clay": "clay_win_rate_10",
    "grass": "grass_win_rate_10",
    "hard": "hard_win_rate_10",
}

# Latest snapshot per player strictly before the as-of date. Point lookup.
_LATEST_SNAPSHOT_SQL = f"""
SELECT * FROM {SILVER_ROLLING_FEATURES}
WHERE player_id = %s
  AND snapshot_date < %s::date
ORDER BY player_match_number DESC
LIMIT 1
"""

# 30-day pre-match activity count, same window semantics as player_matches'
# matches_30d_before ([as_of - 30 days, as_of), strictly). Point lookup.
_MATCHES_30D_SQL = f"""
SELECT COUNT(*) AS n
FROM {SILVER_PLAYER_MATCHES}
WHERE player_id = %s
  AND match_date >= %s::date - INTERVAL '30 days'
  AND match_date < %s::date
"""

# Most recent 5 prior meetings between the canonical pair, deduped to distinct
# match_ids (silver.player_matches has TWO rows per match — one per player
# perspective; the GROUP BY collapses them). a_won = 1 iff the canonical
# a-side (the lower id, always the player_* side) won the meeting — both
# perspective rows of a meeting agree on it, so MAX is safe. Same
# date-granularity rule as match_features.sql: match_date strictly before the
# as-of date; same-date meetings excluded. Params: lower_id, higher_id,
# lower_id, higher_id, as_of_iso (prepared only — ids/dates never appear in
# the SQL text). Point lookup on the pair/date path.
_H2H_PRIOR_SQL = f"""
SELECT match_id, a_won
FROM (
    SELECT
        CASE WHEN player_id < opponent_id THEN player_id ELSE opponent_id END AS a,
        CASE WHEN player_id < opponent_id THEN opponent_id ELSE player_id END AS b,
        match_id,
        match_date,
        MAX(CASE WHEN player_id < opponent_id THEN match_won
                 ELSE 1 - match_won END) AS a_won
    FROM {SILVER_PLAYER_MATCHES}
    WHERE ((%s = player_id AND %s = opponent_id)
        OR (%s = opponent_id AND %s = player_id))
      AND match_date < %s::date
    GROUP BY 1, 2, 3, 4
)
ORDER BY match_date DESC, match_id DESC
LIMIT 5
"""

# Broad aggregate reads below are pg_duckdb acceleration candidates
# (analytical_df): each scans the whole snapshot/profile pool to produce one
# row of aggregates, and its normal-PostgreSQL and pg_duckdb executions are
# result-parity-tested in test_inference_features.py. Point lookups above stay
# on normal PostgreSQL execution.

# On-demand global imputation pool over all snapshots strictly before the
# as-of date. One row; MEDIAN (percentile_cont 0.5) for ranking/streak-related
# values, MEAN for the others. Broad aggregate read.
#
# pg_duckdb compatibility: every aggregate input is cast to DOUBLE PRECISION —
# percentile_cont 0.5 order inputs and AVG inputs alike. rolling_features'
# AVG-based columns (weighted_form_10, win_rate_10, avg_player_rank_10,
# avg_rank_faced_10, the per-surface win rates) are NUMERIC without declared
# precision, which DuckDB cannot bind; the other ratio rates and age are
# already DOUBLE/INTEGER but the explicit casts keep every aggregate input
# uniformly typed. Values are identical to the numeric/integer aggregates
# within float rounding, so pool aggregates (and the features they impute) are
# unchanged. analytical_df additionally sets the documented
# duckdb.convert_unsupported_numeric_to_double=true, which is still required
# because pg_duckdb resolves a scanned column's raw type before expression
# casts.
_POOL_AGG_SQL = f"""
SELECT
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latest_player_ranking::double precision)     AS latest_player_ranking,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latest_player_rank_points::double precision) AS latest_player_rank_points,
    AVG(latest_player_age::double precision)     AS latest_player_age,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY streak::double precision)                      AS streak,
    AVG(weighted_form_10::double precision)     AS weighted_form_10,
    AVG(win_rate_10::double precision)          AS win_rate_10,
    AVG(ace_rate_10::double precision)          AS ace_rate_10,
    AVG(first_serve_pct_10::double precision)   AS first_serve_pct_10,
    AVG(break_points_saved_pct_10::double precision) AS break_points_saved_pct_10,
    AVG(first_serve_win_pct_10::double precision) AS first_serve_win_pct_10,
    AVG(second_serve_win_pct_10::double precision) AS second_serve_win_pct_10,
    AVG(serve_win_pct_10::double precision)     AS serve_win_pct_10,
    AVG(df_rate_10::double precision)           AS df_rate_10,
    AVG(aces_per_svc_game_10::double precision) AS aces_per_svc_game_10,
    AVG(avg_player_rank_10::double precision)   AS avg_player_rank_10,
    AVG(avg_rank_faced_10::double precision)    AS avg_rank_faced_10,
    AVG(clay_win_rate_10::double precision)     AS clay_win_rate_10,
    AVG(grass_win_rate_10::double precision)    AS grass_win_rate_10,
    AVG(hard_win_rate_10::double precision)     AS hard_win_rate_10
FROM {SILVER_ROLLING_FEATURES}
WHERE snapshot_date < %s::date
"""

# MEDIAN of per-player days_since_last_match at the as-of date, over players
# that have an eligible snapshot. One row; NULL when the pool is empty. Date
# subtraction returns whole days in PostgreSQL; the percentile order input is
# cast to DOUBLE PRECISION for pg_duckdb binding (see _POOL_AGG_SQL). Broad
# aggregate read.
_MEDIAN_DAYS_SQL = f"""
SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY days_since::double precision) AS median_days_since
FROM (
    SELECT %s::date - MAX(snapshot_date) AS days_since
    FROM {SILVER_ROLLING_FEATURES}
    WHERE snapshot_date < %s::date
    GROUP BY player_id
)
"""

# MEDIAN of per-player matches_30d at the as-of date, over players that have
# an eligible snapshot (players with zero window matches contribute 0). Broad
# aggregate read. The COUNT-derived order input is cast to DOUBLE PRECISION
# for pg_duckdb binding (see _POOL_AGG_SQL).
_MEDIAN_MATCHES_30D_SQL = f"""
SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY matches_30d::double precision) AS median_matches_30d
FROM (
    SELECT pr.player_id, COUNT(pm.match_id) AS matches_30d
    FROM (
        SELECT DISTINCT player_id
        FROM {SILVER_ROLLING_FEATURES}
        WHERE snapshot_date < %s::date
    ) AS pr
    LEFT JOIN {SILVER_PLAYER_MATCHES} pm
        ON pm.player_id = pr.player_id
       AND pm.match_date >= %s::date - INTERVAL '30 days'
       AND pm.match_date < %s::date
    GROUP BY pr.player_id
)
"""

# Per-player static identity from gold.player_profiles (one row per player).
# Point lookup.
_PROFILE_SQL = f"""
SELECT player_id, height, handedness, turned_pro
FROM {PROFILES_TABLE}
WHERE player_id = %s
"""

# On-demand profile imputation pool over ALL profiles (identity is static, so
# no as-of window applies beyond the years_pro computation): mean left-handed
# rate, and mean years-pro at the as-of date. One row; NULL when no profiles
# exist. Broad aggregate read. Both AVG inputs are cast to DOUBLE PRECISION
# for pg_duckdb binding (see _POOL_AGG_SQL).
_PROFILE_POOL_AGG_SQL = f"""
SELECT
    AVG((CASE WHEN handedness = 'L' THEN 1 ELSE 0 END)::double precision) AS left_handed_rate,
    AVG((%s::int - turned_pro)::double precision) AS avg_years_pro
FROM {PROFILES_TABLE}
"""

_POOL_COUNTS_SQL = f"""
SELECT
    COUNT(*) AS snapshot_pool_rows,
    COUNT(DISTINCT player_id) AS snapshot_pool_players
FROM {SILVER_ROLLING_FEATURES}
WHERE snapshot_date < %s::date
"""

_PROFILE_COUNTS_SQL = f"""
SELECT COUNT(*) AS profile_rows
FROM {PROFILES_TABLE}
"""


# Import-time reference to the real date class. Tests monkeypatch the module
# level `date` name (to fix date.today() for default-as-of tests), so
# `_to_date`'s isinstance checks must use a reference that patching cannot
# shadow.
_DATE_TYPE = date


def _to_date(value: object) -> date:
    """Coerce a PostgreSQL DATE cell (datetime.date/Timestamp/datetime/str) to a
    plain date."""
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, _DATE_TYPE):
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
    """Build one canonical side's values, imputing NULLs and cold starts.

    Keys: "ranking", "rank_points", "age", "weighted_form_10", "win_rate_10",
    "ace_rate_10", "first_serve_pct_10", "break_points_saved_pct_10",
    "first_serve_win_pct_10", "second_serve_win_pct_10", "serve_win_pct_10",
    "df_rate_10", "aces_per_svc_game_10", "rank_trend_10", "avg_rank_faced_10",
    "streak", "days_since_last_match", "matches_30d", "surface_win_rate_10".
    `row` is the side's latest eligible snapshot (None on cold start).
    """

    def cell(snapshot_col: str, default: float) -> float:
        if row is not None:
            value = row.get(snapshot_col)
            # PostgreSQL returns numeric columns as Decimal and float columns
            # as float. NaN self-compares unequal, so `value == value` is the
            # NaN test (Decimal and int are always self-equal).
            if value is not None and isinstance(value, (Real, Decimal)) and value == value:
                return float(value)
        return _agg_or(agg, snapshot_col, default)

    ranking = round(cell("latest_player_ranking", _DEFAULT_RANKING))
    rank_points = round(cell("latest_player_rank_points", _DEFAULT_RANK_POINTS))
    age = cell("latest_player_age", _DEFAULT_AGE)
    avg_rank_10 = cell("avg_player_rank_10", _DEFAULT_RANKING)

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

    surface_win_rate = (
        cell(_SURFACE_TO_SNAPSHOT_COL[surface], _DEFAULT_RATE)
        if surface in _SURFACE_TO_SNAPSHOT_COL
        else _DEFAULT_RATE
    )
    return {
        "ranking": ranking,
        "rank_points": rank_points,
        "age": age,
        "weighted_form_10": cell("weighted_form_10", _DEFAULT_RATE),
        "win_rate_10": cell("win_rate_10", _DEFAULT_RATE),
        "ace_rate_10": cell("ace_rate_10", _DEFAULT_RATE),
        "first_serve_pct_10": cell("first_serve_pct_10", _DEFAULT_RATE),
        "break_points_saved_pct_10": cell("break_points_saved_pct_10", _DEFAULT_RATE),
        "first_serve_win_pct_10": cell("first_serve_win_pct_10", _DEFAULT_RATE),
        "second_serve_win_pct_10": cell("second_serve_win_pct_10", _DEFAULT_RATE),
        "serve_win_pct_10": cell("serve_win_pct_10", _DEFAULT_RATE),
        "df_rate_10": cell("df_rate_10", _DEFAULT_RATE),
        "aces_per_svc_game_10": cell("aces_per_svc_game_10", _DEFAULT_RATE),
        "rank_trend_10": avg_rank_10 - ranking,
        "avg_rank_faced_10": cell("avg_rank_faced_10", _DEFAULT_RANKING),
        "streak": int(cell("streak", _DEFAULT_STREAK)),
        "days_since_last_match": days_since,
        "matches_30d": matches_30d,
        "surface_win_rate_10": surface_win_rate,
    }


def _profile_values(pid: str, as_of_date: date, agg: dict[str, float]) -> dict[str, float]:
    """Fetch one side's profile-derived values from gold.player_profiles.

    Keys: "is_left_handed", "years_pro". A missing profile (or a NULL
    turned_pro/handedness cell) falls back to the on-demand pool aggregates, so
    two unknown players get identical defaults (every profile differential
    collapses to 0) and the row stays NaN-free. years_pro is time-aware: as-of
    year minus turned_pro, not the raw year.
    """
    df = execute_df(_PROFILE_SQL, [pid])
    if df.empty:
        return {
            "is_left_handed": _agg_or(agg, "left_handed_rate", _DEFAULT_LEFT_HANDED_RATE),
            "years_pro": _agg_or(agg, "avg_years_pro", _DEFAULT_YEARS_PRO),
        }
    row = df.iloc[0]
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
    is_indoor: int | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build ONE finalized canonical inference row.

    Returns a 1-row DataFrame with columns exactly:
        [*FEATURE_COLS, "player_id", "opponent_id"]
    (in that order). The canonical lower-id player is always the player_*
    side, matching gold.match_features' orientation. No NaNs in any cell.

    `tournament_level` and `round_encoded` are the integer context features
    (default 0), validated to the same value sets as the CASE expressions in
    dbt/models/gold/match_features.sql.

    `is_indoor` is 1 (indoor), 0 (outdoor), or None (unknown, defaults to 0
    in the feature row).

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
    # Pool reads are broad aggregate scans over silver.rolling_features /
    # gold.player_profiles, so they run through the pg_duckdb analytical path
    # (result parity with normal PostgreSQL is covered by
    # test_inference_features.py::test_pool_aggregates_analytical_parity).
    as_of_iso = as_of_date.isoformat()
    agg = first_row_dict(analytical_df(_POOL_AGG_SQL, [as_of_iso]))
    pool_counts = first_row_dict(analytical_df(_POOL_COUNTS_SQL, [as_of_iso]))
    median_days = _agg_or(
        first_row_dict(analytical_df(_MEDIAN_DAYS_SQL, [as_of_iso, as_of_iso])),
        "median_days_since",
        _DEFAULT_DAYS_SINCE,
    )
    median_matches = _agg_or(
        first_row_dict(analytical_df(_MEDIAN_MATCHES_30D_SQL, [as_of_iso, as_of_iso, as_of_iso])),
        "median_matches_30d",
        _DEFAULT_MATCHES_30D,
    )
    profile_agg = first_row_dict(analytical_df(_PROFILE_POOL_AGG_SQL, [as_of_date.year]))
    profile_counts = first_row_dict(analytical_df(_PROFILE_COUNTS_SQL))

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

    # ── Head-to-head: one query for the canonical pair (player side) ──
    # The SQL returns the most recent 5 prior meetings (deduped to distinct
    # match_ids, strictly before the as-of date) with a_won = 1 iff the
    # canonical player side won. The final contract keeps only the canonical
    # player side's counts (opponent = complement, derived on demand).
    h2h_df = execute_df(_H2H_PRIOR_SQL, [lower_id, higher_id, lower_id, higher_id, as_of_iso])
    h2h_matches = len(h2h_df)
    h2h_player_wins = int(h2h_df["a_won"].sum()) if h2h_matches else 0

    # ── Assemble the canonical row in FEATURE_COLS order ──
    row: dict[str, int | float | str] = {}

    def side(name: str, p: dict[str, int | float], o: dict[str, int | float]) -> int | float:
        """Return the player/opponent value for a per-side feature name."""
        if name in ("player_ranking", "opponent_ranking"):
            return p["ranking"] if name.startswith("player_") else o["ranking"]
        return (p if name.startswith("player_") else o)[
            name.removeprefix("player_").removeprefix("opponent_")
        ]

    # Matchup differences (canonical side minus opponent).
    row["rank_diff"] = player_side["ranking"] - opponent_side["ranking"]
    row["rank_points_diff"] = player_side["rank_points"] - opponent_side["rank_points"]
    row["age_diff"] = player_side["age"] - opponent_side["age"]
    row["win_rate_diff"] = player_side["win_rate_10"] - opponent_side["win_rate_10"]
    row["ace_rate_diff"] = player_side["ace_rate_10"] - opponent_side["ace_rate_10"]
    row["first_serve_pct_diff"] = (
        player_side["first_serve_pct_10"] - opponent_side["first_serve_pct_10"]
    )
    row["break_points_saved_pct_diff"] = (
        player_side["break_points_saved_pct_10"] - opponent_side["break_points_saved_pct_10"]
    )
    row["first_serve_win_pct_diff"] = (
        player_side["first_serve_win_pct_10"] - opponent_side["first_serve_win_pct_10"]
    )
    row["second_serve_win_pct_diff"] = (
        player_side["second_serve_win_pct_10"] - opponent_side["second_serve_win_pct_10"]
    )
    row["serve_win_pct_diff"] = player_side["serve_win_pct_10"] - opponent_side["serve_win_pct_10"]
    row["df_rate_diff"] = player_side["df_rate_10"] - opponent_side["df_rate_10"]
    row["aces_per_svc_game_diff"] = (
        player_side["aces_per_svc_game_10"] - opponent_side["aces_per_svc_game_10"]
    )
    row["rank_trend_diff"] = player_side["rank_trend_10"] - opponent_side["rank_trend_10"]
    row["avg_rank_faced_diff"] = (
        player_side["avg_rank_faced_10"] - opponent_side["avg_rank_faced_10"]
    )
    row["streak_diff"] = player_side["streak"] - opponent_side["streak"]

    # Absolute state values (both sides matter).
    for name in (
        "player_weighted_form_10",
        "opponent_weighted_form_10",
        "player_days_since_last_match",
        "opponent_days_since_last_match",
        "player_matches_30d",
        "opponent_matches_30d",
        "player_surface_win_rate_10",
        "opponent_surface_win_rate_10",
        "player_is_left_handed",
        "opponent_is_left_handed",
        "player_years_pro",
        "opponent_years_pro",
    ):
        row[name] = side(name, player_side, opponent_side)

    # Canonical-player head-to-head history.
    row["player_h2h_matches"] = h2h_matches
    row["player_h2h_wins"] = h2h_player_wins

    # Numeric match context (one-hot surface).
    row["is_clay"] = int(surface == "clay")
    row["is_grass"] = int(surface == "grass")
    row["is_hard"] = int(surface == "hard")
    row["is_carpet"] = int(surface == "carpet")
    row["is_indoor"] = is_indoor if is_indoor is not None else 0
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
        "h2h_prior_meetings": h2h_matches,
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
    is_indoor: int | None = None,
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
        is_indoor=is_indoor,
    )
    return out
