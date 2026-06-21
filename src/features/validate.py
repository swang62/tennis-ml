"""Lightweight DataFrame validation (replaces deepchecks)."""

from __future__ import annotations

import pandas as pd

BRONZE_CAT_FEATURES = ["player_id", "opponent_id", "tournament", "round", "surface"]

# ── Gold table column rules ────────────────────────────────────

GOLD_COLUMN_RULES: dict[str, dict] = {
    # Identity — never null
    "match_id": {"nullable": False, "type": "object"},
    "match_date": {"nullable": False, "type": "datetime64[ns]"},
    "player_id": {"nullable": False, "type": "object"},
    "opponent_id": {"nullable": False, "type": "object"},
    "tournament": {"nullable": False, "type": "object"},
    "round": {"nullable": False, "type": "object"},
    "surface": {"nullable": False, "type": "object"},
    # Rankings — never null, must be > 0
    "player_ranking": {"nullable": False, "type": "int64", "min": 1},
    "opponent_ranking": {"nullable": False, "type": "int64", "min": 1},
    # Match outcome — never null, must be 0 or 1
    "match_won": {"nullable": False, "type": "int64", "allowed": [0, 1]},
    # Surface indicators — never null, must be 0 or 1
    "is_clay": {"nullable": False, "type": "int64", "allowed": [0, 1]},
    "is_grass": {"nullable": False, "type": "int64", "allowed": [0, 1]},
    "is_hard": {"nullable": False, "type": "int64", "allowed": [0, 1]},
    # Base stats — never null, UInt8 range
    "wins_last_10": {"nullable": False, "type": "int64", "min": 0, "max": 10},
    "matches_last_10": {"nullable": False, "type": "int64", "min": 0, "max": 10},
    "aces": {"nullable": False, "type": "int64", "min": 0},
    "double_faults": {"nullable": False, "type": "int64", "min": 0},
    "first_serves_made": {"nullable": False, "type": "int64", "min": 0},
    "total_serve_points": {"nullable": False, "type": "int64", "min": 0},
    "break_points_won": {"nullable": False, "type": "int64", "min": 0},
    "break_points_total": {"nullable": False, "type": "int64", "min": 0},
    # Rate columns — nullable (early matches), if present must be in [0, 1]
    "win_rate_last_10": {"nullable": True, "type": "float64", "min": 0, "max": 1},
    "ace_rate": {"nullable": True, "type": "float64", "min": 0, "max": 1},
    "double_fault_rate": {"nullable": True, "type": "float64", "min": 0, "max": 1},
    "first_serve_pct": {"nullable": True, "type": "float64", "min": 0, "max": 1},
    "break_points_converted_pct": {"nullable": True, "type": "float64", "min": 0, "max": 1},
    # Rolling features — nullable (insufficient history), if present must be in [0, 1]
    "win_rate_5": {"nullable": True, "type": "float64", "min": 0, "max": 1},
    "win_rate_10": {"nullable": True, "type": "float64", "min": 0, "max": 1},
    "win_rate_20": {"nullable": True, "type": "float64", "min": 0, "max": 1},
    "ace_rate_5": {"nullable": True, "type": "float64", "min": 0, "max": 1},
    "ace_rate_10": {"nullable": True, "type": "float64", "min": 0, "max": 1},
    "first_serve_pct_5": {"nullable": True, "type": "float64", "min": 0, "max": 1},
    "first_serve_pct_10": {"nullable": True, "type": "float64", "min": 0, "max": 1},
    "break_pct_5": {"nullable": True, "type": "float64", "min": 0, "max": 1},
    "break_pct_10": {"nullable": True, "type": "float64", "min": 0, "max": 1},
    "surface_win_rate_10": {"nullable": True, "type": "float64", "min": 0, "max": 1},
    # Rolling features — nullable, any positive range
    "avg_opp_rank_10": {"nullable": True, "type": "float64", "min": 0},
    "avg_opp_rank_20": {"nullable": True, "type": "float64", "min": 0},
    "rank_trend_10": {"nullable": True, "type": "float64"},
    "rank_trend_20": {"nullable": True, "type": "float64"},
    # Other rolling — nullable, has defaults in ETL
    "win_streak": {"nullable": True, "type": "int64", "min": 0},
    "days_since_last_match": {"nullable": False, "type": "int64", "min": 0},
    "matches_30d": {"nullable": False, "type": "int64", "min": 0},
}

ALLOWED_SURFACE = {"hard", "clay", "grass"}
ALLOWED_ROUND = {"r128", "r64", "r32", "r16", "qf", "sf", "f"}


def run_ingestion_checks(df: pd.DataFrame) -> dict:
    issues = []

    nulls = df.isnull().sum()
    null_cols = nulls[nulls > 0]
    if not null_cols.empty:
        issues.append(f"Nulls: {null_cols.to_dict()}")

    dupes = df.duplicated(subset=["match_id"]).sum()
    if dupes:
        issues.append(f"{dupes} duplicate match_ids")

    if (df["player_ranking"] <= 0).any():
        issues.append("player_ranking <= 0 found")

    if (df["opponent_ranking"] <= 0).any():
        issues.append("opponent_ranking <= 0 found")

    for col in df.select_dtypes("object"):
        for _name, group in df.groupby(col):
            types = group[col].apply(type).unique()
            if len(types) > 1:
                issues.append(f"Mixed types in {col}: {types}")

    if "match_won" in df.columns and df["match_won"].nunique() > 2:
        issues.append(f"match_won has {df['match_won'].nunique()} values (expected 2)")

    passed = len(issues) == 0
    for issue in issues:
        print(f"  FAIL: {issue}")
    print(f"Ingestion checks: {len(issues)} issues, passed={passed}")
    return {"passed": passed, "results": issues}


def run_feature_checks(df: pd.DataFrame) -> dict:
    issues = []

    for col, rules in GOLD_COLUMN_RULES.items():
        if col not in df.columns:
            issues.append(f"Missing column: {col}")
            continue

        col_data = df[col]

        # Null check
        null_count = col_data.isnull().sum()
        if null_count > 0:
            if not rules.get("nullable", False):
                issues.append(f"{col}: {null_count} nulls (expected 0)")
            else:
                print(f"  NULLs in {col}: {null_count} rows (nullable, OK)")

        # Type check on non-null values
        if not col_data.dropna().empty:
            actual_type = col_data.dropna().infer_objects().dtype
            expected_type = rules.get("type")
            if expected_type and not any(
                actual_type == t
                for t in ([expected_type] if isinstance(expected_type, str) else expected_type)
            ):
                issues.append(f"{col}: expected {expected_type}, got {actual_type}")

        # Range checks
        valid = col_data.dropna()
        if not valid.empty:
            col_min = rules.get("min")
            if col_min is not None and (valid < col_min).any():
                n_below = int((valid < col_min).sum())
                issues.append(f"{col}: values below {col_min} ({n_below} rows)")

            col_max = rules.get("max")
            if col_max is not None and (valid > col_max).any():
                n_above = int((valid > col_max).sum())
                issues.append(f"{col}: values above {col_max} ({n_above} rows)")

            allowed = rules.get("allowed")
            if allowed is not None and not valid.isin(allowed).all():
                bad = valid[~valid.isin(allowed)].unique()
                issues.append(f"{col}: unexpected values {bad.tolist()}")

    # Surface-specific categorical checks
    if "surface" in df.columns:
        bad_surface = set(df["surface"].dropna().unique()) - ALLOWED_SURFACE
        if bad_surface:
            issues.append(f"surface: unexpected values {sorted(bad_surface)}")

    if "round" in df.columns:
        bad_round = set(df["round"].dropna().unique()) - ALLOWED_ROUND
        if bad_round:
            issues.append(f"round: unexpected values {sorted(bad_round)}")

    # Duplicate check
    dupes = df.duplicated(subset=["match_id", "player_id"]).sum()
    if dupes:
        issues.append(f"{dupes} duplicate (match_id, player_id) rows")

    passed = len(issues) == 0
    for issue in issues:
        print(f"  FAIL: {issue}")
    print(f"Feature checks: {len(issues)} issues, passed={passed}")
    return {"passed": passed, "results": issues}
