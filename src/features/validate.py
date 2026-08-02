"""Lightweight DataFrame validation (replaces deepchecks)."""

from __future__ import annotations

import pandas as pd


def run_ingestion_checks(df: pd.DataFrame) -> dict[str, bool | list[str]]:
    """Validate bronze.match_events-shaped rows (player1_*/player2_* columns)."""
    issues = []

    nulls = df.isnull().sum()
    null_cols = nulls[nulls > 0]
    if not null_cols.empty:
        issues.append(f"Nulls: {null_cols.to_dict()}")

    dupes = df.duplicated(subset=["match_id"]).sum()
    if dupes:
        issues.append(f"{dupes} duplicate match_ids")

    for side in ("player1", "player2"):
        if (df[f"{side}_ranking"] <= 0).any():
            issues.append(f"{side}_ranking <= 0 found")

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
