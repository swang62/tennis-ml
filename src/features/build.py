"""Feature column definitions used across training and serving.

Rolling features are pre-computed in the ClickHouse ETL.
H2H, pairwise diffs, and bio embeddings are added in the notebook.
This file re-exports the stable feature list from rolling.py for use
in serving validation and column ordering.
"""

from src.features.rolling import FEATURE_COLS, SEQ_FEATURE_COLS


def get_feature_target(df, feature_cols=None, target_col="match_won"):
    cols = feature_cols or FEATURE_COLS
    return df[cols].copy(), df[target_col].copy()
