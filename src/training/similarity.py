"""Content-based player similarity using calibrated playstyle feature vectors + FAISS storage and retrieval."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import NotRequired, TypedDict

import faiss
import numpy as np
import pandas as pd

from src.constants import (
    BRONZE_PROFILES_TABLE,
    DATA_PROCESSED,
    DEPLOY_ARTIFACTS,
    GOLD_PROFILES_TABLE,
    SIM_EXPERIENCE_K,
    SIM_IDENTITY_WEIGHT,
    SIM_PLAYSTYLE_WEIGHT,
    SIM_REPUTATION_WEIGHT,
    SIM_SURFACE_SHRINK_K,
    SIM_SURFACE_WEIGHT,
)
from src.db.client import to_dataframe

# The similarity index is independent of the prediction models (a dashboard
# feature rebuilt from the fresh snapshot at deploy). Deploy writes it to
# data/deploy, where load() reads it back from the SERVING_* paths.
DEFAULT_INDEX = DATA_PROCESSED / "player_similarity.index"
DEFAULT_METADATA = DATA_PROCESSED / "player_metadata.json"
SERVING_INDEX = DEPLOY_ARTIFACTS / "player_similarity.index"
SERVING_METADATA = DEPLOY_ARTIFACTS / "player_metadata.json"

# Career lifetime playstyle stats per player (from gold.player_profiles).
# Only how a player plays enters this block: serve shape and aggression,
# return strength, and clutch. Surface win rates live in their own calibrated
# block (SURFACE_BLOCK_COLS) with exposure-based shrinkage, and reputation
# (match_count, career_win_rate) is a separate block. Recent rolling match
# performance (win_rate_10, weighted_form_10, streak, and every *_10 signal in
# gold.match_features), bio embeddings, current rank, and identity keys
# (birthplace, name, player_id) never enter the vector. The profile descriptor
# block (PROFILE_COLS) carries bronze height/age/years_pro so physical profile
# is a small calibrated signal, not a biasing one.
PROFILE_COLS = ["height", "age", "years_pro"]
IDENTITY_BLOCK_COLS = ["is_right_handed", "is_two_handed_backhand"]
LIFETIME_PLAYSTYLE_COLS: list[str] = [
    "first_serve_in_pct",
    "overall_serve_points_won_pct",
    "aces_per_service_game",
    "return_points_won_pct",
    "break_point_conversion_pct",
]

# Surface career performance block: win rates on each surface plus exposure
# counts. Rates are exposure-shrunk toward the neutral 0.5 prior
# (SIM_SURFACE_SHRINK_K) so a handful of matches never reads as reliably as a
# full career; counts enter as the same bounded confidence n / (n + K). Column
# names keep their gold names; the win-rate values stored in the vector are
# the shrunk ones.
SURFACE_RATE_COLS = ["hard_win_rate", "clay_win_rate", "grass_win_rate"]
SURFACE_COUNT_COLS = ["hard_matches", "clay_matches", "grass_matches"]
SURFACE_BLOCK_COLS = [*SURFACE_RATE_COLS, *SURFACE_COUNT_COLS]

# Reputation block: career experience (match_count bounded by
# SIM_EXPERIENCE_K) and career win rate. All materialized career signals —
# never current standing or recent rolling form.
DOMINANCE_COLS = ["dominance"]
REPUTATION_BLOCK_COLS = ["match_count", "career_win_rate"]

# Public so deploy can hash exactly the gold lifetime inputs the playstyle
# matrix consumes (its reuse check for the navigation artifacts).
PLAYER_LIFETIME_SQL = f"""
SELECT player_id, {", ".join(LIFETIME_PLAYSTYLE_COLS + SURFACE_BLOCK_COLS + DOMINANCE_COLS + REPUTATION_BLOCK_COLS)}
FROM {GOLD_PROFILES_TABLE}
"""


class PlayerData(TypedDict):
    player_id: str
    display_name: str
    score: NotRequired[str]


# Explicit calibrated block weights (single source of truth; mirrors the SIM_*
# constants in src.constants). Each block is unit-normalized and scaled by its
# weight before the final row L2 normalization, so a block's influence on
# cosine similarity is exactly its weight — never its raw scale or dimension
# count. The primary signals are playstyle, surface, and reputation; identity
# is a small descriptor block.
BLOCK_WEIGHTS: dict[str, float] = {
    "identity": SIM_IDENTITY_WEIGHT,
    "playstyle": SIM_PLAYSTYLE_WEIGHT,
    "surface": SIM_SURFACE_WEIGHT,
    "reputation": SIM_REPUTATION_WEIGHT,
}


def block_slices() -> dict[str, slice]:
    """Map each calibrated block to its column slice in the final vector.

    Fixed-width blocks lead (identity, playstyle, surface, reputation). Tests
    and tooling use this to attribute vector components to blocks.
    """
    widths: list[tuple[str, int]] = [
        ("identity", len(IDENTITY_BLOCK_COLS)),
        ("profile", len(PROFILE_COLS)),
        ("playstyle", len(LIFETIME_PLAYSTYLE_COLS)),
        ("surface", len(SURFACE_BLOCK_COLS)),
        ("dominance", len(DOMINANCE_COLS)),
        ("reputation", len(REPUTATION_BLOCK_COLS)),
    ]
    slices: dict[str, slice] = {}
    start = 0
    for name, width in widths:
        slices[name] = slice(start, start + width)
        start += width
    return slices


def vector_block_norms(vector: object) -> dict[str, float]:
    """Per-block L2 norms of one row vector.

    Because each block is unit-normalized and scaled by its calibrated weight
    before the final row L2 normalization, a fully active block of an
    L2-normalized vector has norm weight / sqrt(sum of squared weights); a
    zero block (cold start) contributes 0.0. This is the calibration contract
    tests assert, so block influence is weight-bound, never dimension-bound.
    """
    arr = np.asarray(vector).astype(np.float32).reshape(-1)
    return {name: float(np.linalg.norm(arr[sl])) for name, sl in block_slices().items()}


def _weighted_block(values: np.ndarray, weight: float) -> np.ndarray:
    """Unit-normalize a block row-wise, then scale it by its calibrated weight.

    Zero rows (cold start) stay zero — faiss.normalize_L2 leaves them
    untouched — so a missing block contributes nothing to similarity.
    """
    block = np.ascontiguousarray(values, dtype=np.float32)
    faiss.normalize_L2(block)
    return block * weight


def build_playstyle_matrix(
    df: pd.DataFrame,
    query: Callable[[str], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Assemble the per-player playstyle vector used by the similarity index.

    The vector is a weighted concatenation of independently calibrated blocks,
    each unit-normalized (or bounded-transformed) before scaling:

    - identity: one-hot handedness + backhand (SIM_IDENTITY_WEIGHT)
    - playstyle: LIFETIME_PLAYSTYLE_COLS lifetime serve/return stats
      (SIM_PLAYSTYLE_WEIGHT)
    - surface: exposure-shrunk hard/clay/grass win rates plus bounded exposure
      counts (SIM_SURFACE_WEIGHT)
    - reputation: career experience (match_count bounded by SIM_EXPERIENCE_K)
      and career win rate, each bounded (SIM_REPUTATION_WEIGHT)

    The primary signals are playstyle, surface, and reputation; identity is a
    small descriptor block. Each block's influence is capped by its explicit
    weight, so raw dimensionality can never dominate the vector. Blocks are
    concatenated and each row L2-normalized once, so FAISS inner-product search
    is cosine similarity. Returns one row per ``df`` row (row order preserved).
    """
    df = df.reset_index(drop=True)
    query = query or to_dataframe
    state = query(PLAYER_LIFETIME_SQL)
    merged = df.merge(state, on="player_id", how="left")
    # Career cells are NULL for players without a match: playstyle, surface
    # rates/counts, and experience impute 0.0 (no signal); career win rate
    # imputes the neutral 0.5 prior (no evidence is not a failure).
    merged["career_win_rate"] = merged["career_win_rate"].fillna(0.5)
    all_numeric = (
        LIFETIME_PLAYSTYLE_COLS + SURFACE_BLOCK_COLS + DOMINANCE_COLS + REPUTATION_BLOCK_COLS
    )
    for name in all_numeric:
        if name not in merged:
            merged[name] = 0.0
    merged[all_numeric] = merged[all_numeric].fillna(0.0).astype(np.float32)
    build_date = pd.Timestamp.today().normalize()

    # Profile metadata (height, birthdate, turned_pro) lives in the bronze
    # profiles snapshot, never in the gold lifetime aggregates
    # (PLAYER_LIFETIME_SQL), so read it from df (the supplied snapshot rows).
    # Missing players default to 0.0 so the descriptor block is a cold-start
    # zero rather than NaN.
    def profile_column(name: str) -> pd.Series:
        if name in df.columns:
            return df[name]
        if name in merged.columns:
            return merged[name]
        return pd.Series(np.nan, index=df.index)

    birthdate = pd.to_datetime(profile_column("birthdate"), errors="coerce")
    birthday_not_reached = (birthdate.dt.month > build_date.month) | (
        (birthdate.dt.month == build_date.month) & (birthdate.dt.day > build_date.day)
    )
    age = build_date.year - birthdate.dt.year - birthday_not_reached
    profile = (
        pd.DataFrame(
            {
                "height": pd.to_numeric(profile_column("height"), errors="coerce"),
                "age": age,
                "years_pro": build_date.year
                - pd.to_numeric(profile_column("turned_pro"), errors="coerce"),
            },
            index=df.index,
        )
        .fillna(0.0)
        .astype(np.float32)
    )
    identity = pd.DataFrame(
        {
            "is_right_handed": (df["handedness"] == "R").astype(np.float32),
            "is_two_handed_backhand": (df["backhand"] == "2H").astype(np.float32),
        },
        index=df.index,
    )

    # Surface block: win rates shrunk toward 0.5 by exposure, plus the bounded
    # exposure counts themselves (confidence signal for the shrunk rates).
    counts = merged[SURFACE_COUNT_COLS].to_numpy(np.float32)
    exposure = counts / (counts + np.float32(SIM_SURFACE_SHRINK_K))
    rates = merged[SURFACE_RATE_COLS].to_numpy(np.float32)
    shrunk = 0.5 + (rates - 0.5) * exposure
    surface_block = np.concatenate([shrunk, exposure], axis=1)

    # Reputation block: career experience and career win rate.
    match_count = merged["match_count"].to_numpy(np.float32)
    experience = match_count / (match_count + np.float32(SIM_EXPERIENCE_K))
    career_rate = merged["career_win_rate"].to_numpy(np.float32)
    reputation_block = np.column_stack([experience, career_rate])

    blocks = [
        _weighted_block(identity.to_numpy(np.float32), SIM_IDENTITY_WEIGHT),
        _weighted_block(profile.to_numpy(np.float32), SIM_IDENTITY_WEIGHT),
        _weighted_block(merged[LIFETIME_PLAYSTYLE_COLS].to_numpy(np.float32), SIM_PLAYSTYLE_WEIGHT),
        _weighted_block(surface_block, SIM_SURFACE_WEIGHT),
        _weighted_block(merged[DOMINANCE_COLS].to_numpy(np.float32), SIM_PLAYSTYLE_WEIGHT),
        _weighted_block(reputation_block, SIM_REPUTATION_WEIGHT),
    ]

    features = np.ascontiguousarray(np.concatenate(blocks, axis=1))
    faiss.normalize_L2(features)
    out = pd.DataFrame(features)
    out.columns = [
        *IDENTITY_BLOCK_COLS,
        *PROFILE_COLS,
        *LIFETIME_PLAYSTYLE_COLS,
        *SURFACE_BLOCK_COLS,
        *DOMINANCE_COLS,
        *REPUTATION_BLOCK_COLS,
    ]
    return out


class PlayerSimilarity:
    """Builds or loads a FAISS index of player playstyle profiles and finds similar players.

    Usage:
        finder = PlayerSimilarity()
        finder.build()
        finder.search("alcaraz")
        finder.search("Carlos Alcaraz")
    """

    def __init__(self):
        self.index: faiss.Index | None = None
        self.players: list[PlayerData] = []
        self.player_ids: list[str] = []

    # ── Build ───────────────────────────────────────

    def build(
        self,
        query: Callable[[str], pd.DataFrame] | None = None,
        *,
        profiles: pd.DataFrame | None = None,
        index_path: Path | None = None,
        metadata_path: Path | None = None,
    ) -> None:
        """Build and save the index using a query helper or supplied snapshot rows."""
        query = query or to_dataframe
        if profiles is None:
            profiles = query(
                f"SELECT player_id, display_name, backhand, handedness, height, birthdate, turned_pro "
                f"FROM {BRONZE_PROFILES_TABLE}"
            )
        profiles = profiles[profiles["player_id"] != ""].reset_index(drop=True)
        if profiles.empty:
            return

        # Career aggregates per player from gold.player_profiles, consumed by
        # the calibrated block vector (identity, lifetime playstyle stats,
        # exposure-shrunk surface performance, reputation). The bronze profile
        # snapshot also feeds the profile descriptor block (height/age/
        # years_pro). Recent rolling match performance, bio embeddings, and
        # current rank are excluded.
        features = build_playstyle_matrix(profiles, query)

        self.index = faiss.IndexFlatIP(features.shape[1])
        self.index.add(np.ascontiguousarray(features.to_numpy(np.float32)))
        self.players = [
            {"player_id": player_id, "display_name": display_name}
            for player_id, display_name in zip(
                profiles["player_id"], profiles["display_name"], strict=True
            )
        ]
        self.player_ids = profiles["player_id"].tolist()

        index_path = index_path or DEFAULT_INDEX
        metadata_path = metadata_path or DEFAULT_METADATA
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "w") as f:
            json.dump(self.players, f)

        print(f"Index ({len(self.players)} players) saved to {index_path}")

    # ── Load saved index ────────────────────────────

    def load(self) -> None:
        """Load a previously saved index from the packaged deploy folder."""
        if not SERVING_INDEX.exists():
            raise FileNotFoundError(f"Index not found at {SERVING_INDEX}. Call build() first.")

        self.index = faiss.read_index(str(SERVING_INDEX))
        with open(SERVING_METADATA) as f:
            self.players = json.load(f)
        self.player_ids = [p["player_id"] for p in self.players]

    # ── Query ───────────────────────────────────────

    def find_by_name(self, display_name: str) -> str | None:
        """Look up a player_id by display name (case-insensitive partial match)."""
        lower = display_name.lower()
        return next(
            (p["player_id"] for p in self.players if p["display_name"].lower() == lower), None
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, str]]:
        """Find players similar to *query* (player_id or display name).

        Returns entries sorted by similarity (highest first), each with
        player_id, display_name, and score (3 decimal places).
        """
        # Load index if not exist
        if self.index is None:
            self.load()

        player_id = query if query in self.player_ids else self.find_by_name(query)
        if player_id is None or self.index is None:
            return []

        player_idx = self.player_ids.index(player_id)
        vector = self.index.reconstruct(player_idx).reshape(1, -1)
        n_results = min(top_k, len(self.player_ids) - 1)
        if n_results < 1:
            return []

        scores, ids = self.index.search(vector, n_results + 1)
        results = []
        for idx, score in zip(ids[0], scores[0], strict=True):
            if idx < 0 or idx == player_idx:
                continue
            entry = dict(self.players[idx])
            entry["score"] = f"{score:.3f}"
            results.append(entry)
            if len(results) == n_results:
                break
        return results
