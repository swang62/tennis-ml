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
    DEPLOY_ARTIFACTS,
    GOLD_PROFILES_TABLE,
    MODELS_ARTIFACTS,
    SIM_EXPERIENCE_K,
    SIM_IDENTITY_WEIGHT,
    SIM_PLAYSTYLE_WEIGHT,
    SIM_REPUTATION_WEIGHT,
    SIM_SURFACE_SHRINK_K,
    SIM_SURFACE_WEIGHT,
)
from src.db.client import to_dataframe

# Deploy rebuilds this dashboard index from the fresh snapshot, independently of models.
DEFAULT_INDEX = MODELS_ARTIFACTS / "player_similarity.index"
DEFAULT_METADATA = MODELS_ARTIFACTS / "player_metadata.json"
SERVING_INDEX = DEPLOY_ARTIFACTS / "player_similarity.index"
SERVING_METADATA = DEPLOY_ARTIFACTS / "player_metadata.json"

# Lifetime playstyle stats form one block; recent form, rank, and identity text do not.
PROFILE_COLS = ["height", "age", "years_pro"]
IDENTITY_BLOCK_COLS = ["is_right_handed", "is_two_handed_backhand"]
LIFETIME_PLAYSTYLE_COLS: list[str] = [
    "first_serve_in_pct",
    "overall_serve_points_won_pct",
    "aces_per_service_game",
    "return_points_won_pct",
    "break_point_conversion_pct",
]

# Surface rates shrink toward 0.5 by exposure; bounded exposure counts are included.
SURFACE_RATE_COLS = ["hard_win_rate", "clay_win_rate", "grass_win_rate"]
SURFACE_COUNT_COLS = ["hard_matches", "clay_matches", "grass_matches"]
SURFACE_BLOCK_COLS = [*SURFACE_RATE_COLS, *SURFACE_COUNT_COLS]

# Reputation uses career experience and career win rate, not current standing or recent form.
DOMINANCE_COLS = ["dominance"]
REPUTATION_BLOCK_COLS = ["match_count", "career_win_rate"]

# Deploy hashes this SQL to decide whether similarity artifacts can be reused.
PLAYER_LIFETIME_SQL = f"""
SELECT player_id, {", ".join(LIFETIME_PLAYSTYLE_COLS + SURFACE_BLOCK_COLS + DOMINANCE_COLS + REPUTATION_BLOCK_COLS)}
FROM {GOLD_PROFILES_TABLE}
"""


class PlayerData(TypedDict):
    player_id: str
    display_name: str
    score: NotRequired[str]


# Normalize each block before applying its calibrated weight.
BLOCK_WEIGHTS: dict[str, float] = {
    "identity": SIM_IDENTITY_WEIGHT,
    "playstyle": SIM_PLAYSTYLE_WEIGHT,
    "surface": SIM_SURFACE_WEIGHT,
    "reputation": SIM_REPUTATION_WEIGHT,
}


def block_slices() -> dict[str, slice]:
    """Map each vector block to its column slice."""
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
    """Return the L2 norm of each block in one vector."""
    arr = np.asarray(vector).astype(np.float32).reshape(-1)
    return {name: float(np.linalg.norm(arr[sl])) for name, sl in block_slices().items()}


def _weighted_block(values: np.ndarray, weight: float) -> np.ndarray:
    """Normalize a block row-wise and apply its calibrated weight."""
    block = np.ascontiguousarray(values, dtype=np.float32)
    faiss.normalize_L2(block)
    return block * weight


def build_playstyle_matrix(
    df: pd.DataFrame,
    query: Callable[[str], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build weighted, row-normalized vectors while preserving ``df`` order."""
    df = df.reset_index(drop=True)
    query = query or to_dataframe
    state = query(PLAYER_LIFETIME_SQL)
    merged = df.merge(state, on="player_id", how="left")
    # Cold starts have no career signal; career win rate uses the neutral prior.
    merged["career_win_rate"] = merged["career_win_rate"].fillna(0.5)
    all_numeric = (
        LIFETIME_PLAYSTYLE_COLS + SURFACE_BLOCK_COLS + DOMINANCE_COLS + REPUTATION_BLOCK_COLS
    )
    for name in all_numeric:
        if name not in merged:
            merged[name] = 0.0
    merged[all_numeric] = merged[all_numeric].fillna(0.0).astype(np.float32)
    build_date = pd.Timestamp.today().normalize()

    # Profile metadata comes from the bronze snapshot, not lifetime aggregates.
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

    # Surface rates use exposure shrinkage, and exposure counts remain features.
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
    """Build or load a FAISS index for player similarity search."""

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

        # Combine lifetime aggregates with the bronze profile descriptor.
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
        """Return similar players sorted by score."""
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
