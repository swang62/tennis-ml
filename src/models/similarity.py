"""Content-based player similarity using text embeddings + FAISS storage and retrieval."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import NotRequired, TypedDict

import faiss
import numpy as np
import pandas as pd

from src.constants import (
    BRONZE_PROFILES_TABLE,
    DATA_PROCESSED,
    DEPLOY_ARTIFACTS,
    PROFILES_TABLE,
)
from src.db.client import to_dataframe

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# The similarity index is independent of the prediction models (a dashboard
# feature rebuilt from the fresh snapshot on every train). Training writes it
# to data/processed (DEFAULT_*) and pipeline.py logs it to MLflow as run
# artifacts; the deploy flow then downloads the pinned artifacts into
# data/deploy and packages them into the Bento, where load() reads them back
# from the SERVING_* paths.
DEFAULT_INDEX = DATA_PROCESSED / "player_similarity.index"
DEFAULT_METADATA = DATA_PROCESSED / "player_metadata.json"
SERVING_INDEX = DEPLOY_ARTIFACTS / "player_similarity.index"
SERVING_METADATA = DEPLOY_ARTIFACTS / "player_metadata.json"

BIO_COL_PREFIX = "bio_"

# Metadata (names/bios/handedness) comes from bronze.player_profiles; the
# lifetime playstyle aggregates below come from gold.player_profiles. Both go
# through the shared query helper so live and snapshot builds use identical SQL.

# Career lifetime playstyle aggregates per player (from gold.player_profiles).
# Only how a player plays enters the vector: serve shape and aggression,
# return strength, clutch, and surface preference. Recent rolling match
# performance (win_rate_10, weighted_form_10, streak, and every *_10 signal
# in gold.match_features) and identity/career attributes (rank, rank points,
# match counts, height, age, turned_pro, birthplace, name, player_id) are
# excluded.
LIFETIME_PLAYSTYLE_COLS: list[str] = [
    # Serve shape and points won.
    "first_serve_in_pct",
    "first_serve_points_won_pct",
    "second_serve_points_won_pct",
    "overall_serve_points_won_pct",
    # Serve aggression/risk: aces and double faults.
    "aces_per_first_serve",
    "aces_per_service_game",
    "double_faults_per_serve_point",
    # Clutch serving.
    "break_points_saved_pct",
    # Return strength and aggression.
    "return_points_won_pct",
    "first_serve_return_points_won_pct",
    "second_serve_return_points_won_pct",
    "break_point_conversion_pct",
    "break_point_opportunities_per_return_game",
    # Surface preference (career win rate on each surface).
    "hard_win_rate",
    "clay_win_rate",
    "grass_win_rate",
]

_PLAYER_LIFETIME_SQL = f"""
SELECT player_id, {", ".join(LIFETIME_PLAYSTYLE_COLS)}
FROM {PROFILES_TABLE}
"""

# Fixed one-hot categories keep the vector layout stable across builds even
# when a category never appears in the data (pd.get_dummies would drop it).
HANDEDNESS_CATEGORIES = ["L", "R"]
BACKHAND_CATEGORIES = ["1H", "2H"]


def _one_hot(df: pd.DataFrame, column: str, categories: list[str]) -> pd.DataFrame:
    """One-hot `column` into fixed `categories`, missing values all zero."""
    out = pd.DataFrame(
        0.0,
        index=df.index,
        columns=[f"{column}_{category}" for category in categories],
        dtype=np.float32,
    )
    for category in categories:
        out[f"{column}_{category}"] = (df[column] == category).astype(np.float32)
    return out


def embed_bio_summaries(profiles: pd.DataFrame, model_name: str = MODEL_NAME) -> pd.DataFrame:
    """Embed summaries; lazy import keeps fastembed out of the serving image."""
    import fastembed

    model = fastembed.TextEmbedding(model_name)
    texts = [s if isinstance(s, str) and s else "" for s in profiles["summary"]]
    embeddings = np.array(list(model.embed(texts)), dtype=np.float32)
    out = pd.DataFrame(embeddings)
    out.columns = [f"{BIO_COL_PREFIX}{i}" for i in range(embeddings.shape[1])]
    out.insert(0, "player_id", profiles["player_id"].to_numpy())
    return out


class PlayerData(TypedDict):
    player_id: str
    display_name: str
    score: NotRequired[str]


class PlayerSimilarity:
    """Builds or loads a FAISS index of player bios and finds similar players.

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
    ) -> None:
        """Build and save the index using the live client or an offline query helper."""
        query = query or to_dataframe
        profiles = query(
            f"SELECT player_id, display_name, backhand, handedness, summary "
            f"FROM {BRONZE_PROFILES_TABLE}"
        )
        profiles = profiles[profiles["player_id"] != ""].reset_index(drop=True)
        if profiles.empty:
            return

        # Career lifetime playstyle aggregates per player from
        # gold.player_profiles: serve shape/aggression, return strength,
        # clutch, and surface preference. Recent rolling match performance
        # and identity/career attributes are excluded.
        state = query(_PLAYER_LIFETIME_SQL)
        df = profiles.merge(state, on="player_id", how="left")
        # Career cells are NULL for players without a match; impute 0.0 so
        # every profiled player is still indexed.
        df[LIFETIME_PLAYSTYLE_COLS] = df[LIFETIME_PLAYSTYLE_COLS].fillna(0.0).astype(np.float32)

        # One-hot handedness/backhand as playstyle descriptors. Fixed
        # categories keep the layout deterministic across datasets and builds.
        encoded = pd.concat(
            [
                _one_hot(df, "handedness", HANDEDNESS_CATEGORIES),
                _one_hot(df, "backhand", BACKHAND_CATEGORIES),
            ],
            axis=1,
        )

        # Shared embedding path (also used by the NN static features in 02_tune_nn).
        embeddings = embed_bio_summaries(pd.DataFrame(df[["player_id", "summary"]]))
        bio_cols = [c for c in embeddings.columns if c != "player_id"]

        features = np.ascontiguousarray(
            pd.concat(
                [
                    encoded,
                    df[LIFETIME_PLAYSTYLE_COLS],
                    embeddings[bio_cols],
                ],
                axis=1,
            ).to_numpy(np.float32)
        )
        faiss.normalize_L2(features)

        self.index = faiss.IndexFlatIP(features.shape[1])
        self.index.add(features)
        self.players = [
            {"player_id": player_id, "display_name": display_name}
            for player_id, display_name in zip(df["player_id"], df["display_name"], strict=True)
        ]
        self.player_ids = df["player_id"].tolist()

        DEFAULT_INDEX.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(DEFAULT_INDEX))
        with open(DEFAULT_METADATA, "w") as f:
            json.dump(self.players, f)

        print(f"Index ({len(self.players)} players) saved to {DEFAULT_INDEX}")

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
