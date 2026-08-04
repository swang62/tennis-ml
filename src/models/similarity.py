"""Content-based player similarity using text embeddings + FAISS storage and retrieval."""

from __future__ import annotations

import json
from typing import NotRequired, TypedDict

import faiss
import numpy as np
import pandas as pd
from fastembed import TextEmbedding

from src.constants import GOLD_ROLLING_FEATURES, PROFILES_TABLE, ROOT
from src.db.client import to_dataframe

MODEL_NAME = "BAAI/bge-small-en-v1.5"

DEFAULT_INDEX = ROOT / "data" / "processed" / "player_similarity.index"
DEFAULT_METADATA = ROOT / "data" / "processed" / "player_metadata.json"

BIO_COL_PREFIX = "bio_"


def embed_bio_summaries(profiles: pd.DataFrame, model_name: str = MODEL_NAME) -> pd.DataFrame:
    """Embed each profile's summary into a player_id -> bio_* embedding frame.

    Pure function of the input frame: no DB access, no disk writes. Empty or
    missing summaries embed as the empty-string vector so every player stays
    joinable. Shared by the FAISS similarity index and the NN static pathway.
    """
    model = TextEmbedding(model_name)
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

    def build(self) -> None:
        """Query player profiles + latest style snapshots, build FAISS index, save to disk, and load in memory."""
        profiles = to_dataframe(
            f"SELECT player_id, display_name, backhand, handedness, summary FROM {PROFILES_TABLE}"
        )
        profiles = profiles[profiles["player_id"] != ""].reset_index(drop=True)
        if profiles.empty:
            return

        # Latest post-match style snapshot per player from gold.rolling_features.
        rolling = to_dataframe(
            f"SELECT player_id, ace_rate_10, first_serve_pct_10, break_points_saved_pct_10,"
            f" clay_win_rate_10, grass_win_rate_10, hard_win_rate_10"
            f" FROM {GOLD_ROLLING_FEATURES}"
            f" QUALIFY ROW_NUMBER() OVER (PARTITION BY player_id"
            f" ORDER BY snapshot_date DESC, match_id DESC) = 1"
        )
        style_cols = [
            "ace_rate_10",
            "first_serve_pct_10",
            "break_points_saved_pct_10",
            "clay_win_rate_10",
            "grass_win_rate_10",
            "hard_win_rate_10",
        ]
        df = profiles.merge(rolling, on="player_id", how="left")
        # Style cells are NULL for players without an eligible snapshot, or
        # without matches on a surface; impute 0.0 so every profiled player is
        # still indexed.
        df[style_cols] = df[style_cols].fillna(0.0).astype(np.float32)

        # One-hot encode categoricals, then stack with style stats and embeddings
        encoded = pd.get_dummies(df[["backhand", "handedness"]]).astype(np.float32)

        # Shared embedding path (also used by the NN static features in 02_tune_nn).
        embeddings = embed_bio_summaries(pd.DataFrame(df[["player_id", "summary"]]))
        bio_cols = [c for c in embeddings.columns if c != "player_id"]

        features = np.ascontiguousarray(
            pd.concat(
                [
                    encoded,
                    df[style_cols],
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

        faiss.write_index(self.index, str(DEFAULT_INDEX))
        with open(DEFAULT_METADATA, "w") as f:
            json.dump(self.players, f)

        print(f"Index ({len(self.players)} players) saved to {DEFAULT_INDEX}")

    # ── Load saved index ────────────────────────────

    def load(self) -> None:
        """Load a previously saved index from disk."""
        if not DEFAULT_INDEX.exists():
            raise FileNotFoundError(f"Index not found at {DEFAULT_INDEX}. Call build() first.")

        self.index = faiss.read_index(str(DEFAULT_INDEX))
        with open(DEFAULT_METADATA) as f:
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
