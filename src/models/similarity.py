"""Content-based player similarity using text embeddings + FAISS storage and retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NotRequired, TypedDict

import faiss
import numpy as np
import pandas as pd
from fastembed import TextEmbedding

from src.db.client import get_client, to_dataframe

PROFILES_TABLE = "gold.player_profiles"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INDEX = ROOT / "data" / "processed" / "player_similarity.index"
DEFAULT_METADATA = ROOT / "data" / "processed" / "player_metadata.json"


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
        """Query player profiles, build FAISS index, save to disk, and load in memory."""
        client = get_client()
        df = to_dataframe(
            (
                f"SELECT player_id, display_name, backhand, play_style,"
                f" handedness, height, turned_pro, summary FROM {PROFILES_TABLE}"
            ),
            client,
        )
        df = df[df["player_id"] != ""].reset_index(drop=True)
        if df.empty:
            return

        model = TextEmbedding(MODEL_NAME)
        summaries = [s if s else "" for s in df["summary"].astype("string")]
        embeddings = np.array(list(model.embed(summaries)), dtype=np.float32)

        # One-hot encode categoricals, numeric features, then stack with embeddings
        encoded = pd.get_dummies(df[["play_style", "backhand", "handedness"]]).astype(np.float32)
        height = pd.to_numeric(df["height"], errors="coerce").fillna(0).astype(np.float32)
        years_pro = (
            (pd.Timestamp.now().year - pd.to_numeric(df["turned_pro"], errors="coerce"))
            .fillna(0)
            .astype(np.float32)
        )

        features = np.ascontiguousarray(
            pd.concat(
                [
                    encoded,
                    height.rename("height"),
                    years_pro.rename("years_pro"),
                    pd.DataFrame(embeddings),
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
