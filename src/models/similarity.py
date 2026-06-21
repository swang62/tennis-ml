"""Content-based player similarity using bio embeddings + FAISS."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from fastembed import TextEmbedding

from src.db.client import get_client, to_dataframe

PROFILES_TABLE = "gold.player_profiles"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INDEX = ROOT / "data" / "processed" / "player_similarity.index"
DEFAULT_NAMES = ROOT / "data" / "processed" / "player_names.json"
DEFAULT_DISPLAY_NAMES = ROOT / "data" / "processed" / "player_display_names.json"


def build_index(
    index_path: str | Path = DEFAULT_INDEX,
    names_path: str | Path = DEFAULT_NAMES,
    display_names_path: str | Path = DEFAULT_DISPLAY_NAMES,
) -> None:
    """Build FAISS similarity index from player profiles and save to disk."""
    client = get_client()
    df = to_dataframe(
        (f"SELECT player_id, display_name, backhand, play_style, summary FROM {PROFILES_TABLE}"),
        client,
    )
    df = df[df["player_id"] != ""].reset_index(drop=True)
    if df.empty:
        return

    player_ids = df["player_id"].tolist()
    display_names_map: dict[str, str] = dict(zip(df["player_id"], df["display_name"], strict=True))

    model = TextEmbedding(MODEL_NAME)
    summaries = [s if s else "" for s in df["summary"].astype("string")]
    embeddings = np.array(list(model.embed(summaries)), dtype=np.float32)

    ohe_cols = ["play_style", "backhand"]
    encoded = pd.get_dummies(
        df.drop(columns=["player_id", "display_name", "summary"]),
        columns=ohe_cols,
    )
    style_features = pd.concat([encoded, pd.DataFrame(embeddings)], axis=1).to_numpy(np.float32)
    faiss.normalize_L2(style_features)

    base_index = faiss.IndexFlatIP(style_features.shape[1])
    index = faiss.IndexIDMap(base_index)
    index.add_with_ids(style_features, np.arange(len(player_ids), dtype=np.int64))

    faiss.write_index(index, str(index_path))
    with open(names_path, "w") as f:
        json.dump(player_ids, f)
    with open(display_names_path, "w") as f:
        json.dump(display_names_map, f)

    print(f"Index ({len(player_ids)} players) saved to {index_path}")


class PlayerSimilarity:
    """Loads a pre-built FAISS index and finds similar players by name."""

    def __init__(
        self,
        index_path: str | Path = DEFAULT_INDEX,
        names_path: str | Path = DEFAULT_NAMES,
        display_names_path: str | Path = DEFAULT_DISPLAY_NAMES,
    ):
        self.index = faiss.read_index(str(index_path))
        with open(names_path) as f:
            self.player_ids: list[str] = json.load(f)
        with open(display_names_path) as f:
            self.display_names: dict[str, str] = json.load(f)

    def __contains__(self, player_id: str) -> bool:
        return player_id in self.player_ids

    def __len__(self) -> int:
        return len(self.player_ids)

    def find_by_name(self, display_name: str) -> str | None:
        """Look up a player_id by display name (case-insensitive partial match)."""
        lower = display_name.lower()
        for pid, dname in self.display_names.items():
            if lower in dname.lower():
                return pid
        return None

    def similar(self, player_id: str, top_k: int = 5) -> list[dict[str, str]]:
        """Return top_k similar players as [{player_id, display_name}, ...].

        Uses vector reconstruction from the FAISS index to find nearest neighbors.
        """
        if player_id not in self.player_ids:
            return []

        player_idx = self.player_ids.index(player_id)
        vector = self.index.reconstruct(player_idx).reshape(1, -1)
        _, ids = self.index.search(vector, top_k + 1)
        results = []
        for i in ids[0]:
            if i != player_idx:
                pid = self.player_ids[i]
                results.append(
                    {
                        "player_id": pid,
                        "display_name": self.display_names.get(pid, pid),
                    }
                )
            if len(results) == top_k:
                break
        return results
