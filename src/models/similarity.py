"""Content-based player similarity using text embeddings + FAISS storage and retrieval."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import NotRequired, TypedDict

import faiss
import numpy as np
import pandas as pd

from src.constants import BRONZE_PROFILES_TABLE, GOLD_TABLE, ROOT
from src.db.client import to_dataframe

MODEL_NAME = "BAAI/bge-small-en-v1.5"

DEFAULT_INDEX = ROOT / "data" / "processed" / "player_similarity.index"
DEFAULT_METADATA = ROOT / "data" / "processed" / "player_metadata.json"

BIO_COL_PREFIX = "bio_"

# Metadata (names/bios/handedness) comes from bronze.player_profiles; the
# style state below comes from gold.match_features. Both go through the shared
# query helper so live and snapshot builds use identical SQL.
_PLAYER_STATE_SQL = f"""
WITH player_side AS (
    SELECT match_id, match_date, surface, player_id AS pid,
           player_weighted_form_10 AS weighted_form_10,
           player_first_serve_pct_10 AS first_serve_pct_10,
           player_first_serve_win_pct_10 AS first_serve_win_pct_10,
           player_second_serve_win_pct_10 AS second_serve_win_pct_10,
           player_serve_win_pct_10 AS serve_win_pct_10,
           player_return_points_won_pct_10 AS return_points_won_pct_10,
           player_surface_win_rate_10 AS surface_win_rate_10
    FROM {GOLD_TABLE}
    UNION ALL
    SELECT match_id, match_date, surface, opponent_id AS pid,
           opponent_weighted_form_10 AS weighted_form_10,
           opponent_first_serve_pct_10 AS first_serve_pct_10,
           opponent_first_serve_win_pct_10 AS first_serve_win_pct_10,
           opponent_second_serve_win_pct_10 AS second_serve_win_pct_10,
           opponent_serve_win_pct_10 AS serve_win_pct_10,
           opponent_return_points_won_pct_10 AS return_points_won_pct_10,
           opponent_surface_win_rate_10 AS surface_win_rate_10
    FROM {GOLD_TABLE}
),
latest_state AS (
    SELECT pid, weighted_form_10,
           first_serve_pct_10, first_serve_win_pct_10,
           second_serve_win_pct_10, serve_win_pct_10,
           return_points_won_pct_10
    FROM (
        SELECT pid, weighted_form_10,
               first_serve_pct_10, first_serve_win_pct_10,
               second_serve_win_pct_10, serve_win_pct_10,
               return_points_won_pct_10,
               ROW_NUMBER() OVER (
                   PARTITION BY pid ORDER BY match_date DESC, match_id DESC
               ) AS rn
        FROM player_side
    ) AS ranked_state
    WHERE rn = 1
),
latest_surface AS (
    SELECT pid, surface, surface_win_rate_10
    FROM (
        SELECT pid, surface, surface_win_rate_10,
               ROW_NUMBER() OVER (
                   PARTITION BY pid, surface ORDER BY match_date DESC, match_id DESC
               ) AS rn
        FROM player_side
    ) AS ranked_surface
    WHERE rn = 1
)
SELECT
    st.pid AS player_id,
    st.weighted_form_10,
    st.first_serve_pct_10,
    st.first_serve_win_pct_10,
    st.second_serve_win_pct_10,
    st.serve_win_pct_10,
    st.return_points_won_pct_10,
    MAX(CASE WHEN ls.surface = 'clay' THEN ls.surface_win_rate_10 END)
        AS clay_win_rate_10,
    MAX(CASE WHEN ls.surface = 'grass' THEN ls.surface_win_rate_10 END)
        AS grass_win_rate_10,
    MAX(CASE WHEN ls.surface = 'hard' THEN ls.surface_win_rate_10 END)
        AS hard_win_rate_10
FROM latest_state st
LEFT JOIN latest_surface ls ON ls.pid = st.pid
GROUP BY st.pid, st.weighted_form_10,
         st.first_serve_pct_10, st.first_serve_win_pct_10,
         st.second_serve_win_pct_10, st.serve_win_pct_10,
         st.return_points_won_pct_10
"""

# Latest rolling style signals; physical and career attributes are excluded.
STYLE_COLS: list[str] = [
    "weighted_form_10",
    "clay_win_rate_10",
    "grass_win_rate_10",
    "hard_win_rate_10",
    "first_serve_pct_10",
    "first_serve_win_pct_10",
    "second_serve_win_pct_10",
    "serve_win_pct_10",
    "return_points_won_pct_10",
]


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

        # Latest pre-match absolute state per player from gold.match_features:
        # weighted form + serve/return percentages from the most recent match
        # and clay/grass/hard win rates from the most recent match on each
        # surface.
        state = query(_PLAYER_STATE_SQL)
        df = profiles.merge(state, on="player_id", how="left")
        # Style cells are NULL for players without a match, or without a match
        # on a surface; impute 0.0 so every profiled player is still indexed.
        df[STYLE_COLS] = df[STYLE_COLS].fillna(0.0).astype(np.float32)

        # One-hot encode categoricals, then stack with style stats and embeddings
        encoded = pd.get_dummies(df[["backhand", "handedness"]]).astype(np.float32)

        # Shared embedding path (also used by the NN static features in 02_tune_nn).
        embeddings = embed_bio_summaries(pd.DataFrame(df[["player_id", "summary"]]))
        bio_cols = [c for c in embeddings.columns if c != "player_id"]

        features = np.ascontiguousarray(
            pd.concat(
                [
                    encoded,
                    df[STYLE_COLS],
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
