"""Content-based player similarity using text embeddings + FAISS storage and retrieval."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import NotRequired, TypedDict

import faiss
import numpy as np
import pandas as pd

from src.constants import (
    BRONZE_PROFILES_TABLE,
    DATA_PROCESSED,
    DEPLOY_ARTIFACTS,
    GOLD_PROFILES_TABLE,
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

# Player-style cluster membership (written by notebooks/eda/02_playstyle_clusters),
# optional. When the artifacts exist, build() appends a deterministic one-hot
# cluster block to the FAISS vector so membership influences similarity, and
# bakes each player's archetype label into the metadata. Labels travel to
# serving inside player_metadata.json — no cluster files are staged by deploy.
DEFAULT_CLUSTERS = DATA_PROCESSED / "cluster_assignments.parquet"
DEFAULT_CLUSTER_LABELS = DATA_PROCESSED / "cluster_descriptions.json"

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
FROM {GOLD_PROFILES_TABLE}
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
    cluster_label: NotRequired[str | None]


def build_playstyle_matrix(
    df: pd.DataFrame,
    query: Callable[[str], pd.DataFrame] | None = None,
    cluster_assignments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble the per-player playstyle vector used by similarity and clustering.

    Returns a DataFrame with one row per ``df`` row (row order preserved) and
    columns ``[one-hot handedness + backhand] + LIFETIME_PLAYSTYLE_COLS +
    [bio_*]``, L2-normalized row-wise. When ``cluster_assignments``
    (player_id, cluster_id) is supplied, a one-hot cluster block is appended so
    players in the same archetype rank more similar — the cluster group is a
    similarity feature. Shared by ``PlayerSimilarity.build`` and the
    playstyle-cluster exploration notebook so both consume identical vectors.
    """
    df = df.reset_index(drop=True)
    query = query or to_dataframe
    state = query(_PLAYER_LIFETIME_SQL)
    merged = df[["player_id"]].merge(state, on="player_id", how="left")
    # Career cells are NULL for players without a match; impute 0.0 so every
    # profiled player is still vectorized.
    merged[LIFETIME_PLAYSTYLE_COLS] = merged[LIFETIME_PLAYSTYLE_COLS].fillna(0.0).astype(np.float32)

    encoded = pd.concat(
        [
            _one_hot(df, "handedness", HANDEDNESS_CATEGORIES),
            _one_hot(df, "backhand", BACKHAND_CATEGORIES),
        ],
        axis=1,
    )
    embeddings = embed_bio_summaries(pd.DataFrame(df[["player_id", "summary"]]))
    bio_cols = [c for c in embeddings.columns if c != "player_id"]

    blocks = [
        encoded,
        merged[LIFETIME_PLAYSTYLE_COLS],
        embeddings[bio_cols],
    ]
    cluster_cols: list[str] = []
    if cluster_assignments is not None and not cluster_assignments.empty:
        # One-hot the cluster id (fixed categories from the full assignment set)
        # so the cluster group is a real similarity feature.
        cluster_ids = sorted(cluster_assignments["cluster_id"].astype(str).unique())
        cluster_cols = [f"cluster_{cid}" for cid in cluster_ids]
        id_to_cluster = dict(
            zip(
                cluster_assignments["player_id"].astype(str),
                cluster_assignments["cluster_id"].astype(str),
                strict=True,
            )
        )
        cluster_frame = pd.DataFrame(
            0.0,
            index=df.index,
            columns=cluster_cols,
            dtype=np.float32,
        )
        for idx, pid in zip(df.index, df["player_id"].astype(str), strict=True):
            cid = id_to_cluster.get(pid)  # a profiled player may lack an assignment
            if cid is not None:
                cluster_frame.loc[idx, f"cluster_{cid}"] = 1.0
        blocks.append(cluster_frame)

    features = np.ascontiguousarray(pd.concat(blocks, axis=1).to_numpy(np.float32))
    faiss.normalize_L2(features)
    out = pd.DataFrame(features)
    out.columns = [*encoded.columns, *LIFETIME_PLAYSTYLE_COLS, *bio_cols, *cluster_cols]
    return out


def _read_cluster_assignments() -> pd.DataFrame | None:
    """Load the (player_id, cluster_id) artifact from data/processed, if present.

    Absent or malformed artifacts return None so the index build never fails
    on a missing cluster file — clustering is an optional feature.
    """
    if not DEFAULT_CLUSTERS.exists():
        return None
    try:
        assignments = pd.read_parquet(DEFAULT_CLUSTERS)
    except (FileNotFoundError, ValueError, KeyError):
        return None
    if not {"player_id", "cluster_id"}.issubset(assignments.columns):
        return None
    return assignments


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
        self.cluster_labels: dict[str, str] = {}

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
        # and identity/career attributes are excluded. Cluster membership is
        # an optional one-hot block appended when the artifact exists.
        assignments = _read_cluster_assignments()
        features = build_playstyle_matrix(profiles, query, assignments)

        self.index = faiss.IndexFlatIP(features.shape[1])
        self.index.add(np.ascontiguousarray(features.to_numpy(np.float32)))
        self.players = [
            {"player_id": player_id, "display_name": display_name}
            for player_id, display_name in zip(
                profiles["player_id"], profiles["display_name"], strict=True
            )
        ]
        self.player_ids = profiles["player_id"].tolist()
        self.cluster_labels = self._load_cluster_labels(profiles["player_id"], assignments)
        self._attach_cluster_labels()

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
        # cluster labels were baked into SERVING_METADATA at build time; no
        # cluster files are staged into the deploy folder, so keep those.
        self.cluster_labels = {}
        self._attach_cluster_labels()

    # ── Query ───────────────────────────────────────

    @staticmethod
    def _load_cluster_labels(
        player_ids: Iterable[str], assignments: pd.DataFrame | None
    ) -> dict[str, str]:
        """Map player_id -> cluster label from the clustering artifacts, if present.

        Returns {} when the cluster-description JSON is absent (e.g. the
        notebook has not produced clusters yet), so similarity never breaks on
        a missing cluster file.
        """
        if assignments is None or not DEFAULT_CLUSTER_LABELS.exists():
            return {}
        try:
            labels = json.loads(DEFAULT_CLUSTER_LABELS.read_text())
        except (FileNotFoundError, ValueError, KeyError):
            return {}
        id_to_cluster = {
            str(row["player_id"]): str(row["cluster_id"]) for _, row in assignments.iterrows()
        }
        return {
            pid: labels.get(cid, f"cluster_{cid}")
            for pid, cid in id_to_cluster.items()
            if pid in set(player_ids)
        }

    def _attach_cluster_labels(self) -> None:
        """Stamp each player entry with its archetype label.

        A label from the current artifacts wins; otherwise a label already
        baked into loaded metadata is kept; otherwise the entry has no label.
        """
        for p in self.players:
            p["cluster_label"] = self.cluster_labels.get(p["player_id"]) or p.get("cluster_label")

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
