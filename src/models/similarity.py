"""Content-based player similarity using text embeddings + FAISS storage and retrieval."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any, NotRequired, TypedDict

import faiss
import numpy as np
import pandas as pd

from src.constants import (
    BRONZE_PROFILES_TABLE,
    DATA_PROCESSED,
    DEPLOY_ARTIFACTS,
    GOLD_PROFILES_TABLE,
    PLAYSTYLE_RANDOM_STATE,
    SIM_BIO_PCA_DIM,
    SIM_BIO_WEIGHT,
    SIM_CLUSTER_WEIGHT,
    SIM_EXPERIENCE_K,
    SIM_IDENTITY_WEIGHT,
    SIM_PLAYSTYLE_WEIGHT,
    SIM_RANK_SCALE,
    SIM_REPUTATION_WEIGHT,
    SIM_SURFACE_SHRINK_K,
    SIM_SURFACE_WEIGHT,
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

# Player-style cluster membership (written by the pipeline after every snapshot
# refresh via build_cluster_artifacts), optional. When the artifacts exist,
# build() appends a deterministic one-hot cluster block to the FAISS vector so
# membership influences similarity, and bakes each player's archetype label
# into the metadata. Labels travel to serving inside player_metadata.json — no
# cluster files are staged by deploy.
DEFAULT_CLUSTERS = DATA_PROCESSED / "cluster_assignments.parquet"
DEFAULT_CLUSTER_LABELS = DATA_PROCESSED / "cluster_descriptions.json"

BIO_COL_PREFIX = "bio_"

# Metadata (names/bios/handedness) comes from bronze.player_profiles; the
# career aggregates below come from gold.player_profiles. Both go through the
# shared query helper so live and snapshot builds use identical SQL.

# Career lifetime playstyle stats per player (from gold.player_profiles).
# Only how a player plays enters this block: serve shape and aggression,
# return strength, and clutch. Surface win rates live in their own calibrated
# block (SURFACE_BLOCK_COLS) with exposure-based shrinkage, and reputation
# (current_rank, match_count, career_win_rate) is a separate block. Recent
# rolling match performance (win_rate_10, weighted_form_10, streak, and every
# *_10 signal in gold.match_features) and identity/bio metadata (height, age,
# turned_pro, birthplace, name, player_id) never enter the vector.
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

# Reputation block: official standing (lower current_rank = stronger, bounded
# by exp(-rank / SIM_RANK_SCALE)), career experience (match_count bounded by
# SIM_EXPERIENCE_K), and career win rate. All materialized career signals —
# never recent rolling form.
REPUTATION_BLOCK_COLS = ["current_rank", "match_count", "career_win_rate"]

# Fixed one-hot categories keep the vector layout stable across builds even
# when a category never appears in the data (pd.get_dummies would drop it).
HANDEDNESS_CATEGORIES = ["L", "R"]
BACKHAND_CATEGORIES = ["1H", "2H"]
IDENTITY_BLOCK_COLS = [f"handedness_{category}" for category in HANDEDNESS_CATEGORIES] + [
    f"backhand_{category}" for category in BACKHAND_CATEGORIES
]

_PLAYER_LIFETIME_SQL = f"""
SELECT player_id, {", ".join(LIFETIME_PLAYSTYLE_COLS + SURFACE_BLOCK_COLS + REPUTATION_BLOCK_COLS)}
FROM {GOLD_PROFILES_TABLE}
"""


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


def reduce_bio_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """PCA-reduce a batch of bio embeddings to at most SIM_BIO_PCA_DIM columns.

    Shared by the matrix and clustering paths: this runs once inside
    ``build_playstyle_matrix``, so both consume exactly the same transformed
    base vector. n_components = min(SIM_BIO_PCA_DIM, n_samples, n_features),
    so tiny batches or narrow embeddings never force a wider block than the
    data supports. Deterministic (full SVD, no randomized solver) and row-order
    preserving; degenerate batches (constant embeddings) reduce to zeros,
    which contribute nothing to similarity. Fitted per build — no transformer
    is persisted, and serving never re-derives it.
    """
    if embeddings.shape[0] == 0:
        return embeddings
    from sklearn.decomposition import PCA

    n_components = min(SIM_BIO_PCA_DIM, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components, svd_solver="full")
    # Constant batches have zero total variance; the ratio is unused, so ignore
    # the 0/0 numpy warning it emits.
    with np.errstate(divide="ignore", invalid="ignore"):
        return pca.fit_transform(embeddings).astype(np.float32)


class PlayerData(TypedDict):
    player_id: str
    display_name: str
    score: NotRequired[str]
    cluster_label: NotRequired[str | None]


# Explicit calibrated block weights (single source of truth; mirrors the SIM_*
# constants in src.constants). Each block is unit-normalized and scaled by its
# weight before the final row L2 normalization, so a block's influence on
# cosine similarity is exactly its weight — never its raw scale or dimension
# count (bio is additionally PCA-reduced to SIM_BIO_PCA_DIM columns and
# weighted 0.05, a minor auxiliary block behind playstyle/surface/reputation).
BLOCK_WEIGHTS: dict[str, float] = {
    "identity": SIM_IDENTITY_WEIGHT,
    "playstyle": SIM_PLAYSTYLE_WEIGHT,
    "surface": SIM_SURFACE_WEIGHT,
    "reputation": SIM_REPUTATION_WEIGHT,
    "bio": SIM_BIO_WEIGHT,
    "cluster": SIM_CLUSTER_WEIGHT,
}


def block_slices(num_bio: int, num_cluster: int = 0) -> dict[str, slice]:
    """Map each calibrated block to its column slice in the final vector.

    Fixed-width blocks lead (identity, playstyle, surface, reputation), the
    bio block follows with PCA-reduced width (at most SIM_BIO_PCA_DIM), and the
    optional cluster one-hot always trails (dynamic width). Tests and tooling
    use this to attribute vector components to blocks.
    """
    widths: list[tuple[str, int]] = [
        ("identity", len(IDENTITY_BLOCK_COLS)),
        ("playstyle", len(LIFETIME_PLAYSTYLE_COLS)),
        ("surface", len(SURFACE_BLOCK_COLS)),
        ("reputation", len(REPUTATION_BLOCK_COLS)),
        ("bio", num_bio),
        ("cluster", num_cluster),
    ]
    slices: dict[str, slice] = {}
    start = 0
    for name, width in widths:
        slices[name] = slice(start, start + width)
        start += width
    return slices


def vector_block_norms(vector: object, num_bio: int, num_cluster: int = 0) -> dict[str, float]:
    """Per-block L2 norms of one row vector.

    Because each block is unit-normalized and scaled by its calibrated weight
    before the final row L2 normalization, a fully active block of an
    L2-normalized vector has norm weight / sqrt(sum of squared weights); a
    zero block (cold start) contributes 0.0. This is the calibration contract
    tests assert, so block influence is weight-bound, never dimension-bound.
    num_bio is the PCA-reduced bio width of the vector being inspected.
    """
    arr = np.asarray(vector).astype(np.float32).reshape(-1)
    return {
        name: float(np.linalg.norm(arr[sl]))
        for name, sl in block_slices(num_bio, num_cluster).items()
    }


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
    cluster_assignments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble the per-player playstyle vector used by similarity and clustering.

    The vector is a weighted concatenation of independently calibrated blocks,
    each unit-normalized (or bounded-transformed) before scaling:

    - identity: one-hot handedness + backhand (SIM_IDENTITY_WEIGHT)
    - playstyle: LIFETIME_PLAYSTYLE_COLS lifetime serve/return stats
      (SIM_PLAYSTYLE_WEIGHT)
    - surface: exposure-shrunk hard/clay/grass win rates plus bounded exposure
      counts (SIM_SURFACE_WEIGHT)
    - reputation: current_rank, match_count, career_win_rate, each bounded
      (SIM_REPUTATION_WEIGHT)
    - bio: summary-text embeddings PCA-reduced to at most SIM_BIO_PCA_DIM
      components, a minor auxiliary block (SIM_BIO_WEIGHT = 0.05)
    - cluster: optional one-hot membership (SIM_CLUSTER_WEIGHT)

    The primary signals are playstyle, surface, and reputation; the bio block
    is a deliberate tiny bonus. Each block's influence is capped by its explicit
    weight, and the bio block additionally cannot exceed SIM_BIO_PCA_DIM
    columns (fit per build over the batch, min(SIM_BIO_PCA_DIM, n_samples,
    n_features)), so raw embedding dimensionality can never dominate the
    vector. Blocks are concatenated and each row L2-normalized once, so FAISS
    inner-product search is cosine similarity. Returns one row per ``df`` row
    (row order preserved). Cluster-absent builds yield the fixed base layout;
    the cluster block trails only when ``cluster_assignments`` is supplied.
    Shared by ``PlayerSimilarity.build``, ``build_cluster_artifacts``, and the
    playstyle-cluster exploration notebook so all consume identical vectors.
    """
    df = df.reset_index(drop=True)
    query = query or to_dataframe
    state = query(_PLAYER_LIFETIME_SQL)
    merged = df[["player_id"]].merge(state, on="player_id", how="left")
    # Career cells are NULL for players without a match: playstyle, surface
    # rates/counts, rank, and experience impute 0.0 (no signal); career win
    # rate imputes the neutral 0.5 prior (no evidence is not a failure).
    merged["career_win_rate"] = merged["career_win_rate"].fillna(0.5)
    all_numeric = LIFETIME_PLAYSTYLE_COLS + SURFACE_BLOCK_COLS + REPUTATION_BLOCK_COLS
    merged[all_numeric] = merged[all_numeric].fillna(0.0).astype(np.float32)

    encoded = pd.concat(
        [
            _one_hot(df, "handedness", HANDEDNESS_CATEGORIES),
            _one_hot(df, "backhand", BACKHAND_CATEGORIES),
        ],
        axis=1,
    )
    embeddings = embed_bio_summaries(pd.DataFrame(df[["player_id", "summary"]]))
    bio_values = reduce_bio_embeddings(
        embeddings[[c for c in embeddings.columns if c != "player_id"]].to_numpy(np.float32)
    )
    bio_cols = [f"{BIO_COL_PREFIX}pca_{i}" for i in range(bio_values.shape[1])]

    # Surface block: win rates shrunk toward 0.5 by exposure, plus the bounded
    # exposure counts themselves (confidence signal for the shrunk rates).
    counts = merged[SURFACE_COUNT_COLS].to_numpy(np.float32)
    exposure = counts / (counts + np.float32(SIM_SURFACE_SHRINK_K))
    rates = merged[SURFACE_RATE_COLS].to_numpy(np.float32)
    shrunk = 0.5 + (rates - 0.5) * exposure
    surface_block = np.concatenate([shrunk, exposure], axis=1)

    # Reputation block: rank standing, career experience, career win rate.
    rank = merged["current_rank"].to_numpy(np.float32)
    rank_signal = np.where(
        rank > 0, np.exp(-rank / np.float32(SIM_RANK_SCALE)), np.float32(0.0)
    ).astype(np.float32)
    match_count = merged["match_count"].to_numpy(np.float32)
    experience = match_count / (match_count + np.float32(SIM_EXPERIENCE_K))
    career_rate = merged["career_win_rate"].to_numpy(np.float32)
    reputation_block = np.column_stack([rank_signal, experience, career_rate])

    cluster_cols: list[str] = []
    cluster_frame: pd.DataFrame | None = None
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

    blocks = [
        _weighted_block(encoded.to_numpy(np.float32), SIM_IDENTITY_WEIGHT),
        _weighted_block(merged[LIFETIME_PLAYSTYLE_COLS].to_numpy(np.float32), SIM_PLAYSTYLE_WEIGHT),
        _weighted_block(surface_block, SIM_SURFACE_WEIGHT),
        _weighted_block(reputation_block, SIM_REPUTATION_WEIGHT),
        _weighted_block(bio_values, SIM_BIO_WEIGHT),
    ]
    if cluster_frame is not None:
        blocks.append(_weighted_block(cluster_frame.to_numpy(np.float32), SIM_CLUSTER_WEIGHT))

    features = np.ascontiguousarray(np.concatenate(blocks, axis=1))
    faiss.normalize_L2(features)
    out = pd.DataFrame(features)
    out.columns = [
        *IDENTITY_BLOCK_COLS,
        *LIFETIME_PLAYSTYLE_COLS,
        *SURFACE_BLOCK_COLS,
        *REPUTATION_BLOCK_COLS,
        *bio_cols,
        *cluster_cols,
    ]
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


def build_cluster_artifacts(
    n_clusters: int,
    labels: dict[str, str],
    query: Callable[[str], pd.DataFrame] | None = None,
    *,
    random_state: int = PLAYSTYLE_RANDOM_STATE,
) -> tuple[pd.DataFrame, Any]:
    """Fit player-archetype clusters and write the two runtime artifacts.

    Shared by the pipeline (after every snapshot refresh, exactly before the
    similarity index build) and the playstyle-cluster EDA notebook. Uses the
    exact playstyle vectors of ``PlayerSimilarity.build`` — without any prior
    cluster assignment, so discovery is never skewed by a previous archetype —
    fits deterministic KMeans, validates that ``labels`` covers exactly the
    fitted cluster ids, then writes ``cluster_assignments.parquet`` plus
    ``cluster_descriptions.json``. Returns ``(assignments, model)`` so the
    notebook can review the very fit that was written.
    """
    from sklearn.cluster import KMeans

    query = query or to_dataframe
    profiles = query(
        f"SELECT player_id, display_name, backhand, handedness, summary "
        f"FROM {BRONZE_PROFILES_TABLE}"
    )
    profiles = profiles[profiles["player_id"] != ""].reset_index(drop=True)
    if profiles.empty:
        return pd.DataFrame(), None

    features = build_playstyle_matrix(profiles, query, cluster_assignments=None).to_numpy(
        np.float32
    )
    model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state).fit(features)
    assignments = pd.DataFrame({"player_id": profiles["player_id"], "cluster_id": model.labels_})

    fitted_ids = {str(c) for c in sorted(assignments["cluster_id"].unique())}
    label_ids = {str(k) for k in labels}
    assert label_ids == fitted_ids, (
        f"cluster_labels keys {sorted(label_ids)} must exactly equal fitted "
        f"cluster ids {sorted(fitted_ids)}"
    )

    DEFAULT_CLUSTERS.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_parquet(DEFAULT_CLUSTERS, index=False)
    DEFAULT_CLUSTER_LABELS.write_text(
        json.dumps({str(k): v for k, v in labels.items()}, indent=2, sort_keys=True)
    )
    print(f"Wrote {DEFAULT_CLUSTERS} ({len(assignments)} players, {len(fitted_ids)} clusters)")
    print(f"Wrote {DEFAULT_CLUSTER_LABELS}")
    return assignments, model


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

        # Career aggregates per player from gold.player_profiles, consumed by
        # the calibrated block vector (identity, lifetime playstyle stats,
        # exposure-shrunk surface performance, reputation, bio embeddings).
        # Recent rolling match performance and identity/bio metadata are
        # excluded. Cluster membership is an optional one-hot block appended
        # when the artifact exists.
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
