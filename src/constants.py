"""Single source of truth for shared paths, names, and environment settings."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    load_dotenv(ROOT / ".env", override=False)


load_env()

# --- Environment Variables -----
IMAGE_NAME = os.getenv("IMAGE_NAME")

# --- PostgreSQL connection contract (single DATABASE_URL) ---
# Applications and dbt derive all connection settings from DATABASE_URL.
DATABASE_URL = os.getenv("DATABASE_URL")


def build_database_url() -> str:
    """Return DATABASE_URL or fail rather than selecting an implicit backend."""
    if not DATABASE_URL:
        raise RuntimeError(
            "missing PostgreSQL configuration: set DATABASE_URL "
            "(e.g. postgresql://user@127.0.0.1:5432/postgres for local trust or "
            "postgresql://user:password@host:5432/db for the Compose stack)"
        )
    return DATABASE_URL


# --- Core directories ---
NOTEBOOKS = ROOT / "notebooks"
PARAMS = NOTEBOOKS / "parameters"
ARTIFACTS = ROOT / "artifacts"
OUTPUTS = ARTIFACTS / "notebooks"
LOGS = ARTIFACTS / "logs"
DATA_PROCESSED = ROOT / "data" / "processed"

# --- Candidate manifest ---
CANDIDATE_MANIFEST = DATA_PROCESSED / "candidate_manifest.json"
PRODUCTION_MODEL = "ensemble_lr_model"
CHAMPION_ALIAS = "champion"

# --- Deployed production Bento (incumbent) endpoint contract ---
# Evaluation and drift make incumbent predictions only through the deploying
# Bento's API-key-protected internal routes (host nginx), never by loading
# incumbent MLflow artifacts. Routes are the nginx allowlist in web/nginx.conf.template.
PRODUCTION_BENTO_URL = os.getenv("PRODUCTION_BENTO_URL", "http://127.0.0.1:8187")
MODEL_INFO_ROUTE = "/api/internal/model-info"
PREDICT_BATCH_ROUTE = "/api/internal/predict-batch"
DRIFT_API_KEY_HEADER = "X-Drift-API-Key"
DRIFT_API_KEY = os.getenv("DRIFT_API_KEY", "")
# Bento rejects batches above this cap (src/features/inference.BULK_MAX_ROWS);
# incumbent scoring chunks to stay beneath it.
INCUMBENT_BATCH_MAX_ROWS = 1000

# --- Champion lineage tags (single source of truth for the tag schema) ---
# 05_evaluate writes these onto the promoted ensemble model version before
# assigning @champion; src/flows/deploy.py reads them back to resolve exact
# base pins. Base models carry no aliases — exact version is the contract.


def build_lineage_tags(base_pins: dict, aux_pins: dict) -> dict[str, str]:
    """Flatten base/aux pins into champion model version tags.

    base_pins is the {name: pin} map consolidated by 03 (registered_model_name,
    version, run_id, model_uri, plus scaler_uri/scaler_hash for linear).
    aux_pins comes from 00 (embeddings/bio_feature_cols URIs + hashes).
    """
    tags: dict[str, str] = {}
    for name, pin in base_pins.items():
        for key in ("registered_model_name", "version", "run_id", "model_uri"):
            tags[f"base_{name}_{key}"] = str(pin[key])
        for key in ("scaler_uri", "scaler_hash"):
            if key in pin:
                tags[f"base_{name}_{key}"] = str(pin[key])
    for key in (
        "embeddings_uri",
        "embeddings_hash",
        "bio_feature_cols_uri",
        "bio_feature_cols_hash",
    ):
        tags[f"aux_{key}"] = str(aux_pins[key])
    return tags


# --- Table names ---
BRONZE_TABLE = "bronze.match_events"
SILVER_PLAYER_MATCHES = "silver.player_matches"
SILVER_ROLLING_FEATURES = "silver.rolling_features"
GOLD_TABLE = "gold.match_features"
TOUR_AVERAGES_TABLE = "gold.tour_averages"
# PROFILES_TABLE is the consumer-facing read relation: inference, similarity,
# and serving all read the stable, dbt-materialized gold.player_profiles.
PROFILES_TABLE = "gold.player_profiles"
# BRONZE_PROFILES_TABLE is the ingest write target: ATP identity loading and
# Wikipedia enrichment UPSERT here; dbt publishes bronze -> gold.
BRONZE_PROFILES_TABLE = "bronze.player_profiles"
