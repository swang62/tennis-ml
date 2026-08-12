"""Single source of truth for shared paths, names, and environment settings."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    load_dotenv(ROOT / ".env", override=True)


load_env()

# --- Environment Variables -----
IMAGE_NAME = os.getenv("IMAGE_NAME")


# --- PostgreSQL connection contract (single DATABASE_URL) ---
def get_database_url() -> str:
    """Return DATABASE_URL or fail rather than selecting an empty db."""
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise RuntimeError("missing PostgreSQL configuration: set DATABASE_URL")
    return DATABASE_URL


# --- Table names ---
BRONZE_TABLE = "bronze.match_events"
BRONZE_PROFILES_TABLE = "bronze.player_profiles"
RANKINGS_TABLE = "bronze.rankings"

SILVER_PLAYER_MATCHES = "silver.player_matches"
SILVER_ROLLING_FEATURES = "silver.rolling_features"

GOLD_TABLE = "gold.match_features"
TOUR_AVERAGES_TABLE = "gold.tour_averages"
PROFILES_TABLE = "gold.player_profiles"

# ---- Config parameters ----
ENRICH_WORKERS = 4
BATCH_MAX_SIZE_ROWS = 1000

# --- Core directories / files ---
NOTEBOOKS = ROOT / "notebooks"
PARAMS = NOTEBOOKS / "parameters"
ARTIFACTS = ROOT / "artifacts"
OUTPUTS = ARTIFACTS / "notebooks"
LOGS = ARTIFACTS / "logs"
DATA_PROCESSED = ROOT / "data" / "processed"
INIT_SQL = ROOT / "infra" / "postgres" / "init.sql"

# --- Candidate manifest ---
CANDIDATE_MANIFEST = DATA_PROCESSED / "candidate_manifest.json"
PRODUCTION_MODEL = "ensemble_lr_model"
CHAMPION_ALIAS = "champion"

# --- Deployed production Bento endpoint ---
PRODUCTION_BENTO_URL = os.getenv("PRODUCTION_BENTO_URL", "http://127.0.0.1:8187")
MODEL_INFO_ROUTE = "/api/internal/model-info"
PREDICT_BATCH_ROUTE = "/api/internal/predict-batch"
DRIFT_API_KEY_HEADER = "X-Drift-API-Key"
DRIFT_API_KEY = os.getenv("DRIFT_API_KEY", "")


# --- Champion lineage tags (single source of truth for the tag schema) ---
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
