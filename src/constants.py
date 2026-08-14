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
# Upper bound for a single bulk inference request (Nginx chunks below this).
BULK_MAX_ROWS = 1000

# --- Core directories / files ---
NOTEBOOKS = ROOT / "notebooks"
PARAMS = NOTEBOOKS / "parameters"
ARTIFACTS = ROOT / "artifacts"
OUTPUTS = ARTIFACTS / "notebooks"
LOGS = ARTIFACTS / "logs"
DATA_PROCESSED = ROOT / "data" / "processed"
INIT_SQL = ROOT / "infra" / "postgres" / "init.sql"
DEPLOY_ARTIFACTS = ROOT / "data" / "deploy"

# Model-tied serving artifacts frozen into DEPLOY_ARTIFACTS on promotion
FROZEN_ARTIFACTS = (
    "linear_scaler.pkl",
    "bio_embeddings.npz",
    "bio_feature_cols.json",
)

# --- Candidate manifest ---
CANDIDATE_MANIFEST = DATA_PROCESSED / "candidate_manifest.json"
PRODUCTION_MODEL = "ensemble_lr_model"
CHAMPION_ALIAS = "champion"

# --- Prefect ---
WORK_POOL_NAME = "tennis-pool"

# --- Deployed production Bento endpoint ---
PRODUCTION_BENTO_URL = os.getenv("PRODUCTION_BENTO_URL", "http://127.0.0.1:8187")
MODEL_INFO_ROUTE = "/api/model_info"
PREDICT_BATCH_ROUTE = "/api/predict_from_ids_bulk"
BENTO_API_KEY_HEADER = "X-API-Key"
BENTO_API_KEY = os.getenv("BENTO_API_KEY", "")


# Fixed base-model order for the ensemble stacker (training and serving).
STACK_ORDER = ("linear", "gbdt", "nn")


# --- Champion lineage tags (single source of truth for the tag schema) ---
# Tag-key names and prefixes used by the flattened `base_*`/`aux_*` champion
# lineage tags; shared by the tag builder here, promotion, and the deploy flow.
LINEAGE_MODEL_NAME_KEY = "registered_model_name"
LINEAGE_VERSION_KEY = "version"
LINEAGE_RUN_ID_KEY = "run_id"
LINEAGE_MODEL_URI_KEY = "model_uri"
LINEAGE_BASE_KEYS = (
    LINEAGE_MODEL_NAME_KEY,
    LINEAGE_VERSION_KEY,
    LINEAGE_RUN_ID_KEY,
    LINEAGE_MODEL_URI_KEY,
)
LINEAGE_SCALER_KEYS = ("scaler_uri", "scaler_hash")
LINEAGE_AUX_KEYS = (
    "embeddings_uri",
    "embeddings_hash",
    "bio_feature_cols_uri",
    "bio_feature_cols_hash",
    "similarity_index_uri",
    "similarity_index_hash",
    "similarity_metadata_uri",
    "similarity_metadata_hash",
)
BASE_TAG_PREFIX = "base_"
AUX_TAG_PREFIX = "aux_"

# Training-data watermark pinned on the champion model version: the latest
# match date present in the training splits at promotion time. Drift checks use
# it as the cutoff for "new" matches instead of the model-registration time.
TRAIN_DATA_MAX_DATE_KEY = "train_data_max_match_date"

# Performance metrics pinned on the champion model version at promotion time.
# Metric tag keys are METRIC_PREFIX + metric name (the 8 in METRIC_NAMES);
# drift reads them with this uniform prefix.
METRIC_PREFIX = "metric_"
METRIC_COMPOSITE_KEY = "metric_promotion_composite"
EVAL_SPLIT_SIZE_KEY = "metric_eval_split_size"
EVAL_MAX_DATE_KEY = "metric_eval_max_date"

# --- Drift monitoring thresholds (single source of truth for the verdict) ---
# Per-feature PSI bands: < DRIFT_PSI_MODERATE = no drift, moderate band between,
# >= DRIFT_PSI_SIGNIFICANT = significant. Drift share is the fraction of
# features with PSI >= DRIFT_PSI_SIGNIFICANT. Prediction PSI and calibration /
# AUC degradation are retrain triggers; the AUC trigger only applies once the
# current window is large enough to mean anything.
DRIFT_PSI_MODERATE = 0.1
DRIFT_PSI_SIGNIFICANT = 0.2
DRIFT_SHARE_THRESHOLD = 0.5
DRIFT_PRED_PSI_THRESHOLD = 0.2
DRIFT_CALIBRATION_DELTA = 0.05
DRIFT_AUC_DROP = 0.05
DRIFT_MIN_N_FOR_AUC = 30

# On-demand reference window bounds: size-matched to the current window,
# floored at DRIFT_REF_MIN and capped at DRIFT_REF_MAX matches.
DRIFT_REF_MIN = 50
DRIFT_REF_MAX = 2000


def build_lineage_tags(
    base_pins: dict[str, dict[str, str]], aux_pins: dict[str, str]
) -> dict[str, str]:
    """Flatten base/aux pins into champion model version tags.

    base_pins is the {name: pin} map consolidated by 03 (registered_model_name,
    version, run_id, model_uri, plus scaler_uri/scaler_hash for linear).
    aux_pins comes from 00 (embeddings/bio_feature_cols URIs + hashes).
    """
    tags: dict[str, str] = {}
    for name, pin in base_pins.items():
        for key in LINEAGE_BASE_KEYS:
            tags[f"{BASE_TAG_PREFIX}{name}_{key}"] = str(pin[key])
        for key in LINEAGE_SCALER_KEYS:
            if key in pin:
                tags[f"{BASE_TAG_PREFIX}{name}_{key}"] = str(pin[key])
    for key in LINEAGE_AUX_KEYS:
        tags[f"{AUX_TAG_PREFIX}{key}"] = str(aux_pins[key])
    return tags


# --- Serving model metadata / manifest keys ---
# BentoML model-metadata key shared by the deploy flow (write) and the serving
# manifest (read back as `bases.<name>.framework`).
FRAMEWORK_KEY = "framework"
