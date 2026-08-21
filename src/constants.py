"""Single source of truth for shared paths, names, and environment settings."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    load_dotenv(ROOT / ".env", override=True)


load_env()


# --- PostgreSQL connection contract (single DATABASE_URL) ---
def get_database_url() -> str:
    """Return DATABASE_URL or fail rather than selecting an empty db."""
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise RuntimeError("missing PostgreSQL configuration: set DATABASE_URL")
    return DATABASE_URL


# --- Table names ---
BRONZE_MATCHES_TABLE = "bronze.match_events"
BRONZE_PROFILES_TABLE = "bronze.player_profiles"
BRONZE_RANKINGS_TABLE = "bronze.rankings"

SILVER_PLAYER_MATCHES = "silver.player_matches"
SILVER_ROLLING_FEATURES = "silver.rolling_features"

GOLD_MATCHES_TABLE = "gold.match_features"
GOLD_PROFILES_TABLE = "gold.player_profiles"
TOUR_AVERAGES_TABLE = "gold.tour_averages"

# ---- Config parameters ----
ENRICH_WORKERS = 4
BATCH_MAX_SIZE_ROWS = 1000
BULK_MAX_ROWS = 1000

# --- Core directories / files ---
NOTEBOOKS = ROOT / "notebooks"
PARAMS = NOTEBOOKS / "parameters"
ARTIFACTS = ROOT / "artifacts"
OUTPUTS = ARTIFACTS / "notebooks"
LOGS = ARTIFACTS / "logs"
DATA_PROCESSED = ROOT / "data" / "processed"
SCHEMA_SQL = ROOT / "infra" / "postgres" / "schema.sql"
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

# --- Chronological train/validation/test split fractions ---
# 01_train_test_split cuts the gold match date range into three contiguous
# chronological bands with these fractions. The validation fraction also sizes
# the within-fold early-stopping holdouts in the 02 tuners; the grouped
# temporal CV folds themselves are not the 94/4/2 split.
TRAIN_FRACTION = 0.94
VAL_FRACTION = 0.04
TEST_FRACTION = 0.02

# The three bands must partition the date span exactly.
assert abs(TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION - 1.0) < 1e-9

# Shared time-forward grouped-CV fold count written by 01 and consumed by
# every 02 tuner.
CV_FOLDS = 5

# --- Deployed production Bento ---
IMAGE_NAME = "tennis-bento"
PRODUCTION_BENTO_URL = os.getenv("PRODUCTION_BENTO_URL", "http://127.0.0.1:8187")
MODEL_INFO_ROUTE = "/api/model_info"
PREDICT_BATCH_ROUTE = "/api/predict_from_ids_bulk"
BENTO_API_KEY_HEADER = "X-API-Key"
BENTO_API_KEY = os.getenv("BENTO_API_KEY", "")


# Fixed base-model order for the ensemble stacker (training and serving).
STACK_ORDER = ("linear", "gbdt", "nn")


# --- Champion lineage tags (single source of truth for the tag schema) ---
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
)
BASE_TAG_PREFIX = "base_"
AUX_TAG_PREFIX = "aux_"
FEATURE_COLS_TAG = "feature_cols"
FEATURE_COLS_HASH_TAG = "feature_cols_hash"

# Probability-calibration artifact. The fitted temperature changes the served
# p_win, so it is pinned on the champion by URI+hash lineage tags at promotion
# time (CALIBRATION_URI_TAG/CALIBRATION_HASH_TAG). Legacy champions lacking
# those tags serve an explicit no-op t=1.0.
CALIBRATION_ARTIFACT = "calibration_t.json"
CALIBRATION_URI_TAG = "calibration_uri"
CALIBRATION_HASH_TAG = "calibration_hash"
CALIBRATION_STATE = DATA_PROCESSED / CALIBRATION_ARTIFACT
PLOTS = ARTIFACTS / "plots"


def build_lineage_tags(
    base_pins: dict[str, dict[str, str]], aux_pins: dict[str, str]
) -> dict[str, str]:
    """Flatten base/aux pins into champion model version tags.

    base_pins is the {name: pin} map consolidated by 03 (registered_model_name,
    version, run_id, model_uri, plus scaler_uri/scaler_hash for linear).
    aux_pins comes from 00 and carries only predictive-model inputs
    (embeddings/bio_feature_cols URIs + hashes); navigation artifacts are
    never pinned on the champion.
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


# Training-data cutoff date: the latest match date present in the training splits at promotion time.
TRAIN_DATA_MAX_DATE_KEY = "train_data_max_match_date"

# Performance metrics pinned on the champion model version at promotion time.
METRIC_PREFIX = "metric_"
EVAL_SPLIT_SIZE_KEY = "metric_eval_split_size"
EVAL_MAX_DATE_KEY = "metric_eval_max_date"

# --- Promotion gate ---
# Either primary metric may improve while the other trails by up to this amount.
PROMOTION_TOLERANCE = 0.01

# --- Drift monitoring thresholds (single source of truth for the verdict) ---
DRIFT_PSI_MODERATE = 0.1
DRIFT_PSI_SIGNIFICANT = 0.3
DRIFT_SHARE_THRESHOLD = 0.75
DRIFT_PRED_PSI_THRESHOLD = 0.3
DRIFT_CALIBRATION_DELTA = 0.1
DRIFT_AUC_DROP = 0.1
DRIFT_MIN_N_FOR_AUC = 30
DRIFT_MIN_N_FOR_CHECK = 10

# On-demand reference window bounds: size-matched to the current window,
# floored at DRIFT_REF_MIN and capped at DRIFT_REF_MAX matches.
DRIFT_REF_MIN = 30
DRIFT_REF_MAX = 10000

# --- Player similarity block calibration (explicit, reviewed weights) ---
# The similarity vector is a weighted concatenation of independently calibrated
# blocks: identity (one-hot), lifetime playstyle stats, surface career
# performance, reputation, and a PCA-reduced bio block.
SIM_IDENTITY_WEIGHT = 0.10
SIM_PLAYSTYLE_WEIGHT = 0.35
SIM_SURFACE_WEIGHT = 0.25
SIM_REPUTATION_WEIGHT = 0.25
SIM_BIO_WEIGHT = 0.05
SIM_BIO_PCA_DIM = 10
SIM_SURFACE_SHRINK_K = 30.0
SIM_RANK_SCALE = 200.0
SIM_EXPERIENCE_K = 100.0


# --- Serving model metadata / manifest keys ---
# BentoML model-metadata key shared by the deploy flow (write) and the serving
# manifest (read back as `bases.<name>.framework`).
FRAMEWORK_KEY = "framework"
