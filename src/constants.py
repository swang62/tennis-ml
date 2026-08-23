"""Single source of truth for shared paths, names, and environment settings."""

import os
import sys
from datetime import date
from enum import StrEnum
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = ROOT / "models"


def load_env() -> None:
    load_dotenv(ROOT / ".env", override=True)


if "pytest" not in sys.modules:
    load_env()


def get_database_url() -> str:
    """Return DATABASE_URL or fail rather than selecting an empty db."""
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise RuntimeError("missing PostgreSQL configuration: set DATABASE_URL")
    return DATABASE_URL


###### Drift Parameters #####

DRIFT_AUC_DROP = 0.1
DRIFT_CALIBRATION_DELTA = 0.1
DRIFT_MIN_N_FOR_AUC = 30
DRIFT_MIN_N_FOR_CHECK = 10
DRIFT_PRED_PSI_THRESHOLD = 0.3
DRIFT_REF_MAX = 10000
DRIFT_REF_MIN = 30
DRIFT_SHARE_THRESHOLD = 0.75
DRIFT_PSI_MODERATE = 0.1
DRIFT_PSI_SIGNIFICANT = 0.3


###### Similarity Parameters #####

SIM_EXPERIENCE_K = 100.0
SIM_IDENTITY_WEIGHT = 0.10
SIM_PLAYSTYLE_WEIGHT = 0.35
SIM_REPUTATION_WEIGHT = 0.25
SIM_SURFACE_SHRINK_K = 30.0
SIM_SURFACE_WEIGHT = 0.25


###### Training Parameters #####

BATCH_MAX_SIZE_ROWS = 1000
BULK_MAX_ROWS = 1000
CV_FOLDS = 5
ENRICH_WORKERS = 4
MIN_TRAINING_DATE = date(1995, 1, 1)
PROMOTION_TOLERANCE = 0.01
RECENCY_HALF_LIFE_DAYS = 8 * 365.25

TRAIN_FRACTION = 0.90
TEST_FRACTION = 0.05
VAL_FRACTION = 0.05

assert abs(TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION - 1.0) < 1e-9


###### Directory Paths #####

ARTIFACTS = ROOT / "artifacts"
CALIBRATION_PLOTS = ARTIFACTS / "plots"
DATA_PROCESSED = ROOT / "data" / "processed"
DEPLOY_ARTIFACTS = MODELS_ROOT / "deploy"
LOGS = ARTIFACTS / "logs"
MODELS_ARTIFACTS = MODELS_ROOT
NOTEBOOKS = ROOT / "notebooks"
OUTPUTS = ARTIFACTS / "notebooks"
PARAMS = NOTEBOOKS / "parameters"
SCHEMA_SQL = ROOT / "infra" / "postgres" / "schema.sql"


###### Database Tables #####

BRONZE_MATCHES_TABLE = "bronze.match_events"
BRONZE_PROFILES_TABLE = "bronze.player_profiles"
BRONZE_RANKINGS_TABLE = "bronze.rankings"

SILVER_PLAYER_MATCHES = "silver.player_matches"
SILVER_ROLLING_FEATURES = "silver.rolling_features"

GOLD_MATCHES_TABLE = "gold.match_features"
GOLD_PROFILES_TABLE = "gold.player_profiles"
GOLD_TOUR_AVERAGES_TABLE = "gold.tour_averages"


###### Training and Evaluation Keys #####

EVAL_MAX_DATE_KEY = "metric_eval_max_date"
EVAL_SPLIT_SIZE_KEY = "metric_eval_split_size"
METRIC_PREFIX = "metric_"
MIN_TRAINING_DATE_KEY = "min_training_date"
RECENCY_CUTOFF_KEY = "recency_cutoff_date"
RECENCY_HALF_LIFE_KEY = "recency_half_life_days"
TEST_FRACTION_KEY = "test_fraction"
TRAIN_DATA_MAX_DATE_KEY = "train_data_max_match_date"
TRAIN_FRACTION_KEY = "train_fraction"
VAL_FRACTION_KEY = "val_fraction"


###### Lineage and Metadata #####

AUX_TAG_PREFIX = "aux_"
BASE_TAG_PREFIX = "base_"
CALIBRATION_HASH_TAG = "calibration_hash"
CALIBRATION_URI_TAG = "calibration_uri"
CHAMPION_CURVE_HASH_TAG = "champion_curve_hash"
CHAMPION_CURVE_URI_TAG = "champion_curve_uri"
FEATURE_COLS_HASH_TAG = "feature_cols_hash"
FEATURE_COLS_TAG = "feature_cols"
LINEAGE_AUX_KEYS = ()
LINEAGE_MODEL_NAME_KEY = "registered_model_name"
LINEAGE_MODEL_URI_KEY = "model_uri"
LINEAGE_RUN_ID_KEY = "run_id"
LINEAGE_VERSION_KEY = "version"
LINEAGE_BASE_KEYS = (
    LINEAGE_MODEL_NAME_KEY,
    LINEAGE_VERSION_KEY,
    LINEAGE_RUN_ID_KEY,
    LINEAGE_MODEL_URI_KEY,
)
LINEAGE_SCALER_KEYS = ("scaler_uri", "scaler_hash")
PIPELINE_SOURCE_RUN_ID_TAG = "pipeline_source_run_id"


###### Artifacts and Model Registry #####

CALIBRATION_ARTIFACT = "calibration_t.json"
CALIBRATION_STATE = MODELS_ARTIFACTS / CALIBRATION_ARTIFACT
CANDIDATE_MANIFEST = MODELS_ARTIFACTS / "candidate_manifest.json"
CHAMPION_ALIAS = "champion"
CHAMPION_CURVE_ARTIFACT = "champion_curves.json"
FROZEN_ARTIFACTS = ("linear_scaler.pkl",)
PRODUCTION_MODEL = "ensemble_lr_model"


###### Bento Inference and Serving #####

BENTO_API_KEY = os.getenv("BENTO_API_KEY", "")
BENTO_API_KEY_HEADER = "X-API-Key"
IMAGE_NAME = "tennis-bento"
MODEL_INFO_ROUTE = "/api/model_info"
PREDICT_BATCH_ROUTE = "/api/predict_from_ids_bulk"
PRODUCTION_BENTO_URL = os.getenv("PRODUCTION_BENTO_URL", "http://127.0.0.1:8187")
STACK_ORDER = ("linear", "gbdt", "nn")
WORK_POOL_NAME = "tennis-pool"


###### Model Framework Selection #####

FRAMEWORK_KEY = "framework"


def build_lineage_tags(
    base_pins: dict[str, dict[str, str]], aux_pins: dict[str, str]
) -> dict[str, str]:
    """Flatten base and auxiliary lineage pins into champion version tags."""
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


class LinearFramework(StrEnum):
    LOGISTIC_REGRESSION = "logistic_regression"
    GAUSSIAN_NAIVE_BAYES = "gaussian_naive_bayes"


class GBDTFramework(StrEnum):
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"


def normalize_linear_framework(value: str) -> LinearFramework:
    """Return a canonical linear framework, accepting legacy tags."""
    aliases = {
        "lr": LinearFramework.LOGISTIC_REGRESSION,
        "nb": LinearFramework.GAUSSIAN_NAIVE_BAYES,
    }
    try:
        return LinearFramework(value)
    except ValueError:
        try:
            return aliases[value]
        except KeyError as exc:
            raise ValueError(f"unsupported linear framework: {value!r}") from exc


def normalize_gbdt_framework(value: str) -> GBDTFramework:
    """Return a canonical GBDT framework, accepting legacy tags."""
    aliases = {"xgb": GBDTFramework.XGBOOST, "lgbm": GBDTFramework.LIGHTGBM}
    try:
        return GBDTFramework(value)
    except ValueError:
        try:
            return aliases[value]
        except KeyError as exc:
            raise ValueError(f"unsupported GBDT framework: {value!r}") from exc
