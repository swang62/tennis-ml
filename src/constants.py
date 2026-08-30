"""Single source of truth for shared paths, names, and environment settings."""

import os
import sys
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any

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
DAYS_SINCE_LAST_MATCH_MAX = 90
ENRICH_WORKERS = 4
OPTUNA_PRUNER_INTERVAL_STEPS = 10
OPTUNA_PRUNER_MIN_TRIALS = 10
OPTUNA_N_TRIALS = 60
OPTUNA_PRUNER_STARTUP_TRIALS = 30
OPTUNA_PRUNER_WARMUP_STEPS = 50
PROMOTION_TOLERANCE = 0.01
RECENCY_HALF_LIFE_DAYS = 8 * 365.25

###### Data split #####

MIN_TRAINING_DATE = date(1990, 1, 1)
TRAIN_FRACTION = 0.9
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
BRONZE_ETL_STATE = "bronze.etl_state"

SILVER_PLAYER_MATCHES = "silver.player_matches"
SILVER_ROLLING_FEATURES = "silver.rolling_features"
SILVER_ELO_SNAPSHOTS = "silver.elo_snapshots"

GOLD_MATCHES_TABLE = "gold.match_features"
GOLD_PROFILES_TABLE = "gold.player_profiles"
GOLD_TOUR_AVERAGES_TABLE = "gold.tour_averages"


###### Elo Parameters #####

ELO_DEFAULT_RATING = 1500.0
ELO_K_BASE = 43.0
ELO_K_MIN = 62.0
ELO_K_DIVISOR = 800.0
ELO_INACTIVITY_GRACE_DAYS = 90
ELO_INACTIVITY_REGRESS_PER_7D = 0.01
ELO_INACTIVITY_REGRESS_CAP = 0.50


###### Model Artifacts #####

CANDIDATE_MANIFEST = MODELS_ARTIFACTS / "candidate_manifest.json"
NN_PREPROCESSING_ARTIFACT = "nn_preprocessing.json"
CALIBRATION_ARTIFACT = "calibration_t.json"
CALIBRATION_STATE = MODELS_ARTIFACTS / CALIBRATION_ARTIFACT
CHAMPION_CURVE_ARTIFACT = "champion_curves.json"
FROZEN_ARTIFACTS = ("linear_scaler.pkl",)


###### Model Registry #####

CHAMPION_ALIAS = "champion"
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
PREFECT_FLOW_TIMEOUT_SECONDS = 60 * 60


def build_lineage_tags(base_pins: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Flatten base-model lineage pins into champion version tags."""
    tags: dict[str, str] = {}
    for name, pin in base_pins.items():
        if name == "nn" and "preprocessing_uri" in pin and "preprocessing_sha256" in pin:
            tags["base_nn_preprocessing_uri"] = str(pin["preprocessing_uri"])
            tags["base_nn_preprocessing_hash"] = str(pin["preprocessing_sha256"])
        for key in ("registered_model_name", "version", "run_id", "model_uri"):
            tags[f"base_{name}_{key}"] = str(pin[key])
        for key in ("scaler_uri", "scaler_hash"):
            if key in pin:
                tags[f"base_{name}_{key}"] = str(pin[key])
        if "selected_framework" in pin:
            tags[f"base_{name}_selected_framework"] = str(pin["selected_framework"])
    return tags


class LinearFramework(StrEnum):
    LOGISTIC_REGRESSION = "logistic_regression"
    GAUSSIAN_NAIVE_BAYES = "gaussian_naive_bayes"
    SGD_CLASSIFIER = "sgd_classifier"


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
