"""Single source of truth for shared paths, names, and environment settings."""

import os
from enum import StrEnum
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    load_dotenv(ROOT / ".env", override=True)


load_env()


def get_database_url() -> str:
    """Return DATABASE_URL or fail rather than selecting an empty db."""
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise RuntimeError("missing PostgreSQL configuration: set DATABASE_URL")
    return DATABASE_URL


BRONZE_MATCHES_TABLE = "bronze.match_events"
BRONZE_PROFILES_TABLE = "bronze.player_profiles"
BRONZE_RANKINGS_TABLE = "bronze.rankings"

SILVER_PLAYER_MATCHES = "silver.player_matches"
SILVER_ROLLING_FEATURES = "silver.rolling_features"

GOLD_MATCHES_TABLE = "gold.match_features"
GOLD_PROFILES_TABLE = "gold.player_profiles"
TOUR_AVERAGES_TABLE = "gold.tour_averages"

ENRICH_WORKERS = 4
BATCH_MAX_SIZE_ROWS = 1000
BULK_MAX_ROWS = 1000

NOTEBOOKS = ROOT / "notebooks"
PARAMS = NOTEBOOKS / "parameters"
ARTIFACTS = ROOT / "artifacts"
OUTPUTS = ARTIFACTS / "notebooks"
LOGS = ARTIFACTS / "logs"
DATA_PROCESSED = ROOT / "data" / "processed"
SCHEMA_SQL = ROOT / "infra" / "postgres" / "schema.sql"

# Generated model artifacts (manifests, calibration, indexes)
# live under root models/; only the selected promoted serving artifacts are
# staged under models/deploy before Bento packaging.
MODELS_ROOT = ROOT / "models"
MODELS_ARTIFACTS = MODELS_ROOT
DEPLOY_ARTIFACTS = MODELS_ROOT / "deploy"

FROZEN_ARTIFACTS = ("linear_scaler.pkl",)

CANDIDATE_MANIFEST = MODELS_ARTIFACTS / "candidate_manifest.json"
PRODUCTION_MODEL = "ensemble_lr_model"
CHAMPION_ALIAS = "champion"

WORK_POOL_NAME = "tennis-pool"

TRAIN_FRACTION = 0.96
VAL_FRACTION = 0.02
TEST_FRACTION = 0.02

assert abs(TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION - 1.0) < 1e-9

CV_FOLDS = 5

IMAGE_NAME = "tennis-bento"
PRODUCTION_BENTO_URL = os.getenv("PRODUCTION_BENTO_URL", "http://127.0.0.1:8187")
MODEL_INFO_ROUTE = "/api/model_info"
PREDICT_BATCH_ROUTE = "/api/predict_from_ids_bulk"
BENTO_API_KEY_HEADER = "X-API-Key"
BENTO_API_KEY = os.getenv("BENTO_API_KEY", "")


STACK_ORDER = ("linear", "gbdt", "nn")


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
LINEAGE_AUX_KEYS = ()
BASE_TAG_PREFIX = "base_"
AUX_TAG_PREFIX = "aux_"
# Exact source run id that produced an immutable frozen selection (tuning run
# for bases, promotion run for the ensemble). Used by the promotion notebook so
# the standing champion resolves identical provenance.
PIPELINE_SOURCE_RUN_ID_TAG = "pipeline_source_run_id"
FEATURE_COLS_TAG = "feature_cols"
FEATURE_COLS_HASH_TAG = "feature_cols_hash"

CALIBRATION_ARTIFACT = "calibration_t.json"
CALIBRATION_URI_TAG = "calibration_uri"
CALIBRATION_HASH_TAG = "calibration_hash"
CALIBRATION_STATE = MODELS_ARTIFACTS / CALIBRATION_ARTIFACT
PLOTS = ARTIFACTS / "plots"

CHAMPION_CURVE_ARTIFACT = "champion_curves.json"
CHAMPION_CURVE_URI_TAG = "champion_curve_uri"
CHAMPION_CURVE_HASH_TAG = "champion_curve_hash"


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


TRAIN_DATA_MAX_DATE_KEY = "train_data_max_match_date"

METRIC_PREFIX = "metric_"
EVAL_SPLIT_SIZE_KEY = "metric_eval_split_size"
EVAL_MAX_DATE_KEY = "metric_eval_max_date"

PROMOTION_TOLERANCE = 0.01

DRIFT_PSI_MODERATE = 0.1
DRIFT_PSI_SIGNIFICANT = 0.3
DRIFT_SHARE_THRESHOLD = 0.75
DRIFT_PRED_PSI_THRESHOLD = 0.3
DRIFT_CALIBRATION_DELTA = 0.1
DRIFT_AUC_DROP = 0.1
DRIFT_MIN_N_FOR_AUC = 30
DRIFT_MIN_N_FOR_CHECK = 10

RECENCY_HALF_LIFE_DAYS = 8 * 365.25

# Recency selection lineage keys: the eight-year half-life and the explicit
# full-snapshot cutoff passed to recency_weights, recorded on selection runs,
# the candidate manifest, and champion version tags.
RECENCY_HALF_LIFE_KEY = "recency_half_life_days"
RECENCY_CUTOFF_KEY = "recency_cutoff_date"

DRIFT_REF_MIN = 30
DRIFT_REF_MAX = 10000

SIM_IDENTITY_WEIGHT = 0.10
SIM_PLAYSTYLE_WEIGHT = 0.35
SIM_SURFACE_WEIGHT = 0.25
SIM_REPUTATION_WEIGHT = 0.25
SIM_SURFACE_SHRINK_K = 30.0
SIM_EXPERIENCE_K = 100.0


FRAMEWORK_KEY = "framework"


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
