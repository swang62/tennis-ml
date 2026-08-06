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
TENNIS_DB_PATH = os.getenv("TENNIS_DB_PATH")

# --- Serving mode: dev (embedded DuckDB) vs production (Quack remote) -----
# `dev` (default) opens the local embedded DB at TENNIS_DB_PATH (or
# data/tennis.duckdb). `production` attaches a remote Quack server as the
# default catalog so existing schema-qualified SQL reaches the served DB.
# Any other value is treated as a configuration error (`src.db.client` fails
# fast rather than silently falling back).
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")

# Quack remote config, only read when ENVIRONMENT == "production".
QUACK_URI = os.getenv("QUACK_URI")
QUACK_TOKEN = os.getenv("QUACK_TOKEN")
QUACK_CATALOG = os.getenv("QUACK_CATALOG", "tennis")

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

# --- Table names ---
BRONZE_TABLE = "bronze.match_events"
SILVER_PLAYER_MATCHES = "silver.player_matches"
GOLD_ROLLING_FEATURES = "gold.rolling_features"
GOLD_TABLE = "gold.match_features"
PROFILES_TABLE = "gold.player_profiles"
