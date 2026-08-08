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
# PostgreSQL is the only operational backend. Application connections use
# exactly one environment variable: DATABASE_URL — a passwordless URL for the
# local Homebrew instance (trust auth) or a password-bearing URL for the
# password-authenticated Compose stack (fixed local-dev credential in
# compose.yaml). There are no POSTGRES_* auth components for applications;
# dbt entry points derive their discrete fields from the same URL
# (src/db/conninfo.py).
DATABASE_URL = os.getenv("DATABASE_URL")


def build_database_url() -> str:
    """Return the application connection URL: the single DATABASE_URL contract.

    Fails fast when it is missing — there is no other backend and no component
    fallback to silently target. The URL is returned verbatim, passwordless
    (Homebrew trust) or password-bearing (Compose), exactly as configured.
    """
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

# --- Table names ---
BRONZE_TABLE = "bronze.match_events"
SILVER_PLAYER_MATCHES = "silver.player_matches"
SILVER_ROLLING_FEATURES = "silver.rolling_features"
GOLD_TABLE = "gold.match_features"
PROFILES_TABLE = "gold.player_profiles"
