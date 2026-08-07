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

# --- PostgreSQL connection contract (single shared credential source) ---
# PostgreSQL is the only operational backend. Host commands and containers
# derive the same DATABASE_URL from these components; only
# POSTGRES_HOST/POSTGRES_PORT differ between endpoints — host commands use the
# configured .env target (the local Homebrew instance) while the Bento
# container on the Compose network uses the service DNS postgres:5432.
# POSTGRES_PASSWORD is a required runtime secret (never defaulted, never
# printed); DATABASE_URL, when provided, overrides the components entirely.
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB", "tennis")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "127.0.0.1")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "6543")
DATABASE_URL = os.getenv("DATABASE_URL")


def build_database_url(host: str | None = None) -> str:
    """Return the PostgreSQL connection URL for the shared component contract.

    An explicit DATABASE_URL wins over the components. Otherwise the effective
    host is the explicit ``host`` argument (used by the Bento container's
    postgres:5432), else POSTGRES_HOST; user, password, database, and port are
    identical for every endpoint, so only the host/port varies.
    """
    if DATABASE_URL:
        return DATABASE_URL
    effective_host = host or POSTGRES_HOST or "127.0.0.1"
    return (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{effective_host}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )


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
