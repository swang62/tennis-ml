"""Single source of truth for shared paths, names, and defaults across flows.

Pure configuration: no imports from other src modules and no side effects on
import (environment loading stays explicit via load_env()).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


# --- Environment accessors ------------------------------------------------
# All env lookups live here.  Accessor functions read os.environ at call time
# so monkeypatched values are always respected.  load_env() must be called
# explicitly (not on import) before first access if .env values are needed.


def load_env() -> None:
    """Load the repo .env file into os.environ (idempotent).

    Existing process-environment values win over .env entries (override=False),
    which is the right precedence: shell / CI / test monkeypatch > .env file.
    """
    load_dotenv(ROOT / ".env", override=False)


def image_name() -> str | None:
    """IMAGE_NAME from the environment."""
    return os.environ.get("IMAGE_NAME")


def registry_push_url() -> str | None:
    """REGISTRY_PUSH_URL from the environment."""
    return os.environ.get("REGISTRY_PUSH_URL")


def tennis_db_path() -> str | None:
    """TENNIS_DB_PATH from the environment (used by tests to redirect DB access)."""
    return os.environ.get("TENNIS_DB_PATH")


# --- Core directories ---
NOTEBOOKS = ROOT / "notebooks"
PARAMS = NOTEBOOKS / "parameters"
ARTIFACTS = ROOT / "artifacts"
OUTPUTS = ARTIFACTS / "notebooks"
DATA_PROCESSED = ROOT / "data" / "processed"

# --- Candidate manifest ---
CANDIDATE_MANIFEST = DATA_PROCESSED / "candidate_manifest.json"
PRODUCTION_MODEL = "ensemble_lr_model"

# --- Table names ---
BRONZE_TABLE = "bronze.match_events"
SILVER_PLAYER_MATCHES = "silver.player_matches"
SILVER_PLAYER_RANKINGS = "silver.player_rankings"
GOLD_ROLLING_FEATURES = "gold.rolling_features"
GOLD_TABLE = "gold.match_features"
PROFILES_TABLE = "gold.player_profiles"
