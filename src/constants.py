"""Single source of truth for shared paths, names, and defaults across flows.

Pure configuration: no imports from other src modules and no side effects on
import (environment loading stays explicit via src.utils.load_env()).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Repo layout ---
NOTEBOOKS = ROOT / "notebooks"
PARAMS = NOTEBOOKS / "parameters"
ARTIFACTS = ROOT / "artifacts"
OUTPUTS = ARTIFACTS / "notebooks"
DATA_PROCESSED = ROOT / "data" / "processed"

# --- Candidate manifest (written by 04, evaluated by 05) ---
CANDIDATE_MANIFEST = DATA_PROCESSED / "candidate_manifest.json"
PRODUCTION_MODEL = "ensemble_lr_model"  # MLflow registered model 05 promotes to

# --- Table names (single source of truth; no hardcoded names elsewhere) ---
# Medallion layers: bronze (raw ingest) -> silver (per-player expansion) ->
# gold (features). Consumer modules import these instead of re-declaring
# schema-qualified table names in SQL or module constants.
BRONZE_TABLE = "bronze.match_events"
SILVER_PLAYER_MATCHES = "silver.player_matches"
SILVER_PLAYER_RANKINGS = "silver.player_rankings"
GOLD_ROLLING_FEATURES = "gold.rolling_features"
GOLD_TABLE = "gold.match_features"  # canonical one-row-per-match training rows
PROFILES_TABLE = "gold.player_profiles"

# --- Repo-local kernelspec used by the pipeline runner ---
KERNEL_NAME = "tennis-ml"
KERNEL_DIR = ROOT / ".jupyter" / "kernels" / KERNEL_NAME
