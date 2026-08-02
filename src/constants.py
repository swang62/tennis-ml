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
PRODUCTION_MODEL = "production_model"  # MLflow registered model 05 promotes to

# --- Gold tables ---
GOLD_TABLE = "gold.match_features"
PROFILES_TABLE = "gold.player_profiles"

# --- Repo-local kernelspec used by the pipeline runner ---
KERNEL_NAME = "tennis-ml"
KERNEL_DIR = ROOT / ".jupyter" / "kernels" / KERNEL_NAME
