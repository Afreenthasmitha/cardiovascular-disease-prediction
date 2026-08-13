"""
config.py — Central configuration for the Cardiovascular Disease Prediction System.
Edit the constants here when adapting to a different dataset.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR         = os.path.join(BASE_DIR, "models")
REPORTS_DIR        = os.path.join(BASE_DIR, "reports")
FIGURES_DIR        = os.path.join(BASE_DIR, "reports", "figures")

DATASET_PATH       = os.path.join(DATA_RAW_DIR, "dataset.csv")
BEST_MODEL_PATH    = os.path.join(MODELS_DIR, "best_model_pipeline.pkl")
MODEL_METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
RESULTS_CSV_PATH   = os.path.join(REPORTS_DIR, "model_results.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Dataset configuration  ← adapt here for different datasets
# ─────────────────────────────────────────────────────────────────────────────

# Set to the exact target column name in your dataset.
# For the cardiovascular disease dataset the target is "cardio".
TARGET_COLUMN: str = "cardio"

# Known target-column names (checked in order; first match wins)
CANDIDATE_TARGET_COLUMNS = [
    "cardio", "target", "heart_disease", "HeartDisease",
    "output", "num", "disease", "label", "class",
]

# Columns to drop unconditionally (e.g. patient IDs)
DROP_COLUMNS: list[str] = ["id", "ID", "patient_id"]

# ─────────────────────────────────────────────────────────────────────────────
# ML configuration
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE   = 42
TEST_SIZE      = 0.20
CV_FOLDS       = 5

# Primary metric used to select the best model
# Options: "roc_auc", "f1", "recall", "accuracy", "precision"
PRIMARY_METRIC = "roc_auc"

# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────
FIGURE_DPI  = 120
FIGURE_SIZE = (10, 6)
COLOR_POS   = "#E74C3C"   # disease present
COLOR_NEG   = "#2ECC71"   # disease absent
PALETTE     = "husl"
