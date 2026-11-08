"""Paths and constants shared by every module."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW_CSV = ROOT / "data" / "raw" / "ai4i2020.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
METRICS_DIR = ROOT / "reports" / "metrics"

SPLIT_CSV = PROCESSED_DIR / "split.csv"
OOF_STAGE1_CSV = PROCESSED_DIR / "oof_stage1.csv"

SEED = 42
N_FOLDS = 5
FDR_CAP = 0.05
N_BOOTSTRAP = 2000

FAMILIES = ("logreg", "xgboost", "lightgbm")
FAMILY_LABELS = {
    "logreg": "Logistic regression",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}


def ensure_dirs() -> None:
    for folder in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
