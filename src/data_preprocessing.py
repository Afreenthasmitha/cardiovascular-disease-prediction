"""
data_preprocessing.py — Dataset loading, inspection, cleaning, and validation.
"""

from __future__ import annotations

import os
import warnings
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a CSV or Excel dataset from *path*.

    Automatically handles both comma-separated and semicolon-separated CSV files.
    After loading, strips whitespace from all column names.

    Returns a pandas DataFrame.
    Raises FileNotFoundError / ValueError on problems.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            # First attempt: auto-detect separator
            df = pd.read_csv(path, sep=None, engine="python")

            # Fallback: if only one column was produced and it contains
            # semicolons, the file is semicolon-separated — retry explicitly.
            if len(df.columns) == 1 and ";" in str(df.columns[0]):
                df = pd.read_csv(path, sep=";")

    except Exception as exc:
        raise ValueError(f"Could not load dataset: {exc}") from exc

    # Normalise column names: strip surrounding whitespace
    df.columns = df.columns.astype(str).str.strip()

    if df.empty:
        raise ValueError("The loaded dataset is empty.")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Target detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_target_column(
    df: pd.DataFrame,
    candidate_names: list[str],
    user_override: str | None = None,
) -> str:
    """
    Return the name of the target column.

    Priority order:
      1. user_override (if provided and exists in df)
      2. First match from candidate_names
      3. Raise an informative error asking the user to specify.
    """
    if user_override:
        if user_override in df.columns:
            return user_override
        raise ValueError(
            f"Target column '{user_override}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    for name in candidate_names:
        if name in df.columns:
            return name

    raise ValueError(
        "Could not automatically detect the target column.\n"
        f"Expected one of: {candidate_names}\n"
        f"Available columns: {list(df.columns)}\n"
        "Please set TARGET_COLUMN in src/config.py."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Inspection
# ──────────────────────────────────────────────────────────────────────────────

def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return a dictionary of dataset inspection results.
    """
    n_rows, n_cols = df.shape

    missing_counts = df.isnull().sum()
    missing_pct    = (missing_counts / n_rows * 100).round(2)
    missing_df     = pd.DataFrame({
        "Missing Count": missing_counts,
        "Missing %": missing_pct,
    }).query("`Missing Count` > 0")

    dup_count = int(df.duplicated().sum())

    numeric_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Potential id columns — single unique value per row and small dtype
    potential_id_cols = [
        c for c in df.columns
        if df[c].nunique() == n_rows and df[c].dtype in (object, "int64")
    ]

    report = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_df": missing_df,
        "total_missing": int(missing_counts.sum()),
        "dup_count": dup_count,
        "dup_pct": round(dup_count / n_rows * 100, 2),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "potential_id_cols": potential_id_cols,
        "describe": df.describe(include="all"),
    }
    return report


# ──────────────────────────────────────────────────────────────────────────────
# Cleaning
# ──────────────────────────────────────────────────────────────────────────────

def drop_identifier_columns(
    df: pd.DataFrame,
    drop_cols: list[str],
) -> Tuple[pd.DataFrame, list[str]]:
    """Drop known identifier / irrelevant columns that exist in df."""
    existing = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing, errors="ignore")
    return df, existing


def handle_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Remove duplicate rows. Returns cleaned df and number removed."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    return df, removed


def handle_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Impute missing values:
      - Numerical  → median
      - Categorical → mode
    Returns the cleaned DataFrame and an imputation report.
    """
    report: Dict[str, Any] = {}
    for col in df.columns:
        n_missing = df[col].isnull().sum()
        if n_missing == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            fill_val = df[col].median()
            strategy = "median"
        else:
            fill_val = df[col].mode()[0]
            strategy = "mode"

        df[col] = df[col].fillna(fill_val)
        report[col] = {
            "missing_count": int(n_missing),
            "strategy": strategy,
            "fill_value": fill_val,
        }

    return df, report


def remove_invalid_bp_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    For blood-pressure columns (ap_hi / ap_lo), remove rows where values
    are physiologically implausible (e.g. <= 0 or > 300).
    Only applied when those columns exist.
    """
    bad_rows = 0
    bp_cols = {"ap_hi": (1, 300), "ap_lo": (1, 200)}
    for col, (lo, hi) in bp_cols.items():
        if col in df.columns:
            mask = (df[col] < lo) | (df[col] > hi)
            bad_rows += int(mask.sum())
            df = df[~mask]
    return df, bad_rows


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """
    Conditionally add engineered features.
    Returns the DataFrame and a list of new column names.
    """
    new_features: list[str] = []

    # BMI — requires height (cm) and weight (kg)
    if "height" in df.columns and "weight" in df.columns:
        height_m = df["height"] / 100.0
        df["bmi"] = (df["weight"] / (height_m ** 2)).round(2)
        new_features.append("bmi")

    # Age in years — if age is stored in days (typical for cardio dataset)
    if "age" in df.columns and df["age"].median() > 365:
        df["age_years"] = (df["age"] / 365.25).round(1)
        new_features.append("age_years")

    # Pulse pressure — systolic minus diastolic
    if "ap_hi" in df.columns and "ap_lo" in df.columns:
        df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
        new_features.append("pulse_pressure")

    return df, new_features


# ──────────────────────────────────────────────────────────────────────────────
# Full pipeline convenience wrapper
# ──────────────────────────────────────────────────────────────────────────────

def full_preprocess(
    df: pd.DataFrame,
    drop_cols: list[str],
    target_col: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run the complete preprocessing sequence.
    Returns the cleaned DataFrame and a summary report dict.
    """
    summary: Dict[str, Any] = {}

    df, dropped = drop_identifier_columns(df, drop_cols)
    summary["dropped_id_cols"] = dropped

    df, dup_removed = handle_duplicates(df)
    summary["duplicates_removed"] = dup_removed

    df, missing_report = handle_missing_values(df)
    summary["missing_imputation"] = missing_report

    df, bad_bp = remove_invalid_bp_values(df)
    summary["invalid_bp_removed"] = bad_bp

    df, new_feats = engineer_features(df)
    summary["engineered_features"] = new_feats

    summary["final_shape"] = df.shape
    summary["target_distribution"] = df[target_col].value_counts().to_dict()

    return df, summary
