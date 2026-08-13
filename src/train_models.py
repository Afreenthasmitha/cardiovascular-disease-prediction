"""
train_models.py — Build scikit-learn Pipelines, train all models,
perform cross-validation, hyperparameter tuning, and save the best pipeline.
"""

from __future__ import annotations

import os
import json
import warnings
import datetime
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from typing import Dict, Any, Tuple, List

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Feature splitting helpers
# ──────────────────────────────────────────────────────────────────────────────

def split_features(
    df: pd.DataFrame,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Split df into X, y and return numeric/categorical column lists.
    Identifies and encodes any remaining object columns in y.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()

    # Ensure binary integer target
    if y.dtype == object:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index, name=target_col)
    else:
        y = y.astype(int)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols     = X.select_dtypes(exclude=[np.number]).columns.tolist()

    return X, y, numeric_cols, cat_cols


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing pipeline builder
# ──────────────────────────────────────────────────────────────────────────────

def build_preprocessor(
    numeric_cols: List[str],
    cat_cols: List[str],
) -> ColumnTransformer:
    """
    ColumnTransformer that:
      - Imputes numerics with median → StandardScaler
      - Imputes categoricals with mode → OrdinalEncoder
    """
    from sklearn.preprocessing import OrdinalEncoder

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    transformers = [("num", numeric_transformer, numeric_cols)]
    if cat_cols:
        transformers.append(("cat", categorical_transformer, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


# ──────────────────────────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────────────────────────

def get_model_definitions() -> Dict[str, Any]:
    """Return a dict of {model_name: estimator} with sensible defaults."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, C=1.0
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=10, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        ),
        "Support Vector Machine": SVC(
            kernel="rbf", probability=True, random_state=42, C=1.0,
            cache_size=1000, max_iter=10000
        ),
    }


def get_param_grids() -> Dict[str, Dict]:
    """Lightweight hyperparameter grids for GridSearchCV."""
    return {
        "Logistic Regression": {"classifier__C": [0.01, 0.1, 1, 10]},
        "K-Nearest Neighbors": {"classifier__n_neighbors": [3, 5, 7, 11, 15]},
        "Decision Tree":       {"classifier__max_depth": [4, 6, 8, None],
                                "classifier__min_samples_leaf": [5, 10, 20]},
        "Random Forest":       {"classifier__n_estimators": [100, 200],
                                "classifier__max_depth": [6, 8, None]},
        "Support Vector Machine": {"classifier__C": [0.1, 1, 10],
                                   "classifier__kernel": ["rbf", "linear"]},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    tune: bool = False,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Tuple[Dict[str, Pipeline], Dict[str, Dict]]:
    """
    Train all models (optionally with GridSearchCV).

    Returns:
      trained_pipelines: {name: fitted Pipeline}
      cv_scores:         {name: {"mean": float, "std": float}}
    """
    model_defs  = get_model_definitions()
    param_grids = get_param_grids()
    cv          = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    trained_pipelines: Dict[str, Pipeline] = {}
    cv_scores: Dict[str, Dict] = {}

    for name, estimator in model_defs.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier",   estimator),
        ])

        if tune and name in param_grids:
            gs = GridSearchCV(
                pipeline,
                param_grids[name],
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
                refit=True,
            )
            gs.fit(X_train, y_train)
            best_pipe = gs.best_estimator_
            cv_mean   = gs.best_score_
            cv_std    = 0.0   # GridSearchCV doesn't expose fold stds directly
        else:
            best_pipe = pipeline
            best_pipe.fit(X_train, y_train)
            scores    = cross_val_score(best_pipe, X_train, y_train,
                                        cv=cv, scoring="roc_auc", n_jobs=-1)
            cv_mean   = scores.mean()
            cv_std    = scores.std()

        trained_pipelines[name] = best_pipe
        cv_scores[name] = {"mean": round(cv_mean, 4), "std": round(cv_std, 4)}

    return trained_pipelines, cv_scores


# ──────────────────────────────────────────────────────────────────────────────
# Saving
# ──────────────────────────────────────────────────────────────────────────────

def save_best_pipeline(
    pipeline: Pipeline,
    model_name: str,
    feature_cols: List[str],
    target_col: str,
    metrics: Dict[str, float],
    dataset_info: Dict[str, Any],
    models_dir: str,
) -> None:
    """Persist the best pipeline and a JSON metadata file."""
    os.makedirs(models_dir, exist_ok=True)
    pkl_path  = os.path.join(models_dir, "best_model_pipeline.pkl")
    meta_path = os.path.join(models_dir, "model_metadata.json")

    joblib.dump(pipeline, pkl_path)

    metadata = {
        "model_name":     model_name,
        "feature_columns": feature_cols,
        "target_column":  target_col,
        "metrics":        metrics,
        "training_date":  datetime.datetime.now().isoformat(),
        "dataset_info":   dataset_info,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4, default=str)


def load_best_pipeline(models_dir: str) -> Tuple[Pipeline, Dict[str, Any]]:
    """Load the saved pipeline and metadata."""
    pkl_path  = os.path.join(models_dir, "best_model_pipeline.pkl")
    meta_path = os.path.join(models_dir, "model_metadata.json")

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"No saved model found at {pkl_path}")

    pipeline = joblib.load(pkl_path)
    metadata: Dict[str, Any] = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)

    return pipeline, metadata
