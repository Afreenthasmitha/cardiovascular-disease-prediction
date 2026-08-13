"""
run_training.py — Standalone script that loads the dataset, runs the full
ML pipeline, evaluates all models, selects the best one, and saves artefacts.

Run from the project root:
    python run_training.py
or optionally with hyperparameter tuning:
    python run_training.py --tune
"""

from __future__ import annotations

import sys
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

# Make src importable when running from project root
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from src.config import (
    DATASET_PATH, FIGURES_DIR, MODELS_DIR, REPORTS_DIR,
    CANDIDATE_TARGET_COLUMNS, TARGET_COLUMN, DROP_COLUMNS,
    RANDOM_STATE, TEST_SIZE, CV_FOLDS, PRIMARY_METRIC, RESULTS_CSV_PATH,
)
from src.data_preprocessing import (
    load_dataset, detect_target_column, inspect_dataset, full_preprocess,
)
from src.eda import generate_all_eda_plots, get_top_correlations
from src.train_models import (
    split_features, build_preprocessor, train_all_models, save_best_pipeline,
)
from src.evaluate_models import (
    evaluate_all_models, select_best_model,
    plot_confusion_matrices, plot_roc_curves, plot_model_comparison,
    save_results_csv,
)
from sklearn.model_selection import train_test_split


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tune", action="store_true",
                   help="Run GridSearchCV hyperparameter tuning (slower)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n📂  Loading dataset …")
    df = load_dataset(DATASET_PATH)
    print(f"    Raw shape: {df.shape}")

    # ── Target detection ──────────────────────────────────────────────────────
    target_col = detect_target_column(df, CANDIDATE_TARGET_COLUMNS, TARGET_COLUMN)
    print(f"    Target column: '{target_col}'")
    print(f"    Class distribution:\n{df[target_col].value_counts().to_string()}")

    # ── Inspection ────────────────────────────────────────────────────────────
    print("\n🔍  Inspecting dataset …")
    info = inspect_dataset(df)
    print(f"    Numeric columns  : {info['numeric_cols']}")
    print(f"    Categorical cols : {info['categorical_cols']}")
    print(f"    Missing values   : {info['total_missing']}")
    print(f"    Duplicates       : {info['dup_count']} ({info['dup_pct']}%)")

    # ── Preprocessing ─────────────────────────────────────────────────────────
    print("\n🔧  Preprocessing …")
    df_clean, prep_summary = full_preprocess(df.copy(), DROP_COLUMNS, target_col)
    print(f"    Duplicates removed    : {prep_summary['duplicates_removed']}")
    print(f"    Invalid BPs removed   : {prep_summary['invalid_bp_removed']}")
    print(f"    Engineered features   : {prep_summary['engineered_features']}")
    print(f"    Final shape           : {prep_summary['final_shape']}")

    # ── EDA & Visualisations ──────────────────────────────────────────────────
    print("\n📊  Generating EDA plots …")
    info2 = inspect_dataset(df_clean)
    saved = generate_all_eda_plots(
        df_clean, target_col, FIGURES_DIR,
        info2["numeric_cols"], info2["categorical_cols"],
    )
    print(f"    Saved {len(saved)} plot(s) to {FIGURES_DIR}")

    # ── Top correlations ──────────────────────────────────────────────────────
    top_corr = get_top_correlations(df_clean, target_col, top_n=10)
    print(f"\n📈  Top correlations with '{target_col}':\n{top_corr.to_string()}")

    # ── Feature/target split ──────────────────────────────────────────────────
    print("\n✂️   Splitting features …")
    X, y, num_cols, cat_cols = split_features(df_clean, target_col)
    print(f"    Features : {list(X.columns)}")
    print(f"    Numeric  : {num_cols}")
    print(f"    Categ.   : {cat_cols}")

    # ── Train / test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"    Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    # ── Train all models ──────────────────────────────────────────────────────
    print(f"\n🤖  Training models (tune={args.tune}) …")
    preprocessor = build_preprocessor(num_cols, cat_cols)
    pipelines, cv_scores = train_all_models(
        X_train, y_train, preprocessor,
        tune=args.tune, cv_folds=CV_FOLDS, random_state=RANDOM_STATE,
    )
    print("\n    Cross-validation ROC-AUC:")
    for n, s in cv_scores.items():
        print(f"      {n:35s}  {s['mean']:.4f} ± {s['std']:.4f}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n📏  Evaluating on test set …")
    results_df, raw_results = evaluate_all_models(pipelines, X_test, y_test)
    print("\n" + results_df[["Accuracy","Precision","Recall","F1 Score","ROC-AUC"]].to_string())

    # ── Best model ───────────────────────────────────────────────────────────
    metric_col_map = {
        "roc_auc": "ROC-AUC", "f1": "F1 Score",
        "recall": "Recall", "accuracy": "Accuracy",
    }
    pk = metric_col_map.get(PRIMARY_METRIC, "ROC-AUC")
    best_name = select_best_model(results_df, pk)
    print(f"\n🏆  Best model ({pk}): {best_name}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n🎨  Generating evaluation plots …")
    plot_confusion_matrices(raw_results, y_test, FIGURES_DIR)
    plot_roc_curves(raw_results, y_test, FIGURES_DIR)
    plot_model_comparison(results_df, FIGURES_DIR)

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\n💾  Saving artefacts …")
    best_pipe = pipelines[best_name]
    metrics_dict = results_df.loc[best_name].to_dict()
    metrics_dict.update(cv_scores.get(best_name, {}))

    save_best_pipeline(
        pipeline=best_pipe,
        model_name=best_name,
        feature_cols=list(X.columns),
        target_col=target_col,
        metrics=metrics_dict,
        dataset_info={"shape": list(df_clean.shape), "target": target_col},
        models_dir=MODELS_DIR,
    )
    save_results_csv(results_df, RESULTS_CSV_PATH)
    print(f"    Model saved  : {MODELS_DIR}/best_model_pipeline.pkl")
    print(f"    Results CSV  : {RESULTS_CSV_PATH}")
    print(f"    Figures dir  : {FIGURES_DIR}")
    print("\n✅  Training complete!\n")


if __name__ == "__main__":
    main()
