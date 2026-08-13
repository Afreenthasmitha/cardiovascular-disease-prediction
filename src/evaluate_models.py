"""
evaluate_models.py — Compute all evaluation metrics and generate comparison visualisations.
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from typing import Dict, List, Tuple, Any

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Per-model metrics
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_pipeline(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> Dict[str, Any]:
    """Return a dict of evaluation metrics for one pipeline."""
    y_pred = pipeline.predict(X_test)

    has_proba = hasattr(pipeline.named_steps["classifier"], "predict_proba")
    if has_proba:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = round(roc_auc_score(y_test, y_proba), 4)
    else:
        y_proba = None
        roc_auc = float("nan")

    return {
        "Model":     model_name,
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1 Score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC":   roc_auc,
        "_y_pred":   y_pred,
        "_y_proba":  y_proba,
    }


def evaluate_all_models(
    pipelines: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Evaluate all pipelines.

    Returns:
      results_df: summary DataFrame (Model, Accuracy, Precision, Recall, F1, ROC-AUC)
      raw_results: full dict including predictions/probabilities
    """
    rows = []
    raw: Dict[str, Dict] = {}
    for name, pipe in pipelines.items():
        r = evaluate_pipeline(pipe, X_test, y_test, name)
        raw[name] = r
        rows.append({k: v for k, v in r.items() if not k.startswith("_")})

    results_df = pd.DataFrame(rows).set_index("Model")
    return results_df, raw


# ──────────────────────────────────────────────────────────────────────────────
# Best model selection
# ──────────────────────────────────────────────────────────────────────────────

def select_best_model(
    results_df: pd.DataFrame,
    primary_metric: str = "ROC-AUC",
) -> str:
    """Return the model name with the highest primary_metric score."""
    col = primary_metric if primary_metric in results_df.columns else "F1 Score"
    return str(results_df[col].idxmax())


# ──────────────────────────────────────────────────────────────────────────────
# Confusion matrix plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(
    raw_results: Dict[str, Dict],
    y_test: pd.Series,
    figures_dir: str,
) -> str:
    """Save a grid of confusion matrices for all models."""
    sns.set_theme(style="whitegrid")
    n = len(raw_results)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
    axes = np.array(axes).flatten()

    class_labels = sorted(y_test.unique())
    display_labels = [f"No Disease ({c})" if c == 0 else f"Disease ({c})" for c in class_labels]

    for i, (name, r) in enumerate(raw_results.items()):
        cm = confusion_matrix(y_test, r["_y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
        disp.plot(ax=axes[i], colorbar=False, cmap="Blues")
        axes[i].set_title(name, fontsize=11, fontweight="bold")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(figures_dir, "07_confusion_matrices.png")
    os.makedirs(figures_dir, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# ROC curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    raw_results: Dict[str, Dict],
    y_test: pd.Series,
    figures_dir: str,
) -> str:
    """Save an overlay ROC curve plot for all models that support predict_proba."""
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("husl", n_colors=len(raw_results))
    fig, ax = plt.subplots(figsize=(9, 7))

    for (name, r), color in zip(raw_results.items(), palette):
        if r["_y_proba"] is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, r["_y_proba"])
        auc_val = r["ROC-AUC"]
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", linewidth=2, color=color)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    path = os.path.join(figures_dir, "08_roc_curve_comparison.png")
    os.makedirs(figures_dir, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Metric comparison bar chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(
    results_df: pd.DataFrame,
    figures_dir: str,
) -> str:
    """Grouped bar chart comparing all models across all metrics."""
    sns.set_theme(style="whitegrid")
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    metrics = [m for m in metrics if m in results_df.columns]

    plot_df = results_df[metrics].reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", ax=ax,
                palette="husl", edgecolor="white")
    ax.set_ylim(0, 1.09)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Metric", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    path = os.path.join(figures_dir, "09_model_comparison.png")
    os.makedirs(figures_dir, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Save results CSV
# ──────────────────────────────────────────────────────────────────────────────

def save_results_csv(results_df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results_df.to_csv(path)
