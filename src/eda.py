"""
eda.py — Exploratory Data Analysis and visualization utilities.
All plots are saved to reports/figures/ automatically.
"""

from __future__ import annotations

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import List, Optional

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: str, dpi: int = 120) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _style() -> None:
    sns.set_theme(style="whitegrid", palette="husl")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


# ──────────────────────────────────────────────────────────────────────────────
# Target distribution
# ──────────────────────────────────────────────────────────────────────────────

def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str,
    figures_dir: str,
    label_map: Optional[dict] = None,
) -> str:
    """Bar + pie chart of target class distribution."""
    _style()
    counts = df[target_col].value_counts().sort_index()
    labels = [label_map.get(k, str(k)) for k in counts.index] if label_map else [str(k) for k in counts.index]
    colors = ["#2ECC71", "#E74C3C"][:len(counts)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Target Variable: Cardiovascular Disease Distribution", fontsize=15, fontweight="bold")

    axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=0.8)
    axes[0].set_xlabel("Class", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Class Count")
    for i, v in enumerate(counts.values):
        pct = v / counts.sum() * 100
        axes[0].text(i, v + counts.max() * 0.01, f"{v:,}\n({pct:.1f}%)", ha="center", fontsize=10)

    axes[1].pie(counts.values, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    axes[1].set_title("Class Proportion")

    path = os.path.join(figures_dir, "01_target_distribution.png")
    _save(fig, path)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Univariate — numerical
# ──────────────────────────────────────────────────────────────────────────────

def plot_numerical_distributions(
    df: pd.DataFrame,
    numeric_cols: List[str],
    target_col: str,
    figures_dir: str,
    max_cols: int = 16,
) -> str:
    """Histogram + KDE for numerical features, coloured by target."""
    _style()
    cols = [c for c in numeric_cols if c != target_col][:max_cols]
    n = len(cols)
    if n == 0:
        return ""
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        for cls, color in zip(sorted(df[target_col].unique()), ["#2ECC71", "#E74C3C"]):
            subset = df[df[target_col] == cls][col].dropna()
            ax.hist(subset, bins=30, alpha=0.5, color=color, label=str(cls), density=True)
            subset.plot.kde(ax=ax, color=color, linewidth=1.5)
        ax.set_title(col, fontsize=11)
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend(title=target_col, fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numerical Feature Distributions by Target", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(figures_dir, "02_numerical_distributions.png")
    _save(fig, path)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Box plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_boxplots(
    df: pd.DataFrame,
    numeric_cols: List[str],
    target_col: str,
    figures_dir: str,
    max_cols: int = 12,
) -> str:
    _style()
    cols = [c for c in numeric_cols if c != target_col][:max_cols]
    n = len(cols)
    if n == 0:
        return ""
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        sns.boxplot(data=df, x=target_col, y=col, ax=ax,
                    palette=["#2ECC71", "#E74C3C"])
        ax.set_title(col, fontsize=11)
        ax.set_xlabel(target_col)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Box Plots: Numerical Features vs Target", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(figures_dir, "03_boxplots.png")
    _save(fig, path)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Categorical features
# ──────────────────────────────────────────────────────────────────────────────

def plot_categorical_distributions(
    df: pd.DataFrame,
    categorical_cols: List[str],
    target_col: str,
    figures_dir: str,
    max_cols: int = 8,
) -> str:
    _style()
    cols = [c for c in categorical_cols if c != target_col][:max_cols]
    n = len(cols)
    if n == 0:
        return ""

    # Also include low-cardinality numeric columns as categorical
    extra = [
        c for c in df.columns
        if c != target_col and c not in categorical_cols
        and df[c].nunique() <= 6 and pd.api.types.is_numeric_dtype(df[c])
    ]
    cols = list(dict.fromkeys(cols + extra))[:max_cols]
    n = len(cols)
    if n == 0:
        return ""

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        ct = pd.crosstab(df[col], df[target_col])
        ct.plot(kind="bar", ax=ax, color=["#2ECC71", "#E74C3C"], edgecolor="white")
        ax.set_title(col, fontsize=11)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title=target_col, fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Categorical Features vs Target", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(figures_dir, "04_categorical_distributions.png")
    _save(fig, path)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Correlation heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(
    df: pd.DataFrame,
    figures_dir: str,
) -> str:
    _style()
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(max(10, len(corr) * 0.9), max(8, len(corr) * 0.8)))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, linewidths=0.5, linecolor="white",
        ax=ax, annot_kws={"size": 8},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=15, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    path = os.path.join(figures_dir, "05_correlation_heatmap.png")
    _save(fig, path)
    return path


def get_top_correlations(df: pd.DataFrame, target_col: str, top_n: int = 10) -> pd.Series:
    """Return top_n features most correlated (abs) with target_col."""
    corr = df.corr(numeric_only=True)
    if target_col not in corr.columns:
        return pd.Series(dtype=float)
    target_corr = corr[target_col].drop(target_col).abs().sort_values(ascending=False)
    return target_corr.head(top_n)


# ──────────────────────────────────────────────────────────────────────────────
# Outlier visualisation (IQR)
# ──────────────────────────────────────────────────────────────────────────────

def plot_outlier_summary(
    df: pd.DataFrame,
    numeric_cols: List[str],
    target_col: str,
    figures_dir: str,
) -> str:
    _style()
    cols = [c for c in numeric_cols if c != target_col]
    outlier_counts: dict[str, int] = {}
    for col in cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        n_out = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        outlier_counts[col] = n_out

    oc = pd.Series(outlier_counts).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(4, len(oc) * 0.5)))
    oc.plot(kind="barh", ax=ax, color="#3498DB", edgecolor="white")
    ax.set_xlabel("Number of Outliers (IQR method)", fontsize=11)
    ax.set_title("Outlier Count per Feature", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(figures_dir, "06_outlier_summary.png")
    _save(fig, path)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Generate all EDA plots in one call
# ──────────────────────────────────────────────────────────────────────────────

def generate_all_eda_plots(
    df: pd.DataFrame,
    target_col: str,
    figures_dir: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> List[str]:
    """Generate and save all EDA plots. Returns list of saved file paths."""
    paths: List[str] = []

    p = plot_target_distribution(df, target_col, figures_dir)
    if p: paths.append(p)

    p = plot_numerical_distributions(df, numeric_cols, target_col, figures_dir)
    if p: paths.append(p)

    p = plot_boxplots(df, numeric_cols, target_col, figures_dir)
    if p: paths.append(p)

    p = plot_categorical_distributions(df, categorical_cols, target_col, figures_dir)
    if p: paths.append(p)

    p = plot_correlation_heatmap(df, figures_dir)
    if p: paths.append(p)

    p = plot_outlier_summary(df, numeric_cols, target_col, figures_dir)
    if p: paths.append(p)

    return paths
