"""
app.py — Cardiovascular Disease Prediction System
Professional multi-page Streamlit dashboard.
"""

from __future__ import annotations

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    DATASET_PATH, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, RESULTS_CSV_PATH,
    CANDIDATE_TARGET_COLUMNS, TARGET_COLUMN, DROP_COLUMNS,
    RANDOM_STATE, TEST_SIZE, CV_FOLDS, PRIMARY_METRIC,
)
from src.data_preprocessing import (
    load_dataset, detect_target_column, inspect_dataset, full_preprocess,
)
from src.eda import (
    generate_all_eda_plots, get_top_correlations,
    plot_target_distribution, plot_correlation_heatmap,
)
from src.train_models import (
    split_features, build_preprocessor, train_all_models,
    save_best_pipeline, load_best_pipeline,
)
from src.evaluate_models import (
    evaluate_all_models, select_best_model,
    plot_confusion_matrices, plot_roc_curves, plot_model_comparison,
    save_results_csv,
)
from src.prediction import predict_cardiovascular_disease
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cardiovascular Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    /* Main background */
    .main .block-container {
        padding-top: 1.5rem;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1rem;
        color: white !important;
    }
    [data-testid="stMetricLabel"] { color: #a0aec0 !important; font-size: 0.85rem; }
    [data-testid="stMetricValue"] { color: #63b3ed !important; font-weight: 700; }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #e74c3c, #c0392b);
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1.2rem 0 0.8rem 0;
    }

    /* Result box */
    .result-positive {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        border-radius: 14px;
        padding: 1.5rem;
        color: white;
        text-align: center;
    }
    .result-negative {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        border-radius: 14px;
        padding: 1.5rem;
        color: white;
        text-align: center;
    }

    /* Disclaimer */
    .disclaimer-box {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        color: #856404;
        font-size: 0.85rem;
    }

    /* Best model badge */
    .best-badge {
        background: linear-gradient(90deg, #f39c12, #e67e22);
        color: white;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Session state helper — cache heavy computations
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_and_preprocess(path: str):
    df_raw  = load_dataset(path)
    target  = detect_target_column(df_raw, CANDIDATE_TARGET_COLUMNS, TARGET_COLUMN)
    info    = inspect_dataset(df_raw)
    df_clean, prep_summary = full_preprocess(df_raw.copy(), DROP_COLUMNS, target)
    return df_raw, df_clean, target, info, prep_summary


@st.cache_resource(show_spinner=False)
def run_training_pipeline(dataset_path: str):
    df_raw, df_clean, target, info, prep_summary = load_and_preprocess(dataset_path)
    X, y, num_cols, cat_cols = split_features(df_clean, target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    preprocessor = build_preprocessor(num_cols, cat_cols)
    pipelines, cv_scores = train_all_models(
        X_train, y_train, preprocessor,
        tune=False, cv_folds=CV_FOLDS, random_state=RANDOM_STATE,
    )
    results_df, raw_results = evaluate_all_models(pipelines, X_test, y_test)

    metric_col_map = {"roc_auc": "ROC-AUC", "f1": "F1 Score",
                      "recall": "Recall", "accuracy": "Accuracy"}
    pk = metric_col_map.get(PRIMARY_METRIC, "ROC-AUC")
    best_name = select_best_model(results_df, pk)

    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    save_best_pipeline(
        pipeline=pipelines[best_name],
        model_name=best_name,
        feature_cols=list(X.columns),
        target_col=target,
        metrics=results_df.loc[best_name].to_dict(),
        dataset_info={"shape": list(df_clean.shape), "target": target},
        models_dir=MODELS_DIR,
    )
    save_results_csv(results_df, RESULTS_CSV_PATH)

    # Save evaluation plots
    plot_confusion_matrices(raw_results, y_test, FIGURES_DIR)
    plot_roc_curves(raw_results, y_test, FIGURES_DIR)
    plot_model_comparison(results_df, FIGURES_DIR)
    generate_all_eda_plots(df_clean, target, FIGURES_DIR, num_cols, cat_cols)

    return (pipelines, results_df, raw_results, cv_scores, best_name,
            X, y, X_train, X_test, y_train, y_test,
            num_cols, cat_cols, target)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ❤️ CardioPredict")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📊 Dataset Analysis", "🔬 Exploratory Data Analysis",
         "🤖 Model Comparison", "🔮 Predict Disease"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    dataset_ok = os.path.exists(DATASET_PATH)
    if dataset_ok:
        st.success("✅ Dataset found")
    else:
        st.error("❌ Dataset not found")
        st.caption(f"Place your dataset at:\n`{DATASET_PATH}`")

    model_ok = os.path.exists(os.path.join(MODELS_DIR, "best_model_pipeline.pkl"))
    if model_ok:
        st.success("✅ Model trained")
    else:
        st.warning("⚠️ Model not trained yet")

    st.markdown("---")
    st.caption("College Project · Educational Use Only")


# ──────────────────────────────────────────────────────────────────────────────
# Page: Home
# ──────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("# ❤️ Cardiovascular Disease Prediction System")
    st.markdown("### *Machine Learning–Based Risk Assessment for Education*")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("🧠 ML Models", "5")
    col2.metric("📈 Metrics Tracked", "5")
    col3.metric("📚 Purpose", "Educational")

    st.markdown("---")
    st.markdown('<div class="section-header">🎯 Project Overview</div>', unsafe_allow_html=True)
    st.markdown("""
Cardiovascular disease (CVD) is the **leading cause of death** worldwide. Early detection through
pattern recognition in clinical data can assist healthcare professionals in identifying at-risk patients.

This system demonstrates how **supervised machine learning** can be applied to structured medical datasets
to build predictive models for cardiovascular disease risk assessment.
    """)

    st.markdown('<div class="section-header">🔄 Machine Learning Workflow</div>', unsafe_allow_html=True)
    steps = [
        ("1️⃣", "Data Loading & Inspection", "Understand dataset shape, types, missing values"),
        ("2️⃣", "Data Preprocessing",         "Handle duplicates, impute missing values, engineer features"),
        ("3️⃣", "Exploratory Data Analysis",  "Visualise distributions, correlations, and class balance"),
        ("4️⃣", "Model Training",              "Train LR, KNN, DT, RF, SVM inside sklearn Pipelines"),
        ("5️⃣", "Model Evaluation",            "Compare Accuracy, Precision, Recall, F1, ROC-AUC"),
        ("6️⃣", "Best Model Selection",        "Choose best model by ROC-AUC and save pipeline"),
        ("7️⃣", "Prediction Interface",        "Enter patient data and get a risk prediction"),
    ]
    for icon, title, desc in steps:
        with st.container():
            c1, c2 = st.columns([1, 8])
            c1.markdown(f"## {icon}")
            c2.markdown(f"**{title}**  \n{desc}")

    st.markdown("---")
    st.markdown("""
<div class="disclaimer-box">
<strong>⚠️ Educational Disclaimer</strong><br>
This application is a machine-learning demonstration created for educational and research purposes.
Its predictions are based on patterns in the supplied dataset and should <strong>not</strong> be
considered a medical diagnosis. The system has not been validated for clinical use.
<strong>Consult a qualified healthcare professional for medical advice or diagnosis.</strong>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Page: Dataset Analysis
# ──────────────────────────────────────────────────────────────────────────────
elif page == "📊 Dataset Analysis":
    st.markdown("# 📊 Dataset Analysis")
    if not dataset_ok:
        st.error(f"Dataset not found at `{DATASET_PATH}`. Please place your CSV there.")
        st.stop()

    with st.spinner("Loading & preprocessing dataset …"):
        df_raw, df_clean, target, info, prep = load_and_preprocess(DATASET_PATH)

    # Overview metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (raw)",       f"{info['n_rows']:,}")
    c2.metric("Columns",          f"{info['n_cols']}")
    c3.metric("Duplicates",       f"{info['dup_count']:,} ({info['dup_pct']}%)")
    c4.metric("Missing values",   f"{info['total_missing']:,}")

    st.markdown('<div class="section-header">📋 Raw Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df_raw.head(20), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">🔧 Data Types</div>', unsafe_allow_html=True)
        dtype_df = pd.DataFrame({
            "Column": df_raw.dtypes.index,
            "Type":   df_raw.dtypes.astype(str).values,
            "Non-Null": df_raw.notnull().sum().values,
        })
        st.dataframe(dtype_df, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">🕳️ Missing Value Summary</div>', unsafe_allow_html=True)
        if info["missing_df"].empty:
            st.success("✅ No missing values detected!")
        else:
            st.dataframe(info["missing_df"], use_container_width=True)

    st.markdown('<div class="section-header">📈 Statistical Summary</div>', unsafe_allow_html=True)
    st.dataframe(df_raw.describe(), use_container_width=True)

    st.markdown('<div class="section-header">🔧 Preprocessing Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duplicates removed",    prep["duplicates_removed"])
    c2.metric("Invalid BPs removed",   prep["invalid_bp_removed"])
    c3.metric("Engineered features",   len(prep["engineered_features"]))
    c4.metric("Final rows",            prep["final_shape"][0])

    if prep["engineered_features"]:
        st.markdown(f"**New features created:** `{'`, `'.join(prep['engineered_features'])}`")

    st.markdown('<div class="section-header">🎯 Target Distribution</div>', unsafe_allow_html=True)
    target_counts = df_clean[target].value_counts().sort_index()
    tc1, tc2 = st.columns([1, 2])
    with tc1:
        td = pd.DataFrame({
            "Class":   target_counts.index,
            "Count":   target_counts.values,
            "Percent": (target_counts.values / target_counts.sum() * 100).round(2),
        })
        st.dataframe(td, use_container_width=True)
    with tc2:
        fig, ax = plt.subplots(figsize=(5, 3))
        colors = ["#2ECC71", "#E74C3C"][:len(target_counts)]
        ax.bar([str(k) for k in target_counts.index], target_counts.values,
               color=colors, edgecolor="white")
        ax.set_xlabel(target); ax.set_ylabel("Count")
        ax.set_title(f"Target: {target}", fontweight="bold")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Page: EDA
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🔬 Exploratory Data Analysis":
    st.markdown("# 🔬 Exploratory Data Analysis")
    if not dataset_ok:
        st.error(f"Dataset not found at `{DATASET_PATH}`.")
        st.stop()

    with st.spinner("Running EDA …"):
        df_raw, df_clean, target, info, prep = load_and_preprocess(DATASET_PATH)
        info2 = inspect_dataset(df_clean)
        generate_all_eda_plots(df_clean, target, FIGURES_DIR,
                               info2["numeric_cols"], info2["categorical_cols"])
        top_corr = get_top_correlations(df_clean, target, top_n=10)

    fig_files = {
        "01_target_distribution.png":      "🎯 Target Distribution",
        "02_numerical_distributions.png":  "📊 Numerical Feature Distributions",
        "03_boxplots.png":                 "📦 Box Plots",
        "04_categorical_distributions.png":"📋 Categorical Features vs Target",
        "05_correlation_heatmap.png":       "🌡️ Correlation Heatmap",
        "06_outlier_summary.png":          "⚡ Outlier Summary",
    }

    for fname, title in fig_files.items():
        fpath = os.path.join(FIGURES_DIR, fname)
        if os.path.exists(fpath):
            st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
            st.image(fpath, use_container_width=True)

    st.markdown('<div class="section-header">📈 Top Feature Correlations with Target</div>',
                unsafe_allow_html=True)
    if not top_corr.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = sns.color_palette("RdYlGn", len(top_corr))[::-1]
        ax.barh(top_corr.index, top_corr.values, color=colors)
        ax.set_xlabel("Absolute Correlation")
        ax.set_title(f"Top Correlations with '{target}'", fontweight="bold")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption(f"**Note:** Correlation does not imply causation.")


# ──────────────────────────────────────────────────────────────────────────────
# Page: Model Comparison
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🤖 Model Comparison":
    st.markdown("# 🤖 Model Comparison")
    if not dataset_ok:
        st.error(f"Dataset not found at `{DATASET_PATH}`.")
        st.stop()

    with st.spinner("Training and evaluating all models … (this may take a minute)"):
        (pipelines, results_df, raw_results, cv_scores, best_name,
         X, y, X_train, X_test, y_train, y_test,
         num_cols, cat_cols, target) = run_training_pipeline(DATASET_PATH)

    st.success(f"🏆 Best model: **{best_name}** (primary metric: {PRIMARY_METRIC.upper()})")

    # Summary metrics
    st.markdown('<div class="section-header">📊 Model Performance Summary</div>', unsafe_allow_html=True)
    display_df = results_df.copy().reset_index()
    display_df.insert(0, "★", display_df["Model"].apply(
        lambda m: "🏆 Best" if m == best_name else ""
    ))
    st.dataframe(display_df.style.format({
        "Accuracy":  "{:.4f}",
        "Precision": "{:.4f}",
        "Recall":    "{:.4f}",
        "F1 Score":  "{:.4f}",
        "ROC-AUC":   "{:.4f}",
    }), use_container_width=True)

    # Cross-validation
    st.markdown('<div class="section-header">🔄 Cross-Validation Results (5-Fold, ROC-AUC)</div>',
                unsafe_allow_html=True)
    cv_df = pd.DataFrame(cv_scores).T.reset_index()
    cv_df.columns = ["Model", "Mean ROC-AUC", "Std"]
    st.dataframe(cv_df.style.format({"Mean ROC-AUC": "{:.4f}", "Std": "{:.4f}"}),
                 use_container_width=True)

    # Evaluation plots
    eval_figs = {
        "09_model_comparison.png":      "📊 Metric Comparison",
        "07_confusion_matrices.png":    "🔢 Confusion Matrices",
        "08_roc_curve_comparison.png":  "📈 ROC Curves",
    }
    for fname, title in eval_figs.items():
        fpath = os.path.join(FIGURES_DIR, fname)
        if os.path.exists(fpath):
            st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
            st.image(fpath, use_container_width=True)

    st.markdown('<div class="section-header">💡 Auto-Generated Insights</div>', unsafe_allow_html=True)
    insights = []
    insights.append(f"• The dataset contains **{len(X):,}** patient records with **{len(X.columns)}** features.")
    class_dist = y.value_counts(normalize=True)
    majority_pct = class_dist.iloc[0] * 100
    insights.append(f"• The target shows a **{majority_pct:.1f}% / {100-majority_pct:.1f}%** class split.")
    best_auc = results_df.loc[best_name, "ROC-AUC"] if "ROC-AUC" in results_df.columns else "N/A"
    insights.append(f"• **{best_name}** achieved the highest ROC-AUC of **{best_auc:.4f}**.")
    best_recall_model = results_df["Recall"].idxmax() if "Recall" in results_df.columns else "N/A"
    insights.append(f"• **{best_recall_model}** achieved the highest Recall — important for minimising missed disease cases.")
    for insight in insights:
        st.markdown(insight)


# ──────────────────────────────────────────────────────────────────────────────
# Page: Predict Disease
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🔮 Predict Disease":
    st.markdown("# 🔮 Cardiovascular Disease Risk Prediction")

    if not dataset_ok:
        st.error(f"Dataset not found at `{DATASET_PATH}`.")
        st.stop()

    if not os.path.exists(os.path.join(MODELS_DIR, "best_model_pipeline.pkl")):
        st.warning("Model not trained yet. Go to **🤖 Model Comparison** first to train the model.")
        st.stop()

    # Load pipeline and metadata
    try:
        pipeline, metadata = load_best_pipeline(MODELS_DIR)
        feature_cols = metadata.get("feature_columns", [])
        model_name   = metadata.get("model_name", "Best Model")
        target_col   = metadata.get("target_column", "cardio")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # Load sample data for ranges
    df_raw, df_clean, target, info, prep = load_and_preprocess(DATASET_PATH)
    info2 = inspect_dataset(df_clean)

    st.markdown(f"**Using model:** `{model_name}`")
    st.markdown("---")
    st.markdown('<div class="section-header">🧑‍⚕️ Enter Patient Information</div>', unsafe_allow_html=True)

    user_input: dict = {}
    cols_per_row = 3
    feat_chunks = [feature_cols[i:i+cols_per_row] for i in range(0, len(feature_cols), cols_per_row)]

    for chunk in feat_chunks:
        row_cols = st.columns(len(chunk))
        for col_widget, feat in zip(row_cols, chunk):
            with col_widget:
                feat_data = df_clean[feat].dropna() if feat in df_clean.columns else pd.Series([0])
                n_unique  = feat_data.nunique()
                is_binary = n_unique <= 2 and feat_data.isin([0, 1]).all()
                is_low_card = n_unique <= 6 and n_unique > 2

                if is_binary:
                    options = sorted(feat_data.unique().tolist())
                    val = st.selectbox(
                        feat,
                        options=options,
                        format_func=lambda x: f"Yes ({x})" if x == 1 else f"No ({x})",
                        key=f"inp_{feat}",
                    )
                elif is_low_card and not pd.api.types.is_float_dtype(feat_data):
                    options = sorted(feat_data.unique().tolist())
                    val = st.selectbox(feat, options=options, key=f"inp_{feat}")
                else:
                    f_min  = float(feat_data.min())
                    f_max  = float(feat_data.max())
                    f_mean = float(feat_data.mean())
                    is_int = pd.api.types.is_integer_dtype(feat_data)
                    val = st.number_input(
                        feat,
                        min_value=f_min,
                        max_value=f_max,
                        value=round(f_mean, 1 if not is_int else 0),
                        step=1.0 if is_int else 0.1,
                        format="%g" if is_int else "%.2f",
                        key=f"inp_{feat}",
                    )
                user_input[feat] = val

    st.markdown("---")

    if st.button("🔮 Predict Cardiovascular Disease", type="primary", use_container_width=True):
        try:
            result = predict_cardiovascular_disease(pipeline, user_input, feature_cols)

            if result["predicted_class"] == 1:
                st.markdown(f"""
<div class="result-positive">
    <h2>⚠️ {result['risk_label']}</h2>
    <h3>Estimated Probability: {result['confidence_pct']}</h3>
    <p>Model: <strong>{model_name}</strong></p>
</div>
""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div class="result-negative">
    <h2>✅ {result['risk_label']}</h2>
    <h3>Estimated Probability: {result['confidence_pct']}</h3>
    <p>Model: <strong>{model_name}</strong></p>
</div>
""", unsafe_allow_html=True)

            st.markdown("---")
            with st.expander("📊 Prediction Details"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted Class",  result["predicted_class"])
                c2.metric("Disease Probability", result["confidence_pct"])
                c3.metric("Model Used",       model_name)

                feat_df = pd.DataFrame(list(user_input.items()), columns=["Feature", "Value"])
                st.dataframe(feat_df, use_container_width=True)

            st.markdown("---")
            st.markdown(f"""
<div class="disclaimer-box">
{result['disclaimer']}
</div>
""", unsafe_allow_html=True)

        except Exception as ex:
            st.error(f"Prediction error: {ex}")
