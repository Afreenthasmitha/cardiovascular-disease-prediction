# Cardiovascular Disease Prediction System

A complete machine-learning project for cardiovascular disease risk assessment, built for educational and college demonstration purposes.

---

## 🎯 Project Overview

This system uses supervised machine learning to predict the likelihood of cardiovascular disease based on patient clinical features. Five algorithms are trained, evaluated, and compared to select the best-performing model.

**Important:** This is an educational project. Predictions are not medical diagnoses.

---

## ✨ Features

- 📂 **Automatic dataset inspection** — detects shape, types, missing values, duplicates
- 🔧 **Smart preprocessing** — imputation, duplicate removal, feature engineering
- 📊 **Comprehensive EDA** — histograms, box plots, correlation heatmap, outlier analysis
- 🤖 **5 ML Models** — Logistic Regression, KNN, Decision Tree, Random Forest, SVM
- 📏 **Complete evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC
- 🔄 **Cross-validation** — 5-fold stratified CV for robust comparison
- 💾 **Model persistence** — saves entire preprocessing + model pipeline
- 🌐 **Streamlit dashboard** — interactive multi-page web application
- 🔮 **Real-time prediction** — patient input form with dynamic fields

---

## 🛠️ Technology Stack

| Component        | Library            |
|------------------|--------------------|
| Data Processing  | pandas, NumPy      |
| Visualisation    | Matplotlib, Seaborn|
| Machine Learning | scikit-learn       |
| Model Saving     | joblib             |
| Web App          | Streamlit          |

---

## 🤖 Machine Learning Algorithms

| Model               | Scalable | Probabilistic |
|---------------------|----------|---------------|
| Logistic Regression | ✅        | ✅             |
| K-Nearest Neighbors | ✅        | ✅             |
| Decision Tree       | ✅        | ✅             |
| Random Forest       | ✅        | ✅             |
| Support Vector Machine | ✅    | ✅             |

---

## 📂 Project Structure

```
cardiovascular-disease-prediction/
│
├── data/
│   ├── raw/
│   │   └── dataset.csv          ← Place your dataset here
│   └── processed/
│
├── notebooks/
│   └── cardiovascular_analysis.ipynb
│
├── src/
│   ├── config.py                ← Dataset/ML configuration
│   ├── data_preprocessing.py   ← Loading, cleaning, imputation
│   ├── eda.py                   ← EDA & visualisation
│   ├── train_models.py          ← Training, CV, saving
│   ├── evaluate_models.py       ← Metrics, plots
│   └── prediction.py            ← Prediction interface
│
├── models/
│   ├── best_model_pipeline.pkl  ← Saved best model
│   └── model_metadata.json
│
├── reports/
│   ├── figures/                 ← Auto-generated plots
│   └── model_results.csv
│
├── app.py                       ← Streamlit application
├── run_training.py              ← CLI training runner
├── requirements.txt
├── README.md
└── SETUP.md
```

---

## 🚀 Installation

### Step 1: Create a virtual environment

```bash
python -m venv venv
```

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📂 Dataset

1. Place your cardiovascular disease dataset (CSV format) at:
   ```
   data/raw/dataset.csv
   ```

2. The system **automatically detects** the target column by checking for:
   `cardio`, `target`, `heart_disease`, `HeartDisease`, `output`, `num`

3. If your target column uses a different name, edit `src/config.py`:
   ```python
   TARGET_COLUMN = "your_target_column_name"
   ```

> **Recommended dataset:** [Cardiovascular Disease Dataset on Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

---

## ▶️ Running the Project

### Option A: Run the Streamlit App

```bash
streamlit run app.py
```

The app trains models automatically when you visit the Model Comparison page.

### Option B: Train from CLI

```bash
python run_training.py
```

With hyperparameter tuning:
```bash
python run_training.py --tune
```

### Option C: Jupyter Notebook

```bash
jupyter notebook notebooks/cardiovascular_analysis.ipynb
```

---

## 📊 Demo Flow

```
Open Application → View Overview → Inspect Dataset →
View EDA → View Correlation Matrix → Compare ML Models →
Identify Best Model → Enter Patient Features →
Click Predict → View Result + Probability → Read Disclaimer
```

---

## ⚠️ Medical Disclaimer

This application is a **machine-learning demonstration** created for educational and research purposes. Its predictions are based on patterns in the supplied dataset and should **not** be considered a medical diagnosis. The system has not been validated for clinical use. **Consult a qualified healthcare professional for medical advice.**

---

## 📜 License

Educational / Open Source — for college project use.
