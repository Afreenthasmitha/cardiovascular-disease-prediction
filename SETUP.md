# SETUP Guide — Windows

Complete setup instructions for the Cardiovascular Disease Prediction System on Windows.

---

## Prerequisites

### Python
- Python 3.11 or later
- Download from: https://www.python.org/downloads/
- During installation, **check "Add Python to PATH"**

### Verify installation
Open PowerShell / Command Prompt:
```bash
python --version
pip --version
```

---

## Step-by-Step Setup

### 1. Open the project folder

```bash
cd "d:\Cardiovascular Disease Prediction"
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

```bash
venv\Scripts\activate
```

You should see `(venv)` in your prompt.

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Add your dataset

Copy your CSV dataset to:
```
data\raw\dataset.csv
```

### 6. Run the application

```bash
streamlit run app.py
```

Your browser should open automatically at `http://localhost:8501`

---

## Troubleshooting

### ❌ `python` not recognized
- Reinstall Python and ensure "Add to PATH" is checked.
- Try `py` instead of `python`.

### ❌ `pip` not recognized
```bash
python -m pip install --upgrade pip
```

### ❌ Virtual environment activation fails
If PowerShell blocks scripts:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### ❌ Package installation errors
Try upgrading pip first:
```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### ❌ Dataset not found error
Ensure the file is at exactly:
```
data\raw\dataset.csv
```
If your file has a different name or is in a different location, edit `src/config.py`:
```python
DATASET_PATH = os.path.join(DATA_RAW_DIR, "your_file.csv")
```

### ❌ Target column not detected
Edit `src/config.py` and set:
```python
TARGET_COLUMN = "your_actual_target_column_name"
```

### ❌ Streamlit not starting
Ensure venv is activated. Then:
```bash
pip install streamlit
streamlit run app.py
```

### ❌ Streamlit opens wrong port
```bash
streamlit run app.py --server.port 8502
```

### ❌ Jupyter not found
```bash
pip install jupyter
jupyter notebook
```

---

## Running the CLI Training Script

```bash
python run_training.py
```

With hyperparameter tuning (slower but more accurate):
```bash
python run_training.py --tune
```

---

## File Locations After Training

| File | Purpose |
|------|---------|
| `models/best_model_pipeline.pkl` | Trained ML pipeline |
| `models/model_metadata.json` | Model info & metrics |
| `reports/model_results.csv` | All model comparison metrics |
| `reports/figures/*.png` | All generated plots |

---

## Support

If you encounter any issue not listed here, check:
1. Python version is 3.11+
2. Virtual environment is activated
3. All packages installed
4. Dataset is at `data/raw/dataset.csv`
5. Target column is correctly configured in `src/config.py`
