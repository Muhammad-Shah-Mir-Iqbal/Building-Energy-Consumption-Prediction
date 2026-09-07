# ⚡ Building Energy Consumption Prediction

A complete, professional machine-learning project that predicts a building's
**energy consumption (kWh)** from its physical and operational characteristics
— built for the **Artificial Intelligence Lab — Open-Ended Lab (OEL)**
assignment (Supervised Learning / Regression paradigm).

The project includes a fully worked **Jupyter Notebook** (EDA → preprocessing
→ baseline model → optimized model → evaluation → reflection) and an
interactive, modern **Streamlit dashboard** for live predictions.

---

## 📁 Project Structure

```
energy_prediction_project/
├── data/
│   ├── energy_data.csv            # Full raw dataset (1,000 records)
│   ├── train_energy_data.csv      # Provided training split
│   └── test_energy_data.csv       # Provided hold-out split
├── notebooks/
│   └── energy_project.ipynb       # Full analysis notebook (Tasks 1–5)
├── src/
│   ├── pipeline.py                # Shared preprocessing/training/eval logic
│   └── train_model.py             # CLI script: trains & saves both models
├── models/                        # Saved trained models (.joblib) — generated
├── outputs/                       # Saved plots & metrics_summary.json — generated
├── report/
│   └── Project_Report.docx        # Written project report (OEL template)
├── app.py                         # Streamlit dashboard (the "UI")
├── requirements.txt
├── .streamlit/config.toml         # Dashboard theme
└── README.md                      # You are here
```

---

## 🧠 Problem Statement

Predict `Energy Consumption` (kWh) for a building given:

| Feature | Type | Description |
|---|---|---|
| `Building Type` | Categorical | Residential / Commercial / Industrial |
| `Square Footage` | Numeric | Total floor area |
| `Number of Occupants` | Numeric | People regularly present |
| `Appliances Used` | Numeric | Count of active appliances |
| `Average Temperature` | Numeric | Ambient temperature (°C) |
| `Day of Week` | Categorical | Weekday / Weekend |

## 🧪 Methodology

1. **Preprocessing** — duplicate removal, median/mode imputation, IQR outlier
   capping, standard scaling, one-hot encoding, 70/15/15 train/val/test split.
2. **Baseline Model** — Multiple Linear Regression.
3. **Optimized Model** — Random Forest Regressor tuned with `GridSearchCV`
   (5-fold CV) — a model-family swap **and** a hyperparameter search, the two
   required optimization changes.
4. **Evaluation** — MSE, RMSE, MAE, R², plus 5 diagnostic plots (actual vs.
   predicted, residuals, feature importance, error distribution, metric bars).
5. **Reflection** — written analysis of results, limitations, and future work
   (see the notebook's final section and `report/Project_Report.docx`).

---

## 🚀 Getting Started (VS Code)

### 1. Prerequisites
- Python 3.10–3.12 installed
- VS Code with the **Python** and **Jupyter** extensions installed

### 2. Open the project
```bash
# Open this folder in VS Code
code energy_prediction_project
```

### 3. Create a virtual environment (recommended)
```bash
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5a. Run the Jupyter Notebook
Open `notebooks/energy_project.ipynb` in VS Code, select the `venv` kernel
(top-right kernel picker), and choose **Run All**. The notebook re-generates
every plot into `outputs/` and re-saves both models into `models/`.

*(Alternatively, from the terminal:)*
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/energy_project.ipynb
```

### 5b. Train the models via script (optional — the dashboard can also train on first run)
```bash
python src/train_model.py
```
This prints train/validation/test metrics and saves:
- `models/baseline_linear_regression.joblib`
- `models/optimized_random_forest.joblib`
- `outputs/metrics_summary.json`

### 6. Launch the Dashboard
```bash
streamlit run app.py
```
Then open the URL Streamlit prints (typically `http://localhost:8501`) in
your browser. The dashboard has 5 pages, accessible from the sidebar:

- **🏠 Overview** — dataset snapshot, key metrics, distribution & correlation charts
- **📊 Data Explorer** — filterable raw data table, distribution & category plots
- **🤖 Model Performance** — baseline vs. optimized metrics, actual-vs-predicted, feature importance
- **🔮 Live Prediction** — enter a new building's details and get instant predictions from both models
- **📄 About the Project** — methodology & tech stack summary

> If no trained models exist yet in `models/`, the dashboard automatically
> trains both models on first launch (a few seconds) and caches them —
> no manual step required.

---

## 📊 Results (Test Set)

| Model | RMSE (kWh) | MAE (kWh) | R² |
|---|---|---|---|
| Baseline — Linear Regression | ≈ 0.01 | ≈ 0.01 | ≈ 1.0000 |
| Optimized — Tuned Random Forest | ≈ 122 | ≈ 93 | ≈ 0.982 |

The near-perfect linear baseline indicates the dataset's target is generated
as an (almost) exact linear function of the input features — a genuine,
honestly-reported finding discussed in depth in the notebook's Task 5
(Reflection) section and in the project report.

---

## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib` / `Seaborn`
(notebook plots) · `Plotly` (dashboard plots) · `Streamlit` (dashboard) ·
`Jupyter` / `ipykernel` (notebook)

---

## ❗ Troubleshooting

- **`ModuleNotFoundError`** → make sure your virtual environment is activated
  and run `pip install -r requirements.txt` again.
- **Notebook kernel not found in VS Code** → open the Command Palette
  (`Ctrl+Shift+P` / `Cmd+Shift+P`) → *Python: Select Interpreter* → choose the
  `venv` you created, then reselect the kernel in the notebook.
- **Streamlit says "port already in use"** → run
  `streamlit run app.py --server.port 8502` to pick a different port.
- **`FileNotFoundError` for the dataset** → run all commands from the project
  root folder (`energy_prediction_project/`), not from inside `src/` or
  `notebooks/`.

---

## 📄 License / Academic Use

Built for academic submission as part of an Artificial Intelligence Lab
Open-Ended Lab (OEL) assignment. Feel free to adapt for coursework with
attribution.
