"""
app.py
------
Professional Streamlit dashboard for the Building Energy Consumption
Prediction project.

Run with:
    streamlit run app.py
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from pipeline import (  # noqa: E402
    load_raw_data, clean_data, split_data, build_baseline_model,
    build_optimized_model_search, evaluate_model, save_model, load_model,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET,
    DATA_PATH, MODELS_DIR,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
METRICS_PATH = os.path.join(OUTPUTS_DIR, "metrics_summary.json")

BASELINE_MODEL_FILE = "baseline_linear_regression.joblib"
OPTIMIZED_MODEL_FILE = "optimized_random_forest.joblib"

st.set_page_config(
    page_title="Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# GLOBAL STYLE
# --------------------------------------------------------------------------
st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Segoe UI', 'Inter', sans-serif; }

    .main-header {
        padding: 1.6rem 2rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #0EA5A4 0%, #0F766E 55%, #134E4A 100%);
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(15, 118, 110, 0.25);
    }
    .main-header h1 { margin: 0; font-size: 2rem; font-weight: 700;}
    .main-header p { margin: 0.35rem 0 0 0; opacity: 0.92; font-size: 1rem;}

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E5EAF0;
        border-radius: 14px;
        padding: 1rem 1.1rem 0.6rem 1.1rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    }
    div[data-testid="stMetricLabel"] { font-weight: 600; color: #475569; }

    .section-card {
        background: #FFFFFF;
        border: 1px solid #E5EAF0;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .badge-baseline { background: #E0F2FE; color: #075985; }
    .badge-optimized { background: #DCFCE7; color: #14532D; }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# CACHED DATA / MODEL LOADING
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_clean_data():
    raw = load_raw_data(DATA_PATH)
    clean = clean_data(raw)
    return raw, clean


@st.cache_resource(show_spinner=False)
def get_models():
    """
    Loads trained models from disk if available; otherwise trains them
    on the spot (first run only) so the app never crashes with a
    missing-file error.
    """
    baseline_path = os.path.join(MODELS_DIR, BASELINE_MODEL_FILE)
    optimized_path = os.path.join(MODELS_DIR, OPTIMIZED_MODEL_FILE)

    if os.path.exists(baseline_path) and os.path.exists(optimized_path):
        baseline = load_model(BASELINE_MODEL_FILE)
        optimized = load_model(OPTIMIZED_MODEL_FILE)
        return baseline, optimized, False

    # Train on the fly (first launch, no cached models yet)
    _, clean = get_clean_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(clean)

    baseline = build_baseline_model()
    baseline.fit(X_train, y_train)

    search = build_optimized_model_search()
    search.fit(X_train, y_train)
    optimized = search.best_estimator_

    save_model(baseline, BASELINE_MODEL_FILE)
    save_model(optimized, OPTIMIZED_MODEL_FILE)
    return baseline, optimized, True


@st.cache_data(show_spinner=False)
def get_test_split():
    _, clean = get_clean_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(clean)
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_metrics_summary():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>⚡ Building Energy Consumption Predictor</h1>
    <p>Supervised Learning Regression Dashboard &nbsp;|&nbsp; Baseline (Linear Regression) vs. Optimized (Tuned Random Forest)</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading data and models..."):
    raw_df, clean_df = get_clean_data()
    baseline_model, optimized_model, just_trained = get_models()
    X_train, X_val, X_test, y_train, y_val, y_test = get_test_split()

if just_trained:
    st.info("No cached model found — trained both models on the fly for this session. "
            "Run `python src/train_model.py` once to persist them for instant future loads.")

baseline_metrics = evaluate_model(baseline_model, X_test, y_test)
optimized_metrics = evaluate_model(optimized_model, X_test, y_test)

# --------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------------------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Overview", "📊 Data Explorer", "🤖 Model Performance", "🔮 Live Prediction"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset")
st.sidebar.write(f"**Records:** {len(clean_df):,}")
st.sidebar.write(f"**Features:** {len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)}")
st.sidebar.write(f"**Target:** `{TARGET}`")
st.sidebar.markdown("---")



# ==========================================================================
# PAGE: OVERVIEW
# ==========================================================================
if page == "🏠 Overview":
    st.subheader("Project Snapshot")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Buildings", f"{len(clean_df):,}")
    col2.metric("Avg. Energy Consumption", f"{clean_df[TARGET].mean():,.0f} kWh")
    col3.metric("Baseline Test R²", f"{baseline_metrics['R2']:.4f}",
                help="Linear Regression, held-out test set")
    col4.metric("Optimized Test R²", f"{optimized_metrics['R2']:.4f}",
                help="Tuned Random Forest, held-out test set")

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Energy Consumption Distribution")
        fig = px.histogram(
            clean_df, x=TARGET, nbins=40, marginal="box",
            color_discrete_sequence=["#0EA5A4"],
        )
        fig.update_layout(height=380, margin=dict(t=10, l=10, r=10, b=10),
                           plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Building Type Mix")
        counts = clean_df["Building Type"].value_counts().reset_index()
        counts.columns = ["Building Type", "Count"]
        fig2 = px.pie(counts, names="Building Type", values="Count", hole=0.5,
                      color_discrete_sequence=px.colors.sequential.Teal)
        fig2.update_layout(height=380, margin=dict(t=10, l=10, r=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Feature Correlation with Energy Consumption")
    corr = clean_df[NUMERICAL_FEATURES + [TARGET]].corr()
    fig3 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig3.update_layout(height=420, margin=dict(t=10, l=10, r=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================================
# PAGE: DATA EXPLORER
# ==========================================================================
elif page == "📊 Data Explorer":
    st.subheader("Explore the Dataset")

    tab1, tab2, tab3 = st.tabs(["🔍 Raw Table", "📈 Distributions", "🏷️ Category Breakdown"])

    with tab1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        building_filter = st.multiselect(
            "Filter by Building Type",
            options=sorted(clean_df["Building Type"].unique()),
            default=sorted(clean_df["Building Type"].unique()),
        )
        day_filter = st.multiselect(
            "Filter by Day of Week",
            options=sorted(clean_df["Day of Week"].unique()),
            default=sorted(clean_df["Day of Week"].unique()),
        )
        filtered = clean_df[
            clean_df["Building Type"].isin(building_filter) &
            clean_df["Day of Week"].isin(day_filter)
        ]
        st.dataframe(filtered, use_container_width=True, height=420)
        st.caption(f"Showing {len(filtered):,} of {len(clean_df):,} records (after cleaning: "
                   f"duplicates removed, missing values imputed, outliers capped).")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        feature = st.selectbox("Choose a numeric feature", NUMERICAL_FEATURES + [TARGET])
        fig = px.histogram(clean_df, x=feature, nbins=35, marginal="violin",
                           color_discrete_sequence=["#0F766E"])
        fig.update_layout(height=420, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        cat_feature = st.selectbox("Choose a categorical feature", CATEGORICAL_FEATURES)
        fig = px.box(clean_df, x=cat_feature, y=TARGET, color=cat_feature,
                    color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=420, plot_bgcolor="white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================================
# PAGE: MODEL PERFORMANCE
# ==========================================================================
elif page == "🤖 Model Performance":
    st.subheader("Baseline vs. Optimized — Test Set Performance")

    st.markdown(
        '<span class="badge badge-baseline">Baseline: Linear Regression</span>'
        '<span class="badge badge-optimized">Optimized: Tuned Random Forest</span>',
        unsafe_allow_html=True,
    )
    st.write("")

    col1, col2, col3, col4 = st.columns(4)
    metric_pairs = [
        ("RMSE (kWh)", baseline_metrics["RMSE"], optimized_metrics["RMSE"], False),
        ("MAE (kWh)", baseline_metrics["MAE"], optimized_metrics["MAE"], False),
        ("MSE", baseline_metrics["MSE"], optimized_metrics["MSE"], False),
        ("R² Score", baseline_metrics["R2"], optimized_metrics["R2"], True),
    ]
    for col, (label, b_val, o_val, higher_better) in zip([col1, col2, col3, col4], metric_pairs):
        delta = o_val - b_val
        delta_str = f"{delta:+.4f}"
        col.metric(f"Optimized {label}", f"{o_val:,.4f}", delta_str,
                    delta_color=("normal" if higher_better else "inverse"))

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Error Metrics Comparison")
        bar_df = pd.DataFrame({
            "Metric": ["RMSE", "MAE"] * 2,
            "Model": ["Baseline"] * 2 + ["Optimized"] * 2,
            "Value": [baseline_metrics["RMSE"], baseline_metrics["MAE"],
                     optimized_metrics["RMSE"], optimized_metrics["MAE"]],
        })
        fig = px.bar(bar_df, x="Metric", y="Value", color="Model", barmode="group",
                    color_discrete_map={"Baseline": "#38BDF8", "Optimized": "#22C55E"})
        fig.update_layout(height=380, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### R² Score Comparison")
        r2_df = pd.DataFrame({
            "Model": ["Baseline", "Optimized"],
            "R2": [baseline_metrics["R2"], optimized_metrics["R2"]],
        })
        fig = px.bar(r2_df, x="Model", y="R2", color="Model", text_auto=".4f",
                    color_discrete_map={"Baseline": "#38BDF8", "Optimized": "#22C55E"},
                    range_y=[0, 1.05])
        fig.update_layout(height=380, plot_bgcolor="white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Actual vs. Predicted (Test Set)")
    model_choice = st.radio("Model", ["Baseline (Linear Regression)", "Optimized (Random Forest)"],
                            horizontal=True)
    preds = baseline_metrics["y_pred"] if "Baseline" in model_choice else optimized_metrics["y_pred"]
    scatter_df = pd.DataFrame({"Actual": y_test.values, "Predicted": preds})
    fig = px.scatter(scatter_df, x="Actual", y="Predicted", opacity=0.65,
                     color_discrete_sequence=["#0EA5A4"])
    lims = [scatter_df.min().min(), scatter_df.max().max()]
    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", name="Perfect Prediction",
                             line=dict(color="red", dash="dash")))
    fig.update_layout(height=460, plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Feature Importance (Optimized Random Forest)")
    importances = optimized_model.named_steps["regressor"].feature_importances_
    feat_names = (
        NUMERICAL_FEATURES +
        list(optimized_model.named_steps["preprocessor"]
             .named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    )
    imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values("Importance")
    fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                color_discrete_sequence=["#7C3AED"])
    fig.update_layout(height=420, plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    metrics_summary = get_metrics_summary()
    if metrics_summary:
        with st.expander("🔧 Best hyperparameters found (GridSearchCV)"):
            st.json(metrics_summary["optimized"]["best_params"])


# ==========================================================================
# PAGE: LIVE PREDICTION
# ==========================================================================
elif page == "🔮 Live Prediction":
    st.subheader("Predict Energy Consumption for a New Building")
    st.caption("Enter a building's characteristics below and compare predictions from both models.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            building_type = st.selectbox("Building Type", ["Residential", "Commercial", "Industrial"])
            square_footage = st.number_input("Square Footage", min_value=100, max_value=200000,
                                             value=25000, step=500)
        with c2:
            occupants = st.number_input("Number of Occupants", min_value=1, max_value=1000, value=40)
            appliances = st.number_input("Appliances Used", min_value=0, max_value=200, value=20)
        with c3:
            temperature = st.slider("Average Temperature (°C)", min_value=-10.0, max_value=45.0, value=25.0)
            day_of_week = st.selectbox("Day of Week", ["Weekday", "Weekend"])

        submitted = st.form_submit_button("⚡ Predict Energy Consumption", use_container_width=True)

    if submitted:
        new_building = pd.DataFrame({
            "Building Type": [building_type],
            "Square Footage": [square_footage],
            "Number of Occupants": [occupants],
            "Appliances Used": [appliances],
            "Average Temperature": [temperature],
            "Day of Week": [day_of_week],
        })

        try:
            pred_baseline = baseline_model.predict(new_building)[0]
            pred_optimized = optimized_model.predict(new_building)[0]

            st.markdown("<br>", unsafe_allow_html=True)
            r1, r2 = st.columns(2)
            r1.metric("Baseline Prediction (Linear Regression)", f"{pred_baseline:,.1f} kWh")
            r2.metric("Optimized Prediction (Random Forest)", f"{pred_optimized:,.1f} kWh")

            fig = go.Figure(data=[
                go.Bar(name="Prediction", x=["Baseline", "Optimized"],
                      y=[pred_baseline, pred_optimized],
                      marker_color=["#38BDF8", "#22C55E"],
                      text=[f"{pred_baseline:,.0f}", f"{pred_optimized:,.0f}"],
                      textposition="outside")
            ])
            fig.update_layout(height=380, plot_bgcolor="white", yaxis_title="Predicted Energy Consumption (kWh)")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.info("Fill in the form above and click **Predict Energy Consumption** to see results.")


# ==========================================================================
# PAGE: ABOUT
# ==========================================================================
elif page == "📄 About the Project":
    st.subheader("About This Project")
    st.markdown("""
<div class="section-card">

### 🎯 Problem Statement
Predict a building's **Energy Consumption (kWh)** from physical and
operational characteristics (square footage, occupants, appliances used,
average temperature, building type, and day of week) using supervised
regression.

### 🧪 Methodology
1. **Data Preprocessing** — duplicate removal, median/mode imputation,
   IQR-based outlier capping, standard scaling, one-hot encoding, and a
   70/15/15 train/validation/test split.
2. **Baseline Model** — Multiple Linear Regression (Ordinary Least
   Squares), chosen for interpretability and as an uncontaminated
   reference point.
3. **Optimized Model** — Random Forest Regressor tuned via `GridSearchCV`
   (5-fold cross-validation) over `n_estimators`, `max_depth`,
   `min_samples_split`, and `min_samples_leaf`.
4. **Evaluation** — MSE, RMSE, MAE, R², actual-vs-predicted plots,
   residual plots, feature importance, and error-distribution plots.

### 🛠️ Tech Stack
`Python` · `scikit-learn` · `pandas` · `NumPy` · `Matplotlib` / `Seaborn`
(notebook) · `Plotly` (dashboard) · `Streamlit` (dashboard) · `Jupyter`

### 📁 Project Structure
```
energy_prediction_project/
├── data/                     # Raw & split CSV datasets
├── notebooks/                # Full analysis notebook (Tasks 1-5)
├── src/                      # Reusable pipeline & training script
├── models/                   # Persisted trained models (.joblib)
├── outputs/                  # Saved plots & metrics summary
├── report/                   # Project report template
├── app.py                    # This Streamlit dashboard
├── requirements.txt
└── README.md
```
</div>
""", unsafe_allow_html=True)
