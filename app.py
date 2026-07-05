"""
app.py
======
Module 9: Interactive Dashboard (Streamlit).

A single-file, multi-tab industrial monitoring dashboard:
- Fleet overview with KPI strip and health-score gauge
- Sensor trend analysis + correlation heatmap
- Anomaly detection visualization (4-detector consensus)
- Explainable AI (SHAP global + per-machine)
- Maintenance recommendations + downloadable report
- Model performance (confusion matrix, ROC, PR curves, model comparison)
- Multi-machine comparison

Run:
    streamlit run app.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

from utils import MODELS_DIR, REPORTS_DIR, load_json
from predict import PredictiveMaintenancePipeline, RAW_SENSOR_NAMES

# ----------------------------------------------------------------------
# Page config + design tokens
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance Control Room",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "bg": "#14171A",
    "panel": "#1E2227",
    "border": "#2C3138",
    "text": "#E6E8EA",
    "text_muted": "#8B939B",
    "amber": "#F2A63A",
    "green": "#3FB68B",
    "blue": "#4C9FE8",
    "red": "#E5484D",
}
RISK_COLOR_MAP = {"Low": COLORS["green"], "Medium": COLORS["blue"], "High": COLORS["amber"], "Critical": COLORS["red"]}

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}
h1, h2, h3 {{
    font-family: 'Oswald', sans-serif !important;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}}
.metric-mono, .metric-mono * {{
    font-family: 'JetBrains Mono', monospace !important;
}}
.kpi-card {{
    background: {COLORS['panel']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 14px 18px;
    border-left: 3px solid {COLORS['amber']};
}}
.kpi-label {{
    color: {COLORS['text_muted']};
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}}
.kpi-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    color: {COLORS['text']};
}}
.risk-pill {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 3px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
}}
hr {{ border-color: {COLORS['border']}; }}
[data-testid="stSidebar"] {{
    background: #101215;
    border-right: 1px solid {COLORS['border']};
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["text"], family="Inter, sans-serif"),
        colorway=[COLORS["amber"], COLORS["blue"], COLORS["green"], COLORS["red"], "#B18CFF", "#7DD3FC"],
        xaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
        yaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
)


# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained model artifacts...")
def load_pipeline():
    return PredictiveMaintenancePipeline()


@st.cache_data(show_spinner="Running inference on sensor data...")
def run_inference(_pipeline, raw_df: pd.DataFrame, latest_only: bool) -> pd.DataFrame:
    return _pipeline.run(raw_df, latest_cycle_only=latest_only)


@st.cache_data
def load_default_fleet_data() -> pd.DataFrame:
    return pd.read_csv("data/live_fleet_snapshot.csv")


@st.cache_data
def load_training_summary() -> dict:
    try:
        return load_json("training_summary")
    except FileNotFoundError:
        return {}


@st.cache_data
def load_curve_data():
    path = REPORTS_DIR / "roc_pr_curve_data.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}


@st.cache_data
def load_shap_importance():
    path = REPORTS_DIR / "global_feature_importance.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ----------------------------------------------------------------------
# Chart helpers
# ----------------------------------------------------------------------
def health_gauge(value: float, risk_level: str) -> go.Figure:
    color = RISK_COLOR_MAP.get(risk_level, COLORS["amber"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"size": 40, "family": "JetBrains Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": COLORS["text_muted"]},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": COLORS["panel"],
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "#3A1F22"},
                {"range": [30, 55], "color": "#3A3320"},
                {"range": [55, 80], "color": "#1E3330"},
                {"range": [80, 100], "color": "#1A3A2E"},
            ],
        },
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=260, margin=dict(t=30, b=10, l=30, r=30))
    return fig


def failure_probability_gauge(proba: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={"suffix": "%", "font": {"size": 32, "family": "JetBrains Mono"}},
        title={"text": "FAILURE PROBABILITY", "font": {"size": 12, "family": "Oswald"}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": COLORS["red"] if proba > 0.5 else COLORS["amber"], "thickness": 0.3},
            "bgcolor": COLORS["panel"],
            "borderwidth": 0,
        },
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=220, margin=dict(t=40, b=10, l=30, r=30))
    return fig


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ CONTROL PANEL")
st.sidebar.markdown("---")

data_source = st.sidebar.radio(
    "Data source", ["Live fleet demo (60 machines)", "Upload CSV"], index=0
)

uploaded_file = None
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload sensor CSV (CMAPSS-style schema)", type=["csv"]
    )
    st.sidebar.caption(
        "Expected columns: unit_id, cycle, and sensor readings "
        "(temperature, pressure, rpm, vibration, voltage, current, etc.)"
    )

latest_only = st.sidebar.checkbox("Show latest cycle per machine only", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("Model artifacts loaded from `/models`. Run `train.py` to retrain.")

# ----------------------------------------------------------------------
# Load pipeline + data + run inference
# ----------------------------------------------------------------------
try:
    pipeline = load_pipeline()
except FileNotFoundError:
    st.error(
        "⚠️ No trained models found in `/models`. Run `python train.py` first, "
        "then relaunch the dashboard."
    )
    st.stop()

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
else:
    raw_df = load_default_fleet_data()

report = run_inference(pipeline, raw_df, latest_only)

# machine selector (populate after we know what's in the report)
machine_ids = sorted(report["unit_id"].dropna().unique().tolist()) if "unit_id" in report.columns else []
selected_machine = st.sidebar.selectbox("Select machine (unit_id)", machine_ids, index=0 if machine_ids else None)

st.sidebar.markdown("---")
report_csv = report.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    "⬇ Download full prediction report (CSV)", data=report_csv,
    file_name="predictive_maintenance_report.csv", mime="text/csv",
)
report_json = report.to_json(orient="records", indent=2).encode("utf-8")
st.sidebar.download_button(
    "⬇ Download full prediction report (JSON)", data=report_json,
    file_name="predictive_maintenance_report.json", mime="application/json",
)

# ----------------------------------------------------------------------
# Header + KPI strip
# ----------------------------------------------------------------------
st.title("Predictive Maintenance Control Room")
st.caption("Real-time machine health monitoring · failure prediction · RUL estimation · anomaly detection")

n_machines = len(report)
n_critical = int((report["risk_level"] == "Critical").sum())
n_high = int((report["risk_level"] == "High").sum())
avg_health = report["health_score"].mean() if n_machines else 0
avg_rul = report["estimated_rul_cycles"].mean() if n_machines else 0

kpi_cols = st.columns(5)
kpi_data = [
    ("MACHINES MONITORED", f"{n_machines}"),
    ("CRITICAL RISK", f"{n_critical}"),
    ("HIGH RISK", f"{n_high}"),
    ("AVG HEALTH SCORE", f"{avg_health:.1f}%"),
    ("AVG RUL (CYCLES)", f"{avg_rul:.0f}"),
]
for col, (label, value) in zip(kpi_cols, kpi_data):
    col.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("")

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab_overview, tab_sensors, tab_anomaly, tab_xai, tab_maint, tab_perf, tab_compare = st.tabs([
    "🏭 Fleet Overview", "📈 Sensor Analysis", "🚨 Anomaly Detection",
    "🔍 Explainable AI", "🛠 Maintenance", "📊 Model Performance", "⚖️ Compare Machines",
])

# ---------------- TAB 1: Fleet Overview ----------------
with tab_overview:
    row = report[report["unit_id"].astype(str) == str(selected_machine)].iloc[0] if selected_machine is not None else None

    if row is not None:
        c1, c2, c3 = st.columns([1, 1, 1.4])
        with c1:
            st.plotly_chart(health_gauge(row["health_score"], row["risk_level"]), width='stretch')
            st.markdown(
                f'<div style="text-align:center;">Health Score — Machine '
                f'<span class="metric-mono">{int(selected_machine)}</span> · '
                f'<span class="risk-pill" style="background:{RISK_COLOR_MAP.get(row["risk_level"])}22;'
                f'color:{RISK_COLOR_MAP.get(row["risk_level"])};">{row["risk_level"]}</span></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.plotly_chart(failure_probability_gauge(row["failure_probability"]), width='stretch')
            st.markdown(
                f'<div class="metric-mono" style="text-align:center;">'
                f'RUL ≈ {row["estimated_rul_cycles"]:.0f} cycles '
                f'({row["estimated_rul_hours"]:.0f} hrs)</div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown("##### Fleet Health Distribution")
            fig = px.histogram(
                report, x="health_score", nbins=20, color="risk_level",
                color_discrete_map=RISK_COLOR_MAP,
                labels={"health_score": "Health Score", "risk_level": "Risk Level"},
            )
            fig.update_layout(template=PLOTLY_TEMPLATE, height=280, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, width='stretch')

    st.markdown("##### Fleet Status Table")
    display_cols = ["unit_id", "cycle", "health_score", "risk_level", "failure_probability",
                     "estimated_rul_cycles", "anomaly_score"]
    display_cols = [c for c in display_cols if c in report.columns]
    styled = report[display_cols].sort_values("health_score")
    st.dataframe(styled, width='stretch', height=320)

    st.markdown("##### Failure Timeline (RUL by machine)")
    if "estimated_rul_cycles" in report.columns:
        timeline = report.sort_values("estimated_rul_cycles")
        fig = px.bar(
            timeline, x="unit_id", y="estimated_rul_cycles", color="risk_level",
            color_discrete_map=RISK_COLOR_MAP,
            labels={"unit_id": "Machine ID", "estimated_rul_cycles": "Remaining Useful Life (cycles)"},
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=350, xaxis_type="category")
        st.plotly_chart(fig, width='stretch')

# ---------------- TAB 2: Sensor Analysis ----------------
with tab_sensors:
    st.markdown(f"##### Sensor Trends — Machine {selected_machine}")
    machine_history = raw_df[raw_df["unit_id"].astype(str) == str(selected_machine)].sort_values("cycle") \
        if "unit_id" in raw_df.columns else raw_df

    available_sensors = [s for s in RAW_SENSOR_NAMES if s in machine_history.columns]
    chosen_sensors = st.multiselect(
        "Sensors to plot", available_sensors, default=available_sensors[:4]
    )
    if chosen_sensors and "cycle" in machine_history.columns:
        fig = go.Figure()
        for s in chosen_sensors:
            fig.add_trace(go.Scatter(x=machine_history["cycle"], y=machine_history[s],
                                      mode="lines", name=s))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=400,
                           xaxis_title="Operating Cycle", yaxis_title="Sensor Reading")
        st.plotly_chart(fig, width='stretch')

    st.markdown("##### Sensor Correlation Heatmap")
    if available_sensors:
        corr = machine_history[available_sensors].corr()
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            aspect="auto",
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=450)
        st.plotly_chart(fig, width='stretch')

# ---------------- TAB 3: Anomaly Detection ----------------
with tab_anomaly:
    st.markdown(f"##### Anomaly Score Timeline — Machine {selected_machine}")
    st.caption("Ensemble of Isolation Forest, One-Class SVM, Local Outlier Factor, and Autoencoder reconstruction error.")

    machine_raw = pd.DataFrame()
    if "unit_id" in raw_df.columns and selected_machine is not None:
        machine_raw = raw_df[raw_df["unit_id"].astype(str) == str(selected_machine)]

    if machine_raw.empty:
        st.info("No historical rows found for this machine in the current dataset; showing latest-snapshot values only.")
        hist_report = report[report["unit_id"].astype(str) == str(selected_machine)] if "unit_id" in report.columns else report
    else:
        hist_report = run_inference(pipeline, machine_raw, latest_only=False)

    if "anomaly_score" in hist_report.columns and "cycle" in hist_report.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_report["cycle"], y=hist_report["anomaly_score"], mode="lines+markers",
            name="Anomaly Score", line=dict(color=COLORS["blue"]),
        ))
        anomalous = hist_report[hist_report.get("is_anomalous_consensus", False) == True]
        if len(anomalous):
            fig.add_trace(go.Scatter(
                x=anomalous["cycle"], y=anomalous["anomaly_score"], mode="markers",
                name="Flagged Anomaly", marker=dict(color=COLORS["red"], size=10, symbol="x"),
            ))
        fig.add_hline(y=0.5, line_dash="dash", line_color=COLORS["amber"], annotation_text="Alert Threshold")
        fig.update_layout(template=PLOTLY_TEMPLATE, height=380,
                           xaxis_title="Operating Cycle", yaxis_title="Ensemble Anomaly Score (0-1)")
        st.plotly_chart(fig, width='stretch')

    detector_cols = [c for c in ["anomaly_isoforest", "anomaly_ocsvm", "anomaly_lof", "anomaly_autoencoder"]
                      if c in hist_report.columns]
    if detector_cols:
        st.markdown("##### Per-Detector Flags (most recent readings)")
        st.dataframe(
            hist_report[["cycle"] + detector_cols].tail(15).set_index("cycle"),
            width='stretch',
        )

# ---------------- TAB 4: Explainable AI ----------------
with tab_xai:
    st.markdown("##### Global Feature Importance (SHAP)")
    shap_df = load_shap_importance()
    explain_cfg = {}
    try:
        explain_cfg = load_json("explainability_config")
    except FileNotFoundError:
        pass

    if shap_df is not None:
        st.caption(
            f"Explaining model: **{explain_cfg.get('explained_model', 'best tree-based model')}** "
            "(SHAP requires a tree-based model; if the deployed predictor is an LSTM, "
            "the best tree-based runner-up is explained here instead)."
        )
        fig = px.bar(
            shap_df.sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature", orientation="h",
            labels={"mean_abs_shap": "Mean |SHAP value|", "feature": "Feature"},
        )
        fig.update_traces(marker_color=COLORS["amber"])
        fig.update_layout(template=PLOTLY_TEMPLATE, height=500, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Run `train.py` to generate SHAP feature-importance data.")

    st.markdown("##### Sensors Driving This Machine's Risk Score")
    if row is not None and "driving_sensors" in row.index:
        drivers = row["driving_sensors"]
        if isinstance(drivers, str):
            import ast
            try:
                drivers = ast.literal_eval(drivers)
            except Exception:
                drivers = []
        if drivers:
            for i, sensor in enumerate(drivers, 1):
                st.markdown(f"**{i}. `{sensor}`** — among the most extreme standardized readings for this machine")
        else:
            st.caption("No dominant sensor identified for this machine's current reading.")

# ---------------- TAB 5: Maintenance ----------------
with tab_maint:
    st.markdown(f"##### Maintenance Recommendation — Machine {selected_machine}")
    if row is not None:
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown(f"""
            **Likely Component:** {row.get('likely_component', 'N/A')}
            **Root Cause:** {row.get('root_cause', 'N/A')}
            **Priority:** {row.get('maintenance_priority', 'N/A')}
            **Recommended Action:** {row.get('recommended_action', 'N/A')}
            """)
        with rc2:
            cost = row.get("estimated_cost_usd", "N/A")
            downtime = row.get("estimated_downtime_hours", "N/A")
            st.markdown(
                f'<div class="kpi-card metric-mono">Estimated Repair Cost<br>'
                f'<span class="kpi-value">{cost}</span> USD</div><br>'
                f'<div class="kpi-card metric-mono">Estimated Downtime<br>'
                f'<span class="kpi-value">{downtime}</span> hours</div>',
                unsafe_allow_html=True,
            )

    st.markdown("##### All Machines — Maintenance Priority Queue")
    priority_order = {"Immediate (schedule within 24 hours)": 0, "Urgent (schedule within 3-5 days)": 1,
                       "Planned (schedule within 2-3 weeks)": 2, "Routine (monitor at next scheduled service)": 3}
    if "maintenance_priority" in report.columns:
        queue = report.copy()
        queue["_sort"] = queue["maintenance_priority"].map(priority_order).fillna(9)
        queue = queue.sort_values("_sort")
        cols = ["unit_id", "risk_level", "likely_component", "root_cause",
                "maintenance_priority", "estimated_cost_usd", "estimated_downtime_hours"]
        cols = [c for c in cols if c in queue.columns]
        st.dataframe(queue[cols], width='stretch', height=400)

# ---------------- TAB 6: Model Performance ----------------
with tab_perf:
    summary = load_training_summary()
    if summary:
        st.markdown(f"##### Deployed Failure-Prediction Model: `{summary.get('best_failure_model', 'N/A')}`")
        metrics_df = pd.DataFrame(summary.get("failure_model_metrics", {})).T
        st.dataframe(metrics_df.style.format("{:.3f}").highlight_max(axis=0, color=f"{COLORS['green']}55"),
                     width='stretch')

        st.markdown("##### Model Comparison — ROC-AUC")
        if not metrics_df.empty:
            fig = px.bar(metrics_df.reset_index(), x="index", y="ROC_AUC",
                         labels={"index": "Model", "ROC_AUC": "ROC-AUC"})
            fig.update_traces(marker_color=COLORS["amber"])
            fig.update_layout(template=PLOTLY_TEMPLATE, height=350)
            st.plotly_chart(fig, width='stretch')

        rul_metrics = summary.get("rul_model_metrics", {})
        if rul_metrics:
            st.markdown("##### RUL Regression Metrics")
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE (cycles)", f"{rul_metrics.get('MAE', 0):.1f}")
            m2.metric("RMSE (cycles)", f"{rul_metrics.get('RMSE', 0):.1f}")
            m3.metric("R²", f"{rul_metrics.get('R2', 0):.3f}")
    else:
        st.info("Run `train.py` to generate model performance metrics.")

    curve_data = load_curve_data()
    if curve_data is not None:
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("##### ROC Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=curve_data["fpr"], y=curve_data["tpr"], mode="lines",
                                      line=dict(color=COLORS["amber"], width=3), name="ROC"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                      line=dict(color=COLORS["text_muted"], dash="dash"), name="Random"))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=350,
                               xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig, width='stretch')
        with cc2:
            st.markdown("##### Precision-Recall Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=curve_data["recall"], y=curve_data["precision"], mode="lines",
                                      line=dict(color=COLORS["blue"], width=3)))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=350,
                               xaxis_title="Recall", yaxis_title="Precision")
            st.plotly_chart(fig, width='stretch')

        st.markdown("##### Confusion Matrix")
        cm = curve_data["confusion_matrix"]
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Oranges",
                         labels=dict(x="Predicted", y="Actual"),
                         x=["No Failure", "Failure"], y=["No Failure", "Failure"])
        fig.update_layout(template=PLOTLY_TEMPLATE, height=350)
        st.plotly_chart(fig, width='stretch')
    else:
        st.caption("ROC/PR/Confusion-matrix data available for tree-based deployed models "
                   "(not generated when the LSTM is the top performer — see training_summary.json).")

# ---------------- TAB 7: Compare Machines ----------------
with tab_compare:
    st.markdown("##### Multi-Machine Comparison")
    compare_ids = st.multiselect(
        "Select machines to compare", machine_ids,
        default=machine_ids[:3] if len(machine_ids) >= 3 else machine_ids,
    )
    if compare_ids:
        compare_df = report[report["unit_id"].isin(compare_ids)]
        metric_choice = st.radio(
            "Metric", ["health_score", "failure_probability", "estimated_rul_cycles", "anomaly_score"],
            horizontal=True,
        )
        fig = px.bar(
            compare_df, x="unit_id", y=metric_choice, color="risk_level",
            color_discrete_map=RISK_COLOR_MAP,
            labels={"unit_id": "Machine ID", metric_choice: metric_choice.replace("_", " ").title()},
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=380, xaxis_type="category")
        st.plotly_chart(fig, width='stretch')

        cols = ["unit_id", "health_score", "risk_level", "failure_probability",
                "estimated_rul_cycles", "anomaly_score", "likely_component", "maintenance_priority"]
        cols = [c for c in cols if c in compare_df.columns]
        st.dataframe(compare_df[cols].set_index("unit_id"), width='stretch')

st.markdown("---")
st.caption(
    "Predictive Maintenance System · Built with scikit-learn, XGBoost, LightGBM, CatBoost, "
    "TensorFlow/Keras, SHAP, and Streamlit · Synthetic demo data schema-compatible with "
    "NASA C-MAPSS and AI4I datasets."
)
