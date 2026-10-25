import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import os
import folium
from datetime import datetime, timedelta
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from docx import Document
from docx.shared import Inches
import streamlit_authenticator as stauth

# ---------------- APP CONFIG ----------------

st.set_page_config(
    page_title="Corzynth | AI Pipeline Corrosion Intelligence",
    page_icon="corzynth_logo.jpeg",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_VERSION = "Corzynth v10.1"
BRAND = "Corzynth"
DEVELOPER = "E. Vellay"
LOGO_FILE = "corzynth_logo.jpeg"

# ---------------- CACHED LOADERS ----------------

@st.cache_resource
def load_corrosion_model():
    try:
        return joblib.load("corrosion_model.joblib")
    except Exception:
        st.error("⚠️ Corrosion model file not found. Ensure 'corrosion_model.joblib' is in the project folder.")
        st.stop()

@st.cache_data
def load_uploaded_csv(file):
    if file.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    return pd.read_csv(file)

# ---------------- LOGIN / START PAGE ----------------

authenticator = stauth.Authenticate(
    st.secrets["credentials"].to_dict(),
    st.secrets["cookie"]["name"],
    st.secrets["cookie"]["key"],
    st.secrets["cookie"]["expiry_days"]
)

if not st.session_state.get("authentication_status"):

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    [data-testid="stImage"] { display: flex; justify-content: center; }
    .login-hero {
        background:
            radial-gradient(circle at 80% 20%, rgba(24,169,153,0.35), transparent 50%),
            linear-gradient(120deg, #06263d 0%, #0b3d62 55%, #10557f 100%);
        border-radius: 28px;
        padding: 50px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        border: 1px solid rgba(255,255,255,0.10);
    }
    .login-hero h1 { font-size: 3rem; font-weight: 800; margin: 10px 0; }
    .login-hero p { font-size: 1.1rem; opacity: 0.9; }
    .login-badge {
        display: inline-block;
        background: rgba(24,169,153,0.2);
        border: 1px solid rgba(24,169,153,0.5);
        color: #7ee2d3;
        padding: 6px 18px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .login-feature {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 20px;
        color: #e6edf3;
        text-align: center;
    }
    .login-feature h4 { color: #7ee2d3; margin-bottom: 6px; }
    .login-feature p { font-size: 0.85rem; opacity: 0.85; margin: 0; }
    </style>
    """, unsafe_allow_html=True)

    logo_col, empty1, empty2 = st.columns([1, 2, 1])
    with logo_col:
        try:
            st.image(LOGO_FILE, width=220)
        except Exception as e:
            st.error(f"Logo error: {e}")

    st.markdown("""
    <div class="login-hero">
        <div class="login-badge">AI-POWERED PIPELINE INTEGRITY PLATFORM</div>
        <h1>Corzynth</h1>
        <p>Predict corrosion. Quantify risk. Forecast remaining life.<br>
        Optimize integrity investment — powered by AI.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="login-feature">
            <h4>🧪 AI Prediction</h4>
            <p>Machine-learning corrosion-rate prediction for pipelines.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="login-feature">
            <h4>🎯 Risk Intelligence</h4>
            <p>Likelihood, consequence, and overall risk scoring.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="login-feature">
            <h4>🌍 Geophysical AI</h4>
            <p>Soil corrosivity, hotspots, and GIS risk mapping.</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="login-feature">
            <h4>💰 ROI & RBI</h4>
            <p>Inspection optimization and corrosion-cost savings.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    login_col1, login_col2, login_col3 = st.columns([1, 1.2, 1])

    with login_col2:
        st.markdown("#### 🔐 Sign in to Corzynth")
        authenticator.login()

    st.markdown(f"""
    <div style="text-align:center; color:#64748b; font-size:0.85rem; margin-top:30px;">
        {APP_VERSION} • AI Pipeline Corrosion Intelligence Platform • Developed by {DEVELOPER}<br>
        Access is provided to authorized users only.
    </div>
    """, unsafe_allow_html=True)

    st.stop()

if st.session_state["authentication_status"] is False:
    st.error("Incorrect username or password.")
    st.stop()

authenticator.logout(location="sidebar")

username = st.session_state["username"]
user_name = st.session_state["name"]
user_role = st.secrets["credentials"]["usernames"][username]["role"]

sidebar_logo_col, sidebar_empty = st.sidebar.columns([1, 0.3])
with sidebar_logo_col:
    try:
        st.image(LOGO_FILE, width=150)
    except Exception as e:
        st.sidebar.error(f"Logo error: {e}")

st.sidebar.success(f"Logged in: {user_name}")
st.sidebar.info(f"Role: {user_role.title()}")
st.sidebar.caption(APP_VERSION)
st.sidebar.info("🔬 PILOT PROGRAM — Evaluation Build")

# ---------------- CORZYNTH THEME ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1500px; }
[data-testid="stImage"] { display: flex; justify-content: center; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #06263d 0%, #0b3d62 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }
.main-header {
    background:
        radial-gradient(circle at 85% 20%, rgba(24,169,153,0.35), transparent 45%),
        linear-gradient(120deg, #06263d 0%, #0b3d62 55%, #10557f 100%);
    border: 1px solid rgba(255,255,255,0.10);
    padding: 38px 40px;
    border-radius: 24px;
    color: white;
    margin-bottom: 28px;
    text-align: center;
}
.main-header h1 { margin: 0; font-size: 2.7rem; font-weight: 800; }
.main-header p { margin: 10px 0 0 0; opacity: 0.88; }
.hero-badge {
    display: inline-block;
    background: rgba(24,169,153,0.18);
    border: 1px solid rgba(24,169,153,0.45);
    color: #7ee2d3;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 14px;
}
.kpi-card {
    background: #ffffff;
    border: 1px solid #e6ebf1;
    border-radius: 18px;
    padding: 22px;
    border-top: 4px solid #18a999;
    height: 100%;
    transition: all 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 18px rgba(11,61,98,0.12);
}
.kpi-value { font-size: 1.55rem; font-weight: 800; color: #0b3d62; }
.kpi-label { font-size: 0.88rem; font-weight: 600; color: #334155; margin-top: 6px; }
.kpi-desc { font-size: 0.78rem; color: #64748b; margin-top: 8px; }
.risk-low { border-top-color: #2ea043; }
.risk-medium { border-top-color: #d29922; }
.risk-high { border-top-color: #cf222e; }
.risk-critical { border-top-color: #8b0000; }
.section-title {
    font-size: 1.25rem;
    font-weight: 750;
    color: #0b3d62;
    margin: 26px 0 14px 0;
}
.professional-note {
    background: linear-gradient(90deg, #f8fafc 0%, #eef6f9 100%);
    border: 1px solid #d7e3ec;
    border-left: 5px solid #18a999;
    padding: 18px 20px;
    border-radius: 14px;
}
.footer {
    text-align: center;
    color: #64748b;
    font-size: 0.82rem;
    margin-top: 40px;
    padding-top: 18px;
    border-top: 1px solid #e6ebf1;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

logo_col, empty1, empty2 = st.columns([1, 2, 1])
with logo_col:
    try:
        st.image(LOGO_FILE, width=170)
    except Exception as e:
        st.error(f"Logo error: {e}")

st.markdown("""
<div class="main-header">
    <div class="hero-badge">AI-POWERED PIPELINE INTEGRITY PLATFORM</div>
    <h1>Corzynth</h1>
    <p>Predict corrosion. Quantify risk. Forecast remaining life. Optimize integrity investment.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- MENU ----------------

if user_role == "admin":
    menu_options = [
        "🏠 Dashboard", "🧪 Single Prediction", "⚙️ Batch Prediction",
        "🎯 Risk Intelligence", "📈 Corrosion Forecast", "🧠 Explainable AI",
        "🌍 Geophysical Analysis", "🔮 What-If Scenario Simulator",
        "📅 RBI Interval Optimizer", "💰 Corrosion Cost & ROI",
        "🛠️ Maintenance Planner", "📊 Model Performance",
        "📈 Corrosion Trend Analyzer", "🗺️ Pipeline Route Risk Map",
        "🔔 Alerts & Notifications", "🛠️ Admin Panel", "📖 User Guide"
    ]
elif user_role == "engineer":
    menu_options = [
        "🏠 Dashboard", "🧪 Single Prediction", "⚙️ Batch Prediction",
        "🎯 Risk Intelligence", "📈 Corrosion Forecast", "🧠 Explainable AI",
        "🌍 Geophysical Analysis", "🔮 What-If Scenario Simulator",
        "📅 RBI Interval Optimizer", "💰 Corrosion Cost & ROI",
        "🛠️ Maintenance Planner", "📈 Corrosion Trend Analyzer",
        "🗺️ Pipeline Route Risk Map", "🔔 Alerts & Notifications", "📖 User Guide"
    ]
else:
    menu_options = ["🏠 Dashboard", "🔔 Alerts & Notifications", "📖 User Guide"]

page = st.sidebar.radio("Select Module", menu_options)

# ---------------- REPORT HELPERS ----------------

def add_report_figure(title, figure):
    if "report_figures" not in st.session_state:
        st.session_state["report_figures"] = []
    st.session_state["report_figures"].append({"title": title, "figure": figure})

def clear_report_figures():
    st.session_state["report_figures"] = []

def convert_figure_to_png(figure):
    try:
        return figure.to_image(format="png", width=1400, height=800, scale=2)
    except Exception:
        return None

def create_word_report(title, summary_text, data_frame, interpretation_points, recommendation_points):
    document = Document()
    document.add_heading(title, level=0)
    document.add_paragraph(f"Report generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}")
    document.add_paragraph(f"Prepared for: {user_name} ({user_role.title()})")
    document.add_paragraph(APP_VERSION)

    try:
        document.add_picture(LOGO_FILE, width=Inches(2.0))
    except Exception:
        pass

    document.add_heading("1. Executive Summary", level=1)
    document.add_paragraph(summary_text)

    if st.session_state.get("report_figures"):
        document.add_heading("2. Visual Analysis", level=1)
        for item in st.session_state["report_figures"]:
            document.add_heading(item["title"], level=2)
            png = convert_figure_to_png(item["figure"])
            if png:
                document.add_picture(io.BytesIO(png), width=Inches(6.5))

    document.add_heading("3. Assessment Results", level=1)
    if data_frame is not None and not data_frame.empty:
        export_df = data_frame.copy()
        if len(export_df) > 50:
            export_df = export_df.head(50)
        table = document.add_table(rows=1, cols=len(export_df.columns))
        table.style = "Table Grid"
        for i, column in enumerate(export_df.columns):
            table.rows[0].cells[i].text = str(column)
        for _, row in export_df.iterrows():
            cells = table.add_row().cells
            for i, column in enumerate(export_df.columns):
                cells[i].text = str(row[column])

    document.add_heading("4. Technical Interpretation", level=1)
    for point in interpretation_points:
        document.add_paragraph(point, style="List Bullet")

    document.add_heading("5. Recommendations", level=1)
    for point in recommendation_points:
        document.add_paragraph(point, style="List Bullet")

    document.add_heading("6. Disclaimer", level=1)
    document.add_paragraph(
        "This report was generated by Corzynth for engineering screening and decision-support purposes. "
        "Final pipeline integrity decisions must be reviewed by qualified engineering professionals."
    )

    buffer = io.BytesIO()
    document.save(buffer)
    buffer.seek(0)
    return buffer

def download_reports(data_frame, csv_name, word_title, summary_text, interpretation_points, recommendation_points, key_prefix):
    csv_data = data_frame.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download CSV Report", data=csv_data, file_name=csv_name,
        mime="text/csv", use_container_width=True, key=f"{key_prefix}_csv"
    )
    if st.button("📄 Generate Word Report with Charts", use_container_width=True, key=f"{key_prefix}_word"):
        st.session_state[f"{key_prefix}_ready"] = True
        st.rerun()
    if st.session_state.get(f"{key_prefix}_ready"):
        word_buffer = create_word_report(word_title, summary_text, data_frame, interpretation_points, recommendation_points)
        st.download_button(
            label="⬇️ Download Word Report (.docx)", data=word_buffer,
            file_name=csv_name.replace(".csv", ".docx"),
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True, key=f"{key_prefix}_download"
        )

# ---------------- ENGINEERING HELPERS ----------------

def classify_corrosion_risk(rate):
    if rate < 0.10:
        return "Low"
    elif rate <= 0.30:
        return "Medium"
    return "High"

def calculate_likelihood_score(row, predicted_rate):
    score = min(predicted_rate / 0.5, 1.0) * 0.40
    if "Soil_Resistivity_ohm_m" in row:
        score += min(1000 / max(row["Soil_Resistivity_ohm_m"], 1), 2) / 2 * 0.15
    if "Soil_pH" in row:
        score += max(0, (7.5 - row["Soil_pH"]) / 4) * 0.10
    if "Moisture_Content_percent" in row:
        score += min(row["Moisture_Content_percent"] / 50, 1) * 0.10
    if "Coating_Condition" in row:
        coating = {"Excellent": 0, "Good": 0.25, "Fair": 0.6, "Poor": 1}
        score += coating.get(row["Coating_Condition"], 0.5) * 0.15
    if "Cathodic_Protection" in row and str(row["Cathodic_Protection"]).lower() == "no":
        score += 0.05
    return min(score, 1.0)

def calculate_consequence_score(row):
    score = 0.30
    if "Operating_Pressure_bar" in row:
        score += min(row["Operating_Pressure_bar"] / 100, 1) * 0.25
    if "Product_Type" in row:
        products = {"Crude Oil": 0.7, "Natural Gas": 1, "Refined Product": 0.85, "Water": 0.3}
        score += products.get(row["Product_Type"], 0.6) * 0.2
    if "Environmental_Sensitivity" in row:
        sensitivity = {"Low": 0, "Medium": 0.5, "High": 1}
        score += sensitivity.get(row["Environmental_Sensitivity"], 0.5) * 0.1
    return min(score, 1.0)

def classify_overall_risk(risk_score, corrosion_risk):
    adjustment = {"Low": 0, "Medium": 0.25, "High": 0.5, "Critical": 0.75}
    score = min(risk_score + adjustment.get(corrosion_risk, 0), 1)
    if score < 0.30:
        return "Low"
    elif score < 0.50:
        return "Medium"
    elif score < 0.70:
        return "High"
    return "Critical"

def get_priority(risk_level):
    return {"Low": "Routine", "Medium": "Planned", "High": "High", "Critical": "Critical"}.get(risk_level, "Routine")

def get_recommended_action(risk_level, remaining_life):
    if risk_level == "Critical":
        return "Immediate ILI inspection, CP review, and repair planning"
    elif risk_level == "High":
        return "Priority inspection and corrosion monitoring"
    elif risk_level == "Medium":
        return "Scheduled inspection and inhibitor review"
    return "Routine monitoring"

# ---------------- INHIBITOR MODEL ----------------

INHIBITOR_EFFICIENCY = {
    "None": 0.0,
    "Imidazoline": 0.90,
    "Amine-based": 0.85,
    "Phosphate Ester": 0.80,
    "Other": 0.75
}

def apply_inhibitor_efficiency(base_rate, dose, inhibitor_type="Imidazoline", residual_ppm=None):
    max_eff = INHIBITOR_EFFICIENCY.get(inhibitor_type, 0.75)
    if dose <= 0 or max_eff == 0:
        efficiency = 0.0
    else:
        dose_factor = dose / (dose + 25)
        residual_factor = 1.0 if residual_ppm is None else min(residual_ppm / max(dose, 1), 1.0)
        efficiency = max_eff * dose_factor * residual_factor
    return base_rate * (1 - efficiency), efficiency

def prepare_model_input(row, model):
    row_data = pd.DataFrame([row])
    row_data = pd.get_dummies(row_data)
    for feature in model.feature_names_in_:
        if feature not in row_data.columns:
            row_data[feature] = 0
    return row_data[model.feature_names_in_]

def encode_batch_data(batch_data, model):
    encoded = pd.get_dummies(batch_data)
    for feature in model.feature_names_in_:
        if feature not in encoded.columns:
            encoded[feature] = 0
    return encoded[model.feature_names_in_]

def get_required_base_columns(model):
    categorical_prefixes = ["Coating_Type", "Coating_Condition", "Cathodic_Protection", "Material_Grade"]
    required = []
    for feature in model.feature_names_in_:
        matched = False
        for prefix in categorical_prefixes:
            if feature.startswith(prefix + "_"):
                if prefix not in required:
                    required.append(prefix)
                matched = True
                break
        if not matched:
            required.append(feature)
    return required

# ---------------- DASHBOARD ----------------

if page == "🏠 Dashboard":

    st.markdown(f"""
    <div class="professional-note">
    <b>Welcome back, {user_name}.</b><br>
    Corzynth turns your pipeline, geophysical, and protection data into clear engineering
    and financial decisions — corrosion prediction, risk, remaining life, inspection planning, and ROI.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Quick Start</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-value">🧪</div>
            <div class="kpi-label">Run a Prediction</div>
            <div class="kpi-desc">Predict corrosion rate and remaining life for one pipeline segment.</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="kpi-card risk-medium">
            <div class="kpi-value">⚙️</div>
            <div class="kpi-label">Analyze a Full Pipeline</div>
            <div class="kpi-desc">Upload a CSV and assess corrosion and risk for all segments.</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="kpi-card risk-high">
            <div class="kpi-value">🔔</div>
            <div class="kpi-label">Check Alerts</div>
            <div class="kpi-desc">See which segments need urgent inspection or mitigation.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Session Status</div>', unsafe_allow_html=True)

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    with col_s1:
        if "batch_results" in st.session_state:
            st.metric("Batch Segments", len(st.session_state["batch_results"]))
        else:
            st.metric("Batch Segments", "—")

    with col_s2:
        if "batch_results" in st.session_state:
            crit = (st.session_state["batch_results"]["Overall_Risk_Level"] == "Critical").sum()
            st.metric("Critical Segments", crit)
        else:
            st.metric("Critical Segments", "—")

    with col_s3:
        if "geo_results" in st.session_state:
            st.metric("Geo Segments", len(st.session_state["geo_results"]))
        else:
            st.metric("Geo Segments", "—")

    with col_s4:
        if "single_prediction" in st.session_state:
            st.metric("Last Prediction", f"{st.session_state['single_prediction']['corrosion_rate']:.3f} mm/yr")
        else:
            st.metric("Last Prediction", "—")

    st.markdown('<div class="section-title">Platform Modules</div>', unsafe_allow_html=True)

    modules = pd.DataFrame({
        "Module": [
            "🧪 Single Prediction", "⚙️ Batch Prediction", "🎯 Risk Intelligence",
            "📈 Corrosion Forecast", "🧠 Explainable AI", "🌍 Geophysical Analysis",
            "🔮 What-If Simulator", "📅 RBI Optimizer", "💰 Cost & ROI",
            "🛠️ Maintenance Planner", "📈 Trend Analyzer", "🗺️ Route Risk Map",
            "🔔 Alerts", "🛠️ Admin Panel"
        ],
        "What it does": [
            "Predict one segment's corrosion, life, and inhibitor dose",
            "Assess corrosion and risk for many segments at once",
            "Likelihood × consequence risk scoring and matrix",
            "Forecast wall-thickness loss and integrity horizon",
            "See which factors drive each prediction",
            "Soil corrosivity, heatmaps, AI hotspots, GIS mapping",
            "Test mitigation strategies before implementation",
            "Optimize inspection intervals and next dates",
            "Avoided loss, net savings, ROI, and payback",
            "Priority actions, due dates, and maintenance plan",
            "Detect increasing, stable, or decreasing corrosion",
            "Draw the pipeline route colored by risk",
            "Automatic critical/high-risk segment alerts",
            "Monitor users, data, and application status"
        ]
    })

    st.dataframe(modules, use_container_width=True, hide_index=True)
    # ---------------- SINGLE PREDICTION ----------------

elif page == "🧪 Single Prediction":

    model = load_corrosion_model()
    clear_report_figures()

    st.markdown('<div class="section-title">Single Pipeline Corrosion Prediction</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        H2S_ppm = st.number_input("H2S concentration (ppm)", min_value=0.0, value=50.0)
        CO2_percent = st.number_input("CO2 concentration (%)", min_value=0.0, value=3.5)
        Operating_Pressure_bar = st.number_input("Operating pressure (bar)", min_value=0.0, value=45.0)
        Operating_Temperature_C = st.number_input("Operating temperature (°C)", value=60.0)
        pH = st.number_input("Fluid pH", min_value=0.0, max_value=14.0, value=6.5)
        Chlorides_mg_L = st.number_input("Chlorides (mg/L)", min_value=0.0, value=15000.0)

    with col2:
        TDS_mg_L = st.number_input("Total dissolved solids (mg/L)", min_value=0.0, value=25000.0)
        Soil_Resistivity_ohm_m = st.number_input("Soil resistivity (ohm-m)", min_value=0.0, value=1200.0)
        Soil_pH = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=7.2)
        Moisture_Content_percent = st.number_input("Soil moisture (%)", min_value=0.0, max_value=100.0, value=18.0)
        Coating_Type = st.selectbox("Coating type", ["FBE", "3LPE", "Coal Tar", "Tape", "None"])
        Coating_Condition = st.selectbox("Coating condition", ["Excellent", "Good", "Fair", "Poor"])

    with col3:
        Cathodic_Protection = st.selectbox("Cathodic protection", ["Yes", "No"])
        CP_Potential_mV = st.number_input("CP potential (mV)", value=-950.0)
        Inhibitor_Type = st.selectbox("Inhibitor type",
            ["Imidazoline", "Amine-based", "Phosphate Ester", "Other", "None"])
        Residual_Inhibitor_ppm = st.number_input("Residual inhibitor (ppm)", min_value=0.0, value=25.0)
        Inhibitor_Dose_ppm = st.number_input("Current inhibitor dose (ppm)", min_value=0.0, value=25.0)
        Service_Years = st.number_input("Service years", min_value=0.0, value=10.0)
        Pipe_Wall_Thickness_mm = st.number_input("Pipe wall thickness (mm)", min_value=0.1, value=8.5)
        Material_Grade = st.selectbox("Material grade", ["API 5L X42", "API 5L X52", "API 5L X60", "API 5L X65"])
        Minimum_Thickness_mm = st.number_input("Minimum acceptable thickness (mm)", min_value=0.1, value=3.0)

    input_row = {
        "H2S_ppm": H2S_ppm, "CO2_percent": CO2_percent,
        "Operating_Pressure_bar": Operating_Pressure_bar,
        "Operating_Temperature_C": Operating_Temperature_C,
        "pH": pH, "Chlorides_mg_L": Chlorides_mg_L, "TDS_mg_L": TDS_mg_L,
        "Soil_Resistivity_ohm_m": Soil_Resistivity_ohm_m, "Soil_pH": Soil_pH,
        "Moisture_Content_percent": Moisture_Content_percent,
        "Coating_Type": Coating_Type, "Coating_Condition": Coating_Condition,
        "Cathodic_Protection": Cathodic_Protection, "CP_Potential_mV": CP_Potential_mV,
        "Inhibitor_Dose_ppm": Inhibitor_Dose_ppm, "Service_Years": Service_Years,
        "Pipe_Wall_Thickness_mm": Pipe_Wall_Thickness_mm, "Material_Grade": Material_Grade
    }

    if st.button("🔍 Predict Corrosion and Integrity", use_container_width=True):
        with st.spinner("Running AI corrosion prediction..."):
            corrosion_rate = model.predict(prepare_model_input(input_row, model))[0]

        corrosion_risk = classify_corrosion_risk(corrosion_rate)
        available_wall = Pipe_Wall_Thickness_mm - Minimum_Thickness_mm
        remaining_life = available_wall / corrosion_rate if corrosion_rate > 0 and available_wall > 0 else 0

        likelihood = calculate_likelihood_score(input_row, corrosion_rate)
        consequence = calculate_consequence_score(input_row)
        overall_score = likelihood * consequence
        overall_risk = classify_overall_risk(overall_score, corrosion_risk)

        st.session_state["single_prediction"] = {
            "input_row": input_row, "corrosion_rate": corrosion_rate,
            "corrosion_risk": corrosion_risk, "remaining_life": remaining_life,
            "likelihood": likelihood, "consequence": consequence,
            "overall_score": overall_score, "overall_risk": overall_risk
        }
        st.success("Corrosion prediction completed successfully.")

    if "single_prediction" in st.session_state:
        prediction = st.session_state["single_prediction"]

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Predicted Corrosion Rate", f"{prediction['corrosion_rate']:.4f} mm/year")
        with col_m2:
            st.metric("Corrosion Risk", prediction["corrosion_risk"])
        with col_m3:
            st.metric("Estimated Remaining Life", f"{prediction['remaining_life']:.1f} years")

        st.success(f"Overall integrity risk level: {prediction['overall_risk']}")
        st.info(get_recommended_action(prediction["overall_risk"], prediction["remaining_life"]))

        target_rate = st.number_input("Target corrosion rate (mm/year)", min_value=0.01, value=0.20)

        if st.button("💉 Optimize Inhibitor Dose", use_container_width=True):
            doses = list(range(0, 201, 5))
            results = []
            for dose in doses:
                adjusted_rate, efficiency = apply_inhibitor_efficiency(
                    prediction["corrosion_rate"], dose, Inhibitor_Type, Residual_Inhibitor_ppm
                )
                results.append({
                    "Inhibitor_Dose_ppm": dose,
                    "Inhibitor_Type": Inhibitor_Type,
                    "Residual_Inhibitor_ppm": Residual_Inhibitor_ppm,
                    "Inhibitor_Efficiency_percent": round(efficiency * 100, 1),
                    "Predicted_Corrosion_Rate_mm_year": round(adjusted_rate, 4)
                })

            optimization_df = pd.DataFrame(results)
            suitable = optimization_df[optimization_df["Predicted_Corrosion_Rate_mm_year"] <= target_rate]

            fig = px.line(optimization_df, x="Inhibitor_Dose_ppm", y="Predicted_Corrosion_Rate_mm_year",
                          markers=True, title=f"Inhibitor Dose Optimization ({Inhibitor_Type})")
            fig.add_hline(y=target_rate, line_dash="dash")
            st.plotly_chart(fig, use_container_width=True)
            add_report_figure("Inhibitor Dose Optimization", fig)

            if not suitable.empty:
                best = suitable.iloc[0]
                st.success(f"Recommended minimum inhibitor dose: {best['Inhibitor_Dose_ppm']:.0f} ppm ({Inhibitor_Type})")
                st.info(f"Predicted corrosion rate: {best['Predicted_Corrosion_Rate_mm_year']:.4f} mm/year")
                report_df = suitable
            else:
                best = optimization_df.loc[optimization_df["Predicted_Corrosion_Rate_mm_year"].idxmin()]
                st.warning("Target corrosion rate was not reached within 0–200 ppm.")
                st.error(f"Lowest predicted rate: {best['Predicted_Corrosion_Rate_mm_year']:.4f} mm/year at {best['Inhibitor_Dose_ppm']:.0f} ppm")
                report_df = optimization_df

            download_reports(
                report_df, "Corzynth_Inhibitor_Optimization_Report.csv",
                "Corzynth Inhibitor Optimization Report",
                f"Inhibitor dose optimization was performed for {Inhibitor_Type} with target corrosion rate {target_rate:.4f} mm/year.",
                ["Inhibitor dose was tested from 0 to 200 ppm.", "Efficiency was adjusted by inhibitor type and residual concentration."],
                ["Apply the recommended inhibitor dose.", "Monitor residual inhibitor levels after implementation."],
                "single_inhibitor"
            )

# ---------------- BATCH PREDICTION ----------------

elif page == "⚙️ Batch Prediction":

    model = load_corrosion_model()
    clear_report_figures()

    st.markdown('<div class="section-title">Batch Pipeline Corrosion Prediction</div>', unsafe_allow_html=True)

    st.markdown("#### 📥 Need a data template?")
    template = pd.DataFrame({
        "Segment_ID": ["SEG-001", "SEG-002"],
        "H2S_ppm": [50, 80],
        "CO2_percent": [3.5, 4.0],
        "Operating_Pressure_bar": [45, 50],
        "Operating_Temperature_C": [60, 65],
        "pH": [6.5, 6.2],
        "Chlorides_mg_L": [15000, 18000],
        "TDS_mg_L": [25000, 28000],
        "Soil_Resistivity_ohm_m": [1200, 800],
        "Soil_pH": [7.2, 6.8],
        "Moisture_Content_percent": [18, 25],
        "Coating_Type": ["FBE", "3LPE"],
        "Coating_Condition": ["Good", "Fair"],
        "Cathodic_Protection": ["Yes", "No"],
        "CP_Potential_mV": [-950, -850],
        "Inhibitor_Dose_ppm": [25, 30],
        "Service_Years": [10, 15],
        "Pipe_Wall_Thickness_mm": [8.5, 8.5],
        "Material_Grade": ["API 5L X52", "API 5L X52"]
    })
    st.download_button(
        "📥 Download Batch Data Template (CSV)",
        template.to_csv(index=False).encode("utf-8"),
        "Corzynth_Batch_Data_Template.csv",
        "text/csv"
    )

    uploaded_batch = st.file_uploader("Upload Pipeline Batch CSV or Excel", type=["csv", "xlsx"])

    if uploaded_batch is not None:
        batch_data = load_uploaded_csv(uploaded_batch)
        st.dataframe(batch_data.head(), use_container_width=True)

        required = get_required_base_columns(model)
        missing = [column for column in required if column not in batch_data.columns]

        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.stop()

        batch_encoded = encode_batch_data(batch_data, model)

        wall_thickness = st.number_input("Pipe wall thickness (mm)", min_value=0.1, value=8.5)
        minimum_thickness = st.number_input("Minimum acceptable thickness (mm)", min_value=0.1, value=3.0)

        if st.button("🔍 Run Batch Prediction and Risk Assessment", use_container_width=True):
            results = []
            for index in range(len(batch_data)):
                model_input = batch_encoded.iloc[[index]]
                rate = model.predict(model_input)[0]
                corrosion_risk = classify_corrosion_risk(rate)

                available_wall = wall_thickness - minimum_thickness
                life = available_wall / rate if rate > 0 and available_wall > 0 else 0

                original = batch_data.iloc[index]
                likelihood = calculate_likelihood_score(original, rate)
                consequence = calculate_consequence_score(original)
                overall_score = likelihood * consequence
                overall_risk = classify_overall_risk(overall_score, corrosion_risk)

                results.append({
                    "Segment_ID": original.get("Segment_ID", index + 1),
                    "Predicted_Corrosion_Rate_mm_year": round(rate, 4),
                    "Corrosion_Risk": corrosion_risk,
                    "Estimated_Remaining_Life_years": round(life, 1),
                    "Likelihood_Score": round(likelihood, 3),
                    "Consequence_Score": round(consequence, 3),
                    "Overall_Risk_Score": round(overall_score, 3),
                    "Overall_Risk_Level": overall_risk,
                    "Inspection_Priority": get_priority(overall_risk),
                    "Recommended_Action": get_recommended_action(overall_risk, life)
                })

            st.session_state["batch_results"] = pd.DataFrame(results)
            st.success("Batch prediction completed successfully.")

        if "batch_results" in st.session_state:
            results_df = st.session_state["batch_results"]

            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            with col_b1:
                st.metric("Total Segments", len(results_df))
            with col_b2:
                st.metric("Critical Segments", (results_df["Overall_Risk_Level"] == "Critical").sum())
            with col_b3:
                st.metric("High-Risk Segments", (results_df["Overall_Risk_Level"] == "High").sum())
            with col_b4:
                st.metric("Average Remaining Life", f"{results_df['Estimated_Remaining_Life_years'].mean():.1f} years")

            st.dataframe(results_df, use_container_width=True)

            download_reports(
                results_df, "Corzynth_Batch_Prediction_Report.csv",
                "Corzynth Batch Prediction and Risk Assessment Report",
                f"AI corrosion predictions and risk assessments were completed for {len(results_df)} pipeline segments.",
                ["Corrosion rate was predicted using the trained machine-learning model.", "Risk was calculated using likelihood, consequence, and corrosion-risk adjustment."],
                ["Prioritize Critical and High-risk segments.", "Review coating and cathodic-protection performance."],
                "batch_prediction"
            )

# ---------------- RISK INTELLIGENCE ----------------

elif page == "🎯 Risk Intelligence":

    clear_report_figures()

    if "batch_results" not in st.session_state:
        st.warning("Please run a Batch Prediction first.")
        st.stop()

    results_df = st.session_state["batch_results"]
    st.markdown('<div class="section-title">Likelihood × Consequence Risk Intelligence</div>', unsafe_allow_html=True)

    risk_counts = results_df["Overall_Risk_Level"].value_counts()

    fig_risk = px.bar(
        x=risk_counts.index, y=risk_counts.values, color=risk_counts.index,
        color_discrete_map={"Low": "#2ea043", "Medium": "#d29922", "High": "#cf222e", "Critical": "#8b0000"},
        title="Pipeline Risk Distribution"
    )
    st.plotly_chart(fig_risk, use_container_width=True)
    add_report_figure("Pipeline Risk Distribution", fig_risk)

    fig_matrix = px.scatter(
        results_df, x="Likelihood_Score", y="Consequence_Score",
        color="Overall_Risk_Level", hover_name="Segment_ID",
        size="Overall_Risk_Score", title="Likelihood vs Consequence Risk Matrix"
    )
    st.plotly_chart(fig_matrix, use_container_width=True)
    add_report_figure("Likelihood vs Consequence Risk Matrix", fig_matrix)

    st.dataframe(results_df.sort_values(by="Overall_Risk_Score", ascending=False).head(10), use_container_width=True)

    download_reports(
        results_df, "Corzynth_Risk_Intelligence_Report.csv",
        "Corzynth Risk Intelligence Report",
        f"Risk intelligence was completed for {len(results_df)} pipeline segments.",
        ["Risk was calculated as likelihood multiplied by consequence.", "Corrosion-risk adjustment was applied."],
        ["Prioritize Critical and High-risk segments.", "Use the risk matrix for inspection planning."],
        "risk_intelligence"
    )

# ---------------- CORROSION FORECAST ----------------

elif page == "📈 Corrosion Forecast":

    clear_report_figures()

    if "batch_results" not in st.session_state:
        st.warning("Please run a Batch Prediction first.")
        st.stop()

    results_df = st.session_state["batch_results"].copy()
    results_df["Segment_ID"] = results_df["Segment_ID"].astype(str)

    segment_id = st.selectbox("Select Pipeline Segment", results_df["Segment_ID"].tolist())
    segment = results_df[results_df["Segment_ID"] == segment_id].iloc[0]

    corrosion_rate = float(segment["Predicted_Corrosion_Rate_mm_year"])
    wall_thickness = 8.5
    minimum_thickness = 3.0
    years = list(range(0, 21))

    forecast_df = pd.DataFrame({
        "Year": [datetime.now().year + year for year in years],
        "Predicted_Wall_Thickness_mm": [
            max(wall_thickness - corrosion_rate * year, minimum_thickness) for year in years
        ]
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_df["Year"], y=forecast_df["Predicted_Wall_Thickness_mm"],
        mode="lines+markers", name="Predicted Wall Thickness"
    ))
    fig.add_hline(y=minimum_thickness, line_dash="dash", line_color="red")
    fig.update_layout(title=f"Corrosion Growth Forecast - Segment {segment_id}")

    st.plotly_chart(fig, use_container_width=True)
    add_report_figure(f"Corrosion Growth Forecast - Segment {segment_id}", fig)
    st.dataframe(forecast_df, use_container_width=True)

    download_reports(
        forecast_df, f"Corzynth_Corrosion_Forecast_{segment_id}.csv",
        f"Corzynth Corrosion Forecast Report - Segment {segment_id}",
        f"Corrosion growth forecast was generated for Segment {segment_id}.",
        ["Wall thickness was forecast using the predicted corrosion rate.", "Minimum allowable thickness was used as the integrity limit."],
        ["Prioritize inspection if remaining life is low.", "Update forecast after next inspection."],
        "forecast"
    )

# ---------------- EXPLAINABLE AI ----------------

elif page == "🧠 Explainable AI":

    model = load_corrosion_model()
    clear_report_figures()

    st.markdown('<div class="section-title">Explainable AI - Corrosion Risk Drivers</div>', unsafe_allow_html=True)

    try:
        importance_df = pd.DataFrame({
            "Feature": model.feature_names_in_,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        fig = px.bar(importance_df.head(12), x="Importance", y="Feature",
                     orientation="h", title="Top 12 Features Influencing Corrosion Prediction")
        fig.update_layout(yaxis={"autorange": "reversed"})
        st.plotly_chart(fig, use_container_width=True)
        add_report_figure("Top Features Influencing Corrosion Prediction", fig)
    except Exception:
        st.warning("This model does not expose feature importance directly.")

    if "batch_results" in st.session_state:
        results_df = st.session_state["batch_results"].copy()
        results_df["Segment_ID"] = results_df["Segment_ID"].astype(str)

        segment_id = st.selectbox("Select Segment to Explain", results_df["Segment_ID"].tolist())
        segment = results_df[results_df["Segment_ID"] == segment_id].iloc[0]

        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            st.metric("Predicted Corrosion Rate", f"{segment['Predicted_Corrosion_Rate_mm_year']:.4f} mm/year")
        with col_e2:
            st.metric("Overall Risk Level", segment["Overall_Risk_Level"])
        with col_e3:
            st.metric("Remaining Life", f"{segment['Estimated_Remaining_Life_years']:.1f} years")
        st.info(segment["Recommended_Action"])

        download_reports(
            pd.DataFrame([segment]), f"Corzynth_Explainable_AI_Report_{segment_id}.csv",
            f"Corzynth Explainable AI Report - Segment {segment_id}",
            f"Explainable AI analysis was completed for Segment {segment_id}.",
            ["Feature importance and risk contribution were analyzed.", "Segment-specific recommendations were generated."],
            [segment["Recommended_Action"], "Validate explanation with field inspection data."],
            "explainable_ai"
        )
# ---------------- GEOPHYSICAL ANALYSIS ----------------

elif page == "🌍 Geophysical Analysis":

    clear_report_figures()

    st.markdown('<div class="section-title">Detailed Geophysical Analysis and GIS Risk Mapping</div>', unsafe_allow_html=True)

    uploaded_geo = st.file_uploader("Upload Geophysical CSV or Excel", type=["csv", "xlsx"])

    if uploaded_geo is not None:
        geo = load_uploaded_csv(uploaded_geo)

        required = ["Segment_ID", "Soil_Type", "Soil_Resistivity_ohm_m", "Soil_pH", "Moisture_Content_percent", "Groundwater_Depth_m"]
        missing = [column for column in required if column not in geo.columns]

        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.stop()

        geo["Geophysical_Risk_Score"] = (
            0.40 * np.minimum(2, 1000 / geo["Soil_Resistivity_ohm_m"].clip(lower=1)) +
            0.30 * np.clip((7.5 - geo["Soil_pH"]) / 2, 0, 2) +
            0.20 * np.clip(geo["Moisture_Content_percent"] / 40, 0, 2) +
            0.10 * np.clip(20 / geo["Groundwater_Depth_m"].clip(lower=1), 0, 2)
        )

        geo["Geophysical_Risk_Level"] = pd.cut(
            geo["Geophysical_Risk_Score"], bins=[0, 0.80, 1.10, 10],
            labels=["Low", "Medium", "High"]
        ).astype(str)

        st.session_state["geo_results"] = geo.copy()

        col_g1, col_g2, col_g3, col_g4 = st.columns(4)
        with col_g1:
            st.metric("Total Segments", len(geo))
        with col_g2:
            st.metric("High-Risk Segments", (geo["Geophysical_Risk_Level"] == "High").sum())
        with col_g3:
            st.metric("Medium-Risk Segments", (geo["Geophysical_Risk_Level"] == "Medium").sum())
        with col_g4:
            st.metric("Low-Risk Segments", (geo["Geophysical_Risk_Level"] == "Low").sum())

        risk_colors = {"Low": "#2ea043", "Medium": "#d29922", "High": "#cf222e"}

        fig_resistivity = px.bar(
            geo.sort_values("Soil_Resistivity_ohm_m"), x="Segment_ID",
            y="Soil_Resistivity_ohm_m", color="Geophysical_Risk_Level",
            color_discrete_map=risk_colors, title="Soil Resistivity by Segment"
        )
        st.plotly_chart(fig_resistivity, use_container_width=True)
        add_report_figure("Soil Resistivity by Segment", fig_resistivity)

        fig_risk_ranking = px.bar(
            geo.sort_values("Geophysical_Risk_Score", ascending=False), x="Segment_ID",
            y="Geophysical_Risk_Score", color="Geophysical_Risk_Level",
            color_discrete_map=risk_colors, title="Pipeline Segments Ranked by Geophysical Risk Score"
        )
        st.plotly_chart(fig_risk_ranking, use_container_width=True)
        add_report_figure("Geophysical Risk Score Ranking", fig_risk_ranking)

        st.markdown("#### 🔥 Geophysical Corrosion Heatmap")

        heatmap_data = geo.set_index("Segment_ID")[
            ["Soil_Resistivity_ohm_m", "Soil_pH", "Moisture_Content_percent", "Groundwater_Depth_m", "Geophysical_Risk_Score"]
        ]
        normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min() + 1e-9)
        normalized["Soil_Resistivity_ohm_m"] = 1 - normalized["Soil_Resistivity_ohm_m"]
        normalized["Groundwater_Depth_m"] = 1 - normalized["Groundwater_Depth_m"]

        fig_heatmap = px.imshow(normalized, aspect="auto", color_continuous_scale="RdYlGn_r",
                                title="Normalized Geophysical Corrosion Risk Heatmap")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        add_report_figure("Geophysical Corrosion Risk Heatmap", fig_heatmap)

        st.markdown("#### 🧠 AI Geophysical Hotspot Detection")

        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans

        hotspot_features = ["Soil_Resistivity_ohm_m", "Soil_pH", "Moisture_Content_percent", "Groundwater_Depth_m"]
        scaled = StandardScaler().fit_transform(geo[hotspot_features])
        geo["AI_Risk_Zone"] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(scaled)

        zone_risk = geo.groupby("AI_Risk_Zone")["Geophysical_Risk_Score"].mean()
        highest_zone = zone_risk.idxmax()
        zone_names = {}
        for zone in sorted(geo["AI_Risk_Zone"].unique()):
            if zone == highest_zone:
                zone_names[zone] = "Critical Hotspot Zone"
            elif zone_risk[zone] >= zone_risk.median():
                zone_names[zone] = "Moderate Risk Zone"
            else:
                zone_names[zone] = "Lower Risk Zone"

        geo["AI_Risk_Zone_Name"] = geo["AI_Risk_Zone"].map(zone_names)
        st.session_state["geo_results"] = geo.copy()

        fig_3d = px.scatter_3d(
            geo, x="Soil_Resistivity_ohm_m", y="Moisture_Content_percent",
            z="Geophysical_Risk_Score", color="AI_Risk_Zone_Name",
            hover_name="Segment_ID", title="3D AI Geophysical Corrosion Risk Map"
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        add_report_figure("3D AI Geophysical Corrosion Risk Map", fig_3d)

        st.markdown("#### 🚨 Critical Geophysical Hotspots")

        threshold = geo["Geophysical_Risk_Score"].quantile(0.80)
        hotspots = geo[geo["Geophysical_Risk_Score"] >= threshold]

        st.dataframe(hotspots[[
            "Segment_ID", "Soil_Type", "Soil_Resistivity_ohm_m", "Soil_pH",
            "Moisture_Content_percent", "Groundwater_Depth_m",
            "Geophysical_Risk_Score", "Geophysical_Risk_Level", "AI_Risk_Zone_Name"
        ]], use_container_width=True)

        st.markdown("#### 🗺️ Geographic Risk Map")

        lat_col = next((column for column in geo.columns if column in ["Latitude", "latitude", "Lat"]), None)
        lon_col = next((column for column in geo.columns if column in ["Longitude", "longitude", "Long", "LON"]), None)

        if lat_col and lon_col:
            geo_map = folium.Map(
                location=[geo[lat_col].mean(), geo[lon_col].mean()],
                zoom_start=11
            )
            colors = {"Low": "green", "Medium": "orange", "High": "red"}

            for _, row in geo.iterrows():
                folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]],
                    radius=9, color=colors.get(row["Geophysical_Risk_Level"], "blue"),
                    fill=True, fill_opacity=0.8,
                    popup=f"Segment {row['Segment_ID']} - {row['Geophysical_Risk_Level']} Risk"
                ).add_to(geo_map)

            st_folium(geo_map, height=600, use_container_width=True)
        else:
            st.warning("Latitude and Longitude columns were not found.")

        download_reports(
            geo, "Corzynth_Detailed_Geophysical_Assessment.csv",
            "Corzynth Detailed Geophysical Assessment Report",
            f"Detailed geophysical corrosion-risk assessment was completed for {len(geo)} pipeline segments.",
            ["Low soil resistivity indicates higher corrosivity.", "Low pH, high moisture, and shallow groundwater increase corrosion risk."],
            ["Prioritize high-risk geophysical segments.", "Review coating and cathodic-protection performance."],
            "geophysical"
        )

# ---------------- WHAT-IF SIMULATOR ----------------

elif page == "🔮 What-If Scenario Simulator":

    clear_report_figures()
    model = load_corrosion_model()

    st.markdown('<div class="section-title">What-If Corrosion Scenario Simulator</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        base_h2s = st.number_input("Current H2S (ppm)", min_value=0.0, value=50.0)
        base_co2 = st.number_input("Current CO2 (%)", min_value=0.0, value=3.5)
        base_temp = st.number_input("Current temperature (°C)", value=60.0)
        base_pressure = st.number_input("Current pressure (bar)", min_value=0.0, value=45.0)

    with col2:
        new_h2s = st.number_input("Proposed H2S (ppm)", min_value=0.0, value=50.0)
        new_co2 = st.number_input("Proposed CO2 (%)", min_value=0.0, value=3.5)
        new_temp = st.number_input("Proposed temperature (°C)", value=60.0)
        new_pressure = st.number_input("Proposed pressure (bar)", min_value=0.0, value=45.0)

    with col3:
        current_inhibitor_type = st.selectbox("Current inhibitor type",
            ["Imidazoline", "Amine-based", "Phosphate Ester", "Other", "None"])
        current_inhibitor = st.number_input("Current inhibitor dose (ppm)", min_value=0.0, value=10.0)
        current_residual = st.number_input("Current residual inhibitor (ppm)", min_value=0.0, value=10.0)
        proposed_inhibitor_type = st.selectbox("Proposed inhibitor type",
            ["Imidazoline", "Amine-based", "Phosphate Ester", "Other", "None"])
        proposed_inhibitor = st.number_input("Proposed inhibitor dose (ppm)", min_value=0.0, value=30.0)
        proposed_residual = st.number_input("Proposed residual inhibitor (ppm)", min_value=0.0, value=30.0)
        current_coating = st.selectbox("Current coating condition", ["Poor", "Fair", "Good", "Excellent"], index=1)
        proposed_coating = st.selectbox("Proposed coating condition", ["Poor", "Fair", "Good", "Excellent"], index=2)
        cp_status = st.selectbox("Cathodic protection", ["Yes", "No"])

    wall_thickness = st.number_input("Pipe wall thickness (mm)", min_value=0.1, value=8.5)
    minimum_thickness = st.number_input("Minimum acceptable thickness (mm)", min_value=0.1, value=3.0)

    if st.button("🔮 Run What-If Simulation", use_container_width=True):
        base_input = {
            "H2S_ppm": base_h2s, "CO2_percent": base_co2,
            "Operating_Pressure_bar": base_pressure, "Operating_Temperature_C": base_temp,
            "pH": 6.5, "Chlorides_mg_L": 15000, "TDS_mg_L": 25000,
            "Soil_Resistivity_ohm_m": 1200, "Soil_pH": 7.2, "Moisture_Content_percent": 18,
            "Coating_Type": "FBE", "Coating_Condition": current_coating,
            "Cathodic_Protection": cp_status, "CP_Potential_mV": -950,
            "Inhibitor_Dose_ppm": current_inhibitor, "Service_Years": 10,
            "Pipe_Wall_Thickness_mm": wall_thickness, "Material_Grade": "API 5L X52"
        }

        new_input = base_input.copy()
        new_input.update({
            "H2S_ppm": new_h2s, "CO2_percent": new_co2,
            "Operating_Temperature_C": new_temp, "Operating_Pressure_bar": new_pressure,
            "Coating_Condition": proposed_coating, "Inhibitor_Dose_ppm": proposed_inhibitor
        })

        base_rate = model.predict(prepare_model_input(base_input, model))[0]
        new_rate = model.predict(prepare_model_input(new_input, model))[0]

        base_rate, _ = apply_inhibitor_efficiency(base_rate, current_inhibitor, current_inhibitor_type, current_residual)
        new_rate, _ = apply_inhibitor_efficiency(new_rate, proposed_inhibitor, proposed_inhibitor_type, proposed_residual)

        available_wall = wall_thickness - minimum_thickness
        base_life = available_wall / base_rate if base_rate > 0 else 0
        new_life = available_wall / new_rate if new_rate > 0 else 0
        reduction = ((base_rate - new_rate) / base_rate * 100) if base_rate > 0 else 0

        comparison_df = pd.DataFrame({
            "Scenario": ["Current", "Proposed"],
            "Corrosion_Rate_mm_year": [base_rate, new_rate],
            "Remaining_Life_years": [base_life, new_life]
        })

        fig = px.bar(comparison_df, x="Scenario", y="Corrosion_Rate_mm_year",
                     color="Scenario", title="Current vs Proposed Corrosion Rate")
        st.plotly_chart(fig, use_container_width=True)
        add_report_figure("Current vs Proposed Corrosion Rate", fig)

        if new_rate < base_rate:
            st.success(f"Corrosion reduced by {reduction:.1f}%. Remaining life improved from {base_life:.1f} to {new_life:.1f} years.")
        else:
            st.warning("The proposed strategy does not improve corrosion performance.")

        download_reports(
            comparison_df, "Corzynth_What_If_Scenario_Report.csv",
            "Corzynth What-If Corrosion Scenario Report",
            f"Current corrosion rate is {base_rate:.4f} mm/year. Proposed strategy reduces it to {new_rate:.4f} mm/year.",
            ["Current and proposed conditions were compared.", "Inhibitor type, dose, and residual concentration were applied to both scenarios."],
            ["Implement the proposed strategy if improvement is acceptable.", "Validate with field corrosion monitoring."],
            "what_if"
        )

# ---------------- RBI OPTIMIZER ----------------

elif page == "📅 RBI Interval Optimizer":

    clear_report_figures()

    if "batch_results" not in st.session_state:
        st.warning("Please run a Batch Prediction first.")
        st.stop()

    results_df = st.session_state["batch_results"].copy()

    st.markdown('<div class="section-title">Risk-Based Inspection Interval Optimizer</div>', unsafe_allow_html=True)

    safety_factor = st.slider("Safety factor", 1.0, 3.0, 1.5, 0.1)
    maximum_interval = st.number_input("Maximum inspection interval (years)", 1, 10, 5)

    results_df["Base_Interval_years"] = results_df["Estimated_Remaining_Life_years"] / safety_factor
    risk_cap = {"Critical": 0.5, "High": 1.0, "Medium": 3.0, "Low": maximum_interval}
    results_df["Risk_Capped_Interval_years"] = results_df["Overall_Risk_Level"].map(risk_cap)
    results_df["Recommended_Inspection_Interval_years"] = results_df[
        ["Base_Interval_years", "Risk_Capped_Interval_years"]
    ].min(axis=1).clip(lower=0.25, upper=maximum_interval)

    results_df["Next_Inspection_Date"] = results_df["Recommended_Inspection_Interval_years"].apply(
        lambda years: (datetime.now() + timedelta(days=int(years * 365))).strftime("%Y-%m-%d")
    )

    fig = px.bar(
        results_df.sort_values("Recommended_Inspection_Interval_years"),
        x="Segment_ID", y="Recommended_Inspection_Interval_years",
        color="Overall_Risk_Level", title="Recommended Inspection Interval by Segment"
    )
    st.plotly_chart(fig, use_container_width=True)
    add_report_figure("Recommended Inspection Interval by Segment", fig)

    st.dataframe(results_df[[
        "Segment_ID", "Overall_Risk_Level", "Estimated_Remaining_Life_years",
        "Recommended_Inspection_Interval_years", "Next_Inspection_Date"
    ]], use_container_width=True)

    download_reports(
        results_df, "Corzynth_RBI_Inspection_Interval_Plan.csv",
        "Corzynth Risk-Based Inspection Interval Plan",
        "RBI inspection intervals were optimized using corrosion rate, remaining life, and risk level.",
        ["Intervals were calculated from remaining life divided by safety factor.", "Risk-based caps prevent long intervals for high-risk segments."],
        ["Use recommended intervals for inspection planning.", "Prioritize Critical and High-risk segments."],
        "rbi"
    )

# ---------------- COST AND ROI ----------------

elif page == "💰 Corrosion Cost & ROI":

    clear_report_figures()

    st.markdown('<div class="section-title">Corrosion Cost, Savings, and ROI Calculator</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        base_rate = st.number_input("Current corrosion rate (mm/year)", min_value=0.001, value=0.3067, format="%.4f")
        optimized_rate = st.number_input("Corrosion rate after mitigation (mm/year)", min_value=0.001, value=0.1963, format="%.4f")
        pipeline_length_km = st.number_input("Pipeline length (km)", min_value=0.1, value=50.0)

    with col2:
        failure_cost = st.number_input("Cost of one major failure (USD)", min_value=0.0, value=2500000.0)
        annual_failure_probability = st.number_input("Annual failure probability without mitigation (%)", 0.0, 100.0, 5.0)
        mitigation_cost = st.number_input("Annual mitigation cost (USD)", min_value=0.0, value=120000.0)

    with col3:
        mitigation_type = st.selectbox("Mitigation strategy", ["Inhibitor optimization", "Coating repair", "Cathodic protection improvement", "Combined integrity program"])
        project_years = st.number_input("Evaluation period (years)", 1, 30, 10)

    if st.button("💰 Calculate Corrosion Savings and ROI", use_container_width=True):
        reduction = base_rate - optimized_rate
        reduction_percent = reduction / base_rate * 100 if base_rate > 0 else 0

        avoided_probability = annual_failure_probability / 100 * (reduction / base_rate) if base_rate > 0 else 0
        annual_avoided_loss = avoided_probability * failure_cost
        annual_net_saving = annual_avoided_loss - mitigation_cost
        total_net_saving = annual_net_saving * project_years
        total_cost = mitigation_cost * project_years
        roi = total_net_saving / total_cost * 100 if total_cost > 0 else 0
        payback = mitigation_cost / annual_net_saving if annual_net_saving > 0 else float("inf")

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Corrosion Reduction", f"{reduction_percent:.1f}%")
        with col_s2:
            st.metric("Annual Avoided Loss", f"${annual_avoided_loss:,.0f}")
        with col_s3:
            st.metric("Annual Net Saving", f"${annual_net_saving:,.0f}")

        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.metric("ROI", f"{roi:.1f}%")
        with col_r2:
            st.metric("Total Net Saving", f"${total_net_saving:,.0f}")
        with col_r3:
            st.metric("Payback Period", f"{payback:.1f} years" if payback != float("inf") else "Not achieved")

        years = list(range(1, int(project_years) + 1))
        roi_df = pd.DataFrame({
            "Year": years,
            "Cumulative_Net_Saving_USD": [annual_net_saving * year for year in years],
            "Cumulative_Mitigation_Cost_USD": [mitigation_cost * year for year in years]
        })

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roi_df["Year"], y=roi_df["Cumulative_Net_Saving_USD"], mode="lines+markers", name="Cumulative Net Saving"))
        fig.add_trace(go.Scatter(x=roi_df["Year"], y=roi_df["Cumulative_Mitigation_Cost_USD"], mode="lines+markers", name="Cumulative Mitigation Cost"))
        fig.update_layout(title="Corrosion Mitigation Investment vs Cumulative Savings")

        st.plotly_chart(fig, use_container_width=True)
        add_report_figure("Corrosion Mitigation Investment vs Cumulative Savings", fig)

        roi_report_df = pd.DataFrame([{
            "Mitigation_Strategy": mitigation_type,
            "Base_Corrosion_Rate_mm_year": base_rate,
            "Optimized_Corrosion_Rate_mm_year": optimized_rate,
            "Corrosion_Reduction_percent": round(reduction_percent, 2),
            "Annual_Avoided_Loss_USD": round(annual_avoided_loss, 0),
            "Annual_Mitigation_Cost_USD": round(mitigation_cost, 0),
            "Annual_Net_Saving_USD": round(annual_net_saving, 0),
            "Total_Net_Saving_USD": round(total_net_saving, 0),
            "ROI_percent": round(roi, 1),
            "Payback_Period_years": round(payback, 2) if payback != float("inf") else "Not achieved"
        }])

        download_reports(
            roi_report_df, "Corzynth_Corrosion_ROI_Report.csv",
            "Corzynth Corrosion Cost, Savings, and ROI Report",
            f"Corrosion mitigation reduces corrosion by {reduction_percent:.1f}%. Estimated annual net saving is ${annual_net_saving:,.0f}.",
            ["Avoided loss was estimated from reduced failure probability and failure cost.", "ROI was calculated over the evaluation period."],
            ["Use this analysis to support integrity-budget approval.", "Validate assumptions with historical data."],
            "roi"
        )

# ---------------- MAINTENANCE PLANNER ----------------

elif page == "🛠️ Maintenance Planner":

    if "batch_results" not in st.session_state:
        st.warning("Please run a Batch Prediction first.")
        st.stop()

    results_df = st.session_state["batch_results"]

    maintenance_df = results_df[[
        "Segment_ID", "Overall_Risk_Level", "Inspection_Priority",
        "Estimated_Remaining_Life_years", "Recommended_Action"
    ]].copy()

    due_days = {"Critical": 7, "High": 30, "Planned": 90, "Routine": 365}
    maintenance_df["Due_Date"] = maintenance_df["Inspection_Priority"].map(due_days).fillna(365).astype(int).apply(
        lambda days: (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
    )
    maintenance_df["Status"] = "Open"

    st.markdown('<div class="section-title">Risk-Based Maintenance Planner</div>', unsafe_allow_html=True)
    st.dataframe(maintenance_df, use_container_width=True)

    download_reports(
        maintenance_df, "Corzynth_Maintenance_Plan.csv",
        "Corzynth Risk-Based Maintenance Plan",
        f"Risk-based maintenance plan was generated for {len(maintenance_df)} pipeline segments.",
        ["Critical segments require immediate action.", "High-risk segments require priority monitoring."],
        ["Complete Critical actions within 7 days.", "Complete High-priority actions within 30 days."],
        "maintenance"
    )

# ---------------- MODEL PERFORMANCE ----------------

elif page == "📊 Model Performance":

    model = load_corrosion_model()
    clear_report_figures()

    st.markdown('<div class="section-title">Machine Learning Model Performance</div>', unsafe_allow_html=True)

    uploaded_test = st.file_uploader("Upload Model Test Dataset", type=["csv", "xlsx"])

    if uploaded_test is not None:
        test_data = load_uploaded_csv(uploaded_test)

        if "Actual_Corrosion_Rate_mm_year" not in test_data.columns:
            st.error("CSV must contain Actual_Corrosion_Rate_mm_year column.")
            st.stop()

        actual = test_data["Actual_Corrosion_Rate_mm_year"]
        predictions = model.predict(encode_batch_data(test_data, model))

        mae = np.mean(np.abs(predictions - actual))
        rmse = np.sqrt(np.mean((predictions - actual) ** 2))
        ss_res = np.sum((actual - predictions) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{r2:.3f}")
        with col2:
            st.metric("RMSE", f"{rmse:.4f}")
        with col3:
            st.metric("MAE", f"{mae:.4f}")

        performance_df = pd.DataFrame({
            "Actual": actual, "Predicted": predictions,
            "Absolute_Error": np.abs(predictions - actual)
        })

        fig = px.scatter(performance_df, x="Actual", y="Predicted", title="Actual vs Predicted Corrosion Rate")
        st.plotly_chart(fig, use_container_width=True)
        add_report_figure("Actual vs Predicted Corrosion Rate", fig)

        download_reports(
            pd.DataFrame([{"R2_Score": round(r2, 3), "RMSE": round(rmse, 4), "MAE": round(mae, 4), "Test_Samples": len(test_data)}]),
            "Corzynth_Model_Performance_Report.csv",
            "Corzynth Model Performance Report",
            f"Model performance was evaluated using {len(test_data)} test samples.",
            ["R² measures explained variation.", "RMSE and MAE measure prediction error."],
            ["Evaluate using larger historical datasets.", "Retrain when new data become available."],
            "model_performance"
        )

# ---------------- CORROSION TREND ANALYZER ----------------

elif page == "📈 Corrosion Trend Analyzer":

    clear_report_figures()

    st.markdown('<div class="section-title">Corrosion Trend Analyzer</div>', unsafe_allow_html=True)

    uploaded_trend = st.file_uploader("Upload Historical Corrosion Trend CSV or Excel", type=["csv", "xlsx"])

    if uploaded_trend is not None:
        trend_data = load_uploaded_csv(uploaded_trend)

        required = ["Segment_ID", "Inspection_Date", "Corrosion_Rate_mm_year"]
        missing = [column for column in required if column not in trend_data.columns]

        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.stop()

        trend_data["Inspection_Date"] = pd.to_datetime(trend_data["Inspection_Date"], errors="coerce")
        trend_data = trend_data.dropna(subset=["Inspection_Date", "Corrosion_Rate_mm_year"])
        trend_data = trend_data.sort_values(by=["Segment_ID", "Inspection_Date"])

        segment_ids = trend_data["Segment_ID"].astype(str).unique().tolist()
        selected_segment = st.selectbox("Select Pipeline Segment", segment_ids)

        segment_trend = trend_data[trend_data["Segment_ID"].astype(str) == selected_segment]

        if len(segment_trend) < 2:
            st.warning("At least two historical readings are required.")
            st.stop()

        x = segment_trend["Inspection_Date"].map(datetime.toordinal)
        y = segment_trend["Corrosion_Rate_mm_year"]

        slope = np.polyfit(x, y, 1)[0]
        first_rate = y.iloc[0]
        last_rate = y.iloc[-1]
        change = (last_rate - first_rate) / first_rate * 100 if first_rate > 0 else 0

        if slope > 0.01:
            trend_status = "Increasing"
            trend_color = "#cf222e"
        elif slope < -0.01:
            trend_status = "Decreasing"
            trend_color = "#2ea043"
        else:
            trend_status = "Stable"
            trend_color = "#d29922"

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("First Rate", f"{first_rate:.4f} mm/year")
        with col2:
            st.metric("Latest Rate", f"{last_rate:.4f} mm/year")
        with col3:
            st.metric("Change", f"{change:.1f}%")
        with col4:
            st.metric("Trend", trend_status)

        fig = px.line(segment_trend, x="Inspection_Date", y="Corrosion_Rate_mm_year",
                      markers=True, title=f"Corrosion Rate Trend - Segment {selected_segment}")
        fig.update_traces(line=dict(color=trend_color, width=3))
        st.plotly_chart(fig, use_container_width=True)
        add_report_figure(f"Corrosion Rate Trend - Segment {selected_segment}", fig)

        trend_summary = []
        for segment in segment_ids:
            data = trend_data[trend_data["Segment_ID"].astype(str) == segment].sort_values(by="Inspection_Date")
            if len(data) < 2:
                continue

            segment_x = data["Inspection_Date"].map(datetime.toordinal)
            segment_y = data["Corrosion_Rate_mm_year"]
            segment_slope = np.polyfit(segment_x, segment_y, 1)[0]
            segment_first = segment_y.iloc[0]
            segment_last = segment_y.iloc[-1]
            segment_change = (segment_last - segment_first) / segment_first * 100 if segment_first > 0 else 0

            if segment_slope > 0.01:
                status = "Increasing"
            elif segment_slope < -0.01:
                status = "Decreasing"
            else:
                status = "Stable"

            trend_summary.append({
                "Segment_ID": segment,
                "First_Corrosion_Rate_mm_year": round(segment_first, 4),
                "Latest_Corrosion_Rate_mm_year": round(segment_last, 4),
                "Change_percent": round(segment_change, 2),
                "Trend_Status": status
            })

        trend_summary_df = pd.DataFrame(trend_summary)
        st.dataframe(trend_summary_df, use_container_width=True)

        fig_summary = px.bar(
            trend_summary_df.sort_values(by="Change_percent", ascending=False),
            x="Segment_ID", y="Change_percent", color="Trend_Status",
            title="Corrosion Rate Change by Segment"
        )
        st.plotly_chart(fig_summary, use_container_width=True)
        add_report_figure("Corrosion Rate Change by Segment", fig_summary)

        download_reports(
            trend_summary_df, "Corzynth_Corrosion_Trend_Report.csv",
            "Corzynth Corrosion Trend Analysis Report",
            f"Historical corrosion trends were analyzed for {len(trend_summary_df)} pipeline segments.",
            ["Positive trend indicates increasing corrosion.", "Negative trend indicates improving conditions."],
            ["Prioritize increasing-corrosion segments.", "Review protection measures for deteriorating segments."],
            "corrosion_trend"
        )

# ---------------- PIPELINE ROUTE RISK MAP ----------------

elif page == "🗺️ Pipeline Route Risk Map":

    st.markdown('<div class="section-title">Pipeline Route Risk Map</div>', unsafe_allow_html=True)

    st.info(
        "Upload a pipeline route CSV containing Segment_ID, Latitude, and Longitude. "
        "Corzynth will draw the pipeline route and color it by corrosion or geophysical risk."
    )

    uploaded_route = st.file_uploader("Upload Pipeline Route CSV or Excel", type=["csv", "xlsx"], key="route_upload")

    if uploaded_route is not None:

        route_data = load_uploaded_csv(uploaded_route)

        required_route_columns = ["Segment_ID", "Latitude", "Longitude"]
        missing_route_columns = [column for column in required_route_columns if column not in route_data.columns]

        if missing_route_columns:
            st.error(f"Missing required columns: {', '.join(missing_route_columns)}")
            st.stop()

        route_data["Segment_ID"] = route_data["Segment_ID"].astype(str)
        route_data["Latitude"] = pd.to_numeric(route_data["Latitude"], errors="coerce")
        route_data["Longitude"] = pd.to_numeric(route_data["Longitude"], errors="coerce")
        route_data = route_data.dropna(subset=["Latitude", "Longitude"])
        route_data = route_data.sort_values(by="Segment_ID")

        risk_source = st.radio(
            "Select Risk Source",
            [
                "Use Batch Prediction Results",
                "Use Geophysical Analysis Results",
                "Use Manual Risk Level"
            ]
        )

        route_map_data = route_data.copy()

        if risk_source == "Use Batch Prediction Results":

            if "batch_results" not in st.session_state:
                st.warning("Please run Batch Prediction first.")
                st.stop()

            batch_results = st.session_state["batch_results"].copy()
            batch_results["Segment_ID"] = batch_results["Segment_ID"].astype(str)

            route_map_data = route_map_data.merge(
                batch_results[["Segment_ID", "Overall_Risk_Level"]],
                on="Segment_ID", how="left"
            )
            risk_column = "Overall_Risk_Level"

        elif risk_source == "Use Geophysical Analysis Results":

            if "geo_results" not in st.session_state:
                st.warning("Please run Geophysical Analysis first.")
                st.stop()

            geo_results = st.session_state["geo_results"].copy()
            geo_results["Segment_ID"] = geo_results["Segment_ID"].astype(str)

            route_map_data = route_map_data.merge(
                geo_results[["Segment_ID", "Geophysical_Risk_Level"]],
                on="Segment_ID", how="left"
            )
            risk_column = "Geophysical_Risk_Level"

        else:
            route_map_data["Manual_Risk_Level"] = "Medium"
            risk_column = "Manual_Risk_Level"

        if risk_column not in route_map_data.columns:
            route_map_data[risk_column] = "Medium"

        route_map_data[risk_column] = route_map_data[risk_column].fillna("Medium").astype(str)

        route_map_data["Latitude_Next"] = route_map_data["Latitude"].shift(-1)
        route_map_data["Longitude_Next"] = route_map_data["Longitude"].shift(-1)
        route_map_data = route_map_data.dropna(subset=["Latitude_Next", "Longitude_Next"])

        route_risk_colors = {
            "Low": "#2ea043", "Medium": "#d29922", "High": "#cf222e",
            "Critical": "#8b0000", "Unknown": "#64748b"
        }

        route_map = folium.Map(
            location=[route_data["Latitude"].mean(), route_data["Longitude"].mean()],
            zoom_start=11, tiles="OpenStreetMap"
        )

        for _, row in route_map_data.iterrows():
            risk_level = str(row[risk_column])
            line_color = route_risk_colors.get(risk_level, "#64748b")

            popup_text = f"<b>Segment ID:</b> {row['Segment_ID']}<br><b>Risk Level:</b> {risk_level}<br>"

            if "Predicted_Corrosion_Rate_mm_year" in row:
                popup_text += f"<b>Corrosion Rate:</b> {row['Predicted_Corrosion_Rate_mm_year']:.4f} mm/year<br>"
            if "Estimated_Remaining_Life_years" in row:
                popup_text += f"<b>Remaining Life:</b> {row['Estimated_Remaining_Life_years']:.1f} years<br>"
            if "Geophysical_Risk_Score" in row:
                popup_text += f"<b>Geophysical Risk Score:</b> {row['Geophysical_Risk_Score']:.3f}<br>"

            folium.PolyLine(
                locations=[
                    [row["Latitude"], row["Longitude"]],
                    [row["Latitude_Next"], row["Longitude_Next"]]
                ],
                color=line_color, weight=7, opacity=0.85,
                popup=folium.Popup(popup_text, max_width=320),
                tooltip=f"Segment {row['Segment_ID']} - {risk_level} Risk"
            ).add_to(route_map)

        for _, row in route_map_data.iterrows():
            risk_level = str(row[risk_column])
            marker_color = route_risk_colors.get(risk_level, "blue")

            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=6, color=marker_color, fill=True,
                fill_color=marker_color, fill_opacity=0.9,
                tooltip=f"Segment {row['Segment_ID']} - {risk_level} Risk"
            ).add_to(route_map)

        st_folium(route_map, height=650, use_container_width=True)

        st.success("Pipeline route risk map generated successfully.")

# ---------------- ALERTS AND NOTIFICATIONS ----------------

elif page == "🔔 Alerts & Notifications":

    st.markdown('<div class="section-title">Corzynth Alerts & Notifications</div>', unsafe_allow_html=True)

    alerts = []

    if "batch_results" in st.session_state:
        batch_results = st.session_state["batch_results"]

        for _, row in batch_results.iterrows():
            if row["Overall_Risk_Level"] == "Critical":
                alerts.append({
                    "Severity": "🔴 Critical", "Segment_ID": row["Segment_ID"],
                    "Alert": "Critical overall risk detected",
                    "Recommended_Action": "Immediate inspection and repair planning",
                    "Due_Date": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
                })

            if row["Estimated_Remaining_Life_years"] <= 5:
                alerts.append({
                    "Severity": "🟠 High", "Segment_ID": row["Segment_ID"],
                    "Alert": f"Low remaining life: {row['Estimated_Remaining_Life_years']:.1f} years",
                    "Recommended_Action": "Prioritize inspection and corrosion monitoring",
                    "Due_Date": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
                })

            if row["Predicted_Corrosion_Rate_mm_year"] >= 0.30:
                alerts.append({
                    "Severity": "🟠 High", "Segment_ID": row["Segment_ID"],
                    "Alert": f"High corrosion rate: {row['Predicted_Corrosion_Rate_mm_year']:.4f} mm/year",
                    "Recommended_Action": "Review inhibitor, coating, and cathodic protection",
                    "Due_Date": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
                })

    if "geo_results" in st.session_state:
        geo_results = st.session_state["geo_results"]
        threshold = geo_results["Geophysical_Risk_Score"].quantile(0.80)
        hotspots = geo_results[geo_results["Geophysical_Risk_Score"] >= threshold]

        for _, row in hotspots.iterrows():
            alerts.append({
                "Severity": "🟠 High", "Segment_ID": row["Segment_ID"],
                "Alert": "Critical geophysical corrosion hotspot",
                "Recommended_Action": "Prioritize coating and cathodic-protection review",
                "Due_Date": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
            })

    if not alerts:
        st.success("No critical alerts were detected. Continue routine monitoring.")
    else:
        alerts_df = pd.DataFrame(alerts)
        severity_order = {"🔴 Critical": 0, "🟠 High": 1, "🟡 Medium": 2}
        alerts_df["Severity_Order"] = alerts_df["Severity"].map(severity_order)
        alerts_df = alerts_df.sort_values(by=["Severity_Order", "Segment_ID"]).drop(columns=["Severity_Order"])

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.metric("Critical Alerts", (alerts_df["Severity"] == "🔴 Critical").sum())
        with col_a2:
            st.metric("High Alerts", (alerts_df["Severity"] == "🟠 High").sum())

        st.dataframe(alerts_df, use_container_width=True)

        download_reports(
            alerts_df, "Corzynth_Alerts_Report.csv",
            "Corzynth Alerts and Notifications Report",
            f"{len(alerts_df)} alerts were generated from corrosion predictions, remaining life, risk levels, and geophysical analysis.",
            ["Critical alerts require immediate attention.", "High alerts should be addressed within 30 days."],
            ["Address Critical alerts within 7 days.", "Address High alerts within 30 days."],
            "alerts"
        )

# ---------------- ADMIN PANEL ----------------

elif page == "🛠️ Admin Panel":

    if user_role != "admin":
        st.error("Access denied. This module is available only to administrators.")
        st.stop()

    st.markdown('<div class="section-title">Corzynth Admin Panel</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Logged-in User", user_name)
    with col2:
        st.metric("User Role", user_role.title())
    with col3:
        st.metric("Application Version", APP_VERSION)

    st.markdown('<div class="section-title">Application Status</div>', unsafe_allow_html=True)

    try:
        load_corrosion_model()
        st.success("AI model loaded successfully.")
    except Exception:
        st.error("AI model could not be loaded.")

    if "batch_results" in st.session_state:
        st.info(f"Batch results available for {len(st.session_state['batch_results'])} segments.")
    else:
        st.warning("No batch prediction results available.")

    if "geo_results" in st.session_state:
        st.info(f"Geophysical results available for {len(st.session_state['geo_results'])} segments.")
    else:
        st.warning("No geophysical analysis results available.")

    st.markdown('<div class="section-title">User Management</div>', unsafe_allow_html=True)
    st.success("You are the sole administrator of Corzynth. All modules, users, and settings are under your control.")

    st.markdown('<div class="section-title">System Maintenance</div>', unsafe_allow_html=True)

    if st.button("🧹 Clear Session Analysis Results", use_container_width=True):
        for key in ["batch_results", "geo_results", "single_prediction", "inhibitor_optimization_done", "report_figures"]:
            if key in st.session_state:
                del st.session_state[key]
        st.success("Session analysis results cleared successfully.")
        st.rerun()

# ---------------- USER GUIDE ----------------

elif page == "📖 User Guide":

    st.markdown('<div class="section-title">Corzynth User Guide</div>', unsafe_allow_html=True)

    st.write("""
    ### 🧪 Single Prediction
    Predict corrosion, remaining life, overall risk, and optimize inhibitor dose by inhibitor type and residual concentration.

    ### ⚙️ Batch Prediction
    Upload a CSV or Excel file with many pipeline segments and predict corrosion and risk for all of them. Download the data template first.

    ### 🎯 Risk Intelligence
    Review likelihood, consequence, overall risk, highest-risk segments, and the risk matrix.

    ### 📈 Corrosion Forecast
    Forecast future wall-thickness loss and remaining life.

    ### 🧠 Explainable AI
    Understand global and segment-specific corrosion-risk drivers.

    ### 🌍 Geophysical Analysis
    Analyze soil corrosivity, moisture, pH, groundwater, heatmaps, AI hotspots, and GIS risk.

    ### 🔮 What-If Scenario Simulator
    Compare current and proposed inhibitor type, dose, residual level, coating, pressure, temperature, and protection strategies.

    ### 📅 RBI Interval Optimizer
    Calculate optimized inspection intervals and next inspection dates.

    ### 💰 Corrosion Cost & ROI
    Calculate avoided loss, net savings, ROI, and payback period.

    ### 🛠️ Maintenance Planner
    Convert risk results into inspection priorities, due dates, and actions.

    ### 📊 Model Performance
    Evaluate model accuracy using actual corrosion-rate data.

    ### 📈 Corrosion Trend Analyzer
    Detect whether corrosion is increasing, stable, or decreasing over time.

    ### 🗺️ Pipeline Route Risk Map
    Draw the pipeline route and color it by corrosion or geophysical risk.

    ### 🔔 Alerts & Notifications
    Automatically identify segments requiring urgent attention.

    ### 🛠️ Admin Panel
    Monitor application status, users, modules, and session data.

    ### ⚠️ Important Note
    Corzynth is a decision-support and screening tool. Final pipeline-integrity decisions must be reviewed by qualified engineering professionals.
    """)

    st.markdown("### 💬 Feedback")
    feedback = st.text_area("Share your feedback or suggestions for Corzynth:")

    if st.button("Submit Feedback"):
        if feedback:
            st.success("Thank you for your feedback!")
        else:
            st.warning("Please enter your feedback before submitting.")

# ---------------- FOOTER ----------------

st.markdown(f"""
<div class="footer">
    <b>{APP_VERSION}</b> • AI Pipeline Corrosion Intelligence Platform<br>
    Developed by {DEVELOPER} • Authorized access only • © {datetime.now().year}
</div>
""", unsafe_allow_html=True)