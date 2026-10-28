# ============================================================
#  CORZYNTH — AI PIPELINE CORROSION INTELLIGENCE PLATFORM
#  Premium Edition v9.5 | Developed by E. Vellay
#  Model: de Waard-Milliams CO2 | Standard: NACE SP0775-2023
#  Units: mm/yr | psi | °F | Logo: corzynth_logo.jpeg
# ============================================================

import os
import io
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import streamlit_authenticator as stauth

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Corzynth | AI Pipeline Corrosion Intelligence",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_VERSION = "Corzynth v9.5 Premium"
DEVELOPER = "E. Vellay"
LOGO_FILE = "corzynth_logo.jpeg"

# ------------------------------------------------------------
# THEME
# ------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1rem; padding-bottom: 3rem; max-width: 1500px; }
    .main-header {
        background: linear-gradient(135deg, #06263d 0%, #0b3d62 55%, #10557f 100%);
        border-radius: 24px; padding: 32px 40px; color: white; margin-bottom: 24px;
    }
    .main-header h1 { margin: 0; font-size: 2.2rem; font-weight: 800; }
    .main-header p { margin: 8px 0 0 0; opacity: 0.88; }
    .hero-badge {
        display: inline-block; background: rgba(24,169,153,0.2);
        border: 1px solid rgba(24,169,153,0.5); color: #7ee2d3;
        padding: 5px 14px; border-radius: 999px; font-size: 0.78rem;
        font-weight: 600; margin-bottom: 12px;
    }
    .section-title { color: #0b3d62; font-size: 1.2rem; font-weight: 750; margin: 24px 0 12px 0; }
    .explain-box {
        background: #f0fdf9; border: 1px solid #99f6e4; border-radius: 14px;
        padding: 16px 20px; margin: 12px 0;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #06263d 0%, #0b3d62 100%);
    }
    section[data-testid="stSidebar"] * { color: #e6edf3 !important; }
    .footer {
        text-align: center; color: #64748b; font-size: 0.8rem;
        margin-top: 40px; padding-top: 16px; border-top: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# TOP BAR — shown ONCE on every page (logo + name + tagline)
# ------------------------------------------------------------
def render_top_bar():
    col_logo, col_text = st.columns([0.3, 3.7])
    with col_logo:
        if os.path.exists(LOGO_FILE):
            st.image(LOGO_FILE, width=90)
    with col_text:
        st.markdown("""
        <div style="display:flex; flex-direction:column; justify-content:center; height:100%;">
            <h1 style="color:#0b3d62; font-size:1.7rem; font-weight:800; margin:0; line-height:1.1;">Corzynth</h1>
            <p style="color:#18a999; font-size:0.9rem; margin:2px 0 0 0;">
                AI Pipeline Corrosion Intelligence · Predict · Assess · Optimize</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# AUTHENTICATION
# ------------------------------------------------------------
try:
    authenticator = stauth.Authenticate(
        st.secrets["credentials"].to_dict(),
        st.secrets["cookie"]["name"],
        st.secrets["cookie"]["key"],
        st.secrets["cookie"]["expiry_days"],
    )
except Exception as e:
    st.error(f"Authentication configuration error: {e}")
    st.info("Check .streamlit/secrets.toml has [cookie] and [credentials] sections.")
    st.stop()

# Show top bar ONCE (before login AND after login)
render_top_bar()

# Login form — centered
col_l, col_c, col_r = st.columns([1, 1.4, 1])
with col_c:
    authenticator.login(location="main", key="Login")

if not st.session_state.get("authentication_status"):
    if st.session_state.get("authentication_status") is False:
        st.error("❌ Incorrect username or password.")
    st.info("🔐 Enter your credentials above to access Corzynth.")
    st.markdown(f"""
    <div class="footer">
        <b>{APP_VERSION}</b> · Developed by <b>{DEVELOPER}</b> ·
        Access restricted to authorized users only.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

username = st.session_state["username"]
name = st.session_state["name"]
user_role = st.secrets["credentials"]["usernames"][username].get("role", "engineer")

authenticator.logout("Logout", "sidebar")
st.sidebar.success(f"✅ Logged in: {username}")
st.sidebar.info(f"👤 Role: {user_role.title()}")
st.sidebar.caption(APP_VERSION)

# ------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("corrosion_model.joblib")
    except Exception:
        return None

model = load_model()

# ------------------------------------------------------------
# ACCURATE ENGINEERING FUNCTIONS
# ------------------------------------------------------------
COATING_FACTOR = {"Excellent": 0.5, "Good": 0.7, "Fair": 1.0, "Poor": 1.5}
COATING_TYPE_FACTOR = {"FBE": 0.85, "3LPE": 0.80, "Coal Tar": 1.0, "Tape": 1.1, "None": 1.5}
PRODUCT_FACTOR = {"Crude Oil": 0.7, "Natural Gas": 1.0, "Refined Product": 0.85, "Water": 0.3}
SENSITIVITY_FACTOR = {"Low": 0.0, "Medium": 0.5, "High": 1.0}

INHIBITOR_TYPES = {
    "Imidazoline (Film-Forming)": {
        "max_eff": 0.92, "persistence_days": 30,
        "best_for": "CO2 sweet corrosion, oil-water systems"},
    "Quaternary Amine (Film-Forming)": {
        "max_eff": 0.90, "persistence_days": 28,
        "best_for": "CO2 + H2S mixed service"},
    "Fatty Acid Amide (Batch)": {
        "max_eff": 0.95, "persistence_days": 90,
        "best_for": "Long-term batch protection of gas lines"},
    "Neutralizing Amine (pH Control)": {
        "max_eff": 0.60, "persistence_days": 7,
        "best_for": "Acid gas / condensate pH control"},
    "Water-Soluble Polyamine": {
        "max_eff": 0.85, "persistence_days": 21,
        "best_for": "Water-dominated production lines"},
    "Vapor-Phase Inhibitor (VPI)": {
        "max_eff": 0.70, "persistence_days": 10,
        "best_for": "Dry gas lines, top-of-line protection"},
}

def nace_classify(rate):
    """NACE SP0775-2023: Low <0.025, Moderate 0.025-0.12, High 0.12-0.25, Severe >0.25 mm/yr."""
    if rate < 0.025: return "Low"
    if rate < 0.12:  return "Moderate"
    if rate < 0.25:  return "High"
    return "Severe"

def de_waard_milliams_rate(temp_c, pressure_bar, co2_percent):
    """De Waard-Milliams (1991) CO2 corrosion rate (mm/yr).
    log(V) = 5.8 - 1710/T + 0.67*log10(pCO2)"""
    if co2_percent <= 0 or pressure_bar <= 0:
        return 0.01
    p_co2 = pressure_bar * (co2_percent / 100.0)
    T_K = temp_c + 273.15
    if T_K <= 0 or p_co2 <= 0:
        return 0.01
    log_v = 5.8 - (1710 / T_K) + 0.67 * math.log10(p_co2)
    return max(10 ** log_v, 0.01)

def inhibitor_efficiency(dose, inhibitor_type):
    info = INHIBITOR_TYPES.get(inhibitor_type, {"max_eff": 0.85})
    max_eff = info["max_eff"]
    if dose <= 0:
        return 0.0
    if dose < 25:
        return dose / 25 * max_eff * 0.55
    if dose < 50:
        return max_eff * 0.55 + (dose - 25) / 25 * max_eff * 0.25
    if dose < 100:
        return max_eff * 0.80 + (dose - 50) / 50 * max_eff * 0.15
    return min(max_eff * 0.95 + (dose - 100) / 100 * max_eff * 0.05, max_eff)

def remaining_inhibitor_life(dose, inhibitor_type, water_cut=50):
    if dose <= 0:
        return 0
    persistence = INHIBITOR_TYPES.get(inhibitor_type, {"persistence_days": 21})["persistence_days"]
    life_days = persistence * (dose / 50) * (1 - water_cut / 200)
    return max(round(life_days, 1), 0.5)

def predict_rate(row):
    """Accurate corrosion rate (mm/yr): de Waard-Milliams + corrections."""
    temp_c = row.get("Operating_Temperature_C", 60)
    pressure_bar = row.get("Operating_Pressure_bar", 45)
    co2_pct = row.get("CO2_percent", 3.5)
    h2s_ppm = row.get("H2S_ppm", 0)
    ph = row.get("pH", 6.5)

    base_rate = de_waard_milliams_rate(temp_c, pressure_bar, co2_pct)

    if ph >= 6.5:
        base_rate *= 0.5
    elif ph >= 5.5:
        base_rate *= 0.75

    if h2s_ppm > 0:
        base_rate *= 0.85 if h2s_ppm < 100 else 1.15

    base_rate *= COATING_FACTOR.get(row.get("Coating_Condition", "Good"), 1.0)
    base_rate *= COATING_TYPE_FACTOR.get(row.get("Coating_Type", "FBE"), 1.0)

    if str(row.get("Cathodic_Protection", "Yes")).lower() == "yes":
        base_rate *= 0.6

    dose = row.get("Inhibitor_Dose_ppm", 0)
    inh_type = row.get("Inhibitor_Type", "Imidazoline (Film-Forming)")
    if dose > 0:
        base_rate *= (1 - inhibitor_efficiency(dose, inh_type))

    return max(round(base_rate, 4), 0.001)

def classify_corrosion_risk(rate):
    return nace_classify(rate)

def likelihood_score(row, rate):
    score = min(rate / 0.5, 1.0) * 0.40
    if "Soil_Resistivity_ohm_m" in row:
        score += min(1000 / max(row["Soil_Resistivity_ohm_m"], 1), 2) / 2 * 0.15
    if "Soil_pH" in row:
        score += max(0, (7.5 - row["Soil_pH"]) / 4) * 0.10
    if "Moisture_Content_percent" in row:
        score += min(row["Moisture_Content_percent"] / 50, 1) * 0.10
    score += COATING_FACTOR.get(row.get("Coating_Condition", "Good"), 1.0) / 1.5 * 0.15
    if str(row.get("Cathodic_Protection", "Yes")).lower() == "no":
        score += 0.05
    return min(score, 1.0)

def consequence_score(row):
    score = 0.30
    if "Operating_Pressure_psi" in row:
        score += min(row["Operating_Pressure_psi"] / 1450, 1) * 0.25
    score += PRODUCT_FACTOR.get(row.get("Product_Type", "Crude Oil"), 0.6) * 0.20
    score += SENSITIVITY_FACTOR.get(row.get("Environmental_Sensitivity", "Medium"), 0.5) * 0.10
    return min(score, 1.0)

def overall_risk(likelihood, consequence, corrosion_risk):
    score = likelihood * consequence
    adj = {"Low": 0.0, "Moderate": 0.15, "High": 0.35, "Severe": 0.5}.get(corrosion_risk, 0.25)
    score = min(score + adj, 1.0)
    if score < 0.30: return "Low", score
    if score < 0.50: return "Medium", score
    if score < 0.70: return "High", score
    return "Critical", score

def get_action(risk, life):
    if risk == "Critical":
        return "Immediate ILI inspection, CP review, and repair planning."
    if risk == "High":
        return "Priority inspection and corrosion monitoring."
    if risk == "Medium":
        return "Scheduled inspection and inhibitor review."
    return "Routine monitoring."

def get_priority(risk):
    return {"Low": "Routine", "Medium": "Planned", "High": "High", "Critical": "Critical"}.get(risk, "Routine")

def explain_prediction(row, rate, risk, life):
    reasons = []
    co2_pct = row.get("CO2_percent", 0)
    pressure_bar = row.get("Operating_Pressure_bar", 0)
    temp_c = row.get("Operating_Temperature_C", 0)
    p_co2 = pressure_bar * (co2_pct / 100)
    reasons.append(f"CO2 partial pressure = {p_co2:.2f} bar — main driver of sweet corrosion.")
    reasons.append(f"Temperature = {temp_c:.0f}°C — accelerates CO2 corrosion kinetics.")
    if row.get("H2S_ppm", 0) > 0:
        reasons.append(f"H2S = {row['H2S_ppm']:.0f} ppm — sour service; check NACE MR0175.")
    if row.get("pH", 7) >= 6.5:
        reasons.append("pH ≥ 6.5 promotes protective carbonate scale.")
    inh = row.get("Inhibitor_Dose_ppm", 0)
    if inh > 0:
        eff = inhibitor_efficiency(inh, row.get("Inhibitor_Type", ""))
        reasons.append(f"Inhibitor ({row.get('Inhibitor_Type','')}) at {inh:.0f} ppm = {eff*100:.0f}% efficiency.")
    else:
        reasons.append("No inhibitor — full uninhibited rate applies.")
    return reasons

def download_csv_button(df, filename, key, label="📥 Download CSV Report"):
    csv = df.to_csv(index=False).encode()
    st.download_button(label, csv, filename, "text/csv", use_container_width=True, key=key)

def download_excel_button(df, filename, key, sheet_name="Results"):
    try:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
        st.download_button("📥 Download Excel Report (.xlsx)", buffer.getvalue(),
                           filename,
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True, key=key)
    except Exception:
        st.caption("Install openpyxl: pip install openpyxl")

# ------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------
menu = [
    "Dashboard", "Single Prediction", "Monte Carlo", "Batch Prediction",
    "Corrosion Forecast", "Risk Intelligence", "What-If Simulator",
    "RBI Optimizer", "Corrosion Cost ROI", "Maintenance Planner",
    "Trend Analyzer", "Geophysical Analysis", "Alerts",
    "Model Performance", "Help & Glossary",
]
if user_role == "admin":
    menu.append("Admin Panel")

page = st.sidebar.radio("📌 Select Module", menu)

# ------------------------------------------------------------
# DASHBOARD (CLEAN)
# ------------------------------------------------------------
if page == "Dashboard":
    st.markdown(f"""
    <div class="main-header">
        <div class="hero-badge">PREMIUM PIPELINE INTEGRITY PLATFORM</div>
        <h1>Welcome, {name}</h1>
        <p>Accurate CO2 corrosion prediction · NACE SP0775 classification ·
        Inhibitor optimization · Risk-based inspection</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ML Model", "Active" if model else "Physics Model")
    c2.metric("Standard", "NACE SP0775-2023")
    c3.metric("Corrosion Model", "De Waard-Milliams")
    c4.metric("Version", "v9.5 Premium")

    st.markdown("")

    st.markdown("### Start Here")
    a1, a2, a3 = st.columns(3)
    with a1:
        with st.container(border=True):
            st.markdown("#### 🔮 New Prediction")
            st.caption("Corrosion rate (mm/yr), remaining life, and optimal inhibitor dose.")
    with a2:
        with st.container(border=True):
            st.markdown("#### 📥 Batch Analysis")
            st.caption("Download template, fill data, analyze hundreds of segments.")
    with a3:
        with st.container(border=True):
            st.markdown("#### 📊 View Results")
            st.caption("Risk matrix, forecasts, inspection plans, and alerts.")

    st.markdown("### Recommended Workflow")
    st.markdown("""
    | Step | Module | What You Get |
    |---|---|---|
    | 1 | Single Prediction | Corrosion rate + inhibitor dose |
    | 2 | Monte Carlo | Uncertainty range (P90 life) |
    | 3 | Batch Prediction | All segments ranked by risk |
    | 4 | RBI Optimizer | Inspection dates |
    | 5 | Cost ROI | Budget justification |
    """)

    st.info("💡 **NACE SP0775-2023:** Low <0.025 · Moderate 0.025–0.12 · "
            "High 0.12–0.25 · Severe >0.25 mm/yr.")

# ------------------------------------------------------------
# SINGLE PREDICTION
# ------------------------------------------------------------
elif page == "Single Prediction":
    st.markdown('<div class="main-header"><div class="hero-badge">SINGLE SEGMENT ANALYSIS</div>'
                '<h1>🔮 Corrosion Prediction & Inhibitor Optimization</h1>'
                '<p>mm/year · psi · °F · de Waard-Milliams · NACE SP0775-2023</p></div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🛢️ Operating Conditions**")
        h2s = st.number_input("H2S (ppm)", 0.0, 10000.0, 50.0)
        co2 = st.number_input("CO2 (%)", 0.0, 100.0, 3.5)
        pressure_psi = st.number_input("Operating Pressure (psi)", 0.0, 5000.0, 650.0)
        temp_f = st.number_input("Operating Temperature (°F)", -50.0, 400.0, 140.0)
        ph = st.number_input("Fluid pH", 0.0, 14.0, 6.5)
        chlorides = st.number_input("Chlorides (mg/L)", 0.0, 100000.0, 15000.0)
    with c2:
        st.markdown("**🌍 Soil / Environment**")
        soil_res = st.number_input("Soil Resistivity (ohm-m)", 1.0, 10000.0, 1200.0)
        soil_ph = st.number_input("Soil pH", 0.0, 14.0, 7.2)
        moisture = st.number_input("Soil Moisture (%)", 0.0, 100.0, 18.0)
        coating_type = st.selectbox("Coating Type", ["FBE", "3LPE", "Coal Tar", "Tape", "None"])
        coating_cond = st.selectbox("Coating Condition", ["Excellent", "Good", "Fair", "Poor"])
        product = st.selectbox("Product Type", ["Crude Oil", "Natural Gas", "Refined Product", "Water"])
    with c3:
        st.markdown("**🧪 Inhibitor & Protection**")
        inhibitor_type = st.selectbox("Inhibitor Type", list(INHIBITOR_TYPES.keys()))
        st.caption(f"Best for: {INHIBITOR_TYPES[inhibitor_type]['best_for']}")
        inhibitor = st.number_input("Current Inhibitor Dose (ppm)", 0.0, 500.0, 25.0)
        residual_inhibitor = st.number_input(
            "Current Residual Inhibitor in Pipeline (ppm)", 0.0, 500.0, 15.0,
            help="How much inhibitor is actually present in the pipe right now. "
                 "This is what protects the pipe today — not the injected dose.")
        cp = st.selectbox("Cathodic Protection", ["Yes", "No"])
        cp_potential = st.number_input("CP Potential (mV)", -2000.0, 0.0, -950.0)
        service_years = st.number_input("Service Years", 0.0, 100.0, 10.0)
        wall_mm = st.number_input("Pipe Wall Thickness (mm)", 0.1, 50.0, 8.5)
        min_wall_mm = st.number_input("Minimum Acceptable Thickness (mm)", 0.1, 50.0, 3.0)
        material = st.selectbox("Material Grade", ["API 5L X42", "API 5L X52", "API 5L X60", "API 5L X65"])

    pressure_bar = pressure_psi * 0.0689476
    temp_c = (temp_f - 32) * 5 / 9

    input_row = {
        "H2S_ppm": h2s, "CO2_percent": co2, "Operating_Pressure_bar": pressure_bar,
        "Operating_Temperature_C": temp_c, "pH": ph, "Chlorides_mg_L": chlorides,
        "Soil_Resistivity_ohm_m": soil_res, "Soil_pH": soil_ph,
        "Moisture_Content_percent": moisture, "Coating_Type": coating_type,
        "Coating_Condition": coating_cond, "Cathodic_Protection": cp,
        "CP_Potential_mV": cp_potential, "Inhibitor_Dose_ppm": inhibitor,
        "Inhibitor_Type": inhibitor_type,
        "Service_Years": service_years, "Pipe_Wall_Thickness_mm": wall_mm,
        "Material_Grade": material, "Product_Type": product,
    }

    if st.button("🔮 Predict Corrosion & Integrity", use_container_width=True):
        with st.spinner("Running de Waard-Milliams prediction..."):
            rate = predict_rate(input_row)
            risk = classify_corrosion_risk(rate)
            available = wall_mm - min_wall_mm
            life = available / rate if rate > 0 else 0
            lik = likelihood_score(input_row, rate)
            con = consequence_score(input_row)
            overall, overall_score = overall_risk(lik, con, risk)
            inh_life = remaining_inhibitor_life(inhibitor, inhibitor_type)

            st.session_state["single_result"] = {
                "rate": rate, "risk": risk, "life": life,
                "wall": wall_mm, "min_wall": min_wall_mm,
                "likelihood": lik, "consequence": con,
                "overall": overall, "overall_score": overall_score,
                "inhibitor_type": inhibitor_type, "inhibitor_dose": inhibitor,
                "inhibitor_life": inh_life, "input": input_row,
            }
        st.success("✅ Prediction completed (de Waard-Milliams model).")

    if "single_result" in st.session_state:
        r = st.session_state["single_result"]
        st.markdown('<div class="section-title">📊 Prediction Results</div>', unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Corrosion Rate", f"{r['rate']:.4f} mm/yr")
        m2.metric("NACE Class", r["risk"])
        m3.metric("Remaining Life", f"{r['life']:.1f} years")
        m4.metric("Overall Risk", r["overall"])
        m5.metric("Inhibitor Life", f"{r['inhibitor_life']:.0f} days")

        st.info(f"**Recommended Action:** {get_action(r['overall'], r['life'])}")

        st.markdown('<div class="section-title">🧠 Why This Result?</div>', unsafe_allow_html=True)
        reasons = explain_prediction(r["input"], r["rate"], r["risk"], r["life"])
        with st.container():
            st.markdown('<div class="explain-box">', unsafe_allow_html=True)
            for reason in reasons:
                st.markdown(f"- {reason}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-title">🧪 Inhibitor Dose Optimization</div>',
                    unsafe_allow_html=True)
        target = st.number_input("Target Corrosion Rate (mm/yr)", 0.01, 2.0, 0.20, 0.01,
                                  key="target_rate")

        if st.button("🧪 Optimize Inhibitor Dose", use_container_width=True):
            doses = list(range(0, 505, 5))
            results = []
            for d in doses:
                eff = inhibitor_efficiency(d, r["inhibitor_type"])
                adj = r["rate"] * (1 - eff)
                life_d = remaining_inhibitor_life(d, r["inhibitor_type"])
                results.append({
                    "Inhibitor_Dose_ppm": d,
                    "Inhibitor_Type": r["inhibitor_type"],
                    "Efficiency_percent": round(eff * 100, 1),
                    "Predicted_Rate_mm_yr": round(adj, 4),
                    "Remaining_Inhibitor_Life_days": life_d,
                })
            opt_df = pd.DataFrame(results)
            suitable = opt_df[opt_df["Predicted_Rate_mm_yr"] <= target]

            fig = px.line(opt_df, x="Inhibitor_Dose_ppm", y="Predicted_Rate_mm_yr",
                          markers=True, title="Inhibitor Dose vs Corrosion Rate (mm/yr)")
            fig.add_hline(y=target, line_dash="dash", line_color="red",
                          annotation_text="Target")
            st.plotly_chart(fig, use_container_width=True)

            if not suitable.empty:
                best = suitable.iloc[0]
                st.success(f"✅ **Recommended dose: {best['Inhibitor_Dose_ppm']:.0f} ppm** "
                           f"({r['inhibitor_type']})")
                st.info(f"Rate: {best['Predicted_Rate_mm_yr']:.4f} mm/yr | "
                        f"Inhibitor life: {best['Remaining_Inhibitor_Life_days']:.0f} days")
            else:
                st.warning("⚠️ Target not reached within 0–500 ppm.")
                st.error("Consider coating repair, CP improvement, or different inhibitor type.")

            download_csv_button(opt_df, "Corzynth_Inhibitor_Optimization.csv", "opt_csv")
            download_excel_button(opt_df, "Corzynth_Inhibitor_Optimization.xlsx", "opt_xlsx")

            
                    # ------------------------------------------------------------
        # RESIDUAL INHIBITOR ANALYSIS
        # ------------------------------------------------------------
        st.markdown('<div class="section-title">🧪 Residual Inhibitor Analysis</div>',
                    unsafe_allow_html=True)

        inh_type = r["inhibitor_type"]
        inh_dose = r["inhibitor_dose"]
        persistence = INHIBITOR_TYPES[inh_type]["persistence_days"]

        # Water cut input (affects how fast inhibitor depletes)
        water_cut = st.slider("Water Cut (%)", 0, 100, 50, 5,
                               key="residual_water_cut",
                               help="Higher water cut = faster inhibitor depletion")

        # Calculate residual concentration decay over time
        # Exponential decay model: C(t) = C0 * exp(-t / half_life_adjusted)
        if inh_dose > 0:
            decay_rate = 1 / (persistence * (inh_dose / 50) * (1 - water_cut / 200))
            days = list(range(0, int(persistence * 2) + 1))
            residual = [inh_dose * math.exp(-d * decay_rate) for d in days]

            residual_df = pd.DataFrame({
                "Day": days,
                "Residual_Concentration_ppm": [round(c, 2) for c in residual],
            })

            # Effective threshold = 25% of initial dose
            threshold = inh_dose * 0.25
            redose_day = next((d for d, c in zip(days, residual) if c < threshold),
                              days[-1])

            # Metrics
            ri1, ri2, ri3, ri4 = st.columns(4)
            ri1.metric("Inhibitor Type", inh_type.split(" (")[0])
            ri2.metric("Initial Dose", f"{inh_dose:.0f} ppm")
            ri3.metric("Re-Dose After", f"{redose_day} days")
            ri4.metric("Effective Threshold", f"{threshold:.1f} ppm")

            # Decay chart
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(
                x=residual_df["Day"], y=residual_df["Residual_Concentration_ppm"],
                mode="lines+markers", name="Residual Concentration",
                line=dict(color="#18a999", width=3)))
            fig_res.add_hline(y=threshold, line_dash="dash", line_color="red",
                              annotation_text="Effective Threshold (25%)")
            fig_res.add_vline(x=redose_day, line_dash="dot", line_color="orange",
                              annotation_text=f"Re-Dose Day {redose_day}")
            fig_res.update_layout(
                title=f"Residual Inhibitor Decay — {inh_type}",
                xaxis_title="Days After Injection",
                yaxis_title="Residual Concentration (ppm)")
            st.plotly_chart(fig_res, use_container_width=True)

            # Re-dosing schedule table
            st.markdown('<div class="section-title">📅 Re-Dosing Schedule</div>',
                        unsafe_allow_html=True)
            redose_dates = []
            current_date = datetime.now()
            for cycle in range(1, 7):  # Next 6 re-dosing cycles
                next_date = current_date + timedelta(days=int(redose_day * cycle))
                redose_dates.append({
                    "Cycle": cycle,
                    "Re-Dose Date": next_date.strftime("%Y-%m-%d"),
                    "Dose (ppm)": f"{inh_dose:.0f}",
                    "Inhibitor Type": inh_type,
                })
            redose_df = pd.DataFrame(redose_dates)
            st.dataframe(redose_df, use_container_width=True)
            download_csv_button(redose_df, "Corzynth_Residual_Inhibitor_Schedule.csv",
                                "residual_csv",
                                label="📥 Download Re-Dosing Schedule (CSV)")

            # Explainable box
            st.markdown("""
            <div class="explain-box">
            - <b>Decay Model:</b> Residual concentration decays exponentially based on
              inhibitor persistence, initial dose, and water cut.<br>
            - <b>Effective Threshold:</b> When residual falls below 25% of the initial
              dose, protection becomes insufficient — re-dose at this point.<br>
            - <b>Water Cut Effect:</b> Higher water cut accelerates depletion
              (inhibitor partitions into the water phase).<br>
            - <b>Recommendation:</b> Re-dose every <b>""" + f"{redose_day} days" + """</b>
              to maintain continuous protection.
            </div>
            """, unsafe_allow_html=True)

            # Alert if re-dose is urgent
            if redose_day <= 7:
                st.error(f"🚨 **URGENT:** With {inh_dose:.0f} ppm {inh_type}, "
                         f"re-dosing is needed every {redose_day} days — "
                         f"consider a higher dose or a longer-persistence inhibitor.")
            elif redose_day <= 21:
                st.warning(f"⚠️ Re-dose every {redose_day} days with current settings.")
            else:
                st.success(f"✅ Good protection interval: re-dose every {redose_day} days.")
        else:
            st.warning("⚠️ No inhibitor dose entered — residual analysis not available. "
                       "Enter an inhibitor dose above to see the decay analysis.")
         
                # Download Report as Word (Charts + Tables)
        from docx import Document
        from docx.shared import Inches, Pt
        import io

        if st.button("📄 Download Word Report (Charts + Tables)", use_container_width=True):
            doc = Document()
            doc.add_heading('Corzynth – Pipeline Corrosion Report', 0)
            doc.add_paragraph('AI-Powered Corrosion Prediction & Inhibitor Optimization')

            # ================= TABLE 1: Corrosion Results =================
            doc.add_heading('1. Corrosion Results', level=1)
            table1 = doc.add_table(rows=6, cols=2)
            table1.style = 'Light Grid Accent 1'
            data1 = [
                ("Corrosion Rate", f"{r['rate']:.4f} mm/yr"),
                ("Remaining Life", f"{r['life']:.1f} years"),
                ("Risk Level", f"{r['risk']}"),
                ("Current Wall Thickness", f"{r['wall']:.2f} mm"),
                ("Minimum Required Wall", f"{r['min_wall']:.2f} mm"),
                ("Wall Loss Margin", f"{r['wall'] - r['min_wall']:.2f} mm"),
            ]
            for i, (k, v) in enumerate(data1):
                table1.rows[i].cells[0].text = k
                table1.rows[i].cells[1].text = v

            # ================= TABLE 2: Inhibitor & Re-dose =================
            doc.add_heading('2. Inhibitor & Re-dose Recommendation', level=1)
            table2 = doc.add_table(rows=5, cols=2)
            table2.style = 'Light Grid Accent 2'
            data2 = [
                ("Inhibitor Type", f"{inh_type}"),
                ("Injected Dose", f"{inh_dose:.0f} ppm"),
                ("Residual Inhibitor", f"{residual_inhibitor:.0f} ppm"),
                ("Re-dose Interval", f"{redose_day} days"),
                ("Recommendation", f"Re-dose every {redose_day} days"),
            ]
            for i, (k, v) in enumerate(data2):
                table2.rows[i].cells[0].text = k
                table2.rows[i].cells[1].text = v

            # ================= CHART 1: Dose vs Corrosion Rate =================
            doc.add_heading('3. Corrosion Rate vs Inhibitor Dose', level=1)
            doses = list(range(0, 105, 5))
            rates_plot = [r['rate'] * (1 - min(d / 200, 0.8)) for d in doses]
            fig1 = px.line(x=doses, y=rates_plot,
                           labels={'x': 'Inhibitor Dose (ppm)', 'y': 'Corrosion Rate (mm/yr)'},
                           title='Corrosion Rate vs Inhibitor Dose')
            img1 = fig1.to_image(format="png", width=900, height=500)
            doc.add_picture(io.BytesIO(img1), width=Inches(6))

            # ================= CHART 2: Residual Decay =================
            doc.add_heading('4. Residual Inhibitor Decay Over Time', level=1)
            days = list(range(0, 91, 5))
            decay = [residual_inhibitor * (0.95 ** (d / 10)) for d in days]
            fig2 = px.line(x=days, y=decay,
                           labels={'x': 'Days', 'y': 'Residual Inhibitor (ppm)'},
                           title='Residual Inhibitor Decay')
            img2 = fig2.to_image(format="png", width=900, height=500)
            doc.add_picture(io.BytesIO(img2), width=Inches(6))

            # ================= TABLE 3: Dose Options =================
            doc.add_heading('5. Inhibitor Dose Options', level=1)
            table3 = doc.add_table(rows=1, cols=3)
            table3.style = 'Light Grid Accent 3'
            hdr = table3.rows[0].cells
            hdr[0].text = "Dose (ppm)"
            hdr[1].text = "Predicted Rate (mm/yr)"
            hdr[2].text = "Meets Target?"
            for d in [0, 25, 50, 75, 100]:
                rate_d = r['rate'] * (1 - min(d / 200, 0.8))
                row = table3.add_row().cells
                row[0].text = f"{d}"
                row[1].text = f"{rate_d:.4f}"
                row[2].text = "✅ Yes" if rate_d <= 0.2 else "—"

            # ================= Save & Download =================
            buffer = io.BytesIO()
            doc.save(buffer)
            buffer.seek(0)
            st.download_button(
                label="📥 Click to Save Word Report",
                data=buffer,
                file_name="Corzynth_Report_Charts_Tables.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            # ------------------------------------------------------------
# MONTE CARLO
# ------------------------------------------------------------
elif page == "Monte Carlo":
    st.markdown('<div class="main-header"><div class="hero-badge">PROBABILISTIC ANALYSIS</div>'
                '<h1>🎲 Monte Carlo Corrosion Simulation</h1>'
                '<p>Quantify uncertainty in corrosion rate (mm/yr) and remaining life.</p></div>',
                unsafe_allow_html=True)

    if "single_result" not in st.session_state:
        st.warning("⚠️ Run a Single Prediction first.")
        st.stop()

    r = st.session_state["single_result"]
    col1, col2 = st.columns(2)
    with col1:
        n_sims = st.selectbox("Number of Simulations", [500, 2000, 5000, 10000], index=1)
    with col2:
        cv_percent = st.slider("Rate Variability (CV %)", 5, 50, 20)

    if st.button("🎲 Run Monte Carlo Simulation", use_container_width=True):
        with st.spinner("Running Monte Carlo simulation..."):
            mean_rate = r["rate"]
            std_rate = mean_rate * cv_percent / 100
            rates = np.random.normal(mean_rate, std_rate, n_sims)
            rates = np.clip(rates, 0.001, None)
            available = r["wall"] - r["min_wall"]
            lives = available / rates
            st.session_state["mc_results"] = {"rates": rates, "lives": lives}
        st.success(f"✅ Simulation completed with {n_sims} runs.")

    if "mc_results" in st.session_state:
        mc = st.session_state["mc_results"]
        rates, lives = mc["rates"], mc["lives"]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Mean Rate (mm/yr)", f"{rates.mean():.4f}")
        k2.metric("Std Dev (mm/yr)", f"{rates.std():.4f}")
        k3.metric("Mean Life", f"{lives.mean():.1f} years")
        k4.metric("P90 Life", f"{np.percentile(lives, 10):.1f} years")

        fig1 = px.histogram(x=rates, nbins=50, title="Corrosion Rate Distribution (mm/yr)",
                            labels={"x": "Corrosion Rate (mm/yr)"})
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.histogram(x=lives, nbins=50, title="Remaining Life Distribution",
                            labels={"x": "Remaining Life (years)"})
        st.plotly_chart(fig2, use_container_width=True)

        prob_low = (lives < 5).mean() * 100
        p1, p2 = st.columns(2)
        p1.metric("P(Life < 5 yrs)", f"{prob_low:.1f}%")
        p2.metric("P(Life < 2 yrs)", f"{(lives < 2).mean()*100:.1f}%")

        st.markdown('<div class="section-title">🧠 Interpretation</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="explain-box">
        - <b>P90 Life</b> = conservative estimate (90% chance actual life exceeds it).<br>
        - <b>{prob_low:.1f}%</b> of simulations predict life under 5 years.<br>
        - If P90 Life is below your inspection interval, prioritize mitigation.
        </div>
        """, unsafe_allow_html=True)

        mc_summary = pd.DataFrame({
            "Statistic": ["Mean Rate (mm/yr)", "Std Dev (mm/yr)", "Mean Life (yrs)", "P90 Life (yrs)"],
            "Value": [f"{rates.mean():.4f}", f"{rates.std():.4f}",
                      f"{lives.mean():.1f}", f"{np.percentile(lives,10):.1f}"],
        })
        download_csv_button(mc_summary, "Corzynth_MonteCarlo_Summary.csv", "mc_csv")

# ------------------------------------------------------------
# BATCH PREDICTION (with Template)
# ------------------------------------------------------------
elif page == "Batch Prediction":
    st.markdown('<div class="main-header"><div class="hero-badge">BATCH ANALYSIS</div>'
                '<h1>📥 Pipeline Batch Corrosion Prediction</h1>'
                '<p>Download template → fill data → upload CSV for bulk analysis.</p></div>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-title">📄 Step 1: Download CSV Template</div>',
                unsafe_allow_html=True)
    template_df = pd.DataFrame({
        "SegmentID": ["SEG-001", "SEG-002", "SEG-003"],
        "H2S_ppm": [50, 120, 30],
        "CO2_percent": [3.5, 6.0, 2.0],
        "Operating_Pressure_psi": [650, 900, 450],
        "Operating_Temperature_F": [140, 180, 100],
        "pH": [6.5, 5.8, 7.0],
        "Chlorides_mg_L": [15000, 25000, 8000],
        "Soil_Resistivity_ohm_m": [1200, 800, 2000],
        "Soil_pH": [7.2, 6.5, 7.8],
        "Moisture_Content_percent": [18, 25, 12],
        "Coating_Type": ["FBE", "3LPE", "Tape"],
        "Coating_Condition": ["Good", "Fair", "Excellent"],
        "Cathodic_Protection": ["Yes", "Yes", "No"],
        "Inhibitor_Type": ["Imidazoline (Film-Forming)",
                           "Fatty Acid Amide (Batch)",
                           "Water-Soluble Polyamine"],
        "Inhibitor_Dose_ppm": [25, 150, 0],
        "Service_Years": [10, 15, 5],
        "Product_Type": ["Crude Oil", "Natural Gas", "Water"],
    })
    st.dataframe(template_df, use_container_width=True)
    st.info("💡 Fill in your values, keep the header row, save as CSV.")
    download_csv_button(template_df, "Corzynth_Batch_Template.csv", "template_csv",
                        label="📥 Download Batch CSV Template")

    st.markdown('<div class="section-title">📤 Step 2: Upload Your Filled CSV</div>',
                unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload Pipeline Batch CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Total rows: {len(df)} | Columns: {len(df.columns)}")

        wall_mm = st.number_input("Wall Thickness (mm)", 0.1, 50.0, 8.5, key="bw")
        min_wall_mm = st.number_input("Min Thickness (mm)", 0.1, 50.0, 3.0, key="bmin")

        if st.button("🚀 Run Batch Prediction", use_container_width=True):
            results = []
            progress = st.progress(0)
            for i, row in df.iterrows():
                row_dict = row.to_dict()
                row_dict["Operating_Pressure_bar"] = row_dict.get("Operating_Pressure_psi", 650) * 0.0689476
                row_dict["Operating_Temperature_C"] = (row_dict.get("Operating_Temperature_F", 140) - 32) * 5 / 9

                rate = predict_rate(row_dict)
                risk = classify_corrosion_risk(rate)
                life = (wall_mm - min_wall_mm) / rate if rate > 0 else 0
                lik = likelihood_score(row_dict, rate)
                con = consequence_score(row_dict)
                overall, score = overall_risk(lik, con, risk)
                inh_life = remaining_inhibitor_life(
                    row_dict.get("Inhibitor_Dose_ppm", 0),
                    row_dict.get("Inhibitor_Type", "Imidazoline (Film-Forming)"))

                results.append({
                    "SegmentID": row_dict.get("SegmentID", i + 1),
                    "Rate_mm_yr": round(rate, 4),
                    "NACE_Class": risk,
                    "Remaining_Life_yrs": round(life, 1),
                    "Inhibitor_Type": row_dict.get("Inhibitor_Type", "N/A"),
                    "Inhibitor_Life_days": inh_life,
                    "Overall_Risk": overall,
                    "Priority": get_priority(overall),
                    "Action": get_action(overall, life),
                })
                progress.progress((i + 1) / len(df))

            st.session_state["batch_results"] = pd.DataFrame(results)
            st.success(f"✅ Completed for {len(results)} segments.")

    if "batch_results" in st.session_state:
        bdf = st.session_state["batch_results"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Segments", len(bdf))
        k2.metric("Severe (NACE)", (bdf["NACE_Class"] == "Severe").sum())
        k3.metric("High Risk", (bdf["Overall_Risk"] == "High").sum())
        k4.metric("Avg Life", f"{bdf['Remaining_Life_yrs'].mean():.1f} yrs")

        st.dataframe(bdf, use_container_width=True)
        download_csv_button(bdf, "Corzynth_Batch_Report.csv", "batch_csv")
        download_excel_button(bdf, "Corzynth_Batch_Report.xlsx", "batch_xlsx", "Batch")

# ------------------------------------------------------------
# CORROSION FORECAST
# ------------------------------------------------------------
elif page == "Corrosion Forecast":
    st.markdown('<div class="main-header"><div class="hero-badge">WALL-THICKNESS FORECAST</div>'
                '<h1>📈 Corrosion Growth Forecast</h1>'
                '<p>Forecast wall-thickness loss (mm) and integrity horizon.</p></div>',
                unsafe_allow_html=True)

    if "single_result" not in st.session_state:
        st.warning("⚠️ Run a Single Prediction first.")
        st.stop()

    r = st.session_state["single_result"]
    years_ahead = st.slider("Forecast Horizon (years)", 5, 50, 20)

    years = list(range(0, years_ahead + 1))
    thickness = [max(r["wall"] - r["rate"] * y, r["min_wall"]) for y in years]
    fc_df = pd.DataFrame({
        "Year": [datetime.now().year + y for y in years],
        "Wall_Thickness_mm": [round(t, 3) for t in thickness],
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc_df["Year"], y=fc_df["Wall_Thickness_mm"],
                             mode="lines+markers", name="Thickness",
                             line=dict(color="#18a999", width=3)))
    fig.add_hline(y=r["min_wall"], line_dash="dash", line_color="red",
                  annotation_text="Minimum Allowable")
    fig.update_layout(title=f"Forecast — {years_ahead} Year Horizon",
                      xaxis_title="Year", yaxis_title="Wall Thickness (mm)")
    st.plotly_chart(fig, use_container_width=True)

    breach = next((y for y, t in zip(years, thickness) if t <= r["min_wall"]), None)
    if breach:
        st.error(f"🚨 Minimum thickness reached in **{breach} years** "
                 f"({datetime.now().year + breach}).")
    else:
        st.success(f"✅ Safe for the full {years_ahead}-year horizon.")

    st.markdown(f"""
    <div class="explain-box">
    - Rate: <b>{r['rate']:.4f} mm/yr</b> · Wall: <b>{r['wall']:.2f} mm</b> ·
      Min: <b>{r['min_wall']:.2f} mm</b><br>
    - Corrosion allowance: <b>{r['wall'] - r['min_wall']:.2f} mm</b> ·
      Life: <b>{r['life']:.1f} years</b>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(fc_df, use_container_width=True)
    download_csv_button(fc_df, "Corzynth_Forecast.csv", "fc_csv")

# ------------------------------------------------------------
# RISK INTELLIGENCE
# ------------------------------------------------------------
elif page == "Risk Intelligence":
    st.markdown('<div class="main-header"><div class="hero-badge">RISK ANALYTICS</div>'
                '<h1>⚠️ Likelihood × Consequence Risk Matrix</h1></div>',
                unsafe_allow_html=True)

    if "batch_results" not in st.session_state:
        st.warning("⚠️ Run Batch Prediction first.")
        st.stop()

    bdf = st.session_state["batch_results"]

    col1, col2 = st.columns(2)
    with col1:
        counts = bdf["Overall_Risk"].value_counts()
        fig1 = px.bar(x=counts.index, y=counts.values, color=counts.index,
                      color_discrete_map={"Low": "#2ea043", "Medium": "#d29922",
                                          "High": "#cf222e", "Critical": "#8b0000"},
                      title="Risk Distribution")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.bar(bdf.sort_values("Remaining_Life_yrs").head(10),
                      x="SegmentID", y="Remaining_Life_yrs", color="Overall_Risk",
                      title="Top 10 Lowest Remaining Life")
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(bdf.sort_values("Remaining_Life_yrs"), use_container_width=True)
    download_csv_button(bdf, "Corzynth_Risk_Report.csv", "risk_csv")

# ------------------------------------------------------------
# WHAT-IF SIMULATOR
# ------------------------------------------------------------
elif page == "What-If Simulator":
    st.markdown('<div class="main-header"><div class="hero-badge">SCENARIO ANALYSIS</div>'
                '<h1>⚖️ What-If Corrosion Simulator</h1>'
                '<p>Compare current vs proposed inhibitor type, dose, coating, CP.</p></div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📌 Current**")
        cur_co2 = st.number_input("CO2 (%)", 0.0, 100.0, 3.5, key="cco2")
        cur_temp_f = st.number_input("Temp (°F)", -50.0, 400.0, 140.0, key="ctemp")
        cur_press_psi = st.number_input("Pressure (psi)", 0.0, 5000.0, 650.0, key="cpress")
        cur_inhib_type = st.selectbox("Inhibitor Type", list(INHIBITOR_TYPES.keys()), key="cit")
        cur_inhib = st.number_input("Inhibitor Dose (ppm)", 0.0, 500.0, 25.0, key="cid")
        cur_coat = st.selectbox("Coating", ["Poor", "Fair", "Good", "Excellent"], key="cc")
        cur_cp = st.selectbox("Cathodic Protection", ["Yes", "No"], key="ccp")
    with c2:
        st.markdown("**🔮 Proposed**")
        new_co2 = st.number_input("CO2 (%)", 0.0, 100.0, 3.5, key="nco2")
        new_temp_f = st.number_input("Temp (°F)", -50.0, 400.0, 140.0, key="ntemp")
        new_press_psi = st.number_input("Pressure (psi)", 0.0, 5000.0, 650.0, key="npress")
        new_inhib_type = st.selectbox("Inhibitor Type", list(INHIBITOR_TYPES.keys()),
                                       index=0, key="nit")
        new_inhib = st.number_input("Inhibitor Dose (ppm)", 0.0, 500.0, 75.0, key="nid")
        new_coat = st.selectbox("Coating", ["Poor", "Fair", "Good", "Excellent"],
                                 index=2, key="nc")
        new_cp = st.selectbox("Cathodic Protection", ["Yes", "No"], key="ncp")

    wall_mm = st.number_input("Wall Thickness (mm)", 0.1, 50.0, 8.5, key="ww")
    min_wall_mm = st.number_input("Min Thickness (mm)", 0.1, 50.0, 3.0, key="wm")

    if st.button("⚖️ Run What-If Simulation", use_container_width=True):
        base_row = {"CO2_percent": cur_co2,
                    "Operating_Temperature_C": (cur_temp_f - 32) * 5 / 9,
                    "Operating_Pressure_bar": cur_press_psi * 0.0689476,
                    "Soil_Resistivity_ohm_m": 1200, "Soil_pH": 7.2,
                    "Moisture_Content_percent": 18,
                    "Coating_Condition": cur_coat, "Cathodic_Protection": cur_cp,
                    "Inhibitor_Dose_ppm": cur_inhib, "Inhibitor_Type": cur_inhib_type}
        new_row = base_row.copy()
        new_row.update({"CO2_percent": new_co2,
                        "Operating_Temperature_C": (new_temp_f - 32) * 5 / 9,
                        "Operating_Pressure_bar": new_press_psi * 0.0689476,
                        "Coating_Condition": new_coat, "Cathodic_Protection": new_cp,
                        "Inhibitor_Dose_ppm": new_inhib, "Inhibitor_Type": new_inhib_type})

        base_rate = predict_rate(base_row)
        new_rate = predict_rate(new_row)
        base_life = (wall_mm - min_wall_mm) / base_rate if base_rate > 0 else 0
        new_life = (wall_mm - min_wall_mm) / new_rate if new_rate > 0 else 0
        reduction = (base_rate - new_rate) / base_rate * 100 if base_rate > 0 else 0

        cmp_df = pd.DataFrame({
            "Scenario": ["Current", "Proposed"],
            "Rate_mm_yr": [round(base_rate, 4), round(new_rate, 4)],
            "Life_yrs": [round(base_life, 1), round(new_life, 1)],
            "Inhibitor": [cur_inhib_type, new_inhib_type],
        })

        fig = px.bar(cmp_df, x="Scenario", y="Rate_mm_yr", color="Scenario",
                     color_discrete_map={"Current": "#cf222e", "Proposed": "#2ea043"},
                     title="Current vs Proposed Corrosion Rate (mm/yr)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cmp_df, use_container_width=True)

        if new_rate < base_rate:
            st.success(f"✅ Corrosion reduced by **{reduction:.1f}%**. "
                       f"Life: {base_life:.1f} → {new_life:.1f} years.")
        else:
            st.warning("⚠️ Proposed scenario does not improve performance.")

        download_csv_button(cmp_df, "Corzynth_WhatIf_Report.csv", "whatif_csv")
        # ------------------------------------------------------------
# RBI OPTIMIZER
# ------------------------------------------------------------
elif page == "RBI Optimizer":
    st.markdown('<div class="main-header"><div class="hero-badge">RISK-BASED INSPECTION</div>'
                '<h1>📅 Inspection Interval Optimizer</h1>'
                '<p>Optimized inspection intervals based on remaining life and NACE risk.</p></div>',
                unsafe_allow_html=True)

    if "batch_results" not in st.session_state:
        st.warning("⚠️ Run Batch Prediction first.")
        st.stop()

    bdf = st.session_state["batch_results"].copy()
    safety = st.slider("Safety Factor", 1.0, 3.0, 1.5, 0.1)
    max_interval = st.number_input("Max Inspection Interval (years)", 1, 10, 5)

    bdf["Base_Interval_yrs"] = bdf["Remaining_Life_yrs"] / safety
    risk_cap = {"Low": max_interval, "Medium": 3.0, "High": 1.0, "Critical": 0.5}
    bdf["Risk_Capped_yrs"] = bdf["Overall_Risk"].map(risk_cap).fillna(max_interval)
    bdf["Recommended_Interval_yrs"] = bdf[["Base_Interval_yrs", "Risk_Capped_yrs"]].min(axis=1).clip(0.25, max_interval)
    bdf["Next_Inspection"] = bdf["Recommended_Interval_yrs"].apply(
        lambda y: (datetime.now() + timedelta(days=int(y * 365))).strftime("%Y-%m-%d"))

    fig = px.bar(bdf.sort_values("Recommended_Interval_yrs"),
                 x="SegmentID", y="Recommended_Interval_yrs", color="Overall_Risk",
                 color_discrete_map={"Low": "#2ea043", "Medium": "#d29922",
                                     "High": "#cf222e", "Critical": "#8b0000"},
                 title="Recommended Inspection Interval by Segment")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">📅 RBI Inspection Plan</div>', unsafe_allow_html=True)
    st.dataframe(bdf[["SegmentID", "Overall_Risk", "Remaining_Life_yrs",
                      "Recommended_Interval_yrs", "Next_Inspection"]],
                 use_container_width=True)

    st.markdown(f"""
    <div class="explain-box">
    - <b>Base Interval</b> = Remaining Life ÷ Safety Factor ({safety})<br>
    - <b>Risk Cap</b>: Critical → 0.5 yrs · High → 1 yr · Medium → 3 yrs · Low → {max_interval} yrs<br>
    - <b>Recommended</b> = minimum of Base and Risk-Capped (0.25–{max_interval} yrs)
    </div>
    """, unsafe_allow_html=True)

    download_csv_button(bdf, "Corzynth_RBI_Plan.csv", "rbi_csv")
    download_excel_button(bdf, "Corzynth_RBI_Plan.xlsx", "rbi_xlsx", "RBI")

# ------------------------------------------------------------
# CORROSION COST ROI
# ------------------------------------------------------------
elif page == "Corrosion Cost ROI":
    st.markdown('<div class="main-header"><div class="hero-badge">FINANCIAL ANALYSIS</div>'
                '<h1>💰 Corrosion Cost, Savings & ROI Calculator</h1></div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        base_rate = st.number_input("Current Corrosion Rate (mm/yr)", 0.001, 5.0, 0.3067, format="%.4f")
        opt_rate = st.number_input("Rate After Mitigation (mm/yr)", 0.001, 5.0, 0.1963, format="%.4f")
        pipeline_km = st.number_input("Pipeline Length (km)", 0.1, 1000.0, 50.0)
    with c2:
        failure_cost = st.number_input("Cost of One Major Failure (USD)", 0.0, 100000000.0, 2500000.0)
        annual_prob = st.number_input("Annual Failure Probability (%)", 0.0, 100.0, 5.0)
        mitigation_cost = st.number_input("Annual Mitigation Cost (USD)", 0.0, 10000000.0, 120000.0)
    with c3:
        mitigation_type = st.selectbox("Mitigation Strategy",
            ["Inhibitor optimization", "Coating repair",
             "Cathodic protection improvement", "Combined integrity program"])
        project_years = st.number_input("Evaluation Period (years)", 1, 30, 10)

    if st.button("💰 Calculate Savings & ROI", use_container_width=True):
        reduction = base_rate - opt_rate
        reduction_pct = reduction / base_rate * 100 if base_rate > 0 else 0
        avoided_prob = annual_prob / 100 * reduction / base_rate if base_rate > 0 else 0
        annual_avoided = avoided_prob * failure_cost
        annual_net = annual_avoided - mitigation_cost
        total_net = annual_net * project_years
        total_cost = mitigation_cost * project_years
        roi = total_net / total_cost * 100 if total_cost > 0 else 0
        payback = mitigation_cost / annual_net if annual_net > 0 else float("inf")

        r1, r2, r3 = st.columns(3)
        r1.metric("Corrosion Reduction", f"{reduction_pct:.1f}%")
        r2.metric("Annual Avoided Loss", f"${annual_avoided:,.0f}")
        r3.metric("Annual Net Saving", f"${annual_net:,.0f}")

        r4, r5, r6 = st.columns(3)
        r4.metric("ROI", f"{roi:.1f}%")
        r5.metric("Total Net Saving", f"${total_net:,.0f}")
        r6.metric("Payback", f"{payback:.1f} yrs" if payback != float("inf") else "Not achieved")

        years = list(range(1, int(project_years) + 1))
        roi_df = pd.DataFrame({
            "Year": years,
            "Cumulative_Net_Saving_USD": [annual_net * y for y in years],
            "Cumulative_Mitigation_Cost_USD": [mitigation_cost * y for y in years],
        })
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roi_df["Year"], y=roi_df["Cumulative_Net_Saving_USD"],
                                 mode="lines+markers", name="Net Saving",
                                 line=dict(color="#2ea043", width=3)))
        fig.add_trace(go.Scatter(x=roi_df["Year"], y=roi_df["Cumulative_Mitigation_Cost_USD"],
                                 mode="lines+markers", name="Mitigation Cost",
                                 line=dict(color="#cf222e", width=3)))
        fig.update_layout(title="Investment vs Cumulative Savings")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class="explain-box">
        - Reduction: ({base_rate:.4f} − {opt_rate:.4f}) mm/yr = {reduction_pct:.1f}%<br>
        - Avoided Loss = {annual_prob}% × {reduction_pct:.1f}% × ${failure_cost:,.0f}<br>
        - Net Saving = Avoided Loss − ${mitigation_cost:,.0f}/yr<br>
        - ROI = Total Net Saving ÷ Total Cost × 100
        </div>
        """, unsafe_allow_html=True)

        roi_report = pd.DataFrame({
            "Strategy": [mitigation_type],
            "Base_Rate_mm_yr": [round(base_rate, 4)],
            "Optimized_Rate_mm_yr": [round(opt_rate, 4)],
            "Reduction_percent": [round(reduction_pct, 2)],
            "Annual_Avoided_USD": [round(annual_avoided, 0)],
            "Annual_Cost_USD": [round(mitigation_cost, 0)],
            "Annual_Net_USD": [round(annual_net, 0)],
            "ROI_percent": [round(roi, 1)],
            "Payback_yrs": [round(payback, 2) if payback != float("inf") else "N/A"],
        })
        download_csv_button(roi_report, "Corzynth_ROI_Report.csv", "roi_csv")

# ------------------------------------------------------------
# MAINTENANCE PLANNER
# ------------------------------------------------------------
elif page == "Maintenance Planner":
    st.markdown('<div class="main-header"><div class="hero-badge">MAINTENANCE</div>'
                '<h1>🔧 Risk-Based Maintenance Planner</h1></div>',
                unsafe_allow_html=True)

    if "batch_results" not in st.session_state:
        st.warning("⚠️ Run Batch Prediction first.")
        st.stop()

    bdf = st.session_state["batch_results"]
    due_days = {"Critical": 7, "High": 30, "Planned": 90, "Routine": 365}
    maint_df = bdf[["SegmentID", "Overall_Risk", "Priority",
                    "Remaining_Life_yrs", "Action"]].copy()
    maint_df["Due_Date"] = maint_df["Priority"].map(due_days).fillna(365).astype(int).apply(
        lambda d: (datetime.now() + timedelta(days=d)).strftime("%Y-%m-%d"))
    maint_df["Status"] = "Open"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Tasks", len(maint_df))
    m2.metric("Critical (7d)", (maint_df["Priority"] == "Critical").sum())
    m3.metric("High (30d)", (maint_df["Priority"] == "High").sum())
    m4.metric("Planned (90d)", (maint_df["Priority"] == "Planned").sum())

    st.dataframe(maint_df, use_container_width=True)
    download_csv_button(maint_df, "Corzynth_Maintenance_Plan.csv", "maint_csv")

# ------------------------------------------------------------
# TREND ANALYZER (with Template)
# ------------------------------------------------------------
elif page == "Trend Analyzer":
    st.markdown('<div class="main-header"><div class="hero-badge">HISTORICAL TRENDS</div>'
                '<h1>📊 Corrosion Trend Analyzer</h1></div>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-title">📄 Download Trend CSV Template</div>',
                unsafe_allow_html=True)
    trend_template = pd.DataFrame({
        "SegmentID": ["SEG-001", "SEG-001", "SEG-001", "SEG-002", "SEG-002"],
        "InspectionDate": ["2024-01-15", "2025-01-15", "2026-01-15", "2024-06-01", "2026-06-01"],
        "CorrosionRate_mm_yr": [0.15, 0.22, 0.31, 0.10, 0.12],
    })
    st.dataframe(trend_template, use_container_width=True)
    download_csv_button(trend_template, "Corzynth_Trend_Template.csv", "trend_template",
                        label="📥 Download Trend CSV Template")

    uploaded = st.file_uploader("Upload Historical Corrosion Trend CSV", type=["csv"], key="trend")
    if uploaded:
        tdf = pd.read_csv(uploaded)
        required = ["SegmentID", "InspectionDate", "CorrosionRate_mm_yr"]
        missing = [c for c in required if c not in tdf.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
            st.stop()

        tdf["InspectionDate"] = pd.to_datetime(tdf["InspectionDate"], errors="coerce")
        tdf = tdf.dropna(subset=["InspectionDate", "CorrosionRate_mm_yr"])
        tdf = tdf.sort_values(["SegmentID", "InspectionDate"])

        segment_ids = tdf["SegmentID"].astype(str).unique().tolist()
        selected = st.selectbox("Select Segment", segment_ids)
        seg = tdf[tdf["SegmentID"].astype(str) == selected]

        if len(seg) < 2:
            st.warning("At least two readings required.")
        else:
            x = seg["InspectionDate"].map(datetime.toordinal)
            y = seg["CorrosionRate_mm_yr"]
            slope = np.polyfit(x, y, 1)[0]
            change = (y.iloc[-1] - y.iloc[0]) / y.iloc[0] * 100 if y.iloc[0] != 0 else 0

            if slope > 0.01:
                status, color = "Increasing 📈", "#cf222e"
            elif slope < -0.01:
                status, color = "Decreasing 📉", "#2ea043"
            else:
                status, color = "Stable ➡️", "#d29922"

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("First Rate", f"{y.iloc[0]:.4f}")
            c2.metric("Latest Rate", f"{y.iloc[-1]:.4f}")
            c3.metric("Change", f"{change:.1f}%")
            c4.metric("Trend", status)

            fig = px.line(seg, x="InspectionDate", y="CorrosionRate_mm_yr",
                          markers=True, title=f"Trend — Segment {selected}")
            fig.update_traces(line=dict(color=color, width=3))
            st.plotly_chart(fig, use_container_width=True)

            trend_summary = []
            for sid in segment_ids:
                data = tdf[tdf["SegmentID"].astype(str) == sid].sort_values("InspectionDate")
                if len(data) < 2:
                    continue
                sx = data["InspectionDate"].map(datetime.toordinal)
                sy = data["CorrosionRate_mm_yr"]
                s_slope = np.polyfit(sx, sy, 1)[0]
                s_change = (sy.iloc[-1] - sy.iloc[0]) / sy.iloc[0] * 100 if sy.iloc[0] != 0 else 0
                s_status = ("Increasing" if s_slope > 0.01
                            else "Decreasing" if s_slope < -0.01 else "Stable")
                trend_summary.append({
                    "SegmentID": sid,
                    "First_Rate": round(sy.iloc[0], 4),
                    "Latest_Rate": round(sy.iloc[-1], 4),
                    "Change_percent": round(s_change, 2),
                    "Trend": s_status,
                })
            trend_df = pd.DataFrame(trend_summary)
            st.dataframe(trend_df, use_container_width=True)
            download_csv_button(trend_df, "Corzynth_Trend_Report.csv", "trend_csv")

# ------------------------------------------------------------
# GEOPHYSICAL ANALYSIS (with Template)
# ------------------------------------------------------------
elif page == "Geophysical Analysis":
    st.markdown('<div class="main-header"><div class="hero-badge">GEOPHYSICAL AI</div>'
                '<h1>🌍 Soil Corrosivity & GIS Risk Mapping</h1></div>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-title">📄 Download Geophysical CSV Template</div>',
                unsafe_allow_html=True)
    geo_template = pd.DataFrame({
        "SegmentID": ["SEG-001", "SEG-002", "SEG-003"],
        "Soil_Resistivity_ohm_m": [1200, 800, 2500],
        "Soil_pH": [7.2, 6.5, 7.8],
        "Moisture_Content_percent": [18, 25, 12],
        "Groundwater_Depth_m": [3.0, 1.5, 5.0],
        "Latitude": [24.7136, 24.7200, 24.7300],
        "Longitude": [46.6753, 46.6800, 46.6900],
    })
    st.dataframe(geo_template, use_container_width=True)
    download_csv_button(geo_template, "Corzynth_Geophysical_Template.csv", "geo_template",
                        label="📥 Download Geophysical CSV Template")

    uploaded = st.file_uploader("Upload Geophysical CSV", type=["csv"], key="geo")
    if uploaded:
        geo = pd.read_csv(uploaded)
        required = ["SegmentID", "Soil_Resistivity_ohm_m", "Soil_pH",
                    "Moisture_Content_percent", "Groundwater_Depth_m"]
        missing = [c for c in required if c not in geo.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
            st.stop()

        geo["Geo_Risk_Score"] = (
            0.40 * np.minimum(2, 1000 / geo["Soil_Resistivity_ohm_m"].clip(lower=1)) +
            0.30 * np.clip((7.5 - geo["Soil_pH"]) / 2, 0, 2) +
            0.20 * np.clip(geo["Moisture_Content_percent"] / 40, 0, 2) +
            0.10 * np.clip(20 / geo["Groundwater_Depth_m"].clip(lower=1), 0, 2)
        )
        geo["Geo_Risk_Level"] = pd.cut(
            geo["Geo_Risk_Score"], bins=[0, 0.80, 1.10, 10],
            labels=["Low", "Medium", "High"]).astype(str)

        st.session_state["geo_results"] = geo

        k1, k2, k3 = st.columns(3)
        k1.metric("Segments", len(geo))
        k2.metric("High Risk", (geo["Geo_Risk_Level"] == "High").sum())
        k3.metric("Medium Risk", (geo["Geo_Risk_Level"] == "Medium").sum())

        fig = px.bar(geo.sort_values("Soil_Resistivity_ohm_m"),
                     x="SegmentID", y="Soil_Resistivity_ohm_m",
                     color="Geo_Risk_Level",
                     color_discrete_map={"Low": "#2ea043", "Medium": "#d29922", "High": "#cf222e"},
                     title="Soil Resistivity by Segment")
        st.plotly_chart(fig, use_container_width=True)

        # AI Hotspots
        st.markdown('<div class="section-title">🤖 AI Hotspot Detection</div>',
                    unsafe_allow_html=True)
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans

        features = ["Soil_Resistivity_ohm_m", "Soil_pH",
                    "Moisture_Content_percent", "Groundwater_Depth_m"]
        scaled = StandardScaler().fit_transform(geo[features])
        geo["AI_Zone"] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(scaled)

        zone_risk = geo.groupby("AI_Zone")["Geo_Risk_Score"].mean()
        highest = zone_risk.idxmax()
        zone_names = {z: ("Critical Hotspot" if z == highest
                          else "Moderate Risk" if zone_risk[z] >= zone_risk.median()
                          else "Lower Risk") for z in geo["AI_Zone"].unique()}
        geo["AI_Zone_Name"] = geo["AI_Zone"].map(zone_names)

        fig3d = px.scatter_3d(geo, x="Soil_Resistivity_ohm_m", y="Moisture_Content_percent",
                              z="Geo_Risk_Score", color="AI_Zone_Name",
                              hover_name="SegmentID",
                              title="3D AI Geophysical Risk Map")
        st.plotly_chart(fig3d, use_container_width=True)

        # GIS Map
        lat_col = next((c for c in geo.columns if c in ["Latitude", "latitude", "Lat"]), None)
        lon_col = next((c for c in geo.columns if c in ["Longitude", "longitude", "Lon"]), None)
        if lat_col and lon_col:
            st.markdown('<div class="section-title">🗺️ Geographic Risk Map</div>',
                        unsafe_allow_html=True)
            import folium
            from streamlit_folium import st_folium

            risk_colors = {"Low": "green", "Medium": "orange", "High": "red"}
            fmap = folium.Map(location=[geo[lat_col].mean(), geo[lon_col].mean()], zoom_start=11)
            for _, row in geo.iterrows():
                folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]], radius=9,
                    color=risk_colors.get(row["Geo_Risk_Level"], "blue"),
                    fill=True, fill_opacity=0.8,
                    popup=f"Segment {row['SegmentID']} — {row['Geo_Risk_Level']} Risk"
                ).add_to(fmap)
            st_folium(fmap, height=600, use_container_width=True)
        else:
            st.warning("Latitude/Longitude not found — GIS map skipped.")

        st.markdown("""
        <div class="explain-box">
        - <b>Low soil resistivity</b> (&lt;500 ohm-m) = corrosive soil → higher risk<br>
        - <b>Low pH</b> (&lt;6.5) = acidic → higher risk<br>
        - <b>High moisture</b> (&gt;40%) = more electrolyte → higher risk<br>
        - <b>Shallow groundwater</b> (&lt;2 m) = more exposure → higher risk<br>
        - <b>AI Hotspots</b>: K-Means clusters similar soil conditions; highest-risk cluster = Critical Hotspot.
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(geo, use_container_width=True)
        download_csv_button(geo, "Corzynth_Geophysical_Assessment.csv", "geo_csv")
        # Download Geophysical Report as Word (Charts + Tables)
        from docx import Document
        from docx.shared import Inches
        import io

        if st.button("📄 Download Geophysical Report (Word)", use_container_width=True):
            doc = Document()
            doc.add_heading('Corzynth – Geophysical Survey Report', 0)
            doc.add_paragraph('DCVG / CIPS / Soil Resistivity Analysis')

            # ================= TABLE 1: Survey Summary =================
            doc.add_heading('1. Survey Summary', level=1)
            table1 = doc.add_table(rows=5, cols=2)
            table1.style = 'Light Grid Accent 1'
            summary = [
                ("Total Survey Points", f"{len(gdf)}"),
                ("Max DCVG (mV)", f"{gdf['DCVG_mV'].max():.1f}"),
                ("Mean DCVG (mV)", f"{gdf['DCVG_mV'].mean():.1f}"),
                ("Severe Coating Faults", f"{(gdf['Coating_Fault_Severity']=='Severe').sum()}"),
                ("Moderate Coating Faults", f"{(gdf['Coating_Fault_Severity']=='Moderate').sum()}"),
            ]
            for i, (k, v) in enumerate(summary):
                table1.rows[i].cells[0].text = k
                table1.rows[i].cells[1].text = v

            # ================= CHART 1: DCVG along pipeline =================
            doc.add_heading('2. DCVG (Coating Fault) Profile', level=1)
            fig_dcv = px.line(gdf, x='Chainage_m', y='DCVG_mV',
                              title='DCVG (mV) vs Chainage',
                              labels={'Chainage_m': 'Chainage (m)', 'DCVG_mV': 'DCVG (mV)'})
            img1 = fig_dcv.to_image(format="png", width=900, height=500)
            doc.add_picture(io.BytesIO(img1), width=Inches(6))

            # ================= CHART 2: CIPS Potential Profile =================
            doc.add_heading('3. CIPS Potential Profile', level=1)
            fig_cips = px.line(gdf, x='Chainage_m', y=['CIPS_On_mV', 'CIPS_Off_mV'],
                               title='CIPS On/Off Potential vs Chainage',
                               labels={'value': 'Potential (mV)', 'Chainage_m': 'Chainage (m)'})
            img2 = fig_cips.to_image(format="png", width=900, height=500)
            doc.add_picture(io.BytesIO(img2), width=Inches(6))

            # ================= CHART 3: Soil Resistivity =================
            doc.add_heading('4. Soil Resistivity Profile', level=1)
            fig_soil = px.line(gdf, x='Chainage_m', y='Soil_Resistivity_ohm_m',
                               title='Soil Resistivity vs Chainage',
                               labels={'Chainage_m': 'Chainage (m)', 'Soil_Resistivity_ohm_m': 'Resistivity (ohm·m)'})
            img3 = fig_soil.to_image(format="png", width=900, height=500)
            doc.add_picture(io.BytesIO(img3), width=Inches(6))

            # ================= TABLE 2: Top 10 Fault Locations =================
            doc.add_heading('5. Top 10 Coating Fault Locations', level=1)
            top_faults = gdf.nlargest(10, 'DCVG_mV')[['Chainage_m', 'DCVG_mV', 'Coating_Fault_Severity']]
            table2 = doc.add_table(rows=1, cols=3)
            table2.style = 'Light Grid Accent 2'
            hdr = table2.rows[0].cells
            hdr[0].text = "Chainage (m)"
            hdr[1].text = "DCVG (mV)"
            hdr[2].text = "Severity"
            for _, row in top_faults.iterrows():
                cells = table2.add_row().cells
                cells[0].text = f"{row['Chainage_m']:.0f}"
                cells[1].text = f"{row['DCVG_mV']:.1f}"
                cells[2].text = f"{row['Coating_Fault_Severity']}"

            # ================= Save & Download =================
            buffer = io.BytesIO()
            doc.save(buffer)
            buffer.seek(0)
            st.download_button(
                label="📥 Click to Save Geophysical Word Report",
                data=buffer,
                file_name="Corzynth_Geophysical_Report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

# ------------------------------------------------------------
# ALERTS
# ------------------------------------------------------------
elif page == "Alerts":
    st.markdown('<div class="main-header"><div class="hero-badge">NOTIFICATIONS</div>'
                '<h1>🚨 Alerts & Notifications</h1></div>',
                unsafe_allow_html=True)

    alerts = []
    if "batch_results" in st.session_state:
        bdf = st.session_state["batch_results"]
        for _, row in bdf[bdf["Overall_Risk"] == "Critical"].iterrows():
            alerts.append(("🚨 CRITICAL", f"Segment {row['SegmentID']} — Immediate action. "
                           f"Life: {row['Remaining_Life_yrs']} yrs."))
        for _, row in bdf[bdf["Overall_Risk"] == "High"].iterrows():
            alerts.append(("⚠️ HIGH", f"Segment {row['SegmentID']} — Priority inspection. "
                           f"Life: {row['Remaining_Life_yrs']} yrs."))

    if "single_result" in st.session_state:
        r = st.session_state["single_result"]
        if r["life"] < 5:
            alerts.append(("🚨 CRITICAL", f"Single prediction: life only {r['life']:.1f} years."))
        elif r["life"] < 10:
            alerts.append(("⚠️ HIGH", f"Single prediction: life {r['life']:.1f} years."))
        if r.get("inhibitor_life", 0) < 7 and r.get("inhibitor_dose", 0) > 0:
            alerts.append(("🧪 INHIBITOR", f"Inhibitor ({r['inhibitor_type']}) life only "
                           f"{r['inhibitor_life']:.0f} days — re-dose soon."))

    if not alerts:
        st.success("✅ No active alerts.")
    else:
        st.markdown(f"### 🔔 {len(alerts)} Active Alert(s)")
        for level, msg in alerts:
            if "CRITICAL" in level:
                st.error(f"{level}: {msg}")
            elif "INHIBITOR" in level:
                st.info(f"{level}: {msg}")
            else:
                st.warning(f"{level}: {msg}")

# ------------------------------------------------------------
# MODEL PERFORMANCE
# ------------------------------------------------------------
elif page == "Model Performance":
    st.markdown('<div class="main-header"><div class="hero-badge">MODEL VALIDATION</div>'
                '<h1>📊 Model Performance</h1></div>',
                unsafe_allow_html=True)

    if model is None:
        st.info("ℹ️ Using de Waard-Milliams physics model (accurate for CO2 corrosion).")
    else:
        st.success("✅ Trained ML model loaded.")
        st.write(f"**Features:** {len(model.feature_names_in_)}")

    uploaded = st.file_uploader("Upload Test Dataset (with ActualCorrosionRate_mm_yr)",
                                type=["csv"], key="perf")
    if uploaded:
        test = pd.read_csv(uploaded)
        if "ActualCorrosionRate_mm_yr" not in test.columns:
            st.error("CSV must contain 'ActualCorrosionRate_mm_yr' column.")
        else:
            actual = test["ActualCorrosionRate_mm_yr"]
            preds = test.drop(columns=["ActualCorrosionRate_mm_yr"]).apply(
                lambda r: predict_rate(r.to_dict()), axis=1)
            mae = np.mean(np.abs(preds - actual))
            rmse = np.sqrt(np.mean((preds - actual) ** 2))
            ss_res = np.sum((actual - preds) ** 2)
            ss_tot = np.sum((actual - actual.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0

            m1, m2, m3 = st.columns(3)
            m1.metric("R² Score", f"{r2:.3f}")
            m2.metric("RMSE (mm/yr)", f"{rmse:.4f}")
            m3.metric("MAE (mm/yr)", f"{mae:.4f}")

            fig = px.scatter(x=actual, y=preds, labels={"x": "Actual", "y": "Predicted"},
                             title="Actual vs Predicted (mm/yr)")
            fig.add_trace(go.Scatter(x=[actual.min(), actual.max()],
                                     y=[actual.min(), actual.max()],
                                     mode="lines", name="Perfect Fit",
                                     line=dict(dash="dash", color="red")))
            st.plotly_chart(fig, use_container_width=True)

            perf_df = pd.DataFrame({
                "Metric": ["R2", "RMSE_mm_yr", "MAE_mm_yr", "Samples"],
                "Value": [round(r2, 3), round(rmse, 4), round(mae, 4), len(test)],
            })
            download_csv_button(perf_df, "Corzynth_Model_Performance.csv", "perf_csv")

# ------------------------------------------------------------
# HELP & GLOSSARY
# ------------------------------------------------------------
elif page == "Help & Glossary":
    st.markdown('<div class="main-header"><div class="hero-badge">DOCUMENTATION</div>'
                '<h1>❓ Help & Glossary</h1></div>',
                unsafe_allow_html=True)

    st.markdown("""
    ### 📖 Key Terms

    | Term | Definition |
    |---|---|
    | **Corrosion Rate** | Metal loss speed in **mm/year** |
    | **De Waard-Milliams** | CO2 corrosion model: log(V) = 5.8 − 1710/T + 0.67·log10(pCO2) |
    | **NACE SP0775-2023** | Low <0.025 · Moderate 0.025–0.12 · High 0.12–0.25 · Severe >0.25 mm/yr |
    | **Remaining Life** | Years until wall reaches minimum allowable |
    | **Inhibitor Types** | Imidazoline, Quaternary Amine, Fatty Acid Amide, Neutralizing Amine, Polyamine, VPI |
    | **Remaining Inhibitor Life** | Days until re-dosing needed |
    | **RBI** | Risk-Based Inspection |
    | **Monte Carlo** | Uncertainty simulation |
    | **P90 Life** | Conservative (10th percentile) life estimate |
    | **ILI** | In-Line Inspection (smart pigging) |
    | **CP** | Cathodic Protection |
    | **ROI** | Return on Investment |
    | **NACE MR0175** | Sour service (H2S) standard |
    | **API 570** | Piping inspection code |

    ### 🎯 Workflow
    1. **Single Prediction** → rate, life, inhibitor dose
    2. **Monte Carlo** → uncertainty (after Single)
    3. **Batch Prediction** → many segments (download template first)
    4. **Forecast** → wall loss over time (after Single)
    5. **Risk Intelligence** → prioritize (after Batch)
    6. **What-If** → compare scenarios
    7. **RBI** → inspection dates (after Batch)
    8. **ROI** → budget justification
    9. **Maintenance** → action plan (after Batch)
    10. **Trend** → historical analysis (download template)
    11. **Geophysical** → soil/GIS (download template)

    ### ⚠️ Disclaimer
    Uses **de Waard-Milliams CO2 model** and **NACE SP0775-2023** for engineering
    screening only. Final decisions require qualified engineering review.
    """)

# ------------------------------------------------------------
# ADMIN PANEL
# ------------------------------------------------------------
elif page == "Admin Panel":
    st.markdown('<div class="main-header"><div class="hero-badge">ADMINISTRATOR</div>'
                '<h1>⚙️ Admin Panel</h1></div>',
                unsafe_allow_html=True)

    st.info(f"Logged in as: **{username}** ({user_role})")

    s1, s2, s3 = st.columns(3)
    s1.metric("ML Model", "Active" if model else "Physics Model")
    s2.metric("Python", "3.13")
    s3.metric("Version", APP_VERSION.split()[-1])

    st.markdown('<div class="section-title">👥 Users</div>', unsafe_allow_html=True)
    users_df = pd.DataFrame([
        {"Username": u, "Name": d.get("name", ""), "Email": d.get("email", ""),
         "Role": d.get("role", "engineer")}
        for u, d in st.secrets["credentials"]["usernames"].items()
    ])
    st.dataframe(users_df, use_container_width=True)

    st.markdown('<div class="section-title">📁 Project Files</div>', unsafe_allow_html=True)
    st.write([f for f in os.listdir(".") if os.path.isfile(f)])

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.markdown(f"""
<div class="footer">
    <b>{APP_VERSION}</b> · AI Pipeline Corrosion Intelligence Platform<br>
    Model: <b>de Waard-Milliams (CO2)</b> · Classification: <b>NACE SP0775-2023</b><br>
    Developed by <b>{DEVELOPER}</b> · For engineering screening and decision-support only.<br>
    Final integrity decisions require qualified engineering review.
</div>
""", unsafe_allow_html=True)