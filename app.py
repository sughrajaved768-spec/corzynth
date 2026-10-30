# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CORZYNTH v10.0             ║
# ║  de Waard-Milliams + Material Metallurgy + NACE MR0175                   ║
# ║  LOGIN: admin / corzynth2026                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import streamlit as st
import pandas as pd
import numpy as np
import os, io, base64
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Corzynth", page_icon="corzynth_logo.jpeg", layout="wide")

DESIGN_LIFE = 40
LOGO = "corzynth_logo.jpeg"

FLUID_TYPES = ["🛢️ Crude Oil", "⛽ Natural Gas", "💧 Water",
               "🫧 Multiphase (Oil+Gas+Water)"]
FLUID_FACTOR = {"🛢️ Crude Oil": 0.85, "⛽ Natural Gas": 0.70,
                "💧 Water": 1.30, "🫧 Multiphase (Oil+Gas+Water)": 1.15}
INHIBITOR_TYPES = ["Film-Forming Amine", "Imidazoline", "Quaternary Amine",
                   "Phosphate Ester", "Batch Inhibitor", "Mixed Film/Neutralizer"]
COATING_CONDITIONS = ["Excellent", "Good", "Fair", "Poor"]
COATING_FACTOR = {"Excellent": 0.3, "Good": 0.5, "Fair": 0.8, "Poor": 1.0}
MATERIAL_GRADES = ["API 5L X42", "API 5L X52", "API 5L X60", "API 5L X65",
                   "API 5L X70", "Carbon Steel A106", "Stainless 316L", "Duplex 2205"]
MATERIAL_FACTOR = {"API 5L X42": 1.05, "API 5L X52": 1.00, "API 5L X60": 0.97,
                   "API 5L X65": 0.95, "API 5L X70": 0.93,
                   "Carbon Steel A106": 1.00, "Stainless 316L": 0.30,
                   "Duplex 2205": 0.20}

# ══════════════════════════════════════════════════════════════════════════
#  LOGIN
# ══════════════════════════════════════════════════════════════════════════

def check_login(u, p):
    return u == "admin" and p == "corzynth2026"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("""
    <style>
    .stApp {background:linear-gradient(135deg,#0b1f3a,#123a5c);}
    .login-card {max-width:440px;margin:30px auto;padding:35px;border-radius:22px;
        background:rgba(255,255,255,0.08);backdrop-filter:blur(10px);
        border:1px solid rgba(255,255,255,0.15);text-align:center;}
    .login-logo {width:110px;height:110px;border-radius:50%;object-fit:cover;
        border:3px solid #2e86c1;margin-bottom:12px;}
    .login-title {font-size:34px;font-weight:800;color:white;margin:8px 0 2px 0;}
    .login-sub {color:#8fd3ff;font-size:14px;margin-bottom:22px;}
    </style>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        logo_html = ""
        if os.path.exists(LOGO):
            b64 = base64.b64encode(open(LOGO, "rb").read()).decode()
            logo_html = f'<img src="data:image/jpeg;base64,{b64}" class="login-logo">'
        st.markdown(f"""
        <div class="login-card">
            {logo_html}
            <div class="login-title">CORZYNTH</div>
            <div class="login-sub">Corrosion & Chemical Intelligence Platform</div>
        </div>
        """, unsafe_allow_html=True)
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("🔐 Sign In", type="primary", use_container_width=True):
            if check_login(u, p):
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════
#  TRANSPARENT PHYSICS ENGINE — 13 VERIFIABLE STEPS
# ══════════════════════════════════════════════════════════════════════════

def full_calculation(v):
    """de Waard-Milliams (1991) with fugacity correction + material metallurgy.
       Every step returned for full transparency."""
    steps = {}

    # 1. Temperature
    T = (v["temp"] - 32) * 5 / 9 + 273.15
    steps["1. Temperature"] = f"{v['temp']}°F = {T:.1f} K"

    # 2. Pressure
    P_bar = min(v["pressure"] * 0.0689476, 250)
    steps["2. Pressure"] = f"{v['pressure']} psi = {P_bar:.1f} bar"

    # 3. CO2 partial pressure
    pco2 = max(0.001, v["co2"] / 100 * P_bar)
    steps["3. CO2 Partial Pressure"] = (
        f"{v['co2']}% × {P_bar:.1f} bar = {pco2:.3f} bar")

    # 4. Fugacity coefficient
    a = 10 ** (P_bar * (0.0031 - 1.4 / T))
    steps["4. Fugacity Coefficient"] = (
        f"a = 10^({P_bar:.1f} × (0.0031 - 1.4/{T:.1f})) = {a:.3f}")

    # 5. CO2 fugacity
    fco2 = a * pco2
    steps["5. CO2 Fugacity"] = f"fCO2 = {a:.3f} × {pco2:.3f} = {fco2:.3f} bar"

    # 6. Base bare-steel rate
    log_v = 5.8 - 1710 / T + 0.67 * np.log10(fco2)
    base_rate = 10 ** log_v
    steps["6. Base Rate (bare steel)"] = (
        f"log V = 5.8 - 1710/{T:.1f} + 0.67×log({fco2:.3f}) = {log_v:.3f} "
        f"→ {base_rate:.3f} mm/yr")

    # 7. pH correction
    ph_factor = 1.3 if v["ph"] < 4.5 else (0.6 if v["ph"] > 6.5 else 1.0)
    rate_ph = base_rate * ph_factor
    steps["7. pH Correction"] = f"pH {v['ph']} → ×{ph_factor} = {rate_ph:.3f} mm/yr"

    # 8. H2S sour-service correction
    h2s_factor = 1.3 if v["h2s"] > 500 else 1.0
    rate_h2s = rate_ph * h2s_factor
    steps["8. H2S Sour Service"] = (
        f"H2S {v['h2s']} ppm → ×{h2s_factor} = {rate_h2s:.3f} mm/yr")

    # 9. Inhibitor
    inh_eff = min(0.95, 0.5 + v["dose"] / 100)
    rate_inh = rate_h2s * (1 - inh_eff)
    steps["9. Inhibitor"] = (
        f"{v['dose']} ppm ({v.get('itype','Film-Forming Amine')}) → "
        f"{inh_eff*100:.0f}% reduction → {rate_inh:.3f} mm/yr")

    # 10. Coating
    cm = COATING_FACTOR[v["coating_condition"]]
    rate_coat = rate_inh * cm
    steps["10. Coating Protection"] = (
        f"{v['coating_condition']} → ×{cm} = {rate_coat:.3f} mm/yr")

    # 11. Cathodic protection
    cpm = 0.5 if v["cp_on"] == "Yes" else 1.0
    rate_cp = rate_coat * cpm
    steps["11. Cathodic Protection"] = (
        f"CP {v['cp_on']} → ×{cpm} = {rate_cp:.3f} mm/yr")

    # 12. Fluid type
    fm = FLUID_FACTOR[v["fluid"]]
    rate_fluid = rate_cp * fm
    steps["12. Fluid Type"] = f"{v['fluid']} → ×{fm} = {rate_fluid:.3f} mm/yr"

    # 13. Material grade metallurgy
    mf = MATERIAL_FACTOR.get(v.get("material", "API 5L X52"), 1.0)
    final_rate = max(0.01, rate_fluid * mf)
    steps["13. Material Grade"] = (
        f"{v.get('material','API 5L X52')} → ×{mf} = {final_rate:.3f} mm/yr")

    # Remaining life — capped at design life
    life_raw = v["wt"] / final_rate
    life = min(life_raw, DESIGN_LIFE)
    steps["14. Remaining Life"] = (
        f"{v['wt']}mm ÷ {final_rate:.3f} = {life_raw:.1f} yrs "
        f"→ capped at {DESIGN_LIFE} yrs design life")

    if final_rate < 0.2:
        risk, emoji = "LOW", "🟢"
    elif final_rate < 0.5:
        risk, emoji = "MEDIUM", "🟡"
    else:
        risk, emoji = "HIGH", "🔴"

    return final_rate, life, life_raw, risk, emoji, steps


def ai_explanation(v, rate, life_raw, risk):
    coat_pct = {"Excellent": "70%", "Good": "50%", "Fair": "20%", "Poor": "0%"}
    mf = MATERIAL_FACTOR.get(v.get("material", "API 5L X52"), 1.0)
    lines = []
    lines.append(f"**Why the corrosion rate is {rate:.3f} mm/yr:**")
    lines.append(f"- Bare steel in your CO₂ environment would corrode at the "
                 f"de Waard-Milliams base rate — CO₂ partial pressure of "
                 f"{v['co2']}% at {v['pressure']} psi drives this.")
    lines.append(f"- Your {v['dose']} ppm {v.get('itype','inhibitor')} reduces "
                 f"corrosion by {min(95, 50 + v['dose'])}%.")
    lines.append(f"- {v['coating_condition']} coating cuts the remaining rate by "
                 f"{coat_pct[v['coating_condition']]}.")
    lines.append(f"- Cathodic protection ({v['cp_on']}) provides "
                 f"{'an additional 50% reduction' if v['cp_on']=='Yes' else 'no reduction'}.")
    lines.append(f"- {v['fluid']} service factor applied.")
    lines.append(f"- {v.get('material','API 5L X52')} material provides "
                 f"{'standard' if mf >= 1.0 else 'enhanced'} corrosion resistance "
                 f"(factor ×{mf}).")
    lines.append("")
    lines.append(f"**What this means for your pipeline ({v['wt']} mm wall):**")
    if life_raw > DESIGN_LIFE:
        lines.append(f"- Corrosion alone would take {life_raw:.0f} years to penetrate "
                     f"the wall — beyond the {DESIGN_LIFE}-year design life.")
        lines.append("- **Corrosion is NOT your limiting factor.** Ageing, coating "
                     "degradation, and mechanical integrity will govern. "
                     "Plan a re-assessment at design life.")
    elif life_raw > 20:
        lines.append(f"- Corrosion will consume the wall in {life_raw:.1f} years.")
        lines.append(f"- **Action:** maintain the inhibitor program, verify CP "
                     f"annually, plan inspection at 50% of remaining life "
                     f"({life_raw/2:.0f} yrs).")
    else:
        lines.append(f"- Corrosion will consume the wall in only {life_raw:.1f} years.")
        lines.append("- **Urgent action:** increase inhibitor dose, verify CP "
                     "rectifier output, upgrade to corrosion-resistant alloy "
                     "(13Cr / duplex), schedule intelligent pigging within 12 months.")
    lines.append("")
    lines.append(f"**Risk: {risk}** — " + {
        "LOW": "Continue routine monitoring.",
        "MEDIUM": "Increase monitoring frequency; review inhibitor residual monthly.",
        "HIGH": "Immediate mitigation: dose review, CP survey, and inspection."}[risk])
    return "\n".join(lines)


def sensitivity_analysis(v, rate):
    factors = {
        "CO₂ ±50%": ("co2", v["co2"] * 0.5, v["co2"] * 1.5),
        "Temperature ±50°F": ("temp", max(32, v["temp"] - 50), v["temp"] + 50),
        "Inhibitor Dose ±50%": ("dose", max(0, v["dose"] * 0.5), v["dose"] * 1.5),
        "pH ±0.5": ("ph", max(3, v["ph"] - 0.5), min(10, v["ph"] + 0.5)),
        "Material Grade": ("material", "Carbon Steel A106", "Duplex 2205"),
    }
    rows = []
    for name, (key, lo, hi) in factors.items():
        r_lo, _, _, _, _, _ = full_calculation({**v, key: lo})
        r_hi, _, _, _, _, _ = full_calculation({**v, key: hi})
        rows.append({"Factor": name,
                     "Rate Low": f"{r_lo:.3f} mm/yr",
                     "Rate High": f"{r_hi:.3f} mm/yr",
                     "Impact": f"{abs(r_hi - r_lo) / rate * 100:.0f}%"})
    return pd.DataFrame(rows)


def find_optimal_dose(v, target_rate):
    doses = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150]
    for d in doses:
        vd = {**v, "dose": d}
        r, _, _, _, _, _ = full_calculation(vd)
        if r <= target_rate:
            return d, r
    return 150, full_calculation({**v, "dose": 150})[0]


def make_redose_table(optimal_dose, decay, min_res):
    init_res = max(optimal_dose, min_res * 2)
    days_to_min = max(1, int((init_res - min_res) / decay))
    rows = []
    day = 0
    for cycle in range(1, 7):
        rows.append({"Cycle": cycle, "Injection Day": day,
                     "Dose (ppm)": optimal_dose,
                     "Initial Residual (ppm)": f"{init_res:.0f}",
                     "Days to Min Residual": days_to_min})
        day += days_to_min
    return pd.DataFrame(rows), days_to_min


def make_life_projection(wt, rate, life_raw):
    """Native Streamlit chart + year-by-year table — always displays."""
    max_year = int(min(life_raw + 5, 60))
    proj_rows = []
    for yr in range(0, max_year + 1, 2):
        loss = rate * yr
        remaining_wall = max(0, wt - loss)
        status = ("🔴 Failed" if remaining_wall <= 0 else
                  "🟡 Critical" if remaining_wall < wt * 0.25 else
                  "🟢 Healthy")
        proj_rows.append({"Year": yr, "Metal Loss (mm)": round(loss, 2),
                          "Remaining Wall (mm)": round(remaining_wall, 2),
                          "Status": status})
    proj_df = pd.DataFrame(proj_rows)
    chart_df = proj_df.set_index("Year")[["Metal Loss (mm)", "Remaining Wall (mm)"]]
    return chart_df, proj_df

# ══════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════

def logo_b64():
    if os.path.exists(LOGO):
        return base64.b64encode(open(LOGO, "rb").read()).decode()
    return ""

st.markdown("""
<style>
.main-header {background:linear-gradient(135deg,#0b1f3a,#1a5276,#2e86c1);
    padding:30px;border-radius:18px;color:white;margin-bottom:25px;
    display:flex;align-items:center;gap:25px;}
.header-logo {width:90px;height:90px;border-radius:16px;object-fit:cover;
    border:3px solid rgba(255,255,255,0.4);}
.header-text h1 {margin:0;font-size:38px;font-weight:800;}
.header-text p {color:#8fd3ff;margin:6px 0 0 0;font-size:15px;}
.calc-step {background:#f8f9fa;padding:8px 15px;border-left:4px solid #2e86c1;
    margin:6px 0;border-radius:6px;font-family:monospace;font-size:14px;}
</style>
""", unsafe_allow_html=True)


def render_header():
    lb = logo_b64()
    logo_html = f'<img src="data:image/jpeg;base64,{lb}" class="header-logo">' if lb else ''
    st.markdown(f"""
    <div class="main-header">
        {logo_html}
        <div class="header-text">
            <h1>CORZYNTH</h1>
            <p>Corrosion & Chemical Intelligence Platform | NACE MR0175 | de Waard-Milliams</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


page = st.sidebar.radio("Navigate", [
    "🔮 Prediction", "🧪 Chemical Advisor", "💊 Dose Optimizer",
    "💰 ROI Calculator", "🌍 Geophysical", "🏛️ NACE Compliance",
    "🗺️ GIS Map", "🔔 Alerts", "ℹ️ About"])

# ══════════════════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════════════════

if page == "🔮 Prediction":
    render_header()
    st.info("🔬 **Transparent Engineering** — every calculation step is shown "
            "so you can verify the result.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🔧 Operating Conditions")
        fluid = st.selectbox("Material Flowing in Pipe", FLUID_TYPES)
        pressure = st.number_input("Pressure (psi)", 0, 5000, 1000, 50)
        temp = st.number_input("Temperature (°F)", 32, 400, 150, 5)
        wt = st.number_input("Wall Thickness (mm)", 1.0, 50.0, 8.0, 0.5)
        material = st.selectbox("Pipeline Material Grade", MATERIAL_GRADES)
        coating_condition = st.selectbox("Coating Condition", COATING_CONDITIONS)
        cp_on = st.selectbox("Cathodic Protection", ["Yes", "No"])
    with c2:
        st.subheader("🧪 Chemistry & Inhibitor")
        co2 = st.number_input("CO₂ (%)", 0.0, 100.0, 2.0, 0.5)
        h2s = st.number_input("H₂S (ppm)", 0, 20000, 100, 50)
        ph = st.number_input("pH", 3.0, 10.0, 6.5, 0.1)
        itype = st.selectbox("Inhibitor Type", INHIBITOR_TYPES)
        dose = st.number_input("Inhibitor Dose (ppm)", 0, 500, 25, 5)
        residual = st.number_input("Inhibitor Residual (ppm)", 0, 200, 10, 5)

    if st.button("🔮 Calculate Corrosion Rate & Life", type="primary",
                 use_container_width=True):
        v = dict(fluid=fluid, pressure=pressure, temp=temp, wt=wt,
                 material=material, coating_condition=coating_condition,
                 cp_on=cp_on, co2=co2, h2s=h2s, ph=ph, dose=dose,
                 itype=itype, residual=residual)
        rate, life, life_raw, risk, emoji, steps = full_calculation(v)

        # ── RESULTS ──
        st.subheader("📊 Integrity Assessment")
        m1, m2, m3 = st.columns(3)
        m1.metric("Corrosion Rate", f"{rate:.3f} mm/yr")
        life_text = (f"≥ {DESIGN_LIFE} yrs" if life_raw > DESIGN_LIFE
                     else f"{life:.1f} yrs")
        m2.metric("Remaining Life", life_text,
                  "🟢 Beyond design life" if life_raw > DESIGN_LIFE else
                  ("🟢 Good" if life > 25 else ("🟡 Plan" if life > 10 else "🔴 Urgent")))
        m3.metric("Risk", f"{emoji} {risk}")

        # ── NACE ──
        if h2s > 500:
            st.warning("🔴 **NACE MR0175: Sour Service** — H₂S > 500 ppm, "
                       "use compliant materials")
        elif rate > 0.5:
            st.warning("🔴 **High Corrosion** — increase inhibitor or verify CP")
        else:
            st.success("✅ NACE MR0175: No compliance flags")

        # ── DOSE OPTIMIZATION ──
        st.subheader("💊 Best Inhibitor Dose Optimization")
        target = st.number_input("Your Target Rate (mm/yr)", 0.01, 1.0, 0.1, 0.01,
                                 key="pred_target")
        opt_dose, opt_rate = find_optimal_dose(v, target)
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a5276,#2e86c1);padding:20px;
             border-radius:14px;color:white;margin:15px 0;">
            <h3 style="margin:0;color:#ffd700;">✅ BEST DOSE: {opt_dose} ppm</h3>
            <p style="margin:5px 0 0 0;">Achieves {opt_rate:.3f} mm/yr
            (target: {target} mm/yr) with {itype}</p>
        </div>
        """, unsafe_allow_html=True)
        if opt_rate > target:
            st.info("💡 **Inhibitor alone cannot reach your target.** Consider "
                    "upgrading coating condition, strengthening CP, or using a "
                    "corrosion-resistant alloy (13Cr / Duplex 2205).")

        # ── RE-DOSING SCHEDULE ──
        st.subheader("🔄 Re-dosing Schedule")
        decay = st.number_input("Residual Decay (ppm/day)", 1, 50, 10, key="pred_decay")
        min_res = st.number_input("Min Effective Residual (ppm)", 1, 100, 10,
                                  key="pred_minres")
        redose_df, redose_day = make_redose_table(opt_dose, decay, min_res)
        c1, c2, c3 = st.columns(3)
        c1.metric("Initial Residual", f"{max(opt_dose, min_res*2):.0f} ppm")
        c2.metric("Re-dose Every", f"{redose_day} days")
        c3.metric("Re-dose Amount", f"{opt_dose} ppm")
        st.dataframe(redose_df, use_container_width=True)

        # ── TRANSPARENT CALCULATION ──
        st.subheader("🔬 Calculation Breakdown — Verify Every Step")
        st.caption("de Waard-Milliams (1991): "
                   "log₁₀V = 5.8 − 1710/T + 0.67·log₁₀(fCO₂)")
        for step_name, step_calc in steps.items():
            st.markdown(f'<div class="calc-step"><b>{step_name}:</b> {step_calc}</div>',
                        unsafe_allow_html=True)

        # ── AI EXPLANATION ──
        st.subheader("🤖 AI Engineering Explanation")
        st.markdown(ai_explanation(v, rate, life_raw, risk))

        # ── SENSITIVITY ──
        st.subheader("📊 Sensitivity Analysis — What Affects Your Rate Most")
        st.dataframe(sensitivity_analysis(v, rate), use_container_width=True)
        st.caption("Higher Impact % = that input changes your corrosion rate "
                   "the most. Focus control efforts there.")

        # ── LIFE PROJECTION ──
        st.subheader("📈 Life Projection")
        chart_df, proj_df = make_life_projection(wt, rate, life_raw)
        st.line_chart(chart_df)
        st.caption(f"🔴 Metal loss grows at {rate:.3f} mm/yr | "
                   f"🟢 Wall thickness = {wt} mm | "
                   f"🔵 Design life = {DESIGN_LIFE} yrs | "
                   f"Corrosion end of life = {life_raw:.1f} yrs")
        st.subheader("📋 Year-by-Year Wall Loss Projection")
        st.dataframe(proj_df, use_container_width=True)
        k1, k2, k3 = st.columns(3)
        k1.metric("50% Wall Loss", f"{(wt*0.5/rate):.1f} yrs")
        k2.metric("75% Wall Loss", f"{(wt*0.75/rate):.1f} yrs")
        k3.metric("Full Penetration",
                  f"{life_raw:.1f} yrs" if life_raw <= DESIGN_LIFE
                  else f"> {DESIGN_LIFE} yrs (beyond design life)")

# ══════════════════════════════════════════════════════════════════════════
#  CHEMICAL ADVISOR
# ══════════════════════════════════════════════════════════════════════════

elif page == "🧪 Chemical Advisor":
    render_header()
    st.header("🧪 Chemical Advisor — Smart Buyer's Guide")
    chem_type = st.selectbox("What chemical do you need?", [
        "🛡️ Corrosion Inhibitor", "💧 Demulsifier", "🦠 Biocide",
        "🧴 Scale Inhibitor", "🫧 Antifoam", "⛽ Drag Reducing Agent"])
    c1, c2, c3, c4 = st.columns(4)
    t_wc = c1.number_input("Water Cut (%)", 0, 100, 50, 5)
    t_temp = c2.number_input("Temperature (°F)", 32, 400, 150, 5)
    t_salinity = c3.number_input("Salinity TDS (ppm)", 0, 200000, 50000, 5000)
    t_api = c4.number_input("Crude API Gravity", 10, 50, 30, 1)
    c5, c6, c7 = st.columns(3)
    t_ph = c5.number_input("Water pH", 3.0, 10.0, 7.0, 0.1)
    t_h2s = c6.number_input("H₂S (ppm)", 0, 20000, 500, 50)
    t_co2 = c7.number_input("CO₂ (%)", 0.0, 100.0, 3.0, 0.5)

    if st.button("🔍 Analyze & Recommend", type="primary", use_container_width=True):
        if chem_type == "💧 Demulsifier":
            rec_product = ("Asphaltene-Dispersing EO/PO" if (t_api < 20 and t_wc > 60)
                           else "High-MW EO/PO Block Copolymer")
            rec_dose = f"{max(15, int(t_wc/3))}-{max(30, int(t_wc/2))} ppm"
            real_eff = "70-85% water removal in 30 min"
        elif chem_type == "🛡️ Corrosion Inhibitor":
            rec_product = ("Imidazoline H₂S Inhibitor" if t_h2s > 500
                           else "Film-Forming Amine")
            rec_dose = f"{max(25, int(t_h2s/100))} ppm" if t_h2s > 100 else "25-75 ppm"
            real_eff = "80-92% protection"
        elif chem_type == "🦠 Biocide":
            rec_product = "Glutaraldehyde Biocide"
            rec_dose = "50-200 ppm"
            real_eff = "85-95% SRB kill"
        elif chem_type == "🧴 Scale Inhibitor":
            rec_product = "HEDP Phosphonate"
            rec_dose = "5-20 ppm"
            real_eff = "85-95% scale prevention"
        elif chem_type == "🫧 Antifoam":
            rec_product = "Silicone Antifoam"
            rec_dose = "5-20 ppm"
            real_eff = "90-98% foam knockdown"
        else:
            rec_product = "High-MW Polymer DRA"
            rec_dose = "10-50 ppm"
            real_eff = "30-50% friction reduction"

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a5276,#2e86c1);padding:28px;
             border-radius:16px;color:white;margin:20px 0;">
            <h2 style="margin:0;color:#fff;">✅ USE THIS CHEMICAL</h2>
            <h3 style="margin:12px 0 6px 0;color:#ffd700;">🏆 {rec_product}</h3>
            <p style="margin:0;font-size:17px;">Dose: <b>{rec_dose}</b> | 
               Real Efficiency: <b>{real_eff}</b></p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("⚠️ SUPPLIER TRAP ALERTS")
        st.error("""
**🚨 TRAP #1 — "99% effective"** → Real is 60-92%. Demand bottle test.
**🚨 TRAP #2 — "Works everywhere"** → Field-specific. Test YOUR crude.
**🚨 TRAP #3 — "More = better"** → Over-dosing worsens results.
**🚨 TRAP #4 — Hidden dilution** → Check ACTIVE INGREDIENT % and cost per ACTIVE ppm.
        """)

# ══════════════════════════════════════════════════════════════════════════
#  OTHER PAGES
# ══════════════════════════════════════════════════════════════════════════

elif page == "💊 Dose Optimizer":
    render_header()
    st.header("💊 Dose Optimizer & Re-dosing Schedule")
    st.info("Use the 🔮 Prediction page — Best Dose and Re-dosing Schedule "
            "are built in.")

elif page == "💰 ROI Calculator":
    render_header()
    st.header("💰 ROI Calculator")
    c1, c2, c3 = st.columns(3)
    pipe_km = c1.number_input("Pipeline Length (km)", 1, 1000, 50, 5)
    fail_cost = c2.number_input("Failure Cost ($/event)", 10000, 10000000, 500000, 10000)
    years = c3.number_input("Period (years)", 1, 30, 10, 1)
    c4, c5 = st.columns(2)
    dose = c4.number_input("Dose (ppm)", 0, 500, 50, 5)
    chem_cost = c5.number_input("Chemical Cost ($/L)", 1, 100, 15, 1)

    if st.button("💰 Calculate ROI", type="primary", use_container_width=True):
        v0 = dict(fluid="🛢️ Crude Oil", pressure=1000, temp=150, wt=8,
                  material="API 5L X52", coating_condition="Good", cp_on="Yes",
                  co2=2, h2s=100, ph=6.5, dose=0)
        r0, _, _, _, _, _ = full_calculation(v0)
        r1, _, _, _, _, _ = full_calculation({**v0, "dose": dose})
        f0 = r0 * years * 0.1
        f1 = r1 * years * 0.1
        savings = (f0 - f1) * fail_cost
        daily = pipe_km * 1000 * 0.0001 * dose / 1000
        spend = daily * 365 * years * chem_cost
        net = savings - spend
        roi = (net / spend * 100) if spend > 0 else 0
        a, b, c = st.columns(3)
        a.metric("Rate w/o Inhibitor", f"{r0:.3f} mm/yr")
        b.metric("Rate with Inhibitor", f"{r1:.3f} mm/yr")
        c.metric("Net Benefit", f"${net:,.0f}")
        d1, d2 = st.columns(2)
        d1.metric("Program Cost", f"${spend:,.0f}")
        d2.metric("ROI", f"{roi:.0f}%")
        if roi > 0:
            st.success(f"✅ Inhibitor program saves ${net:,.0f} over {years} years")
        else:
            st.warning("⚠️ Review program economics")

elif page == "🌍 Geophysical":
    render_header()
    st.header("🌍 Geophysical Data — Upload & Map")
    up = st.file_uploader("Upload Geophysical CSV", type="csv")
    if up:
        gdf = pd.read_csv(up)
        st.success(f"✅ {len(gdf)} survey points loaded")
        st.dataframe(gdf.head(10))
        latc = [c for c in gdf.columns if 'lat' in c.lower()]
        lonc = [c for c in gdf.columns if 'lon' in c.lower()]
        if latc and lonc:
            md = gdf[[latc[0], lonc[0]]].rename(
                columns={latc[0]: 'latitude', lonc[0]: 'longitude'})
            st.map(md, size=2000)
    else:
        st.info("Upload a CSV with Latitude, Longitude, Soil_Resistivity_ohm_m, "
                "Soil_pH, Moisture_Content_percent, CP_Potential_mV, DCVG_mV")

elif page == "🏛️ NACE Compliance":
    render_header()
    st.header("🏛️ NACE Compliance Checker")
    c1, c2 = st.columns(2)
    cp = c1.number_input("CP Potential (mV)", -1500, -500, -950, 10)
    dcvg = c2.number_input("DCVG Coating Defect (mV)", 0, 400, 50, 5)
    ok_cp = cp <= -850
    ok_dcvg = dcvg < 100
    st.write(f"**Cathodic Protection:** {'✅ PASS' if ok_cp else '❌ FAIL'} "
             f"(threshold -850 mV)")
    st.write(f"**Coating Integrity:** {'✅ PASS' if ok_dcvg else '❌ FAIL'} "
             f"(threshold 100 mV)")
    if ok_cp and ok_dcvg:
        st.success("✅ FULL NACE COMPLIANCE")
    else:
        st.error("❌ Remediation required")

elif page == "🗺️ GIS Map":
    render_header()
    st.header("🗺️ GIS Injection Points Map")
    up = st.file_uploader("Upload Injection Points CSV", type="csv", key="inj")
    if up:
        idf = pd.read_csv(up)
        st.success(f"✅ {len(idf)} injection points loaded")
        st.dataframe(idf.head(10))
        latc = [c for c in idf.columns if 'lat' in c.lower()]
        lonc = [c for c in idf.columns if 'lon' in c.lower()]
        if latc and lonc:
            md = idf[[latc[0], lonc[0]]].rename(
                columns={latc[0]: 'latitude', lonc[0]: 'longitude'})
            st.map(md, size=2000)
    else:
        st.info("Upload Corzynth_Injection_Points.csv with latitude/longitude columns")

elif page == "🔔 Alerts":
    render_header()
    st.header("🔔 Alerts")
    st.info("Alerts appear automatically on the 🔮 Prediction page when "
            "H₂S > 500 ppm, rate > 0.5 mm/yr, or sour service is detected.")

elif page == "ℹ️ About":
    render_header()
    st.markdown("""
**Corzynth v10.0 —  Transparent Engineering **

Every calculation is shown step-by-step:
1. Temperature → Kelvin | 2. Pressure → bar | 3. CO₂ partial pressure  
4. Fugacity coefficient | 5. CO₂ fugacity | 6. Base rate (de Waard-Milliams)  
7. pH correction | 8. H₂S sour service | 9. Inhibitor reduction  
10. Coating protection | 11. Cathodic protection | 12. Fluid type  
13. **Material grade metallurgy** | 14. Remaining life (capped at 40-yr design life)

**Unbeatable features competitors don't have:**
- 🤖 AI Engineering Explanation — plain-English "why" + action plan
- 📊 Sensitivity Analysis — which input affects your rate most (incl. material)
- 🏗️ Material metallurgy — X42 to X70, Stainless 316L, Duplex 2205
- 📈 Year-by-year wall loss projection table + milestones
- 🔬 Fully transparent, verifiable physics — no black box
- 💊 Dose optimizer with plateau detection ("inhibitor alone can't fix this")
""")