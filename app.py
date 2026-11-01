# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CORZYNTH v10.6                                                          ║
# ║  Physics-Based Corrosion Model — de Waard-Milliams (1991)                ║
# ║  + Material Metallurgy + NACE MR0175                                     ║
# ║  NEW: Residual Inhibitor in Quarts (used in calculation)                 ║
# ║       + Specific Well/Plant Scope Notes                                  ║
# ║       + Corrosion-Rate-Optimized Daily Quarts + Day-by-Day Schedule      ║
# ║       + Unified Flow Calculator + Save Data                              ║
# ║       + COMPLETE Word Report (all page material)                         ║
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

import streamlit_authenticator as stauth


def check_login(u, p):
    try:
        creds = st.secrets["credentials"]["usernames"]
        if u in creds:
            stored_hash = creds[u]["password"]
            try:
                return stauth.Hasher.check_pw(p, stored_hash)
            except AttributeError:
                import bcrypt
                return bcrypt.checkpw(p.encode(), stored_hash.encode())
    except Exception:
        pass
    return False

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
#  TRANSPARENT PHYSICS ENGINE — de Waard-Milliams, FULLY VERIFIABLE
# ══════════════════════════════════════════════════════════════════════════


def full_calculation(v):
    """Physics-based de Waard-Milliams (1991) calculation with fugacity
    correction + material metallurgy. Inhibitor protection is limited by
    the RESIDUAL inhibitor in the pipe. Every step returned for transparency."""
    steps = {}

    T = (v["temp"] - 32) * 5 / 9 + 273.15
    steps["1. Temperature"] = f"{v['temp']}°F = {T:.1f} K"

    P_bar = min(v["pressure"] * 0.0689476, 250)
    steps["2. Pressure"] = f"{v['pressure']} psi = {P_bar:.1f} bar"

    pco2 = max(0.001, v["co2"] / 100 * P_bar)
    steps["3. CO2 Partial Pressure"] = (
        f"{v['co2']}% × {P_bar:.1f} bar = {pco2:.3f} bar")

    a = 10 ** (P_bar * (0.0031 - 1.4 / T))
    steps["4. Fugacity Coefficient"] = (
        f"a = 10^({P_bar:.1f} × (0.0031 - 1.4/{T:.1f})) = {a:.3f}")

    fco2 = a * pco2
    steps["5. CO2 Fugacity"] = f"fCO2 = {a:.3f} × {pco2:.3f} = {fco2:.3f} bar"

    log_v = 5.8 - 1710 / T + 0.67 * np.log10(fco2)
    base_rate = 10 ** log_v
    steps["6. Base Rate (bare steel)"] = (
        f"log V = 5.8 - 1710/{T:.1f} + 0.67×log({fco2:.3f}) = {log_v:.3f} "
        f"→ {base_rate:.3f} mm/yr")

    ph_factor = 1.3 if v["ph"] < 4.5 else (0.6 if v["ph"] > 6.5 else 1.0)
    rate_ph = base_rate * ph_factor
    steps["7. pH Correction"] = f"pH {v['ph']} → ×{ph_factor} = {rate_ph:.3f} mm/yr"

    h2s_factor = 1.3 if v["h2s"] > 500 else 1.0
    rate_h2s = rate_ph * h2s_factor
    steps["8. H2S Sour Service"] = (
        f"H2S {v['h2s']} ppm → ×{h2s_factor} = {rate_h2s:.3f} mm/yr")

    # 9. Inhibitor — protection limited by RESIDUAL in the pipe
    residual_ppm = v.get("residual_ppm", 0.0)
    effective_ppm = min(v["dose"], residual_ppm)
    inh_eff = min(0.95, 0.5 + effective_ppm / 100)
    if residual_ppm < 10:
        inh_eff *= 0.5
    rate_inh = rate_h2s * (1 - inh_eff)
    steps["9. Inhibitor (Residual-Limited)"] = (
        f"Dose {v['dose']:.0f} ppm | Residual {residual_ppm:.0f} ppm "
        f"({v.get('residual_qt', 0):.1f} qt) → effective {effective_ppm:.0f} ppm "
        f"→ {inh_eff*100:.0f}% reduction → {rate_inh:.3f} mm/yr")

    cm = COATING_FACTOR[v["coating_condition"]]
    rate_coat = rate_inh * cm
    steps["10. Coating Protection"] = (
        f"{v['coating_condition']} → ×{cm} = {rate_coat:.3f} mm/yr")

    cpm = 0.5 if v["cp_on"] == "Yes" else 1.0
    rate_cp = rate_coat * cpm
    steps["11. Cathodic Protection"] = (
        f"CP {v['cp_on']} → ×{cpm} = {rate_cp:.3f} mm/yr")

    fm = FLUID_FACTOR[v["fluid"]]
    rate_fluid = rate_cp * fm
    steps["12. Fluid Type"] = f"{v['fluid']} → ×{fm} = {rate_fluid:.3f} mm/yr"

    mf = MATERIAL_FACTOR.get(v.get("material", "API 5L X52"), 1.0)
    final_rate = max(0.01, rate_fluid * mf)
    steps["13. Material Grade"] = (
        f"{v.get('material','API 5L X52')} → ×{mf} = {final_rate:.3f} mm/yr")

    water_cut = v.get("water_cut", 50)
    wc_factor = 0.3 if water_cut < 20 else (0.7 if water_cut < 40 else 1.00)
    final_rate = max(0.01, final_rate * wc_factor)
    steps["14. Water Cut Correction"] = (
        f"{water_cut}% water → ×{wc_factor:.2f} = {final_rate:.3f} mm/yr")

    velocity = v.get("velocity", 2.0)
    vel_factor = 0.9 if velocity < 1.0 else (1.00 if velocity <= 3.0 else
                 (1.15 if velocity <= 6.0 else 1.30))
    final_rate = max(0.01, final_rate * vel_factor)
    steps["15. Velocity Correction"] = (
        f"{velocity} m/s → ×{vel_factor:.2f} = {final_rate:.3f} mm/yr")

    chlorides = v.get("chlorides", 10000)
    cl_factor = 1.00 if chlorides <= 10000 else (1.10 if chlorides <= 50000 else
                 (1.20 if chlorides <= 100000 else 1.30))
    final_rate = max(0.01, final_rate * cl_factor)
    steps["16. Chloride Correction"] = (
        f"{chlorides:,} mg/L → ×{cl_factor:.2f} = {final_rate:.3f} mm/yr")

    if v["temp"] > 150 and v["ph"] >= 6.0:
        final_rate = max(0.01, final_rate * 0.6)
        steps["17. FeCO3 Scale Credit"] = (
            f"Temp {v['temp']}°F + pH {v['ph']} → ×0.60 = {final_rate:.3f} mm/yr")

    life_raw = v["wt"] / final_rate
    life = min(life_raw, DESIGN_LIFE)
    steps["18. Remaining Life"] = (
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
    residual_ppm = v.get("residual_ppm", 0.0)
    lines = []
    lines.append(f"**Why the corrosion rate is {rate:.3f} mm/yr:**")
    lines.append(f"- Bare steel in your CO₂ environment would corrode at the "
                 f"de Waard-Milliams base rate — CO₂ partial pressure of "
                 f"{v['co2']}% at {v['pressure']} psi drives this.")
    lines.append(f"- Your residual inhibitor is {residual_ppm:.0f} ppm "
                 f"({v.get('residual_qt', 0):.1f} quarts in the system) — "
                 f"protection is limited by this residual, not just the "
                 f"daily dose of {v['dose']:.0f} ppm.")
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
    lines.append("")
    lines.append("📍 **Note: This result applies to THIS specific well/plant only** "
                 "— based on its exact inputs. Other wells/plants need their own "
                 "calculation.")
    return "\n".join(lines)


def sensitivity_analysis(v, rate):
    factors = {
        "CO₂ ±50%": ("co2", v["co2"] * 0.5, v["co2"] * 1.5),
        "Temperature ±50°F": ("temp", max(32, v["temp"] - 50), v["temp"] + 50),
        "Inhibitor Dose ±50%": ("dose", max(0, v["dose"] * 0.5), v["dose"] * 1.5),
        "Residual Inhibitor ±50%": ("residual_ppm",
                                    max(0, v.get("residual_ppm", 0) * 0.5),
                                    v.get("residual_ppm", 0) * 1.5),
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


def make_redose_table(optimal_dose, decay, min_res, water_bbl_d, corrosion_rate):
    """DAY-BY-DAY schedule — quarts OPTIMIZED by corrosion rate.
    Residual shown in BOTH ppm and quarts."""
    water_L_day = water_bbl_d * 159
    qt_per_ppm = water_L_day / 1e6 / 0.946 if water_L_day > 0 else 0.0

    if corrosion_rate < 0.2:
        severity_mult = 0.8
    elif corrosion_rate < 0.5:
        severity_mult = 1.0
    else:
        severity_mult = 1.5

    base_qt = optimal_dose * qt_per_ppm * severity_mult
    init_res = max(optimal_dose, min_res * 2)

    rows = []
    day = 0
    for cycle in range(1, 4):
        residual = init_res
        rows.append({"Day": day, "Cycle": cycle,
                     "Quarts to Add Today": f"{base_qt:.2f} qt",
                     "Action": "💉 START / RE-DOSE",
                     "Residual (ppm)": f"{residual:.0f}",
                     "Residual (quarts equiv.)": f"{residual * qt_per_ppm:.2f} qt",
                     "Status": "🟢 Fresh dose"})
        day += 1
        while residual > min_res:
            residual -= decay
            if residual < min_res * 2:
                qt_today = base_qt * 1.5
                action = "📈 INCREASE — boost dose"
            else:
                qt_today = base_qt
                action = "➕ Add daily amount"
            if residual <= min_res:
                rows.append({"Day": day, "Cycle": cycle,
                             "Quarts to Add Today": f"{qt_today:.2f} qt",
                             "Action": "🔴 RE-DOSE TODAY",
                             "Residual (ppm)": f"{max(residual, 0):.0f}",
                             "Residual (quarts equiv.)":
                                 f"{max(residual, 0) * qt_per_ppm:.2f} qt",
                             "Status": "🔴 Re-dose day"})
                break
            rows.append({"Day": day, "Cycle": cycle,
                         "Quarts to Add Today": f"{qt_today:.2f} qt",
                         "Action": action,
                         "Residual (ppm)": f"{residual:.0f}",
                         "Residual (quarts equiv.)": f"{residual * qt_per_ppm:.2f} qt",
                         "Status": "🟢 Protecting"})
            day += 1
    return pd.DataFrame(rows), day, base_qt


def make_life_projection(wt, rate, life_raw):
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
#  UNIFIED FLOW CALCULATOR (4 FLUID TYPES) — NO INHIBITOR
# ══════════════════════════════════════════════════════════════════════════


def calculate_velocity(flow_fluid, flow_rate, pipe_dia_in, line_psi, temp_f,
                       oil_bbl, water_bbl):
    """Velocity from fluid type + flow rate + diameter. NO inhibitor."""
    area_m2 = 3.1416 * (pipe_dia_in * 0.0254) ** 2 / 4
    T_K = (temp_f - 32) * 5 / 9 + 273.15

    if flow_fluid == "⛽ Natural Gas":
        q_actual = (flow_rate * 1e6 * 0.028317 / 86400
                    * (14.7 / max(line_psi, 14.7))
                    * (T_K / 288.7) * 0.9)
    elif flow_fluid == "🫧 Multiphase (Oil+Gas+Water)":
        gas_m3s = (flow_rate * 1e6 * 0.028317 / 86400
                   * (14.7 / max(line_psi, 14.7))
                   * (T_K / 288.7) * 0.9)
        liq_m3s = (oil_bbl + water_bbl) * 0.159 / 86400
        q_actual = gas_m3s + liq_m3s
    else:
        q_actual = flow_rate * 0.159 / 86400

    return (q_actual / area_m2) if area_m2 > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════
#  SAVE DATA TO CSV DATABASE
# ══════════════════════════════════════════════════════════════════════════


def save_result_csv(v, rate, life_raw, risk, filename="corzynth_results.csv"):
    """Save inputs + results to CSV (append mode)."""
    row = {
        "Saved At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Fluid": v.get("fluid", ""),
        "Pressure (psi)": v.get("pressure", ""),
        "Temperature (F)": v.get("temp", ""),
        "Wall Thickness (mm)": v.get("wt", ""),
        "Material": v.get("material", ""),
        "Coating": v.get("coating_condition", ""),
        "CP": v.get("cp_on", ""),
        "CO2 (%)": v.get("co2", ""),
        "H2S (ppm)": v.get("h2s", ""),
        "pH": v.get("ph", ""),
        "Chlorides (mg/L)": v.get("chlorides", ""),
        "Inhibitor Type": v.get("itype", ""),
        "Inhibitor (quarts/day)": v.get("daily_qt", ""),
        "Dose (ppm)": round(v.get("dose", 0), 1),
        "Residual Inhibitor (quarts)": v.get("residual_qt", ""),
        "Residual (ppm)": round(v.get("residual_ppm", 0), 1),
        "Velocity (m/s)": round(v.get("velocity", 0), 3),
        "Water Cut (%)": v.get("water_cut", ""),
        "Corrosion Rate (mm/yr)": round(rate, 4),
        "Remaining Life (yrs)": round(life_raw, 1),
        "Risk": risk,
    }
    df_new = pd.DataFrame([row])
    if os.path.exists(filename):
        df_new.to_csv(filename, mode="a", header=False, index=False)
    else:
        df_new.to_csv(filename, index=False)
    return filename


# ══════════════════════════════════════════════════════════════════════════
#  COMPLETE WORD REPORT — ALL PAGE MATERIAL
# ══════════════════════════════════════════════════════════════════════════


def make_word_report(proj_df, wt, rate, life_raw, v, steps, redose_df,
                     sens_df, opt_dose, opt_qt_day, risk):
    """Complete engineering report — ALL material from the page:
    inputs, results, dose optimization, day-by-day schedule,
    calculation breakdown, sensitivity, and life projection."""
    from docx import Document
    from io import BytesIO

    doc = Document()
    doc.add_heading("Corzynth — Complete Corrosion Integrity Report", 0)
    doc.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    doc.add_paragraph("📍 Scope: This report applies to the SPECIFIC well/plant "
                      "whose inputs were entered — not the whole field.")

    # ── 1. INPUTS ──
    doc.add_heading("1. Well / Plant Inputs", level=1)
    inputs_data = [
        ("Fluid Type", v.get("fluid", "")),
        ("Pressure (psi)", v.get("pressure", "")),
        ("Temperature (°F)", v.get("temp", "")),
        ("Wall Thickness (mm)", v.get("wt", "")),
        ("Material Grade", v.get("material", "")),
        ("Coating Condition", v.get("coating_condition", "")),
        ("Cathodic Protection", v.get("cp_on", "")),
        ("CO2 (%)", v.get("co2", "")),
        ("H2S (ppm)", v.get("h2s", "")),
        ("pH", v.get("ph", "")),
        ("Chlorides (mg/L)", v.get("chlorides", "")),
        ("Flow Velocity (m/s)", round(v.get("velocity", 0), 3)),
        ("Water Cut (%)", v.get("water_cut", "")),
        ("Inhibitor Type", v.get("itype", "")),
        ("Inhibitor Per Day (quarts)", v.get("daily_qt", "")),
        ("Inhibitor Dose (ppm)", round(v.get("dose", 0), 1)),
        ("Residual Inhibitor (quarts)", v.get("residual_qt", "")),
        ("Residual Inhibitor (ppm)", round(v.get("residual_ppm", 0), 1)),
    ]
    t = doc.add_table(rows=1, cols=2)
    t.style = "Light Grid Accent 1"
    t.rows[0].cells[0].text = "Input"
    t.rows[0].cells[1].text = "Value"
    for name, val in inputs_data:
        cells = t.add_row().cells
        cells[0].text = str(name)
        cells[1].text = str(val)

    # ── 2. RESULTS ──
    doc.add_heading("2. Integrity Assessment Results", level=1)
    doc.add_paragraph(f"Corrosion Rate: {rate:.3f} mm/yr")
    doc.add_paragraph(f"Remaining Life: {life_raw:.1f} years "
                      f"(design life cap: 40 years)")
    doc.add_paragraph(f"Risk Level: {risk}")

    # ── 3. DOSE OPTIMIZATION ──
    doc.add_heading("3. Best Inhibitor Dose — Optimized by Corrosion Rate", level=1)
    doc.add_paragraph(f"Optimized Dose: {opt_dose} ppm = {opt_qt_day:.2f} quarts/day")
    doc.add_paragraph(f"Optimized for corrosion rate {rate:.3f} mm/yr "
                      f"using {v.get('itype', 'inhibitor')}.")
    doc.add_paragraph("Higher corrosion rate → higher daily quarts. "
                      "The day-by-day schedule boosts the dose as residual "
                      "drops near the minimum.")

    # ── 4. DAY-BY-DAY RE-DOSING SCHEDULE ──
    doc.add_heading("4. Re-dosing Schedule — Day by Day", level=1)
    doc.add_paragraph("Daily quarts optimized by corrosion rate; residual "
                      "tracked in both ppm and quarts.")
    t = doc.add_table(rows=1, cols=len(redose_df.columns))
    t.style = "Light Grid Accent 1"
    for i, col in enumerate(redose_df.columns):
        t.rows[0].cells[i].text = str(col)
    for _, row in redose_df.iterrows():
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)

    # ── 5. FULL CALCULATION BREAKDOWN ──
    doc.add_heading("5. Calculation Breakdown — Every Step", level=1)
    doc.add_paragraph("Physics-based de Waard-Milliams (1991): "
                      "log10 V = 5.8 - 1710/T + 0.67 log10(fCO2)")
    for step_name, step_calc in steps.items():
        p = doc.add_paragraph()
        run = p.add_run(f"{step_name}: ")
        run.bold = True
        p.add_run(step_calc)

    # ── 6. SENSITIVITY ANALYSIS ──
    doc.add_heading("6. Sensitivity Analysis", level=1)
    t = doc.add_table(rows=1, cols=len(sens_df.columns))
    t.style = "Light Grid Accent 1"
    for i, col in enumerate(sens_df.columns):
        t.rows[0].cells[i].text = str(col)
    for _, row in sens_df.iterrows():
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)

    # ── 7. LIFE PROJECTION ──
    doc.add_heading("7. Year-by-Year Life Projection", level=1)
    t = doc.add_table(rows=1, cols=len(proj_df.columns))
    t.style = "Light Grid Accent 1"
    for i, col in enumerate(proj_df.columns):
        t.rows[0].cells[i].text = str(col)
    for _, row in proj_df.iterrows():
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)

    # ── 8. ENGINEERING RECOMMENDATIONS ──
    doc.add_heading("8. Engineering Recommendations", level=1)
    if life_raw > 40:
        doc.add_paragraph("Corrosion is NOT the limiting factor — plan "
                          "re-assessment at design life.")
    elif life_raw > 20:
        doc.add_paragraph("Maintain inhibitor program, verify CP annually, "
                          "plan inspection at 50% of remaining life.")
    else:
        doc.add_paragraph("URGENT: increase inhibitor dose, verify CP, "
                          "consider corrosion-resistant alloy, "
                          "schedule intelligent pigging within 12 months.")

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer


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
    st.info("🔬 **Transparent Physics-Based Engineering** — every calculation "
            "step is shown so you can verify the result.")

    # ═══ SCOPE NOTE — SPECIFIC WELL/PLANT ═══
    st.warning("📍 **Scope: Specific Well / Plant Only** — this calculation "
               "applies to the ONE well or plant whose inputs you enter below. "
               "It does NOT represent the whole field, gathering system, or "
               "network. Run a separate calculation for each well/plant.")

    if "auto_velocity" not in st.session_state:
        st.session_state["auto_velocity"] = None
    if "auto_water_cut" not in st.session_state:
        st.session_state["auto_water_cut"] = None
    if "dose_water_bbl" not in st.session_state:
        st.session_state["dose_water_bbl"] = 100.0

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

        auto_wc = st.session_state["auto_water_cut"]
        if auto_wc is not None:
            water_cut = st.number_input(
                "Water Cut (%) — ⚙️ auto-calculated, editable",
                0, 100, int(round(auto_wc)), 1, key="wc_main")
            st.caption("✅ Auto-calculated from oil + water rates.")
        else:
            water_cut = st.number_input("Water Cut (%)", 0, 100, 50, 5,
                                        key="wc_main")

        auto_vel = st.session_state["auto_velocity"]
        if auto_vel is not None:
            velocity = st.number_input(
                "Flow Velocity (m/s) — ⚙️ auto-calculated, editable",
                0.0, 15.0, float(round(auto_vel, 3)), 0.1, key="vel_main")
            st.caption("✅ Auto-calculated from fluid type + flow rate + diameter.")
        else:
            velocity = st.number_input("Flow Velocity (m/s)", 0.0, 15.0, 2.0, 0.5,
                                       key="vel_main")

    with c2:
        st.subheader("🧪 Chemistry & Inhibitor")
        co2 = st.number_input("CO₂ (%)", 0.0, 100.0, 2.0, 0.5)
        h2s = st.number_input("H₂S (ppm)", 0, 20000, 100, 50)
        ph = st.number_input("pH", 3.0, 10.0, 6.5, 0.1)
        chlorides = st.number_input("Chlorides (mg/L)", 0, 200000, 10000, 1000)
        itype = st.selectbox("Inhibitor Type", INHIBITOR_TYPES)

        # ── Daily inhibitor: QUARTS/DAY ──
        st.markdown("**💊 Inhibitor — enter daily amount (quarts/day)**")
        cq1, cq2 = st.columns(2)
        daily_qt = cq1.number_input("Inhibitor Per Day (quarts)", 0.0, 500.0,
                                    5.0, 0.5, key="dose_qt")
        water_for_dose = st.session_state["dose_water_bbl"]
        calc_ppm = (daily_qt * 0.946 * 1e6 / (water_for_dose * 159)
                    if water_for_dose > 0 else 0.0)
        dose = round(calc_ppm, 1)
        cq2.metric("= Equivalent Dose", f"{dose:,.0f} ppm",
                   f"at {water_for_dose:.0f} bbl/d water")

        # ── Residual inhibitor: QUARTS (used in calculation) ──
        st.markdown("**💊 Residual Inhibitor — enter in quarts (in system now)**")
        cr1, cr2 = st.columns(2)
        residual_qt = cr1.number_input("Residual Inhibitor (quarts)", 0.0, 500.0,
                                       2.0, 0.5, key="residual_qt")
        residual_ppm = (residual_qt * 0.946 * 1e6 / (water_for_dose * 159)
                        if water_for_dose > 0 else 0.0)
        cr2.metric("= Residual", f"{residual_ppm:,.0f} ppm",
                   f"{residual_qt:.1f} qt in system")
        st.caption(f"💡 {residual_qt} quarts residual ÷ {water_for_dose:.0f} bbl/d "
                   f"water = {residual_ppm:,.0f} ppm residual — this is the "
                   f"protection actually available in the pipe right now, and "
                   f"it drives the corrosion prediction.")

    if st.button("🔮 Calculate Corrosion Rate & Life", type="primary",
                 use_container_width=True):
        v = dict(fluid=fluid, pressure=pressure, temp=temp, wt=wt,
                 material=material, coating_condition=coating_condition,
                 cp_on=cp_on, co2=co2, h2s=h2s, ph=ph, dose=dose,
                 itype=itype, residual_ppm=residual_ppm,
                 residual_qt=residual_qt, daily_qt=daily_qt,
                 velocity=velocity, water_cut=water_cut, chlorides=chlorides)
        rate, life, life_raw, risk, emoji, steps = full_calculation(v)

        # ═══ SCOPE NOTE — RESULT ═══
        st.caption("📍 **Result applies to this specific well/plant only** — "
                   "based on the exact inputs entered (fluid, CO2, H2S, "
                   "temperature, velocity, water cut, inhibitor dose and "
                   "residual). Other wells/plants need their own calculation.")

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

        if chlorides > 50000:
            st.warning("⚠️ **High Chlorides** — increased pitting risk. "
                       "Consider alloy upgrade (13Cr / Duplex) or tighter monitoring.")

        # ── DOSE OPTIMIZATION — BY CORROSION RATE ──
        st.subheader("💊 Best Inhibitor Dose — Optimized by Corrosion Rate")
        target = st.number_input("Your Target Rate (mm/yr)", 0.01, 1.0, 0.1, 0.01,
                                 key="pred_target")
        opt_dose, opt_rate = find_optimal_dose(v, target)
        water_L_day = water_for_dose * 159
        qt_per_ppm = water_L_day / 1e6 / 0.946 if water_L_day > 0 else 0.0
        opt_qt_day = opt_dose * qt_per_ppm
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a5276,#2e86c1);padding:20px;
             border-radius:14px;color:white;margin:15px 0;">
            <h3 style="margin:0;color:#ffd700;">✅ BEST DOSE: {opt_dose} ppm
                = {opt_qt_day:.2f} quarts/day</h3>
            <p style="margin:5px 0 0 0;">Optimized for corrosion rate
            {rate:.3f} mm/yr → target {target} mm/yr with {itype}</p>
        </div>
        """, unsafe_allow_html=True)
        if opt_rate > target:
            st.info("💡 **Inhibitor alone cannot reach your target.** Consider "
                    "upgrading coating condition, strengthening CP, or using a "
                    "corrosion-resistant alloy (13Cr / Duplex 2205).")

        # ── DAY-BY-DAY RE-DOSING — QUARTS OPTIMIZED ──
        st.subheader("🔄 Re-dosing Schedule — Day by Day (Quarts Optimized)")
        decay = st.number_input("Residual Decay (ppm/day)", 1, 50, 10, key="pred_decay")
        min_res = st.number_input("Min Effective Residual (ppm)", 1, 100, 10,
                                  key="pred_minres")
        redose_df, redose_day, base_qt = make_redose_table(
            opt_dose, decay, min_res, water_for_dose, rate)
        c1, c2, c3 = st.columns(3)
        c1.metric("Base Daily Dose", f"{base_qt:.2f} quarts/day",
                  f"optimized for {rate:.2f} mm/yr")
        c2.metric("Pump Setting", f"{base_qt * 946 / 1440:.2f} mL/min",
                  "continuous injection")
        c3.metric("Re-dose Check", f"Day {redose_day}",
                  f"residual ≤ {min_res} ppm")
        st.dataframe(redose_df, use_container_width=True)
        st.success(f"✅ **Optimized daily dosing:** base {base_qt:.2f} quarts/day "
                   f"for {rate:.2f} mm/yr corrosion. As residual drops near the "
                   f"{min_res} ppm minimum, the daily quarts increase 1.5× to "
                   f"maintain protection. Residual is tracked in BOTH ppm and "
                   f"quarts.")

        # ── TRANSPARENT CALCULATION ──
        st.subheader("🔬 Calculation Breakdown — Verify Every Step")
        st.caption("Physics-based de Waard-Milliams (1991): "
                   "log₁₀V = 5.8 − 1710/T + 0.67·log₁₀(fCO₂)")
        for step_name, step_calc in steps.items():
            st.markdown(f'<div class="calc-step"><b>{step_name}:</b> {step_calc}</div>',
                        unsafe_allow_html=True)

        # ═══ SCOPE NOTE — BREAKDOWN ═══
        st.info("📍 **This breakdown is for the specific well/plant entered** — "
                "every step used ITS conditions only (including its residual "
                "inhibitor in quarts). To assess another well or plant, change "
                "the inputs and recalculate — do not apply this result to the "
                "whole system.")

        # ── ENGINEERING EXPLANATION ──
        st.subheader("🤖 Engineering Explanation")
        st.markdown(ai_explanation(v, rate, life_raw, risk))

        # ── SENSITIVITY ──
        st.subheader("📊 Sensitivity Analysis — What Affects Your Rate Most")
        sens_df = sensitivity_analysis(v, rate)
        st.dataframe(sens_df, use_container_width=True)
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

        # ── DOWNLOAD COMPLETE WORD REPORT (ALL MATERIAL) ──
        word_file = make_word_report(proj_df, wt, rate, life_raw, v, steps,
                                     redose_df, sens_df, opt_dose,
                                     opt_qt_day, risk)
        st.download_button(
            label="📄 Download Complete Word Report (All Material)",
            data=word_file,
            file_name="Corzynth_Complete_Report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

        # ── SAVE DATA ──
        st.subheader("💾 Save This Calculation")
        st.caption("Saves all inputs + results to a growing CSV database — "
                   "for well tracking, reporting, and future AI model training.")
        if st.button("💾 Save Results to Database", type="primary",
                     use_container_width=True, key="save_calc"):
            saved_file = save_result_csv(v, rate, life_raw, risk)
            st.success(f"✅ Saved to {saved_file} — download it below anytime.")

        if os.path.exists("corzynth_results.csv"):
            with open("corzynth_results.csv", "rb") as f:
                st.download_button(
                    label="📥 Download All Saved Results (CSV)",
                    data=f,
                    file_name="corzynth_results.csv",
                    mime="text/csv",
                    key="dl_results")

    # ════════════════════════════════════════════════════════════════════
    #  ⚙️ UNIFIED FLOW CALCULATOR — 4 FLUID TYPES (NO INHIBITOR)
    # ════════════════════════════════════════════════════════════════════
    st.subheader("⚙️ Flow Calculator — Velocity & Water Cut (Auto to Prediction)")
    st.caption("Pick the fluid type, enter its flow rate + pipe diameter → "
               "velocity and water cut auto-calculated and pushed to the "
               "Prediction inputs. Inhibitor is separate (quarts/day).")

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        flow_fluid = st.selectbox("Fluid Type", [
            "💧 Water", "🛢️ Crude Oil", "⛽ Natural Gas",
            "🫧 Multiphase (Oil+Gas+Water)"], key="fc_fluid")
        pipe_dia = st.number_input("Pipe Diameter (inches)", 1.0, 60.0, 10.0, 0.5,
                                   key="fc_dia")
    with fc2:
        if flow_fluid == "⛽ Natural Gas":
            flow_rate = st.number_input("Gas Flow Rate (MMscfd)", 0.0, 500.0,
                                        60.0, 1.0, key="fc_rate_gas")
        elif flow_fluid == "💧 Water":
            flow_rate = st.number_input("Water Flow Rate (bbl/d)", 0.0, 50000.0,
                                        100.0, 10.0, key="fc_rate_water")
        elif flow_fluid == "🛢️ Crude Oil":
            flow_rate = st.number_input("Oil Flow Rate (bbl/d)", 0.0, 50000.0,
                                        100.0, 10.0, key="fc_rate_oil")
        else:
            flow_rate = st.number_input("Gas Flow Rate (MMscfd)", 0.0, 500.0,
                                        60.0, 1.0, key="fc_rate_multi")
        line_p = st.number_input("Line Pressure (psi)", 14.7, 5000.0, 1000.0, 50.0,
                                 key="fc_p")
    with fc3:
        oil_bbl = st.number_input("Oil Rate (bbl/d)", 0.0, 50000.0, 0.0, 10.0,
                                  key="fc_oil")
        water_bbl = st.number_input("Water Rate (bbl/d)", 0.0, 50000.0, 100.0, 10.0,
                                    key="fc_water")

    total_liquid = oil_bbl + water_bbl
    auto_water_cut = (water_bbl / total_liquid * 100) if total_liquid > 0 else 0.0
    st.metric("💧 Water Cut (auto-calculated)", f"{auto_water_cut:.1f}%",
              f"from {water_bbl:.0f} bbl/d water + {oil_bbl:.0f} bbl/d oil")

    if st.button("⚙️ Calculate Velocity & Water Cut", type="primary",
                 use_container_width=True, key="fc_calc"):
        velocity = calculate_velocity(flow_fluid, flow_rate, pipe_dia, line_p,
                                      temp, oil_bbl, water_bbl)

        st.session_state["auto_velocity"] = velocity
        st.session_state["auto_water_cut"] = auto_water_cut
        st.session_state["dose_water_bbl"] = water_bbl

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a5276,#2e86c1);padding:22px;
             border-radius:14px;color:white;margin:15px 0;">
            <h3 style="margin:0;color:#ffd700;">
                ✅ Velocity: {velocity:.3f} m/s &nbsp;|&nbsp;
                Water Cut: {auto_water_cut:.1f}%
            </h3>
            <p style="margin:8px 0 0 0;">{flow_fluid} at {flow_rate:,.0f} 
            {'MMscfd' if flow_fluid in ('⛽ Natural Gas',
             '🫧 Multiphase (Oil+Gas+Water)') else 'bbl/d'} 
            in {pipe_dia}" pipe — pushed to Prediction inputs.</p>
        </div>
        """, unsafe_allow_html=True)

        k1, k2 = st.columns(2)
        k1.metric("Velocity", f"{velocity:.3f} m/s",
                  "🟢 Safe" if velocity < 15 else "🔴 High")
        k2.metric("Water Cut", f"{auto_water_cut:.1f}%",
                  f"from {water_bbl:.0f} bbl/d water")

        st.info("🔄 **Velocity and Water Cut pushed to the Prediction inputs "
                "above** — scroll up and click "
                "“🔮 Calculate Corrosion Rate & Life”.")


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
    st.info("Use the 🔮 Prediction page — Best Dose optimized by corrosion "
            "rate (in quarts/day), day-by-day Re-dosing Schedule, and "
            "residual inhibitor in quarts are built in.")


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
**Corzynth v10.6 — Transparent Physics-Based Engineering**


Every calculation is shown step-by-step using the de Waard-Milliams (1991)
corrosion equation:
1. Temperature → Kelvin | 2. Pressure → bar | 3. CO₂ partial pressure  
4. Fugacity coefficient | 5. CO₂ fugacity | 6. Base rate (de Waard-Milliams)  
7. pH correction | 8. H₂S sour service | 9. Inhibitor reduction  
10. Coating protection | 11. Cathodic protection | 12. Fluid type  
13. **Material grade metallurgy** | 14. Remaining life (capped at 40-yr design life)


**New in v10.6:**
- 💊 **Residual Inhibitor in QUARTS** — entered in inputs, converted to
  ppm, and USED in the corrosion calculation (Step 9): low residual means
  weaker protection, higher rate, shorter life
- 📍 **Specific Well/Plant Scope** — clear notes that the result applies
  to the ONE well/plant entered, not the whole field
- 💊 **Daily quarts OPTIMIZED by corrosion rate** — higher rate → more
  quarts/day; schedule boosts dose as residual drops
- ⚙️ **Unified Flow Calculator** — 4 fluid types → velocity AND water cut
  auto-calculated and pushed to Prediction inputs (no inhibitor in velocity)
- 💾 **Save Data** — every calculation saved to a growing CSV database
- 📄 **COMPLETE Word Report** — ALL page material: inputs (incl. residual
  quarts), results, optimized dose, day-by-day schedule, all 18 calculation
  steps, sensitivity, life projection, and recommendations


**Roadmap:** Physics-based model today → Save Data builds the field
database → AI/ML layer trained on collected data to refine predictions.


**Unbeatable features competitors don't have:**
- 🤖 Engineering Explanation — plain-English "why" + action plan
- 📊 Sensitivity Analysis — which input affects your rate most (incl. residual)
- 🏗️ Material metallurgy — X42 to X70, Stainless 316L, Duplex 2205
- 📈 Year-by-year wall loss projection table + milestones
- 🔬 Fully transparent, verifiable physics — no black box
- 💊 Dose optimizer with plateau detection ("inhibitor alone can't fix this")
""")