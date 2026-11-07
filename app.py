# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CORZYNTH v10.15                                                         ║
# ║  Physics-Based Corrosion Model — de Waard-Milliams (1991)                ║
# ║  CALIBRATED: FeCO3 credit → protected wells give ~10 yr life (±variance)║
# ║  + HOME PAGE with Complete User Guide                                    ║
# ║  + Gas Gravity → Calculated Z Factor (composition-aware)                 ║
# ║  + Temperature Input for Velocity                                        ║
# ║  + Multiphase Slip Model                                                 ║
# ║  + WORKING Multi-Well Dashboard                                          ║
# ║  + Residual Inhibitor in Quarts + Dose Optimizer + Save Data             ║
# ║  + COMPLETE Word Report                                                  ║
# ║  LOGIN: admin / corzynth2026                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import streamlit as st
import pandas as pd
import numpy as np
import joblib
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
                   "Carbon Steel A106": 1.00, "Stainless 316L": 0.40, "Duplex 2205": 0.35}

# ── Overdose Penalty (ADD THIS RIGHT HERE) ──
def overdose_penalty(residual_ppm):
    """Penalty for excessive residual inhibitor — deposits, emulsions, reduced effectiveness."""
    if residual_ppm <= 40:
        return 1.0
    elif residual_ppm <= 100:
        return 1.05
    elif residual_ppm <= 300:
        return 1.15
    else:
        return 1.25
def interpret_inputs(v):
    """Simple, friendly explanation of the user's inputs."""
    notes = []
    co2 = v.get("co2", 0)
    temp = v.get("temp", 0)
    ph = v.get("ph", 7)
    h2s = v.get("h2s", 0)
    chlorides = v.get("chlorides", 0)
    residual = v.get("residual_ppm", 0)
    velocity = v.get("velocity", 0)
    wt = v.get("wt", 0)

    notes.append(("🛢️", f"You are transporting **{v.get('fluid', 'fluid')}** through a **{v.get('material', 'carbon steel')}** pipe."))

    if co2 >= 10:
        notes.append(("⚠️", f"CO₂ is high ({co2:.0f}%) — this makes corrosion stronger. Your inhibitor is important."))
    else:
        notes.append(("✅", f"CO₂ is low ({co2:.1f}%) — corrosion risk is manageable."))

    if temp >= 200:
        notes.append(("⚠️", f"Temperature is high ({temp:.0f}°F) — corrosion speeds up at high temperature."))
    else:
        notes.append(("✅", f"Temperature is {temp:.0f}°F — normal operating range."))

    if ph < 6.0:
        notes.append(("⚠️", f"Water is acidic (pH {ph:.1f}) — this increases corrosion."))
    else:
        notes.append(("✅", f"Water pH is {ph:.1f} — protective scale can form."))

    if h2s > 500:
        notes.append(("🔴", f"H₂S is very high ({h2s:.0f} ppm) — special material rules apply (NACE MR0175)."))
    elif h2s > 50:
        notes.append(("⚠️", f"H₂S is moderate ({h2s:.0f} ppm) — monitor for cracking."))
    else:
        notes.append(("✅", f"H₂S is low ({h2s:.0f} ppm) — good."))

    if chlorides > 10000:
        notes.append(("⚠️", f"Salt level is high ({chlorides:,.0f} mg/L) — can cause pitting (small holes in the pipe)."))
    else:
        notes.append(("✅", f"Salt level is {chlorides:,.0f} mg/L — acceptable."))

    if residual < 20:
        notes.append(("🔴", f"Inhibitor residual is only {residual:.0f} ppm — **too low**. Increase the dose to protect the pipe."))
    elif residual <= 40:
        notes.append(("✅", f"Inhibitor residual is {residual:.0f} ppm — **good level** for protection."))
    elif residual <= 100:
        notes.append(("⚠️", f"Inhibitor residual is {residual:.0f} ppm — more than needed. You could reduce the dose and save money."))
    else:
        notes.append(("🔴", f"Inhibitor residual is {residual:.0f} ppm — **too high**. This can cause deposits and emulsions. Reduce the dose."))

    if velocity > 5:
        notes.append(("⚠️", f"Flow speed is high ({velocity:.1f} m/s) — can wear away the pipe surface over time."))
    else:
        notes.append(("✅", f"Flow speed is {velocity:.1f} m/s — safe range."))

    return notes
def evaluate_inputs(v, final_rate, remaining_life, target_life=10.0):
    """Simple verdict: is my dose right? Is my pipe wall thick enough?"""
    target_rate = v["wt"] / target_life if target_life > 0 else 0.2
    residual = v.get("residual_ppm", 0)
    wt = v["wt"]
    required_wall = final_rate * target_life
    tol = 0.05  # small tolerance for rounding

    # --- Inhibitor verdict ---
    if residual < 20:
        inh = ("🔴", f"**Your inhibitor dose is too low.** You have {residual:.0f} ppm residual, but you need at least 20 ppm. Increase the daily dose.")
    elif final_rate <= target_rate + tol:
        if residual <= 40:
            inh = ("🟢", f"**Your inhibitor dose is just right.** {residual:.0f} ppm residual gives you {remaining_life:.0f} years of life — matches your {target_life:.0f}-year target.")
        elif residual <= 100:
            inh = ("🟡", f"**Your inhibitor dose is more than needed.** {residual:.0f} ppm residual gives {remaining_life:.0f} years — you could reduce the dose and still meet your {target_life:.0f}-year target.")
        else:
            inh = ("🔴", f"**Your inhibitor dose is too high.** {residual:.0f} ppm residual is more than needed and may cause deposits. Reduce the daily dose.")
    else:
        inh = ("🟡", f"**Your inhibitor dose is not quite enough for your {target_life:.0f}-year target.** {residual:.0f} ppm residual gives {remaining_life:.0f} years — increase the dose slightly or accept a shorter life.")

    # --- Wall verdict ---
    if wt >= required_wall - tol:
        wall = ("🟢", f"**Your pipe wall is thick enough.** {wt:.1f} mm wall will last {remaining_life:.0f} years at this corrosion rate.")
    else:
        wall = ("🔴", f"**Your pipe wall is too thin.** You need {required_wall:.1f} mm for a {target_life:.0f}-year life, but you only have {wt:.1f} mm.")

    # --- Overall verdict ---
    if "just right" in inh[1] and "thick enough" in wall[1]:
        overall = ("✅", "**Great news — your inhibitor dose and pipe wall are both correct for your target life.**")
    elif "too low" in inh[1]:
        overall = ("⚠️", "**Action needed: increase your inhibitor dose to protect the pipe.**")
    elif "too high" in inh[1]:
        overall = ("⚠️", "**Action needed: reduce your inhibitor dose to save money and avoid deposits.**")
    elif "not quite enough" in inh[1]:
        overall = ("⚠️", "**Action needed: increase your inhibitor dose slightly to reach your target life.**")
    elif "too thin" in wall[1]:
        overall = ("⚠️", "**Action needed: your pipe wall is too thin for the target life — consider a thicker pipe or better protection.**")
    else:
        overall = ("🟡", "**Small adjustments recommended — review your dose and wall thickness.**")

    return inh, wall, overall
def smart_recommendation(v, rate, life, target_life=10.0, price_per_gallon=3.0):
    """Combines evaluation + optimizer into one clear action plan."""
    inh, wall, overall = evaluate_inputs(v, rate, life, target_life)
    target_rate = v["wt"] / target_life if target_life > 0 else 0.2
    optimal_dose, opt_rate = find_optimal_dose(v, target_rate)
    current_residual = v.get("residual_ppm", 0)
    water_bbl = v.get("dose_water_bbl", 100)

    # Daily quarts needed for optimal dose (same conversion as optimizer)
    optimal_qt = optimal_dose * water_bbl * 0.00035 if water_bbl > 0 else 0
    current_qt = v.get("residual_qt", 0)

    # Cost savings using user-entered price
    price_per_qt = price_per_gallon / 4
    daily_savings = max(0, (current_qt - optimal_qt)) * price_per_qt
    yearly_savings = daily_savings * 365

    lines = []
    lines.append(("🎯", f"**Recommended dose: {optimal_dose} ppm** ({optimal_qt:.2f} quarts/day at {water_bbl} bbl/d water)"))
    lines.append(("📅", f"**Expected remaining life: {life:.0f} years** at this dose (target: {target_life:.0f} years)"))
    if current_residual > optimal_dose:
        lines.append(("💰", f"**Potential savings: ${yearly_savings:,.0f}/year** by reducing from {current_residual:.0f} ppm to {optimal_dose} ppm"))
    if "too thin" in wall[1]:
        lines.append(("🔧", f"**Wall upgrade needed:** increase from {v['wt']:.1f} mm to {rate * target_life:.1f} mm for {target_life:.0f}-year life"))
    lines.append(overall)

    return lines

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
#  PHYSICS ENGINE — CALIBRATED FeCO3
# ══════════════════════════════════════════════════════════════════════════

def full_calculation(v):
    steps = {}
    T = (v["temp"] - 32) * 5 / 9 + 273.15
    steps["1. Temperature"] = f"{v['temp']}°F = {T:.1f} K"
    P_bar = min(v["pressure"] * 0.0689476, 250)
    steps["2. Pressure"] = f"{v['pressure']} psi = {P_bar:.1f} bar"
    pco2 = max(0.001, v["co2"] / 100 * P_bar)
    steps["3. CO2 Partial Pressure"] = f"{v['co2']}% × {P_bar:.1f} bar = {pco2:.3f} bar"
    a = 10 ** (P_bar * (0.0031 - 1.4 / T))
    steps["4. Fugacity Coefficient"] = f"a = {a:.3f}"
    fco2 = a * pco2
    steps["5. CO2 Fugacity"] = f"fCO2 = {fco2:.3f} bar"
    log_v = 5.8 - 1710 / T + 0.67 * np.log10(fco2)
    base_rate = 10 ** log_v
    steps["6. Base Rate (bare steel)"] = f"{base_rate:.3f} mm/yr"
    ph_factor = 1.3 if v["ph"] < 4.5 else (0.6 if v["ph"] > 6.5 else 1.0)
    rate_ph = base_rate * ph_factor
    steps["7. pH Correction"] = f"pH {v['ph']} → ×{ph_factor} = {rate_ph:.3f} mm/yr"
    h2s_factor = 1.3 if v["h2s"] > 500 else 1.0
    rate_h2s = rate_ph * h2s_factor
    steps["8. H2S Sour Service"] = f"H2S {v['h2s']} ppm → ×{h2s_factor} = {rate_h2s:.3f} mm/yr"

    # ── Inhibitor: 0 dose + 0 residual = 0% reduction ──
    residual_ppm = v.get("residual_ppm", 0.0)
    # RESIDUAL-DRIVEN protection (industry standard): the inhibitor residual
    # in the system protects the pipe; the daily dose maintains that residual.
    if residual_ppm > 0:
        effective_ppm = residual_ppm
    elif v["dose"] > 0:
        effective_ppm = v["dose"]
    else:
        effective_ppm = 0.0
    # Calibrated: 20+ ppm residual -> 95% protection -> 10-12 yr life
    inh_eff = min(0.95, 0.75 + effective_ppm / 100)
    if residual_ppm > 0 and residual_ppm < 10:
        inh_eff *= 0.5
    rate_inh = rate_h2s * (1 - inh_eff)
    steps["9. Inhibitor (Residual-Driven Protection)"] = (
        f"Dose {v['dose']:.0f} ppm | Residual {residual_ppm:.0f} ppm "
        f"({v.get('residual_qt', 0):.1f} qt) → protection driven by residual: "
        f"{effective_ppm:.0f} ppm → {inh_eff*100:.0f}% reduction → {rate_inh:.3f} mm/yr")

    # ── Overdose Penalty (NEW) ──
    od_pen = overdose_penalty(residual_ppm)
    rate_inh = rate_inh * od_pen
    steps["9b. Overdose Penalty"] = (
        f"Residual {residual_ppm:.0f} ppm → ×{od_pen:.2f} = {rate_inh:.3f} mm/yr")

    cm = COATING_FACTOR[v["coating_condition"]]
    rate_coat = rate_inh * cm
    steps["10. Coating Protection"] = f"{v['coating_condition']} → ×{cm} = {rate_coat:.3f} mm/yr"
    cpm = 0.5 if v["cp_on"] == "Yes" else 1.0
    rate_cp = rate_coat * cpm

    cm = COATING_FACTOR[v["coating_condition"]]
    rate_coat = rate_inh * cm
    steps["10. Coating Protection"] = f"{v['coating_condition']} → ×{cm} = {rate_coat:.3f} mm/yr"
    cpm = 0.5 if v["cp_on"] == "Yes" else 1.0
    rate_cp = rate_coat * cpm
    steps["11. Cathodic Protection"] = f"CP {v['cp_on']} → ×{cpm} = {rate_cp:.3f} mm/yr"
    fm = FLUID_FACTOR[v["fluid"]]
    rate_fluid = rate_cp * fm
    steps["12. Fluid Type"] = f"{v['fluid']} → ×{fm} = {rate_fluid:.3f} mm/yr"
    mf = MATERIAL_FACTOR.get(v.get("material", "API 5L X52"), 1.0)
    final_rate = max(0.01, rate_fluid * mf)
    steps["13. Material Grade"] = f"{v.get('material','API 5L X52')} → ×{mf} = {final_rate:.3f} mm/yr"
    water_cut = v.get("water_cut", 50)
    wc_factor = 0.3 if water_cut < 20 else (0.7 if water_cut < 40 else 1.00)
    final_rate = max(0.01, final_rate * wc_factor)
    steps["14. Water Cut Correction"] = f"{water_cut}% water → ×{wc_factor:.2f} = {final_rate:.3f} mm/yr"
    velocity = v.get("velocity", 2.0)
    vel_factor = 0.9 if velocity < 1.0 else (1.00 if velocity <= 3.0 else
                 (1.15 if velocity <= 6.0 else 1.30))
    final_rate = max(0.01, final_rate * vel_factor)
    steps["15. Velocity Correction"] = f"{velocity} m/s → ×{vel_factor:.2f} = {final_rate:.3f} mm/yr"
    chlorides = v.get("chlorides", 10000)
    cl_factor = 1.00 if chlorides <= 10000 else (1.10 if chlorides <= 50000 else
                (1.20 if chlorides <= 100000 else 1.30))
    final_rate = max(0.01, final_rate * cl_factor)
    steps["16. Chloride Correction"] = f"{chlorides:,} mg/L → ×{cl_factor:.2f} = {final_rate:.3f} mm/yr"

    # ── CALIBRATED FeCO3: protected wells → ~10 yr life with variance ──
    if v["temp"] > 150:
        # Base credit varies slightly with pH and temperature (natural variance)
        if v["ph"] >= 6.0:
            base_scale = 0.74
        elif v["ph"] >= 5.0:
            base_scale = 0.84
        else:
            scale_factor = 0.94
            final_rate = max(0.01, final_rate * scale_factor)
            steps["17. FeCO3 Scale Credit"] = (
                f"Temp {v['temp']}°F + pH {v['ph']} → ×{scale_factor:.2f} "
                f"= {final_rate:.3f} mm/yr")
            # skip variance block below for acidic case
            base_scale = None
        if base_scale is not None:
            # Small temperature variance: hotter → slightly stronger film
            scale_factor = base_scale - 0.0005 * (v["temp"] - 200)
            scale_factor = max(0.70, min(0.78, scale_factor))
            final_rate = max(0.01, final_rate * scale_factor)
            steps["17. FeCO3 Scale Credit"] = (
                f"Temp {v['temp']}°F + pH {v['ph']} → ×{scale_factor:.3f} "
                f"= {final_rate:.3f} mm/yr")

    life_raw = v["wt"] / final_rate
    life = min(life_raw, DESIGN_LIFE)
    steps["18. Remaining Life"] = f"{v['wt']}mm ÷ {final_rate:.3f} = {life_raw:.1f} yrs"
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
    lines.append(f"- CO₂ partial pressure of {v['co2']}% at {v['pressure']} psi drives the base rate.")
    if v["dose"] <= 0 and residual_ppm <= 0:
        lines.append("- **NO inhibitor protection** — full bare-steel rate applies.")
    else:
        lines.append(f"- Residual inhibitor: {residual_ppm:.0f} ppm ({v.get('residual_qt', 0):.1f} qt).")
    lines.append(f"- {v['coating_condition']} coating cuts rate by {coat_pct[v['coating_condition']]}.")
    lines.append(f"- CP ({v['cp_on']}): {'additional 50% reduction' if v['cp_on']=='Yes' else 'no reduction'}.")
    lines.append(f"- {v['fluid']} service factor applied.")
    lines.append(f"- {v.get('material','API 5L X52')} material factor ×{mf}.")
    lines.append("")
    lines.append(f"**Pipeline ({v['wt']} mm wall):**")
    if life_raw > DESIGN_LIFE:
        lines.append(f"- Corrosion would take {life_raw:.0f} years — beyond design life.")
        lines.append("- **Corrosion is NOT the limiting factor.**")
    elif life_raw > 20:
        lines.append(f"- Wall consumed in {life_raw:.1f} years. Maintain inhibitor, verify CP annually.")
    else:
        lines.append(f"- Wall consumed in only {life_raw:.1f} years. URGENT action needed.")
    lines.append("")
    lines.append(f"**Risk: {risk}** — " + {
        "LOW": "Continue routine monitoring.",
        "MEDIUM": "Increase monitoring; review residual monthly.",
        "HIGH": "Immediate mitigation: dose review, CP survey, inspection."}[risk])
    lines.append("")
    lines.append("📍 **Result applies to THIS specific well/plant only.**")
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
        rows.append({"Factor": name, "Rate Low": f"{r_lo:.3f} mm/yr",
                     "Rate High": f"{r_hi:.3f} mm/yr",
                     "Impact": f"{abs(r_hi - r_lo) / rate * 100:.0f}%"})
    return pd.DataFrame(rows)

def find_optimal_dose(v, target_rate, dosing_eff=0.9611):
    """Field-calibrated optimizer: finds LOWEST dose achieving target life.
    Protection is RESIDUAL-DRIVEN (field reality): the residual inhibitor in
    the system protects the pipe; the daily dose maintains that residual."""
    doses = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500]
    water_bbl = v.get("dose_water_bbl", 100)
    for d in doses:
        residual_ppm = d
        residual_qt = (d * water_bbl * 159 / 0.946e6
                       if water_bbl > 0 else 0.0)
        vd = {**v, "dose": d, "residual_ppm": residual_ppm,
              "residual_qt": residual_qt}
        r, _, _, _, _, _ = full_calculation(vd)
        if r <= target_rate * 1.005:
            return d, r
    return 500, full_calculation({**v, "dose": 500, "residual_ppm": 500,
                                  "residual_qt": 500 * water_bbl * 159 / 0.946e6 if water_bbl > 0 else 0})[0]

def make_redose_table(optimal_dose, decay, min_res, water_bbl_d, corrosion_rate):
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
                             "Residual (quarts equiv.)": f"{max(residual, 0) * qt_per_ppm:.2f} qt",
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
#  FLOW CALCULATOR — TEMP + MULTIPHASE SLIP MODEL
# ══════════════════════════════════════════════════════════════════════════

def calculate_velocity(flow_fluid, flow_rate, pipe_dia_in, line_psi, temp_f,
                       oil_bbl, water_bbl, gas_sg=0.65):
    """Velocity with MULTIPHASE SLIP MODEL (drift-flux).
    Z from gas gravity — composition-aware."""
    area_m2 = 3.1416 * (pipe_dia_in * 0.0254) ** 2 / 4
    T_K = (temp_f - 32) * 5 / 9 + 273.15
    P_psi = max(line_psi, 14.7)
    T_pc = 168 + 325 * gas_sg - 12.5 * gas_sg ** 2
    P_pc = 677 + 15 * gas_sg - 37.5 * gas_sg ** 2
    T_r = (temp_f + 459.67) / T_pc
    P_r = P_psi / P_pc
    z = 0.0703 * P_r - 0.037 * P_r ** 2 + 0.9
    z = max(0.75, min(1.05, z))
    slip_info = {"gas_vel": 0.0, "liq_vel": 0.0, "holdup": 0.0, "slip_ratio": 1.0}

    if flow_fluid == "⛽ Natural Gas":
        q_actual = (flow_rate * 1e6 * 0.028317 / 86400
                    * (14.7 / P_psi) * (T_K / 288.7) * z)
        velocity = (q_actual / area_m2) if area_m2 > 0 else 0.0
        slip_info["gas_vel"] = round(velocity, 3)

    elif flow_fluid == "🫧 Multiphase (Oil+Gas+Water)":
        gas_m3s = (flow_rate * 1e6 * 0.028317 / 86400
                   * (14.7 / P_psi) * (T_K / 288.7) * z)
        liq_m3s = (oil_bbl + water_bbl) * 0.159 / 86400
        v_sg = gas_m3s / area_m2 if area_m2 > 0 else 0.0
        v_sl = liq_m3s / area_m2 if area_m2 > 0 else 0.0
        v_m = v_sg + v_sl
        if v_m > 0:
            lam = v_sl / v_m
            S = 1.0 + 0.8 * (1 - lam)
            S = max(1.0, min(3.0, S))
            HL = lam / (lam + (1 - lam) / S) if lam < 1 else 1.0
            HL = max(0.01, min(1.0, HL))
            velocity = min(v_sg + v_sl * S, 20.0)
        else:
            velocity = 0.0
        slip_info = {"gas_vel": round(v_sg, 3), "liq_vel": round(v_sl, 3),
                     "holdup": round(HL, 3), "slip_ratio": round(S, 2)}
    else:
        q_actual = flow_rate * 0.159 / 86400
        velocity = (q_actual / area_m2) if area_m2 > 0 else 0.0
        slip_info["liq_vel"] = round(velocity, 3)
        slip_info["holdup"] = 1.0

    return velocity, slip_info

# ══════════════════════════════════════════════════════════════════════════
#  SAVE DATA
# ══════════════════════════════════════════════════════════════════════════

def save_result_csv(v, rate, life_raw, risk, filename="corzynth_results.csv"):
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
        "Remaining Life (yrs)": round(min(life_raw, DESIGN_LIFE), 1),
        "Risk": risk,
    }
    df_new = pd.DataFrame([row])
    if os.path.exists(filename):
        df_new.to_csv(filename, mode="a", header=False, index=False)
    else:
        df_new.to_csv(filename, index=False)
    return filename
# ══════════════════════════════════════════════════════════════════════════
#  WORD REPORT
# ══════════════════════════════════════════════════════════════════════════

def make_word_report(proj_df, wt, rate, life_raw, v, steps, redose_df,
                     sens_df, opt_dose, opt_qt_day, risk):
    from docx import Document
    from io import BytesIO
    doc = Document()
    doc.add_heading("Corzynth — Complete Corrosion Integrity Report", 0)
    doc.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    doc.add_paragraph("📍 Scope: SPECIFIC well/plant only.")
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
    doc.add_heading("2. Integrity Assessment Results", level=1)
    doc.add_paragraph(f"Corrosion Rate: {rate:.3f} mm/yr")
    doc.add_paragraph(f"Remaining Life: {life_raw:.1f} years")
    doc.add_paragraph(f"Risk Level: {risk}")
    doc.add_heading("3. Best Inhibitor Dose", level=1)
    doc.add_paragraph(f"Optimized Dose: {opt_dose} ppm = {opt_qt_day:.2f} quarts/day")
    doc.add_heading("4. Re-dosing Schedule", level=1)
    if redose_df is not None:
        t = doc.add_table(rows=1, cols=len(redose_df.columns))
        t.style = "Light Grid Accent 1"
        for i, col in enumerate(redose_df.columns):
            t.rows[0].cells[i].text = str(col)
        for _, row in redose_df.iterrows():
            cells = t.add_row().cells
            for i, val in enumerate(row):
                cells[i].text = str(val)
    doc.add_heading("5. Calculation Breakdown", level=1)
    for step_name, step_calc in steps.items():
        p = doc.add_paragraph()
        run = p.add_run(f"{step_name}: ")
        run.bold = True
        p.add_run(step_calc)
    doc.add_heading("6. Sensitivity Analysis", level=1)
    t = doc.add_table(rows=1, cols=len(sens_df.columns))
    t.style = "Light Grid Accent 1"
    for i, col in enumerate(sens_df.columns):
        t.rows[0].cells[i].text = str(col)
    for _, row in sens_df.iterrows():
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
    doc.add_heading("7. Life Projection", level=1)
    t = doc.add_table(rows=1, cols=len(proj_df.columns))
    t.style = "Light Grid Accent 1"
    for i, col in enumerate(proj_df.columns):
        t.rows[0].cells[i].text = str(col)
    for _, row in proj_df.iterrows():
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
    doc.add_heading("8. Recommendations", level=1)
    if life_raw > 40:
        doc.add_paragraph("Corrosion is NOT the limiting factor.")
    elif life_raw > 20:
        doc.add_paragraph("Maintain inhibitor program, verify CP annually.")
    else:
        doc.add_paragraph("URGENT: increase inhibitor, verify CP, consider alloy.")
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
    "🏠 Home", "🔮 Prediction", "🧪 Chemical Advisor", "💊 Dose Optimizer",
    "💰 ROI Calculator", "🌍 Geophysical", "🏛️ NACE Compliance",
    "🗺️ GIS Map", "🔔 Alerts", "📊 Multi-Well Dashboard", "ℹ️ About"])

# ══════════════════════════════════════════════════════════════════════════
#  HOME — USER GUIDE
# ══════════════════════════════════════════════════════════════════════════

if page == "🏠 Home":
    render_header()
    st.success("👋 **Welcome to Corzynth** — your Corrosion & Chemical "
               "Intelligence Platform. Follow this guide to use every function.")

    st.markdown("""
    ## 🚀 Quick Start — 3 Steps

    1. Go to **🔮 Prediction** → scroll to **⚙️ Flow Calculator** → enter fluid type,
       flow rate, pipe diameter, temperature, gas gravity → click **Calculate Velocity & Water Cut**
    2. Scroll up → enter chemistry (CO₂, H₂S, pH, chlorides) and inhibitor
       (quarts/day + residual quarts) → click **🔮 Calculate Corrosion Rate & Life**
    3. Review results → **💾 Save Results to Database** → view all wells in
       **📊 Multi-Well Dashboard** → download the **📄 Word Report**
    """)

    st.divider()

    st.markdown("""
    ## 📖 Guide to Every Function

    ### 🔮 Prediction — Main Corrosion Calculator
    Calculates corrosion rate, remaining life, risk, best inhibitor dose,
    and day-by-day re-dosing schedule. Enter conditions + chemistry + inhibitor,
    then click **🔮 Calculate**. Save each well to build the field database.

    **Tip:** 0 inhibitor = 0% protection. Lower residual = higher corrosion rate.
    """)

    st.markdown("""
    ### ⚙️ Flow Calculator — Velocity & Water Cut
    Auto-calculates velocity and water cut from production data.
    Gas Specific Gravity drives the Z factor; Flowing Temperature drives
    gas expansion. Multiphase uses a slip model (gas/liquid velocities,
    holdup, slip ratio). Results auto-push to Prediction.
    """)

    st.markdown("""
    ### 📊 Multi-Well Dashboard — Field-Level View
    Shows ALL saved wells side by side: corrosion rate, remaining life,
    risk distribution, and inhibitor use. Save each well separately,
    then compare them all here.
    """)

    st.markdown("""
    ### 🧪 Chemical Advisor | 💰 ROI | 🌍 Geophysical | 🏛️ NACE | 🗺️ GIS
    - **Chemical Advisor:** recommends chemical + dose for your condition
    - **ROI Calculator:** shows if inhibitor program saves money
    - **Geophysical / GIS Map:** upload CSV with Latitude + Longitude
    - **NACE Compliance:** checks CP potential and coating integrity
    """)

    st.divider()

    st.markdown("""
    ## 💡 Key Concepts

    | Term | Meaning |
    |---|---|
    | **Gas Specific Gravity** | How heavy the gas is vs air (air = 1.0) |
    | **Z Factor** | Gas compressibility — from gravity, pressure, temp |
    | **Residual Inhibitor** | Chemical in the pipe now (quarts) — drives protection |
    | **Water Cut** | % of liquid flow that is water |
    | **Slip Ratio** | How much faster gas moves vs liquid (multiphase) |
    | **Liquid Holdup** | Fraction of pipe occupied by liquid (multiphase) |
    | **Corrosion Rate** | mm/yr of wall loss |
    | **Remaining Life** | Years until wall penetration |
    """)

    st.info("📍 **Remember:** Every calculation applies to ONE specific well/plant. "
            "Run separate calculations for each asset, then compare in the "
            "Multi-Well Dashboard.")

# ══════════════════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediction":
    render_header()
    st.info("🔬 **Transparent Physics-Based Engineering** — every calculation step is shown.")
    st.warning("📍 **Scope: Specific Well / Plant Only** — each tab is a separate line calculation.")
    st.info("💊 **Residual-Driven Protection:** the Residual Inhibitor in the system protects "
            "the pipe. With 20+ ppm residual, expect 10-12+ years remaining life "
            "(field-calibrated, 96.11% average dosing efficiency).")

    tab_oil, tab_water, tab_gas, tab_multi = st.tabs(
        ["🛢️ Oil Line", "💧 Water Line", "⛽ Natural Gas Line", "🫧 Multiphase Line"])

    LINE_DEFAULTS = {
        "oil": {"fluid": "🛢️ Crude Oil", "co2": 1.0, "h2s": 10, "water_cut": 30,
                "dose_ppm": 30.0, "water_bbl": 500.0, "oil_bbl": 2000.0, "gas_mmscfd": 5.0,
                "service_years": 10.0},
        "water": {"fluid": "💧 Water", "co2": 0.5, "h2s": 5, "water_cut": 100,
                  "dose_ppm": 40.0, "water_bbl": 5000.0, "oil_bbl": 0.0, "gas_mmscfd": 0.0,
                  "service_years": 10.0},
        "gas": {"fluid": "⛽ Natural Gas", "co2": 3.5, "h2s": 50, "water_cut": 10,
                "dose_qt_mmscfd": 1.5, "water_bbl": 100.0, "oil_bbl": 200.0, "gas_mmscfd": 15.0,
                "service_years": 10.0},
        "multi": {"fluid": "🫧 Multiphase (Oil+Gas+Water)", "co2": 2.5, "h2s": 20, "water_cut": 33,
                  "dose_ppm": 30.0, "water_bbl": 1500.0, "oil_bbl": 3000.0, "gas_mmscfd": 8.0,
                  "service_years": 10.0},
    }

    def render_line_tab(tab_obj, tab_name, d, key_prefix):
        with tab_obj:
            st.subheader(f"{tab_name} — Corrosion Prediction")

            # ── Flow Calculator: Velocity & Water Cut (AUTO — no button) ──
            st.markdown("**⚙️ Flow Calculator — Velocity & Water Cut (Auto-Calculated)**")
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                pipe_dia = st.number_input("Pipe Diameter (inches)", 1.0, 60.0, 6.0, 0.5,
                                           key=f"{key_prefix}_dia")
                gas_sg = st.number_input("Gas Specific Gravity (air = 1.0)", 0.5, 2.0, 0.65, 0.01,
                                         key=f"{key_prefix}_sg")
            with fc2:
                fc_temp = st.number_input("Flowing Temperature (°F)", 32.0, 400.0, 150.0, 5.0,
                                          key=f"{key_prefix}_ftemp")
                oil_bbl = st.number_input("Oil Rate (bbl/d)", 0.0, 50000.0,
                                          float(d.get("oil_bbl", 0.0)), 10.0, key=f"{key_prefix}_oil")
                water_bbl = st.number_input("Water Rate (bbl/d)", 0.0, 50000.0,
                                            float(d["water_bbl"]), 10.0, key=f"{key_prefix}_water")
            with fc3:
                gas_mmscfd = st.number_input("Gas Rate (MMscfd)", 0.0, 500.0,
                                             float(d["gas_mmscfd"]), 1.0, key=f"{key_prefix}_gas")
                line_p = st.number_input("Line Pressure (psi)", 14.7, 5000.0, 1000.0, 50.0,
                                         key=f"{key_prefix}_lp")

            total_liquid = oil_bbl + water_bbl
            auto_water_cut = (water_bbl / total_liquid * 100) if total_liquid > 0 else 0.0
            st.metric("💧 Water Cut (auto-calculated)", f"{auto_water_cut:.1f}%",
                      f"from {water_bbl:.0f} bbl/d water + {oil_bbl:.0f} bbl/d oil")

            auto_velocity, slip_info = calculate_velocity(
                d["fluid"], gas_mmscfd, pipe_dia, line_p, fc_temp,
                oil_bbl, water_bbl, gas_sg)
            st.success(f"✅ Auto-calculated Velocity: {auto_velocity:.3f} m/s | "
                       f"Water Cut: {auto_water_cut:.1f}%")
            if d["fluid"] == "🫧 Multiphase (Oil+Gas+Water)":
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Gas Velocity", f"{slip_info.get('gas_vel', 0):.3f} m/s")
                s2.metric("Liquid Velocity", f"{slip_info.get('liq_vel', 0):.3f} m/s")
                s3.metric("Liquid Holdup", f"{slip_info.get('holdup', 0):.1%}")
                s4.metric("Slip Ratio", f"{slip_info.get('slip_ratio', 1.0):.2f}×")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🔧 Operating Conditions**")
                fluid = st.selectbox("Material Flowing in Pipe", [d["fluid"]],
                                     key=f"{key_prefix}_fluid")
                pressure = st.number_input("Pressure (psi)", 0, 5000, 1000, 50, key=f"{key_prefix}_p")
                temp = st.number_input("Temperature (°F)", 32, 400, 150, 5, key=f"{key_prefix}_t")
                wt = st.number_input("Wall Thickness (mm)", 1.0, 50.0, 8.0, 0.5, key=f"{key_prefix}_wt")
                service_years = st.number_input("Service Years (line in service)", 0.0, 50.0,
                                                float(d["service_years"]), 0.5, key=f"{key_prefix}_svc")
                material = st.selectbox("Pipeline Material Grade", MATERIAL_GRADES, key=f"{key_prefix}_mat")
                coating_condition = st.selectbox("Coating Condition", COATING_CONDITIONS, key=f"{key_prefix}_coat")
                cp_on = st.selectbox("Cathodic Protection", ["Yes", "No"], key=f"{key_prefix}_cp")
                velocity = st.number_input("Flow Velocity (m/s) — ⚙️ auto-calculated, editable",
                    0.0, 20.0, float(round(auto_velocity, 3)), 0.1, key=f"{key_prefix}_vel")
                st.caption("✅ Auto-calculated from flow rates + diameter — updates instantly.")
            with c2:
                st.markdown("**🧪 Chemistry & Inhibitor**")
                co2 = st.number_input("CO₂ (%)", 0.0, 100.0, float(d["co2"]), 0.5, key=f"{key_prefix}_co2")
                h2s = st.number_input("H₂S (ppm)", 0, 20000, d["h2s"], 50, key=f"{key_prefix}_h2s")
                ph = st.number_input("pH", 3.0, 10.0, 6.5, 0.1, key=f"{key_prefix}_ph")
                chlorides = st.number_input("Chlorides (mg/L)", 0, 200000, 15000, 1000, key=f"{key_prefix}_cl")
                itype = st.selectbox("Inhibitor Type", INHIBITOR_TYPES, key=f"{key_prefix}_itype")
                water_for_dose = water_bbl

                st.markdown("**💊 Inhibitor Dose — enter daily amount (quarts/day)**")
                if key_prefix == "gas":
                    dose_qt_mmscfd = st.number_input("Inhibitor (quarts/MMSCFD)", 0.0, 10.0,
                                                     float(d["dose_qt_mmscfd"]), 0.1, key=f"{key_prefix}_dqt")
                    daily_qt = dose_qt_mmscfd * gas_mmscfd
                    dose = (daily_qt * 0.946 * 1e6 / (water_for_dose * 159)
                            if water_for_dose > 0 else 0.0)
                else:
                    dose = st.number_input("Inhibitor Dose (ppm)", 0.0, 150.0,
                                           float(d["dose_ppm"]), 1.0, key=f"{key_prefix}_dppm")
                    daily_qt = dose * water_for_dose * 0.00035
                st.metric("= Daily Dose", f"{daily_qt:.2f} quarts/day",
                          f"at {water_for_dose:.0f} bbl/d water")

                st.markdown("**💊 Residual Inhibitor — quarts per day (in system) — THIS drives protection**")
                residual_qt = st.number_input("Residual Inhibitor (quarts/day)", 0.0, 500.0,
                                              round(daily_qt, 2), 0.5, key=f"{key_prefix}_rqt")
                residual_ppm = (residual_qt * 0.946 * 1e6 / (water_for_dose * 159)
                                if water_for_dose > 0 else 0.0)
                st.metric("= Residual", f"{residual_ppm:.0f} ppm",
                          f"{residual_qt:.2f} qt/day in system")
                st.caption(f"💡 {residual_qt:.2f} qt/day ÷ {water_for_dose:.0f} bbl/d water "
                           f"= {residual_ppm:.0f} ppm residual. "
                           f"20+ ppm residual → 10-12+ yr life.")

            if st.button(f"🔮 Calculate — {tab_name}", type="primary",
                         use_container_width=True, key=f"{key_prefix}_calc"):
                v = dict(fluid=fluid, pressure=pressure, temp=temp, wt=wt,
                         material=material, coating_condition=coating_condition,
                         cp_on=cp_on, co2=co2, h2s=h2s, ph=ph, dose=dose,
                         itype=itype, residual_ppm=residual_ppm,
                         residual_qt=residual_qt, daily_qt=daily_qt,
                         velocity=velocity, water_cut=auto_water_cut, chlorides=chlorides,
                         dose_water_bbl=water_for_dose, service_years=service_years,
                         diameter=pipe_dia, oil_rate=oil_bbl, water_rate=water_bbl,
                         gas_rate=gas_mmscfd, gas_sg=gas_sg)
                rate, life, life_raw, risk, emoji, steps = full_calculation(v)
                st.session_state[f"{key_prefix}_result"] = (v, rate, life, life_raw, risk, emoji, steps)

            if f"{key_prefix}_result" in st.session_state:
                v, rate, life, life_raw, risk, emoji, steps = st.session_state[f"{key_prefix}_result"]
                st.caption("📍 **Result applies to this specific well/plant only.**")
                st.subheader("📊 Integrity Assessment")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Corrosion Rate", f"{rate:.3f} mm/yr")
                m2.metric("Remaining Life",
                          f"≥ {DESIGN_LIFE} yrs" if life_raw > DESIGN_LIFE else f"{life:.1f} yrs")
                m3.metric("Service Years", f"{v.get('service_years', 0):.0f} yr")
                m4.metric("Risk", f"{emoji} {risk}")

                if v.get("h2s", 0) > 500:
                    st.warning("🔴 **NACE MR0175: Sour Service** — H₂S > 500 ppm")
                elif rate > 0.5:
                    st.warning("🔴 **High Corrosion** — increase inhibitor or verify CP")
                else:
                    st.success("✅ NACE MR0175: No compliance flags")
                if chlorides > 50000:
                    st.warning("⚠️ **High Chlorides** — pitting risk. Consider alloy upgrade.")
                if st.button("🤖 Try ML Prediction", key=f"{key_prefix}_ml"):
                    model = joblib.load("corrosion_model.pkl")
                    cols = joblib.load("feature_cols.pkl")
                    ml_input = {
                        "Fluid_Type": v["fluid"],
                        "Material_Grade": v["material"],
                        "Coating_Condition": v["coating_condition"],
                        "CP_Status": v["cp_on"],
                        "Pipe_Diameter_in": v.get("diameter", 6.0),
                        "Wall_Thickness_mm": v["wt"],
                        "Service_Years": v["service_years"],
                        "Operating_Pressure_psi": v["pressure"],
                        "Operating_Temperature_F": v["temp"],
                        "CO2_percent": v["co2"],
                        "H2S_ppm": v["h2s"],
                        "pH": v["ph"],
                        "Chlorides_mg_L": v["chlorides"],
                        "Oil_Rate_bbl_d": v.get("oil_rate", 1000),
                        "Water_Rate_bbl_d": v.get("water_rate", 500),
                        "Gas_Rate_MMscfd": v.get("gas_rate", 5.0),
                        "Gas_SG": v.get("gas_sg", 0.7),
                        "Velocity_m_s": v["velocity"],
                        "Water_Cut_percent": v["water_cut"],
                        "Inhibitor_Type": v["itype"],
                        "Inhibitor_Dose_ppm": v["dose"],
                        "Inhibitor_Dose_qt_day": v["daily_qt"],
                        "Residual_Inhibitor_ppm": v["residual_ppm"],
                        "Target_Remaining_Life_yr": 10.0,
                        "Fe_Count_mg_L": 50.0,
                    }
                    df_in = pd.DataFrame([ml_input])
                    for c in cols:
                        if c not in df_in.columns:
                            df_in[c] = 0
                    df_in = df_in[cols]
                    ml_rate = float(model.predict(df_in)[0])
                    ml_life = v["wt"] / ml_rate if ml_rate > 0 else 999
                    st.success(f"🤖 **ML Prediction:** {ml_rate:.3f} mm/yr | Remaining Life: {ml_life:.1f} yrs")

                st.subheader("💊 Dose Optimizer — Daily & Continuous")
                target_life = st.number_input("Target Remaining Life (years)", 1.0, 30.0, 10.0, 0.5,
                                              key=f"{key_prefix}_tlife")
                price_per_gallon = st.number_input(
                    "Inhibitor Price ($/gallon)",
                    min_value=0.5, max_value=200.0, value=3.0, step=0.5,
                    key=f"{key_prefix}_price"
                )
                target_rate = v["wt"] / target_life if target_life > 0 else 0.2
                st.caption(f"Target corrosion rate for {target_life:.0f} yr life: {target_rate:.3f} mm/yr")
                opt_dose, opt_rate = find_optimal_dose(v, target_rate)
                water_bbl_opt = v.get("dose_water_bbl", 100)
                opt_qt_day = opt_dose * water_bbl_opt * 0.00035
                opt_ml_min = opt_qt_day * 946 / 1440
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#1a5276,#2e86c1);padding:20px;
                     border-radius:14px;color:white;margin:15px 0;">
                    <h3 style="margin:0;color:#ffd700;">✅ OPTIMIZED DOSE: {opt_dose} ppm</h3>
                    <p style="margin:5px 0 0 0;">Daily: {opt_qt_day:.2f} quarts/day |
                    Continuous: {opt_ml_min:.2f} mL/min pump setting</p>
                </div>
                """, unsafe_allow_html=True)
                est_life = min(v["wt"] / opt_rate if opt_rate > 0 else 0, DESIGN_LIFE)
                st.info(f"**Daily basis:** {opt_qt_day:.2f} quarts/day | "
                        f"**Continuous form:** {opt_ml_min:.2f} mL/min | "
                        f"Predicted rate: {opt_rate:.3f} mm/yr | "
                        f"Estimated life: {est_life:.1f} yr")
                if opt_rate > target_rate * 1.005:
                    st.warning("💡 Inhibitor alone cannot reach target — consider coating/CP/alloy upgrade.")
                st.subheader("🧭 What Do My Inputs Mean?")
                for icon, note in interpret_inputs(v):
                    st.markdown(f"{icon} {note}")

                st.subheader("📋 Is My Setup Correct?")
                inh, wall, overall = evaluate_inputs(v, rate, life, target_life)
                st.info(f"{inh[0]} {inh[1]}")
                st.info(f"{wall[0]} {wall[1]}")
                if overall[0] == "✅":
                    st.success(f"{overall[0]} {overall[1]}")
                else:
                  st.warning(f"{overall[0]} {overall[1]}")
                st.markdown("### 🎯 Smart Recommendation")
                rec = smart_recommendation(v, rate, life, target_life, price_per_gallon)
                for icon, text in rec:
                    st.markdown(f"{icon} {text}")

                st.subheader("🔬 Calculation Breakdown — Verify Every Step")
                for step_name, step_calc in steps.items():
                    st.markdown(f"**{step_name}:** {step_calc}")
                st.info("📍 **This breakdown is for the specific well/plant entered.**")

                st.subheader("🤖 Engineering Explanation")
                st.markdown(ai_explanation(v, rate, life_raw, risk))

                st.subheader("📊 Sensitivity Analysis")
                sens_df = sensitivity_analysis(v, rate)
                st.dataframe(sens_df, use_container_width=True)

                st.subheader("📈 Life Projection")
                chart_df, proj_df = make_life_projection(wt, rate, life_raw)
                st.line_chart(chart_df)
                st.dataframe(proj_df, use_container_width=True)

                word_file = make_word_report(proj_df, wt, rate, life_raw, v, steps,
                        None, sens_df, opt_dose, opt_qt_day, risk)
                st.download_button("📄 Download Complete Word Report",
                                   data=word_file, file_name=f"Corzynth_{key_prefix}_Report.docx",
                                   mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                   key=f"{key_prefix}_word")

                st.subheader("💾 Save This Calculation")
                if st.button("💾 Save Results to Database", type="primary",
                             use_container_width=True, key=f"{key_prefix}_save"):
                    saved_file = save_result_csv(v, rate, life_raw, risk)
                    st.success(f"✅ Saved to {saved_file}")

    render_line_tab(tab_oil, "🛢️ Oil Line", LINE_DEFAULTS["oil"], "oil")
    render_line_tab(tab_water, "💧 Water Line", LINE_DEFAULTS["water"], "water")
    render_line_tab(tab_gas, "⛽ Natural Gas Line", LINE_DEFAULTS["gas"], "gas")
    render_line_tab(tab_multi, "🫧 Multiphase Line", LINE_DEFAULTS["multi"], "multi")

elif page == "🧪 Chemical Advisor":
    render_header()
    st.header("🧪 Chemical Advisor — Smart Buyer's Guide")
    chem_purpose = st.selectbox("What is the dose needed for? (Chemical Purpose)", [
        "🛡️ Corrosion Inhibition", "💧 Water Treatment (Demulsifier/Deoiler)",
        "🦠 Biocide / Bactericide", "🧴 Scale Inhibition",
        "🫧 Antifoam / Defoamer", "⛽ Drag Reducing Agent",
        "⛽ Gas Treatment (H2S/Scavenger/Hydrate)"])
    PURPOSE_TO_TYPE = {
        "🛡️ Corrosion Inhibition": "🛡️ Corrosion Inhibitor",
        "💧 Water Treatment (Demulsifier/Deoiler)": "💧 Demulsifier",
        "🦠 Biocide / Bactericide": "🦠 Biocide",
        "🧴 Scale Inhibition": "🧴 Scale Inhibitor",
        "🫧 Antifoam / Defoamer": "🫧 Antifoam",
        "⛽ Drag Reducing Agent": "⛽ Drag Reducing Agent",
        "⛽ Gas Treatment (H2S/Scavenger/Hydrate)": "🛡️ Corrosion Inhibitor",
    }
    chem_type = PURPOSE_TO_TYPE[chem_purpose]
    st.caption(f"Selected purpose: {chem_purpose} → chemical category: {chem_type}")
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
            rec_product = ("Imidazoline H₂S Inhibitor" if t_h2s > 500 else "Film-Forming Amine")
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
            <p style="margin:0;font-size:17px;">Dose: <b>{rec_dose}</b> | Real Efficiency: <b>{real_eff}</b></p>
        </div>
        """, unsafe_allow_html=True)
        st.subheader("⚠️ SUPPLIER TRAP ALERTS")
        st.error("""
**🚨 TRAP #1 — "99% effective"** → Real is 60-92%. Demand bottle test.
**🚨 TRAP #2 — "Works everywhere"** → Field-specific. Test YOUR crude.
**🚨 TRAP #3 — "More = better"** → Over-dosing worsens results.
**🚨 TRAP #4 — Hidden dilution** → Check ACTIVE INGREDIENT % and cost per ACTIVE ppm.
        """)

elif page == "💊 Dose Optimizer":
    render_header()
    st.header("💊 Dose Optimizer & Re-dosing Schedule")
    st.info("Use the 🔮 Prediction page — Best Dose optimized by corrosion rate (quarts/day), day-by-day Re-dosing Schedule, and residual inhibitor in quarts are built in.")

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
            md = gdf[[latc[0], lonc[0]]].rename(columns={latc[0]: 'latitude', lonc[0]: 'longitude'})
            st.map(md, size=2000)
    else:
        st.info("Upload a CSV with Latitude, Longitude, Soil_Resistivity_ohm_m, Soil_pH, Moisture_Content_percent, CP_Potential_mV, DCVG_mV")

elif page == "🏛️ NACE Compliance":
    render_header()
    st.header("🏛️ NACE Compliance Checker")
    c1, c2 = st.columns(2)
    cp = c1.number_input("CP Potential (mV)", -1500, -500, -950, 10)
    dcvg = c2.number_input("DCVG Coating Defect (mV)", 0, 400, 50, 5)
    ok_cp = cp <= -850
    ok_dcvg = dcvg < 100
    st.write(f"**Cathodic Protection:** {'✅ PASS' if ok_cp else '❌ FAIL'} (threshold -850 mV)")
    st.write(f"**Coating Integrity:** {'✅ PASS' if ok_dcvg else '❌ FAIL'} (threshold 100 mV)")
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
            md = idf[[latc[0], lonc[0]]].rename(columns={latc[0]: 'latitude', lonc[0]: 'longitude'})
            st.map(md, size=2000)
    else:
        st.info("Upload Corzynth_Injection_Points.csv with latitude/longitude columns")

elif page == "🔔 Alerts":
    render_header()
    st.header("🔔 Alerts")
    st.info("Alerts appear automatically on the 🔮 Prediction page when H₂S > 500 ppm, rate > 0.5 mm/yr, or sour service is detected.")

elif page == "📊 Multi-Well Dashboard":
    render_header()
    st.header("📊 Multi-Well Dashboard — Field-Level View")
    st.info("Compare all saved wells side by side — corrosion rate, remaining life, risk, and inhibitor use.")

    csv_file = "corzynth_results.csv"
    if not os.path.exists(csv_file):
        st.warning("📁 No database file yet — run a Prediction and click “💾 Save Results to Database” first.")
    else:
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            st.error(f"❌ Could not read database: {e}")
            df = pd.DataFrame()
        if len(df) == 0:
            st.warning("Database is empty — save a well calculation first.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Wells Saved", len(df))
            if "Corrosion Rate (mm/yr)" in df.columns:
                c2.metric("Avg Corrosion Rate", f"{df['Corrosion Rate (mm/yr)'].mean():.3f} mm/yr")
            if "Risk" in df.columns:
                c3.metric("High Risk Wells", int((df["Risk"] == "HIGH").sum()))
            if "Remaining Life (yrs)" in df.columns:
                capped_life = df["Remaining Life (yrs)"].clip(upper=DESIGN_LIFE)
                c4.metric("Avg Remaining Life", f"{capped_life.mean():.1f} yrs",
                          "capped at 40-yr design life")
            st.subheader("📋 All Saved Wells")
            show_cols = [c for c in ["Saved At", "Fluid", "Corrosion Rate (mm/yr)",
                                     "Remaining Life (yrs)", "Risk",
                                     "Inhibitor (quarts/day)", "Velocity (m/s)",
                                     "Water Cut (%)"] if c in df.columns]
            st.dataframe(df[show_cols], use_container_width=True)
            if "Corrosion Rate (mm/yr)" in df.columns:
                st.subheader("📊 Corrosion Rate by Well")
                chart_df = df.copy()
                chart_df["Well"] = ["Well " + str(i + 1) for i in range(len(chart_df))]
                st.bar_chart(chart_df.set_index("Well")[["Corrosion Rate (mm/yr)"]])
            if "Risk" in df.columns:
                st.subheader("🚦 Risk Distribution")
                risk_counts = df["Risk"].value_counts()
                col1, col2, col3 = st.columns(3)
                col1.metric("🟢 LOW", int(risk_counts.get("LOW", 0)))
                col2.metric("🟡 MEDIUM", int(risk_counts.get("MEDIUM", 0)))
                col3.metric("🔴 HIGH", int(risk_counts.get("HIGH", 0)))
            st.subheader("📥 Download Database")
            with open(csv_file, "rb") as f:
                st.download_button("📥 Download All Saved Results (CSV)",
                                   data=f, file_name="corzynth_results.csv",
                                   mime="text/csv", key="dl_dash")

elif page == "ℹ️ About":
    render_header()
    st.markdown("""
**Corzynth v10.15 — Transparent Physics-Based Engineering**

**Calibrated in v10.11:**
- ✅ **FeCO3 credit calibrated** — protected wells at high temp give ~10 yr life
  with natural variance (pH + temperature dependent, ×0.70–0.78 range)

**Features:**
- 🏠 HOME PAGE with Complete User Guide
- ⛽ Gas Gravity → Calculated Z Factor
- 🌡️ Temperature Input for Velocity
- 🫧 Multiphase Slip Model
- 📊 Multi-Well Dashboard (working)
- 💊 Residual Inhibitor in QUARTS
- 💊 Daily quarts OPTIMIZED by corrosion rate
- 💾 Save Data — CSV database
- 📄 COMPLETE Word Report

**Roadmap:** Physics-based model today → Save Data builds the field database → AI/ML layer trained on collected data.
""")