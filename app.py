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
from corzynth_features import add_features
import json
import csv, re, math, time, hashlib
 
SAVED_INPUTS_FILE = "saved_inputs.json"
 
 
def save_inputs(key_prefix, inputs):
    all_inputs = {}
 
    if os.path.exists(SAVED_INPUTS_FILE):
        try:
            with open(SAVED_INPUTS_FILE, "r") as f:
                all_inputs = json.load(f)
        except Exception:
            all_inputs = {}
 
    all_inputs[key_prefix] = inputs
 
    with open(SAVED_INPUTS_FILE, "w") as f:
        json.dump(all_inputs, f, indent=2)
 
 
def load_inputs(key_prefix):
    if os.path.exists(SAVED_INPUTS_FILE):
        try:
            with open(SAVED_INPUTS_FILE, "r") as f:
                all_inputs = json.load(f)
            return all_inputs.get(key_prefix, {})
        except Exception:
            return {}
    return {}
 
 
def user_csv_file():
    u = st.session_state.get("username", "guest")
    safe = "".join(c for c in u if c.isalnum() or c in "-_") or "guest"
    if safe != u:
        # different usernames must never share a file (e.g. "a.b" vs "ab")
        import hashlib
        safe += "_" + hashlib.sha256(u.encode("utf-8")).hexdigest()[:8]
    return f"corzynth_results_{safe}.csv"
 
 
def save_result_csv(v, rate, life_raw, risk, filename=None):
    if filename is None:
        filename = user_csv_file()
 
    row = {
        "Saved At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Well / Stream Name": v.get("well_name", ""),
        "Fluid": v.get("fluid", ""),
        "Pressure (psi)": v.get("pressure", 0),
        "Temperature (F)": v.get("temp", 0),
        "Wall Thickness (mm)": v.get("wt", 0),
        "Material": v.get("material", ""),
        "Coating": v.get("coating_condition", ""),
        "CP": v.get("cp_on", ""),
        "CO2 (%)": v.get("co2", 0),
        "H2S (ppm)": v.get("h2s", 0),
        "pH": v.get("ph", 0),
        "Chlorides (mg/L)": v.get("chlorides", 0),
        "Inhibitor Type": v.get("itype", ""),
        "Inhibitor (quarts/day)": v.get("daily_qt", 0),
        "Dose (ppm)": round(v.get("dose", 0), 1),
        "Residual Inhibitor (quarts)": v.get("residual_qt", 0),
        "Residual (ppm)": round(v.get("residual_ppm", 0), 1),
        "Velocity (m/s)": round(v.get("velocity", 0), 3),
        "Water Cut (%)": v.get("water_cut", 0),
        "Corrosion Rate (mm/yr)": round(rate, 4),
        "Remaining Life (yrs)": round(min(life_raw, DESIGN_LIFE), 1),
        "Risk": risk,
    }
 
    df_new = pd.DataFrame([row])
 
    if os.path.exists(filename):
        df_new.to_csv(filename, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df_new.to_csv(filename, index=False, encoding="utf-8-sig")
 
    audit_log("save_result", detail=f"{v.get('well_name', '')} -> {filename}")
    return filename
 
 
# ── Well Data File Loader (CSV / Excel → Prediction inputs) ──
# Users download the CSV template (button on the Prediction page), fill one row
# per well and upload it. Headers are matched by name, so column order does not
# matter. Units are converted automatically using the Water Rate:
#   • Dose given as ppm OR quarts/day OR quarts/MMscfd → the unit the line needs
#   • Residual given as quarts/day OR ppm              → quarts/day (app input)
# Security: files are read in memory only (never saved), size / row limits
# apply, every value is range-checked, and text from the file is sanitised
# before it is shown or saved.
WELL_MAX_BYTES = 2 * 1024 * 1024
WELL_MAX_ROWS = 500
 
WELL_HEADER_MAP = [
    ("startswith", "wellstreamname", "name"),
    ("startswith", "fluid", "fluid"),
    ("startswith", "pipediameter", "dia"),
    ("startswith", "gasspecificgravity", "sg"),
    ("startswith", "flowingtemperature", "ftemp"),
    ("startswith", "linepressure", "lp"),
    ("startswith", "oilrate", "oil"),
    ("startswith", "waterrate", "water"),
    ("startswith", "gasrate", "gas"),
    ("startswith", "wallthickness", "wt"),
    ("startswith", "minrequiredwall", "min_wall"),
    ("startswith", "serviceyears", "svc"),
    ("startswith", "materialgrade", "mat"),
    ("startswith", "coatingcondition", "coat"),
    ("startswith", "cathodicprotection", "cp"),
    ("exact", "co2", "co2"),
    ("exact", "co", "co2"),
    ("exact", "h2s", "h2s"),
    ("exact", "hs", "h2s"),
    ("exact", "h2sppm", "h2s"),
    ("exact", "hsppm", "h2s"),
    ("exact", "ph", "ph"),
    ("startswith", "chlorides", "cl"),
    ("startswith", "inhibitortype", "itype"),
    ("startswith", "inhibitordoseppm", "dppm"),
    ("startswith", "inhibitordosequarts", "dqd"),
    ("startswith", "inhibitorquartsmmscfd", "dqt"),
    ("startswith", "residualinhibitorppm", "rppm"),
    ("startswith", "residualppm", "rppm"),
    ("startswith", "residualinhibitorquarts", "rqt"),
]
 
# suffix: (label, min, max, kind) — limits match the number_input widgets
WELL_NUM_SPEC = {
    "dia": ("Pipe Diameter", 1.0, 60.0, "float"),
    "sg": ("Gas Specific Gravity", 0.5, 2.0, "float"),
    "ftemp": ("Flowing Temperature", 32.0, 400.0, "float"),
    "lp": ("Line Pressure", 14.7, 5000.0, "float"),
    "oil": ("Oil Rate", 0.0, 50000.0, "float"),
    "water": ("Water Rate", 0.0, 50000.0, "float"),
    "gas": ("Gas Rate", 0.0, 500.0, "float"),
    "wt": ("Wall Thickness", 1.0, 50.0, "float"),
    "min_wall": ("Min Required Wall", 0.1, 50.0, "float"),
    "svc": ("Service Years", 0.0, 50.0, "float"),
    "co2": ("CO2", 0.0, 100.0, "float"),
    "h2s": ("H2S", 0, 20000, "int"),
    "ph": ("pH", 3.0, 10.0, "float"),
    "cl": ("Chlorides", 0, 200000, "int"),
    "dppm": ("Inhibitor Dose (ppm)", 0.0, 150.0, "float"),
    "dqd": ("Inhibitor Dose (quarts/day)", 0.0, 1e9, "float"),
    "dqt": ("Inhibitor (quarts/MMscfd)", 0.0, 10.0, "float"),
    "rqt": ("Residual Inhibitor (quarts/day)", 0.0, 500.0, "float"),
    "rppm": ("Residual Inhibitor (ppm)", 0.0, 1e9, "float"),
}
# dose / residual fields are unit-converted, not copied straight into a widget
WELL_DERIVED = {"dppm", "dqd", "dqt", "rqt", "rppm"}
 
WELL_TEMPLATE_HEADERS = [
    "Well / Stream Name", "Fluid", "Pipe Diameter (in)", "Gas Specific Gravity",
    "Flowing Temperature (F)", "Line Pressure (psi)", "Oil Rate (bbl/d)",
    "Water Rate (bbl/d)", "Gas Rate (MMscfd)", "Wall Thickness (mm)",
    "Min Required Wall (mm)", "Service Years", "Material Grade",
    "Coating Condition", "Cathodic Protection", "CO2 (%)", "H2S (ppm)", "pH",
    "Chlorides (mg/L)", "Inhibitor Type", "Inhibitor Dose (ppm)",
    "Inhibitor Dose (quarts/day)", "Inhibitor (quarts/MMscfd) - gas lines only",
    "Residual Inhibitor (quarts/day)", "Residual Inhibitor (ppm)",
]
WELL_TEMPLATE_EXAMPLES = [
    ["EXAMPLE Oil Well", "Crude Oil", 6, 0.65, 150, 1000, 2000, 500, 5, 8, 3, 10,
     "API 5L X52", "Good", "Yes", 1.0, 10, 6.5, 15000, "Film-Forming Amine",
     30, "", "", "", 30],
    ["EXAMPLE Water Line", "Water", 6, 0.65, 140, 800, 0, 5000, 0, 8, 3, 10,
     "API 5L X52", "Good", "Yes", 0.5, 5, 6.5, 15000, "Film-Forming Amine",
     "", 33.6, "", 33.6, ""],
    ["EXAMPLE Gas Line", "Natural Gas", 8, 0.70, 160, 1200, 200, 100, 15, 9, 3.5, 5,
     "API 5L X60", "Fair", "No", 3.5, 50, 6.0, 20000, "Imidazoline",
     "", "", 1.5, "", 30],
    ["EXAMPLE Multiphase Line", "Multiphase", 6, 0.65, 150, 1000, 3000, 1500, 8, 8, 3, 10,
     "API 5L X52", "Good", "Yes", 2.5, 20, 6.5, 15000, "Film-Forming Amine",
     30, "", "", 7.5, ""],
]
 
 
def well_template_csv():
    """CSV template users can fill in and upload (plain ASCII, opens in Excel)."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["# CORZYNTH WELL DATA TEMPLATE - one row per well. Do not rename or delete the header row."])
    w.writerow(["# Dose: fill ONE of Inhibitor Dose (ppm) / Inhibitor Dose (quarts/day) / Inhibitor (quarts/MMscfd - gas lines). "
                "Residual: fill ONE of Residual Inhibitor (quarts/day) / Residual Inhibitor (ppm). "
                "Corzynth converts the units automatically using the Water Rate."])
    w.writerow(["# Use plain numbers only (no units, no formulas). Delete the EXAMPLE rows before uploading your own wells."])
    w.writerow(["# Fluid: Crude Oil / Natural Gas / Water / Multiphase. Material Grade, Coating Condition and Inhibitor Type "
                "must match the app lists. Cathodic Protection: Yes or No."])
    w.writerow(WELL_TEMPLATE_HEADERS)
    for row in WELL_TEMPLATE_EXAMPLES:
        w.writerow(row)
    return buf.getvalue().encode("utf-8")
 
 
def _well_norm(x):
    first = (str(x).splitlines() or [""])[0]
    return re.sub(r"[^a-z0-9]", "", first.lower())
 
 
def _well_clean_cell(c):
    """Remove control characters and cap the length of every cell."""
    c = re.sub(r"[\x00-\x09\x0b-\x1f\x7f]", " ", str(c))
    return c.strip()[:200]
 
 
def _well_safe_text(x, n=60):
    """Text that came from a file: keep letters, digits and basic punctuation only,
    and never start with a spreadsheet-formula character."""
    t = re.sub(r"[^\w\s\-./()#&,:']", "", str(x))
    t = re.sub(r"\s+", " ", t).strip().lstrip("- ")
    return t[:n]
 
 
def read_well_file(uploaded):
    """Read an uploaded .csv / .xlsx well-data sheet → list of well dicts."""
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if len(data) > WELL_MAX_BYTES:
        raise ValueError("The file is larger than 2 MB. Please upload a smaller file.")
 
    if name.endswith(".xlsx"):
        xl = pd.ExcelFile(io.BytesIO(data))
        sheet = "Well Data" if "Well Data" in xl.sheet_names else xl.sheet_names[0]
        raw = xl.parse(sheet, header=None, dtype=object,
                       nrows=WELL_MAX_ROWS + 20).fillna("")
        rows = [[_well_clean_cell(c) for c in r] for r in raw.values.tolist()]
    elif name.endswith(".csv"):
        text = None
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                text = data.decode(enc)
                break
            except Exception:
                continue
        delim = ";" if text.count(";") > text.count(",") else ","
        rows = [[_well_clean_cell(c) for c in r]
                for r in csv.reader(io.StringIO(text), delimiter=delim)]
        if delim == ";":
            rows = [[re.sub(r"^(-?\d+),(\d+)$", r"\1.\2", c) for c in r] for r in rows]
    else:
        raise ValueError("Only .csv or .xlsx files are accepted.")
 
    hdr_idx = None
    for i, r in enumerate(rows):
        if any(_well_norm(c).startswith("wellstreamname") for c in r):
            hdr_idx = i
            break
    if hdr_idx is None:
        return []
 
    colmap = {}
    for j, h in enumerate(rows[hdr_idx]):
        n = _well_norm(h)
        for mode, pat, suffix in WELL_HEADER_MAP:
            if (mode == "exact" and n == pat) or (mode == "startswith" and n.startswith(pat)):
                if suffix not in colmap:
                    colmap[suffix] = j
                break
 
    wells = []
    for r in rows[hdr_idx + 1:]:
        rec = {}
        for suffix, j in colmap.items():
            if j < len(r) and r[j] != "":
                rec[suffix] = r[j]
        if "name" in rec:
            safe_name = _well_safe_text(rec["name"], 60)
            if safe_name:
                rec["name"] = safe_name
            else:
                del rec["name"]
        if not any(k != "name" for k in rec):
            continue
        wells.append(rec)
 
    if len(wells) > WELL_MAX_ROWS:
        raise ValueError(f"More than {WELL_MAX_ROWS} wells in one file. Split it into smaller files.")
    return wells
 
 
def well_line_type(fluid_text):
    """Which Prediction tab a well belongs to (None = unknown → shown in all)."""
    f = re.sub(r"[^a-z]", "", str(fluid_text).lower())
    if not f:
        return None
    if "multi" in f:
        return "multi"
    if "gas" in f:
        return "gas"
    if "water" in f:
        return "water"
    if "oil" in f or "crude" in f:
        return "oil"
    return None
 
 
def _well_float(x):
    try:
        v = float(str(x).replace(",", "").strip())
    except Exception:
        return None
    return v if math.isfinite(v) else None
 
 
def _well_pick(options, text):
    t = re.sub(r"\s+", "", str(text).lower())
    for o in options:
        if re.sub(r"\s+", "", o.lower()) == t:
            return o
    return None
 
 
def apply_well_to_inputs(key_prefix, rec, d):
    """Callback: write one well's values into the Prediction-tab widgets,
    converting dose / residual units automatically."""
    ss = st.session_state
    warns, infos, vals = [], [], {}
 
    def _clamp(label, x, lo, hi):
        c = min(max(x, lo), hi)
        if c != x:
            warns.append(f"{label}: {x:g} is outside the allowed range ({lo:g} to {hi:g}) — set to {c:g}.")
        return c
 
    def _raw(k):
        if k not in rec:
            return None
        x = _well_float(rec[k])
        if x is None:
            warns.append(f"{WELL_NUM_SPEC[k][0]}: “{_well_safe_text(rec[k], 30)}” is not a valid number — skipped.")
            return None
        return max(x, 0.0)
 
    # 1) plain numeric inputs
    for k, (label, lo, hi, kind) in WELL_NUM_SPEC.items():
        if k in WELL_DERIVED or k not in rec:
            continue
        x = _well_float(rec[k])
        if x is None:
            warns.append(f"{label}: “{_well_safe_text(rec[k], 30)}” is not a valid number — skipped.")
            continue
        c = _clamp(label, x, lo, hi)
        vals[k] = int(round(c)) if kind == "int" else float(c)
 
    water = vals.get("water", ss.get(f"{key_prefix}_water", d["water_bbl"]))
    gasr = vals.get("gas", ss.get(f"{key_prefix}_gas", d["gas_mmscfd"]))
 
    # 2) inhibitor DOSE — accepts ppm, quarts/day or quarts/MMscfd
    r_dppm, r_dqt, r_dqd = _raw("dppm"), _raw("dqt"), _raw("dqd")
    daily_qt = None
 
    if key_prefix == "gas":
        dqt = None
        if r_dqt is not None:
            dqt = r_dqt
        elif r_dqd is not None:
            if gasr > 0:
                dqt = r_dqd / gasr
                infos.append(f"Dose {r_dqd:g} quarts/day → {dqt:.3f} quarts/MMscfd (gas rate {gasr:g} MMscfd).")
            else:
                warns.append("Gas Rate is 0 — the quarts/day dose cannot be converted; dose left unchanged.")
        elif r_dppm is not None:
            if gasr > 0 and water > 0:
                dqt = ppm_to_qt_day(r_dppm, water) / gasr
                infos.append(f"Dose {r_dppm:g} ppm → {dqt:.3f} quarts/MMscfd "
                             f"(water {water:g} bbl/d, gas {gasr:g} MMscfd).")
            else:
                warns.append("Water or Gas Rate is 0 — the ppm dose cannot be converted; dose left unchanged.")
        if dqt is not None:
            vals["dqt"] = round(_clamp("Inhibitor (quarts/MMscfd)", dqt, 0.0, 10.0), 3)
        dqt_now = vals.get("dqt", ss.get(f"{key_prefix}_dqt", d["dose_qt_mmscfd"]))
        daily_qt = dqt_now * gasr
    else:
        dppm = None
        if r_dppm is not None:
            dppm = r_dppm
        elif r_dqd is not None:
            if water > 0:
                dppm = qt_day_to_ppm(r_dqd, water)
                infos.append(f"Dose {r_dqd:g} quarts/day → {dppm:.1f} ppm (water {water:g} bbl/d).")
            else:
                warns.append("Water Rate is 0 — the quarts/day dose cannot be converted; dose left unchanged.")
        elif r_dqt is not None:
            if water > 0:
                dppm = qt_day_to_ppm(r_dqt * gasr, water)
                infos.append(f"Dose {r_dqt:g} quarts/MMscfd × {gasr:g} MMscfd → {dppm:.1f} ppm (water {water:g} bbl/d).")
            else:
                warns.append("Water Rate is 0 — the dose cannot be converted; dose left unchanged.")
        if dppm is not None:
            vals["dppm"] = round(_clamp("Inhibitor Dose (ppm)", dppm, 0.0, 150.0), 1)
        dppm_now = vals.get("dppm", ss.get(f"{key_prefix}_dppm", d["dose_ppm"]))
        daily_qt = ppm_to_qt_day(dppm_now, water)
 
    # 3) RESIDUAL — accepts quarts/day or ppm; the app input is quarts/day
    r_rqt, r_rppm = _raw("rqt"), _raw("rppm")
    rqt = None
    if r_rqt is not None:
        rqt = r_rqt
        if r_rppm is not None:
            infos.append("Both residual quarts/day and residual ppm were given — quarts/day was used.")
    elif r_rppm is not None:
        if water > 0:
            rqt = r_rppm * water * 159 / 0.946e6
            infos.append(f"Residual {r_rppm:g} ppm → {rqt:.2f} quarts/day (water {water:g} bbl/d).")
        else:
            warns.append("Water Rate is 0 — the residual ppm cannot be converted; residual set equal to the daily dose.")
    if rqt is None:
        rqt = daily_qt
        infos.append("No residual given — residual set equal to the daily dose.")
    vals["rqt"] = round(_clamp("Residual Inhibitor (quarts/day)", rqt, 0.0, 500.0), 4)
 
    for k, val in vals.items():
        ss[f"{key_prefix}_{k}"] = val
 
    if rec.get("name"):
        ss[f"{key_prefix}_wellname"] = rec["name"]
 
    for suffix, options, label in (("mat", MATERIAL_GRADES, "Material Grade"),
                                   ("coat", COATING_CONDITIONS, "Coating Condition"),
                                   ("itype", INHIBITOR_TYPES, "Inhibitor Type")):
        if suffix in rec:
            hit = _well_pick(options, rec[suffix])
            if hit:
                ss[f"{key_prefix}_{suffix}"] = hit
            else:
                warns.append(f"{label}: “{_well_safe_text(rec[suffix], 30)}” not recognised — left unchanged.")
 
    if "cp" in rec:
        t = str(rec["cp"]).strip().lower()
        if t in ("yes", "y", "true", "1"):
            ss[f"{key_prefix}_cp"] = "Yes"
        elif t in ("no", "n", "false", "0"):
            ss[f"{key_prefix}_cp"] = "No"
        else:
            warns.append(f"Cathodic Protection: “{_well_safe_text(rec['cp'], 30)}” not recognised — left unchanged.")
 
    audit_log("load_well", detail=f"{key_prefix} | {rec.get('name', '')}")
    ss[f"{key_prefix}_loadmsg"] = (rec.get("name") or "well", warns, infos)
# ── End Well Data File Loader ──
 
 
# ══════════════════════════════════════════════════════════════════════════
#  ADVANCED ANALYTICS & SECURITY ADD-ONS
#  Additive only — the physics engine (full_calculation) is not modified.
#   • Confidence range around the model result + chance of reaching target life
#   • Measured wall loss from UT readings → actual rate, API 570-style interval
#   • Pit / metal-loss defect assessment (modified B31G, 0.85dL)
#   • Run ID + engine version for reproducible, auditable results
#   • Login lock-out, audit log, sign-out
#   • Word-report extension (original report function untouched)
# ══════════════════════════════════════════════════════════════════════════
ENGINE_VERSION = "10.16"
AUDIT_FILE = "corzynth_audit_log.csv"
LOGIN_MAX_FAILS = 5
LOGIN_LOCK_SECONDS = 300
 
SMYS_MPA = {"API 5L X42": 289, "API 5L X52": 359, "API 5L X60": 414,
            "API 5L X65": 448, "API 5L X70": 483, "Carbon Steel A106": 241}
 
 
# ── Audit log ──────────────────────────────────────────────────────────────
def audit_log(event, user=None, detail=""):
    """Append one line to the audit log. Never raises, never logs passwords."""
    try:
        if user is None:
            user = st.session_state.get("username", "anonymous")
 
        def _clean(x, n):
            t = re.sub(r"[^\w\s\-./()#&,:@']", "", str(x))
            t = re.sub(r"\s+", " ", t).strip().lstrip("=+-@ ")
            return t[:n]
 
        row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               _clean(user, 64), _clean(event, 40), _clean(detail, 200)]
        is_new = not os.path.exists(AUDIT_FILE)
        with open(AUDIT_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(["Time", "User", "Event", "Detail"])
            w.writerow(row)
    except Exception:
        pass
 
 
# ── Login lock-out (shared across sessions while the server is running) ───
@st.cache_resource
def _login_guard():
    return {"fails": {}, "locked": {}}
 
 
def _login_key(u):
    return str(u).strip().lower()[:64]
 
 
def login_locked(u):
    """Seconds left on a lock-out (0 = not locked)."""
    g = _login_guard()
    k = _login_key(u)
    until = g["locked"].get(k)
    if not until:
        return 0
    remain = until - time.time()
    if remain <= 0:
        g["locked"].pop(k, None)
        g["fails"].pop(k, None)
        return 0
    return remain
 
 
def login_failed(u):
    """Count a failed attempt; lock the name after LOGIN_MAX_FAILS."""
    g = _login_guard()
    k = _login_key(u)
    if len(g["fails"]) > 5000:
        g["fails"].clear()
    g["fails"][k] = g["fails"].get(k, 0) + 1
    if g["fails"][k] >= LOGIN_MAX_FAILS:
        g["locked"][k] = time.time() + LOGIN_LOCK_SECONDS
        g["fails"][k] = 0
        return True
    return False
 
 
def login_ok(u):
    g = _login_guard()
    k = _login_key(u)
    g["fails"].pop(k, None)
    g["locked"].pop(k, None)
 
 
# ── Reproducibility ────────────────────────────────────────────────────────
def run_id(v):
    s = json.dumps(v, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:10].upper()
 
 
# ── Confidence range (one simple, consistent band around the model result) ─
def confidence_range(rate, avail_wall, target_life, sigma=0.25):
    """Likely range around the model's corrosion rate.
    The model's own scatter is treated as log-normal (sigma 0.25 → the 10–90%
    band is about ×/÷ 1.38). Always centred on the model result."""
    k = math.exp(1.2816 * sigma)
    rate_lo, rate_hi = rate / k, rate * k
 
    def _life(r):
        return avail_wall / r if (r > 0 and avail_wall > 0) else 0.0
 
    if avail_wall <= 0:
        p_meet = 0.0
    elif target_life <= 0:
        p_meet = 1.0
    else:
        rate_needed = avail_wall / target_life
        zz = math.log(rate_needed / rate) / sigma
        p_meet = 0.5 * (1 + math.erf(zz / math.sqrt(2)))   # P(rate <= needed)
    return {"rate_lo": rate_lo, "rate_hi": rate_hi, "rate": float(rate),
            "life_exp": _life(rate), "life_best": _life(rate_lo),
            "life_worst": _life(rate_hi), "p_meet": float(p_meet),
            "target_life": float(target_life)}
 
 
# ── Defect assessment — modified B31G (0.85 dL) ────────────────────────────
def b31g_modified_failure_pressure(od_mm, t_mm, smys_mpa, depth_mm, length_mm):
    """Predicted failure pressure (MPa) of a corrosion defect, modified B31G.
    Returns (Pf_MPa, Folias_M, d_over_t)."""
    if od_mm <= 0 or t_mm <= 0:
        return 0.0, 1.0, 0.0
    dt = depth_mm / t_mm
    if dt >= 1.0:
        return 0.0, 1.0, dt
    z = length_mm ** 2 / (od_mm * t_mm)
    if z <= 50:
        M = math.sqrt(1 + 0.6275 * z - 0.003375 * z ** 2)
    else:
        M = 0.032 * z + 3.3
    sigma_flow = smys_mpa + 69.0
    pf_intact = 2 * t_mm * sigma_flow / od_mm
    ratio = (1 - 0.85 * dt) / (1 - 0.85 * dt / M)
    return max(pf_intact * ratio, 0.0), M, dt
 
 
def b31g_projection(od_mm, t_mm, smys_mpa, d0_mm, length_mm, maop_psi,
                    growth_mm_yr, required_margin, years=40.0, step=0.5):
    """Year-by-year safety margin (Pf / MAOP) as the defect deepens."""
    maop_mpa = maop_psi * 0.00689476
    rows, limit_year = [], None
    for i in range(int(years / step) + 1):
        yr = i * step
        depth = d0_mm + growth_mm_yr * yr
        pf, M, dt = b31g_modified_failure_pressure(od_mm, t_mm, smys_mpa, depth, length_mm)
        margin = pf / maop_mpa if maop_mpa > 0 else float("inf")
        rows.append({"Year": yr, "Depth (mm)": round(depth, 2),
                     "d/t (%)": round(dt * 100, 1), "Pf (psi)": round(pf / 0.00689476, 0),
                     "Margin": round(margin, 2)})
        if limit_year is None and (margin < required_margin or dt > 0.80):
            limit_year = yr
    return rows, limit_year
 
 
# ── Word report extension ──────────────────────────────────────────────────
def make_word_report_plus(proj_df, wt, rate, life_raw, v, steps, redose_df,
                          sens_df, opt_dose, opt_qt_day, risk, extras=None):
    """Original report + an 'Advanced Analytics' section.
    make_word_report() itself is unchanged."""
    from docx import Document
    from io import BytesIO
 
    base = make_word_report(proj_df, wt, rate, life_raw, v, steps, redose_df,
                            sens_df, opt_dose, opt_qt_day, risk)
    if not extras:
        return base
    base.seek(0)
    doc = Document(base)
 
    def _table(header, rows):
        t = doc.add_table(rows=1, cols=len(header))
        t.style = "Light Grid Accent 1"
        for i, h in enumerate(header):
            t.rows[0].cells[i].text = str(h)
        for r in rows:
            cells = t.add_row().cells
            for i, c in enumerate(r):
                cells[i].text = str(c)
 
    doc.add_heading("13. Advanced Analytics — Confidence & Verification", level=1)
    doc.add_paragraph(f"Engine version: {ENGINE_VERSION} | Run ID: {extras.get('run_id', '')} | "
                      f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    doc.add_paragraph("Validation status: the physics model is a published CO2 correlation with "
                      "engineering calibration factors. It has not been validated against this "
                      "operator's field data unless the measured-data section below is filled in. "
                      "Use as engineering decision-support, not as a substitute for professional judgment.")
 
    conf = extras.get("conf")
    if conf:
        doc.add_heading("Confidence Range", level=2)
        _table(["Measure", "Value"], [
            ("Corrosion rate — likely range (mm/yr)",
             f"{conf['rate_lo']:.3f} to {conf['rate_hi']:.3f} (model {conf['rate']:.3f})"),
            ("Remaining life — likely range (yr)",
             f"{conf['life_worst']:.1f} to {conf['life_best']:.1f} (expected {conf['life_exp']:.1f})"),
            (f"Chance of reaching the {conf['target_life']:.0f}-year target",
             f"{conf['p_meet'] * 100:.0f}%"),
        ])
        doc.add_paragraph("The range assumes the model itself is accurate to about ±38% (typical for "
                          "CO2 corrosion correlations). Replace this assumption with measured UT data "
                          "where available.")
 
    ut = extras.get("ut")
    if ut:
        doc.add_heading("Measured Wall Loss (UT)", level=2)
        _table(["Measure", "Value"], [
            ("Earlier / latest thickness (mm)", f"{ut['t1']:.2f} / {ut['t2']:.2f}"),
            ("Interval (years)", f"{ut['years']:.2f}"),
            ("Measured corrosion rate (mm/yr)", f"{ut['rate_m']:.3f}"),
            ("Model rate (mm/yr)", f"{ut['rate_model']:.3f}"),
            ("Measured ÷ model", f"{ut['calib']:.2f}"),
            ("Governing rate = max(model, measured) (mm/yr)", f"{ut['gov_rate']:.3f}"),
            ("Remaining life at governing rate (yr)", f"{ut['life_gov']:.1f}"),
            ("Suggested next inspection (API 570 style: half life, max 10 yr)", f"{ut['interval']:.1f} yr"),
        ])
 
    bg = extras.get("b31g")
    if bg:
        doc.add_heading("Defect Assessment (modified B31G)", level=2)
        _table(["Measure", "Value"], [
            ("OD / wall / SMYS", f"{bg['od']:.1f} mm / {bg['t']:.1f} mm / {bg['smys']:.0f} MPa"),
            ("Defect depth / length", f"{bg['d0']:.2f} mm / {bg['length']:.0f} mm"),
            ("Defect depth ÷ wall", f"{bg['dt0'] * 100:.1f}%"),
            ("Predicted failure pressure (psi)", f"{bg['pf_psi']:.0f}"),
            ("MAOP (psi)", f"{bg['maop']:.0f}"),
            ("Safety margin (Pf ÷ MAOP) / required", f"{bg['margin0']:.2f} / {bg['required']:.2f}"),
            ("Years until margin or d/t limit is reached",
             "beyond 40 yr" if bg['limit_year'] is None else f"{bg['limit_year']:.1f}"),
        ])
        doc.add_paragraph("Modified B31G applies to corrosion metal loss in carbon / low-alloy line pipe "
                          "(not cracking). Valid for defect depth up to 80% of wall.")
 
    doc.add_heading("References", level=2)
    for ref in ("de Waard & Milliams (1991), Carbonic acid corrosion of steel, Corrosion 31(5).",
                "ASME B31G (modified, 0.85dL) — remaining strength of corroded pipe.",
                "API 570 — piping inspection intervals; AMPP (NACE) MR0175 / ISO 15156 — sour service."):
        doc.add_paragraph(ref, style="List Bullet")
 
    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf
# ── End Advanced Analytics & Security Add-ons ──
 
 
def smart_input(label, input_type="text", key=None, default="", options=None, **kwargs):
    """
    One function for all inputs.
    Automatically keeps the value when the user changes tabs.
    """
    if key is None:
        key = label.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
 
    if input_type == "text":
        return st.text_input(label, value=st.session_state.get(key, default), key=key, **kwargs)
 
    if input_type == "number":
        return st.number_input(label, value=st.session_state.get(key, default), key=key, **kwargs)
 
    if input_type == "select":
        return st.selectbox(label, options, index=options.index(st.session_state.get(key, default)) if st.session_state.get(key, default) in options else 0, key=key, **kwargs)
 
    if input_type == "checkbox":
        return st.checkbox(label, value=st.session_state.get(key, default), key=key, **kwargs)
 
 
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
# ── Unit Conversion Helpers ──
def ppm_to_qt_day(ppm, water_bbl_day):
    """Convert product-volume ppm to US liquid quarts per day.
 
    Basis: petroleum barrel = 42 US gallons; 1 US gallon = 4 US liquid quarts.
    Therefore: 1 bbl = 168 quarts.
 
    This converts the injected product volume concentration relative to the
    water volume. It is NOT a measurement of active ingredient and is NOT a
    substitute for a measured residual inhibitor concentration.
    """
    if water_bbl_day <= 0:
        return 0.0
    return float(ppm) * float(water_bbl_day) * 168.0 / 1_000_000.0
 
 
def qt_day_to_ppm(qt_day, water_bbl_day):
    """Convert US liquid quarts per day to product-volume ppm on water volume.
 
    This is a dosing conversion only. It should not be used to claim an
    actual residual inhibitor concentration unless the calculation has been
    validated for the specific chemical, water rate, and system.
    """
    if water_bbl_day <= 0:
        return 0.0
    return float(qt_day) * 1_000_000.0 / (
        float(water_bbl_day) * 168.0
    )
   
def get_available_wall(v):
    """Return the wall thickness available for future corrosion.
 
    This is the only thickness basis that should be used for:
    - remaining life
    - target corrosion rate
    - optimizer
    - wall evaluation
    - ML remaining-life display
 
    A minimum required thickness of zero is not a valid acceptance limit.
    """
    current_wall = float(v.get("wt", 0.0))
    min_required_wall = float(v.get("min_required_wall", 0.0))
 
    if min_required_wall <= 0:
        return 0.0
 
    return max(0.0, current_wall - min_required_wall)
 
# ── Overdose Penalty (Literature-Calibrated) ──
def overdose_penalty(residual_ppm):
    """Penalty for excessive residual inhibitor — deposits, emulsions,
    reduced effectiveness. Literature: Gharbi et al. (2021) found deposits
    and performance degradation at higher inhibitor doses."""
    if residual_ppm <= 50:
        return 1.00
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
 
    if velocity <= 3.0:
        notes.append(f"✅ Flow speed is {velocity:.1f} m/s — safe range.")
    elif velocity <= 6.0:
        notes.append(f"⚠️ Flow speed is {velocity:.1f} m/s — slightly high; velocity correction ×1.15 applies.")
    else:
        notes.append(f"🔴 Flow speed is {velocity:.1f} m/s — high; velocity correction ×1.30 applies.")
    return notes
def evaluate_inputs(v, final_rate, remaining_life, target_life=10.0):
    """Simple verdict: is my dose right? Is my pipe wall thick enough"""
    target_rate = v["wt"] / target_life if target_life > 0 else 0.2
    residual = v.get("residual_ppm", 0)
    wt = v["wt"]
    required_wall = final_rate * target_life
    tol = 0.05  # small tolerance for rounding
 
    # --- Inhibitor verdict (Literature-Calibrated) ---
    if residual < 10:
        inh = ("🔴", f"**Your inhibitor residual is too low** ({residual:.0f} ppm). "
                     f"Increase the daily dose to reach at least 20 ppm residual.")
    elif residual < 20:
        inh = ("🟡", f"**Your inhibitor residual is borderline** ({residual:.0f} ppm). "
                     f"Increase slightly to reach the 20-50 ppm optimal zone.")
    elif residual <= 50:
        if final_rate <= target_rate + tol:
            inh = ("🟢", f"**Your inhibitor dose is just right.** {residual:.0f} ppm residual "
                         f"gives {remaining_life:.0f} years of life — matches your "
                         f"{target_life:.0f}-year target.")
        else:
            inh = ("🟡", f"**Your inhibitor residual is in the optimal zone** ({residual:.0f} ppm), "
                         f"but the wall is still too thin for {target_life:.0f} years. "
                         f"Consider wall upgrade or better CP/coating.")
    elif residual <= 100:
        inh = ("🟡", f"**Your inhibitor residual is above optimal** ({residual:.0f} ppm). "
                     f"You could reduce to 20-50 ppm and save money without losing protection.")
    else:
        inh = ("🔴", f"**Your inhibitor residual is too high** ({residual:.0f} ppm). "
                     f"This can cause deposits and emulsions. Reduce the dose to 20-50 ppm.")
 
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
    optimal_dose, opt_rate, opt_status = find_optimal_dose(v, target_rate)
    current_residual = v.get("residual_ppm", 0)
    water_bbl = v.get("dose_water_bbl", 100)
 
    # Daily quarts needed for optimal dose (same conversion as optimizer)
    optimal_qt = ppm_to_qt_day(optimal_dose, water_bbl)
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
 
def make_life_projection_chart(proj_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io
 
    # Auto-detect column names
    year_col = next((c for c in proj_df.columns if "year" in c.lower()), proj_df.columns[0])
    wall_col = next((c for c in proj_df.columns if "remaining" in c.lower() and "wall" in c.lower()),
                    proj_df.columns[1] if len(proj_df.columns) > 1 else proj_df.columns[0])
 
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(proj_df[year_col], proj_df[wall_col],
            marker="o", linewidth=2.5, color="#0b3d62", label="Remaining Wall")
    ax.axhline(y=0, color="#e74c3c", linestyle="--", linewidth=2, label="Failure Limit")
    ax.set_xlabel("Year")
    ax.set_ylabel("Remaining Wall")
    ax.set_title("Pipeline Life Projection — Year by Year")
    ax.legend()
    ax.grid(alpha=0.3)
 
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf
 
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
            wait_s = login_locked(u)
            if wait_s > 0:
                audit_log("login_locked", user=u)
                st.error(f"🔒 Too many failed attempts. Try again in {int(wait_s // 60) + 1} minute(s).")
            elif check_login(u, p):
                login_ok(u)
                audit_log("login_ok", user=u)
                st.session_state.logged_in = True
                st.session_state.username = u               
                st.rerun()
            else:
                locked_now = login_failed(u)
                audit_log("login_failed", user=u)
                st.error("🔒 Too many failed attempts. Try again in 5 minutes." if locked_now
                         else "❌ Invalid username or password")
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
 
        # ── Inhibitor: Literature-Calibrated Residual-Driven Protection ──
    # Basis: Gharbi et al. (2021) — film-forming inhibitors, 10-100 ppm typical;
    # Nouryon Armohib — 97.8-98.4% inhibition at 10-20 ppm active CI.
    residual_ppm = v.get("residual_ppm", 0.0)
 
    if residual_ppm > 0:
        effective_ppm = residual_ppm
    elif v["dose"] > 0:
        effective_ppm = v["dose"]
    else:
        effective_ppm = 0.0
 
    # Literature-calibrated efficiency curve:
    # 0-10 ppm: partial film (50-80%)
    # 10-20 ppm: strong film (80-95%, Nouryon: 97.8-98.4%)
    # 20-50 ppm: optimal plateau (95%)
    # 50-100 ppm: plateau, no added benefit
    # >100 ppm: overdose — efficiency degrades (deposits/emulsions)
    if effective_ppm < 10:
        inh_eff = 0.50 + (effective_ppm / 10) * 0.30
    elif effective_ppm < 20:
        inh_eff = 0.80 + ((effective_ppm - 10) / 10) * 0.15
    elif effective_ppm <= 100:
        inh_eff = 0.95
    else:
        inh_eff = max(0.85, 0.95 - ((effective_ppm - 100) / 200) * 0.10)
 
    if 0 < residual_ppm < 10:
        inh_eff *= 0.5
 
    rate_inh = rate_h2s * (1 - inh_eff)
    steps["9. Inhibitor (Literature-Calibrated Residual Protection)"] = (
        f"Dose {v['dose']:.0f} ppm | Residual {residual_ppm:.0f} ppm "
        f"({v.get('residual_qt', 0):.1f} qt) → effective {effective_ppm:.0f} ppm "
        f"→ {inh_eff*100:.1f}% reduction → {rate_inh:.3f} mm/yr")
 
        # ── Overdose Penalty (Literature-Based) ──
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
 
    # ── Residual Film Synergy (field-calibrated) ──
    if residual_ppm >= 20:
        synergy_factor = 0.70
    elif residual_ppm >= 10:
        synergy_factor = 0.85
    else:
        synergy_factor = 1.00
 
    if synergy_factor < 1.0:
        final_rate = max(0.01, final_rate * synergy_factor)
        steps["17b. Residual Film Synergy"] = (
            f"Residual {residual_ppm:.0f} ppm + FeCO3 scale → "
            f"×{synergy_factor:.2f} = {final_rate:.3f} mm/yr")
 
        # ── Field Calibration Factor ──
    # Calibrated against measured field corrosion-rate data.
    # Do not change these values without documented field calibration.
    FIELD_CALIBRATION_FACTORS = {
        "🛢️ Crude Oil": 1.00,
        "⛽ Natural Gas": 1.00,
        "💧 Water": 1.00,
        "🫧 Multiphase (Oil+Gas+Water)": 0.625,
    }
 
    calibration_factor = FIELD_CALIBRATION_FACTORS.get(v["fluid"], 1.00)
    final_rate = max(0.01, final_rate * calibration_factor)
 
    steps["17c. Field Calibration"] = (
        f"{v['fluid']} field calibration ×{calibration_factor:.3f} "
        f"= {final_rate:.3f} mm/yr"
    )        
 
    # ── Remaining life based on available wall thickness ──
    min_required_wall = float(v.get("min_required_wall", 3.0))
    available_wall = max(0.0, float(v["wt"]) - min_required_wall)
 
    if final_rate > 0 and available_wall > 0:
        life_raw = available_wall / final_rate
    else:
        life_raw = 0.0
 
    life = min(life_raw, DESIGN_LIFE)
 
    steps["18. Remaining Life"] = (
        f"({v['wt']:.1f} mm actual - {min_required_wall:.1f} mm minimum required) "
        f"÷ {final_rate:.3f} mm/yr = {life_raw:.1f} yrs"
    )
    life_for_risk = round(life_raw, 1)
 
    if life_for_risk > 20:
        risk, emoji = "LOW", "🟢"
    elif life_for_risk >= 10:
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
        lines.append(f"- Minimum required thickness is projected to be reached in {life_raw:.1f} years. Maintain inhibitor, verify CP annually.")
    elif round(life_raw, 1) >= 10:
        lines.append(f"- Minimum required thickness is projected to be reached in {life_raw:.1f} years. Maintain inhibitor program and monitor residual monthly.")
    else:
        lines.append(f"- Minimum required thickness is projected to be reached in only {life_raw:.1f} years. Urgent review of corrosion management is required.")
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
    """Literature-calibrated, residual-aware optimizer.
    Finds LOWEST realistic dose (5-50 ppm) achieving target life.
    If residual >= 20 ppm, protection is already at the 95% cap —
    returns current state instead of recommending more chemical.
    Returns: (recommended_dose, achieved_rate, status)
    status: 'optimal', 'at_target', 'cannot_reach',
            'overdose', 'severe_overdose'
    """
    residual_ppm = v.get("residual_ppm", 0.0)
    doses = [5, 10, 15, 20, 25, 30, 40, 50]
    water_bbl = v.get("dose_water_bbl", 100)
 
    # Severe overdose (>300 ppm residual)
    if residual_ppm > 300:
        r, _, _, _, _, _ = full_calculation(v)
        return 50, r, "severe_overdose"
 
    # Overdose (101-300 ppm residual)
    if residual_ppm > 100:
        r, _, _, _, _, _ = full_calculation(v)
        return 50, r, "overdose"
 
    # Already in optimal protection zone (20-100 ppm residual)
    if residual_ppm >= 20:
        r, _, _, _, _, _ = full_calculation(v)
        if r <= target_rate * 1.005:
            return residual_ppm, r, "at_target"
        return residual_ppm, r, "cannot_reach"
 
    # Low/no residual — search realistic doses
    for d in doses:
        residual_qt = (d * water_bbl * 159 / 0.946e6
                       if water_bbl > 0 else 0.0)
        vd = {**v, "dose": d, "residual_ppm": d,
              "residual_qt": residual_qt}
        r, _, _, _, _, _ = full_calculation(vd)
        if r <= target_rate * 1.005:
            return d, r, "optimal"
 
    # No realistic dose reaches target — return best option
    best_d, best_r = 50, None
    for d in doses:
        residual_qt = (d * water_bbl * 159 / 0.946e6
                       if water_bbl > 0 else 0.0)
        vd = {**v, "dose": d, "residual_ppm": d,
              "residual_qt": residual_qt}
        r, _, _, _, _, _ = full_calculation(vd)
        if best_r is None or r < best_r:
            best_d, best_r = d, r
    return best_d, best_r, "cannot_reach"
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
        # FIX: Use actual liquid rates (oil + water), not flow_rate (which receives gas rate)
        q_actual = (oil_bbl + water_bbl) * 0.159 / 86400
        velocity = (q_actual / area_m2) if area_m2 > 0 else 0.0
        slip_info["liq_vel"] = round(velocity, 3)
        slip_info["holdup"] = 1.0
 
    return velocity, slip_info
 
def make_word_report(proj_df, wt, rate, life_raw, v, steps, redose_df,
                     sens_df, opt_dose, opt_qt_day, risk):
    from docx import Document
    from docx.shared import Inches
    from io import BytesIO
 
    doc = Document()
    well_name = v.get("well_name", "").strip()
    title_suffix = f" — {well_name}" if well_name else ""
 
    doc.add_heading(f"Corzynth — Complete Corrosion Integrity Report{title_suffix}", 0)
    doc.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    doc.add_paragraph("📍 Scope: SPECIFIC well/plant only.")
 
    # ── 1. Well / Plant Inputs ──
    doc.add_heading("1. Well / Plant Inputs", level=1)
    inputs_data = [
        ("Well / Stream Name", v.get("well_name", "")),
        ("Fluid Type", v.get("fluid", "")),
        ("Pressure (psi)", v.get("pressure", "")),
        ("Temperature (°F)", v.get("temp", "")),
        ("Wall Thickness (mm)", v.get("wt", "")),
        ("Minimum Required Wall (mm)", v.get("min_required_wall", "")),
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
 
    # ── 2. Integrity Assessment Results ──
    doc.add_heading("2. Integrity Assessment Results", level=1)
    doc.add_paragraph(f"Corrosion Rate: {rate:.3f} mm/yr")
    doc.add_paragraph(f"Remaining Life: {life_raw:.1f} years")
    doc.add_paragraph(f"Service Years: {v.get('service_years', '')} yr")
    doc.add_paragraph(f"Risk Level: {risk}")
    if v.get("h2s", 0) >= 500:
        doc.add_paragraph("⚠️ NACE MR0175 Sour Service: H2S >= 500 ppm")
    elif rate >= 0.5:
        doc.add_paragraph("⚠️ Corrosion rate is above 0.5 mm/yr. Review inhibitor residual, corrosion monitoring, and protection performance.")
    else:
        doc.add_paragraph("✅ NACE MR0175: No compliance flags")
    if v.get("chlorides", 0) > 50000:
        doc.add_paragraph("⚠️ High Chlorides — pitting risk. Consider alloy upgrade.")
 
    # ── 3. Dose Optimizer ──
    doc.add_heading("3. Dose Optimizer — Daily & Continuous", level=1)
    residual_ppm = v.get("residual_ppm", 0)
    if residual_ppm > 300:
        doc.add_paragraph(f"🚨 SEVERE OVERDOSE — Residual: {residual_ppm:.0f} ppm (>300 ppm). Overdose penalty ×1.25 active — corrosion is worse, not better.")
        doc.add_paragraph("Action: Cut dose immediately to ~50 ppm residual.")
    elif residual_ppm > 100:
        doc.add_paragraph(f"⚠️ OVERDOSE — Residual: {residual_ppm:.0f} ppm (>100 ppm). Overdose penalty ×1.15 active.")
    elif residual_ppm > 40:
        doc.add_paragraph(f"⚠️ Slight Overdose — Residual: {residual_ppm:.0f} ppm (>40 ppm). Overdose penalty ×1.05 active.")
    else:
        doc.add_paragraph(f"✅ OPTIMAL RESIDUAL: {residual_ppm:.0f} ppm — Literature-calibrated protection zone (20-50 ppm = 95%+ inhibition efficiency).")
    doc.add_paragraph(f"Daily basis: {v.get('daily_qt', 0):.2f} quarts/day | Predicted rate: {rate:.3f} mm/yr | Estimated life: {life_raw:.1f} yr")
 
    doc.add_heading("3b. Why the Penalty Applies", level=2)
    penalty_table = doc.add_table(rows=1, cols=2)
    penalty_table.style = "Light Grid Accent 1"
    penalty_table.rows[0].cells[0].text = "Residual"
    penalty_table.rows[0].cells[1].text = "Penalty"
    for res_range, pen in [("0–40 ppm", "×1.00"), ("41–100 ppm", "×1.05"),
                            ("101–300 ppm", "×1.15"), (">300 ppm", "×1.25")]:
        cells = penalty_table.add_row().cells
        cells[0].text = res_range
        cells[1].text = pen
    doc.add_paragraph("Beyond ~40 ppm, more inhibitor gives no benefit and can disrupt the protective film.")
 
    # ── 4. What Do My Inputs Mean? ──
    doc.add_heading("4. What Do My Inputs Mean?", level=1)
    for note in interpret_inputs(v):
        if isinstance(note, tuple):
            doc.add_paragraph(f"{note[0]} {note[1]}")
        else:
            doc.add_paragraph(note)
 
    # ── 5. Ideal / Optimal Ranges ──
    doc.add_heading("5. Ideal / Optimal Ranges", level=1)
    def status_icon(ok):
        return "✅ OK" if ok else "⚠️ Review"
    ideal_rows = [
        ("CO2", "< 5%", f"{v.get('co2', 0):.1f}%", v.get('co2', 0) < 5),
        ("Temperature", "< 200°F", f"{v.get('temp', 0):.0f}°F", v.get('temp', 0) < 200),
        ("pH", "6.0–7.0", f"{v.get('ph', 0):.1f}", 6.0 <= v.get('ph', 0) <= 7.0),
        ("H2S", "< 50 ppm", f"{v.get('h2s', 0):.0f} ppm", v.get('h2s', 0) < 50),
        ("Chlorides", "< 10,000 mg/L", f"{v.get('chlorides', 0):,.0f} mg/L", v.get('chlorides', 0) < 10000),
        ("Residual Inhibitor", "20–50 ppm", f"{v.get('residual_ppm', 0):.0f} ppm", 20 <= v.get('residual_ppm', 0) <= 50),
        ("Velocity", "≤ 3.0 m/s", f"{v.get('velocity', 0):.2f} m/s", v.get('velocity', 0) <= 3.0),
        ("Water Cut", "< 40%", f"{v.get('water_cut', 0):.0f}%", v.get('water_cut', 0) < 40),
    ]
    t = doc.add_table(rows=1, cols=4)
    t.style = "Light Grid Accent 1"
    for i, h in enumerate(["Input", "Ideal Range", "Your Value", "Status"]):
        t.rows[0].cells[i].text = h
    for name, ideal, val, ok in ideal_rows:
        cells = t.add_row().cells
        cells[0].text = name
        cells[1].text = ideal
        cells[2].text = val
        cells[3].text = status_icon(ok)
 
    # ── 6. Is My Setup Correct? ──
    doc.add_heading("6. Is My Setup Correct?", level=1)
    inh, wall, overall = evaluate_inputs(v, rate, life_raw, 10.0)
    doc.add_paragraph(f"{inh[0]} {inh[1]}")
    doc.add_paragraph(f"{wall[0]} {wall[1]}")
    doc.add_paragraph(f"{overall[0]} {overall[1]}")
 
    # ── 7. Smart Recommendation ──
    doc.add_heading("7. Smart Recommendation", level=1)
    if well_name:
        doc.add_paragraph(f"📍 Well / Stream: {well_name}")
    rec = smart_recommendation(v, rate, life_raw, 10.0, v.get("pricepergallon", 3.0))
    for icon, text in rec:
        doc.add_paragraph(f"{icon} {text}")
 
    # ── 8. Calculation Breakdown ──
    doc.add_heading("8. Calculation Breakdown — Verify Every Step", level=1)
    for step_name, step_calc in steps.items():
        p = doc.add_paragraph()
        run = p.add_run(f"{step_name}: ")
        run.bold = True
        p.add_run(step_calc)
    doc.add_paragraph("📍 This breakdown is for the specific well/plant entered.")
 
    # ── 9. Engineering Explanation ──
    doc.add_heading("9. Engineering Explanation", level=1)
    doc.add_paragraph(ai_explanation(v, rate, life_raw, risk))
 
    # ── 10. Sensitivity Analysis ──
    doc.add_heading("10. Sensitivity Analysis", level=1)
    t = doc.add_table(rows=1, cols=len(sens_df.columns))
    t.style = "Light Grid Accent 1"
    for i, col in enumerate(sens_df.columns):
        t.rows[0].cells[i].text = str(col)
    for _, row in sens_df.iterrows():
        cells = t.add_row().cells
        for i, col in enumerate(sens_df.columns):
            cells[i].text = str(row[col])
 
    # ── 11. Life Projection ──
    doc.add_heading("11. Life Projection — Year by Year", level=1)
    table_cols = [c for c in proj_df.columns if c.lower() != "color"]
    t = doc.add_table(rows=1, cols=len(table_cols))
    t.style = "Light Grid Accent 1"
    for i, col in enumerate(table_cols):
        t.rows[0].cells[i].text = str(col)
    for _, row in proj_df.iterrows():
        cells = t.add_row().cells
        for i, col in enumerate(table_cols):
            cells[i].text = str(row[col])
 
    # ── 11b. Life Projection Chart ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io as _io
 
    year_col = next((c for c in proj_df.columns if "year" in c.lower()), proj_df.columns[0])
    wall_col = next((c for c in proj_df.columns if "remaining" in c.lower() and "wall" in c.lower()),
                    proj_df.columns[1] if len(proj_df.columns) > 1 else proj_df.columns[0])
 
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(proj_df[year_col], proj_df[wall_col],
            marker="o", linewidth=2.5, color="#0b3d62", label="Remaining Wall")
    ax.axhline(y=0, color="#e74c3c", linestyle="--", linewidth=2, label="Failure Limit")
    ax.set_xlabel("Year")
    ax.set_ylabel("Remaining Wall")
    ax.set_title("Pipeline Life Projection — Year by Year")
    ax.legend()
    ax.grid(alpha=0.3)
 
    chart_buf = _io.BytesIO()
    fig.savefig(chart_buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    chart_buf.seek(0)
 
    doc.add_heading("11b. Life Projection Chart", level=2)
    doc.add_picture(chart_buf, width=Inches(6.0))
 
    # ── 12. Recommendations ──
    doc.add_heading("12. Recommendations", level=1)
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
 
PAGES = [
    "🏠 Home", "🔮 Prediction", "🧪 Chemical Advisor", "💊 Dose Optimizer",
    "💰 ROI Calculator", "🌍 Geophysical", "🏛️ NACE Compliance",
    "🗺️ GIS Map", "🔔 Alerts", "📊 Multi-Well Dashboard", "ℹ️ About"]
if st.session_state.get("username") == "admin":
    PAGES.insert(len(PAGES) - 1, "🛡️ Audit Log")
page = st.sidebar.radio("Navigate", PAGES)
if st.sidebar.button("🚪 Sign Out", key="signout_btn"):
    audit_log("logout")
    st.session_state.logged_in = False
    st.session_state.pop("username", None)
    st.rerun()
 
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
 
        well_records = []
        with st.expander("📂 Load Well Data from CSV / Excel File (optional)", expanded=False):
            st.caption("1) Download the CSV template and fill one row per well — give the inhibitor dose as "
                       "ppm **or** quarts/day, and the residual as quarts/day **or** ppm; Corzynth converts "
                       "the units automatically using the water rate. 2) Upload the filled file. "
                       "3) Open a line tab, pick a well and click **Load Into Inputs**. "
                       "Corrosion rate, life and risk are still calculated by the app. "
                       "🔒 Your file is read in memory only and is never stored. Limits: 2 MB, 500 wells.")
            st.download_button("📄 Download CSV Template",
                               data=well_template_csv(),
                               file_name="Corzynth_Well_Data_Template.csv",
                               mime="text/csv", key="well_tpl_dl")
            well_file = st.file_uploader("Upload well data file (.csv or .xlsx)",
                                         type=["csv", "xlsx"], key="well_file_upl")
            if well_file is not None:
                try:
                    well_records = read_well_file(well_file)
                    if well_records:
                        st.success(f"✅ {len(well_records)} well(s) found in the file.")
                    else:
                        st.warning("No well rows found. Check that the header row contains "
                                   "“Well / Stream Name” and that the rows below it are filled.")
                except Exception as e:
                    st.error(f"❌ Could not read the file: {e}")
 
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
 
                # ── Load a well from the uploaded CSV / Excel file ──
                tab_wells = [w for w in well_records
                             if well_line_type(w.get("fluid", "")) in (None, key_prefix)]
                if tab_wells:
                    lw1, lw2 = st.columns([3, 1])
                    pick = lw1.selectbox(
                        "📂 Load well from uploaded file",
                        list(range(len(tab_wells))),
                        format_func=lambda i: tab_wells[i].get("name") or f"Row {i + 1}",
                        key=f"{key_prefix}_wellpick"
                    )
                    lw2.button(
                        "📥 Load Into Inputs",
                        key=f"{key_prefix}_wellload",
                        use_container_width=True,
                        on_click=apply_well_to_inputs,
                        args=(key_prefix, tab_wells[pick], d)
                    )
                    load_msg = st.session_state.pop(f"{key_prefix}_loadmsg", None)
                    if load_msg:
                        st.success(f"✅ Loaded “{load_msg[0]}” into the {tab_name} inputs — "
                                   f"review them, then click Calculate.")
                        for n in load_msg[2]:
                            st.info(n)
                        for n in load_msg[1]:
                            st.warning(n)
                saved = load_inputs(key_prefix)
 
                # ── Flow Calculator: Velocity & Water Cut (AUTO — no button) ──
                st.markdown("**⚙️ Flow Calculator — Velocity & Water Cut (Auto-Calculated)**")
 
                fc1, fc2, fc3 = st.columns(3)
 
                with fc1:
                    pipe_dia = st.number_input(
                        "Pipe Diameter (inches)",
                        min_value=1.0,
                        max_value=60.0,
                        value=6.0,
                        step=0.5,
                        key=f"{key_prefix}_dia"
                    )
 
                    gas_sg = st.number_input(
                        "Gas Specific Gravity (air = 1.0)",
                        min_value=0.5,
                        max_value=2.0,
                        value=0.65,
                        step=0.01,
                        key=f"{key_prefix}_sg"
                    )
 
                with fc2:
                    fc_temp = st.number_input(
                        "Flowing Temperature (°F)",
                        min_value=32.0,
                        max_value=400.0,
                        value=150.0,
                        step=5.0,
                        key=f"{key_prefix}_ftemp"
                    )
 
                    oil_bbl = st.number_input(
                        "Oil Rate (bbl/d)",
                        min_value=0.0,
                        max_value=50000.0,
                        value=float(d.get("oil_bbl", 0.0)),
                        step=10.0,
                        key=f"{key_prefix}_oil"
                    )
 
                    water_bbl = st.number_input(
                        "Water Rate (bbl/d)",
                        min_value=0.0,
                        max_value=50000.0,
                        value=float(d["water_bbl"]),
                        step=10.0,
                        key=f"{key_prefix}_water"
                    )
 
                with fc3:
                    gas_mmscfd = st.number_input(
                        "Gas Rate (MMscfd)",
                        min_value=0.0,
                        max_value=500.0,
                        value=float(d["gas_mmscfd"]),
                        step=1.0,
                        key=f"{key_prefix}_gas"
                    )
 
                    line_p = st.number_input(
                        "Line Pressure (psi)",
                        min_value=14.7,
                        max_value=5000.0,
                        value=1000.0,
                        step=50.0,
                        key=f"{key_prefix}_lp"
                    )
 
                total_liquid = oil_bbl + water_bbl
                auto_water_cut = (water_bbl / total_liquid * 100) if total_liquid > 0 else 0.0
 
                st.metric(
                    "💧 Water Cut (auto-calculated)",
                    f"{auto_water_cut:.1f}%",
                    f"from {water_bbl:.0f} bbl/d water + {oil_bbl:.0f} bbl/d oil"
                )
 
                auto_velocity, slip_info = calculate_velocity(
                    d["fluid"],
                    gas_mmscfd,
                    pipe_dia,
                    line_p,
                    fc_temp,
                    oil_bbl,
                    water_bbl,
                    gas_sg
                )
 
                st.success(
                    f"✅ Auto-calculated Velocity: {auto_velocity:.3f} m/s | "
                    f"Water Cut: {auto_water_cut:.1f}%"
                )
 
                if d["fluid"] == "🫧 Multiphase (Oil+Gas+Water)":
                    s1, s2, s3, s4 = st.columns(4)
 
                    s1.metric("Gas Velocity", f"{slip_info.get('gas_vel', 0):.3f} m/s")
                    s2.metric("Liquid Velocity", f"{slip_info.get('liq_vel', 0):.3f} m/s")
                    s3.metric("Liquid Holdup", f"{slip_info.get('holdup', 0):.1%}")
                    s4.metric("Slip Ratio", f"{slip_info.get('slip_ratio', 1.0):.2f}×")
 
                # Sync Flow Calculator values into Operating Conditions
                st.session_state[f"{key_prefix}_t"] = float(fc_temp)
                st.session_state[f"{key_prefix}_p"] = float(line_p)
                st.session_state[f"{key_prefix}_vel"] = float(round(auto_velocity, 3))
 
                c1, c2 = st.columns(2)
 
                with c1:
                    st.markdown("**🔧 Operating Conditions**")
 
                    well_name = st.text_input(
                        "Well / Stream Name",
                        value="",
                        placeholder="e.g., Well ABC-01 or North Line",
                        key=f"{key_prefix}_wellname"
                    )
 
                    fluid = st.selectbox(
                        "Material Flowing in Pipe",
                        [d["fluid"]],
                        key=f"{key_prefix}_fluid"
                    )
 
                    pressure = st.number_input(
                        "Pressure (psi)",
                        min_value=0,
                        max_value=5000,
                        value=1000,
                        step=50,
                        key=f"{key_prefix}_p"
                    )
 
                    temp = st.number_input(
                        "Temperature (°F)",
                        min_value=32,
                        max_value=400,
                        value=150,
                        step=5,
                        key=f"{key_prefix}_t"
                    )
 
                    wt = st.number_input(
                        "Wall Thickness (mm)",
                        min_value=1.0,
                        max_value=50.0,
                        value=8.0,
                        step=0.5,
                        key=f"{key_prefix}_wt"
                    )
 
                    min_required_wall = st.number_input(
                        "Minimum Required Wall Thickness (mm)",
                        min_value=0.1,
                        max_value=50.0,
                        value=3.0,
                        step=0.1,
                        key=f"{key_prefix}_min_wall",
                        help="Minimum thickness required by design, pressure, and structural requirements."
                    )
 
                    service_years = st.number_input(
                        "Service Years (line in service)",
                        min_value=0.0,
                        max_value=50.0,
                        value=float(d["service_years"]),
                        step=0.5,
                        key=f"{key_prefix}_svc"
                    )
 
                    material = st.selectbox(
                        "Pipeline Material Grade",
                        MATERIAL_GRADES,
                        key=f"{key_prefix}_mat"
                    )
 
                    coating_condition = st.selectbox(
                        "Coating Condition",
                        COATING_CONDITIONS,
                        key=f"{key_prefix}_coat"
                    )
 
                    cp_on = st.selectbox(
                        "Cathodic Protection",
                        ["Yes", "No"],
                        key=f"{key_prefix}_cp"
                    )
 
                    velocity = st.number_input(
                        "Flow Velocity (m/s) — ⚙️ auto-calculated, editable",
                        min_value=0.0,
                        max_value=20.0,
                        value=float(round(auto_velocity, 3)),
                        step=0.1,
                        key=f"{key_prefix}_vel"
                    )
 
                    st.caption("✅ Auto-calculated from flow rates + diameter — updates instantly.")
 
                with c2:
                    st.markdown("**🧪 Chemistry & Inhibitor**")
 
                    co2 = st.number_input(
                        "CO₂ (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(d["co2"]),
                        step=0.5,
                        key=f"{key_prefix}_co2"
                    )
 
                    h2s = st.number_input(
                        "H₂S (ppm)",
                        min_value=0,
                        max_value=20000,
                        value=d["h2s"],
                        step=50,
                        key=f"{key_prefix}_h2s"
                    )
 
                    ph = st.number_input(
                        "pH",
                        min_value=3.0,
                        max_value=10.0,
                        value=6.5,
                        step=0.1,
                        key=f"{key_prefix}_ph"
                    )
 
                    chlorides = st.number_input(
                        "Chlorides (mg/L)",
                        min_value=0,
                        max_value=200000,
                        value=15000,
                        step=1000,
                        key=f"{key_prefix}_cl"
                    )
 
                    itype = st.selectbox(
                        "Inhibitor Type",
                        INHIBITOR_TYPES,
                        key=f"{key_prefix}_itype"
                    )
 
                    water_for_dose = water_bbl
 
                    st.markdown("**💊 Inhibitor Dose — enter daily amount (quarts/day)**")
 
                    if key_prefix == "gas":
                        dose_qt_mmscfd = st.number_input(
                            "Inhibitor (quarts/MMSCFD)",
                            min_value=0.0,
                            max_value=10.0,
                            value=float(d["dose_qt_mmscfd"]),
                            step=0.1,
                            key=f"{key_prefix}_dqt"
                        )
 
                        daily_qt = dose_qt_mmscfd * gas_mmscfd
 
                        dose = (
                            daily_qt * 0.946 * 1e6 / (water_for_dose * 159)
                            if water_for_dose > 0
                            else 0.0
                        )
 
                    else:
                        dose = st.number_input(
                            "Inhibitor Dose (ppm)",
                            min_value=0.0,
                            max_value=150.0,
                            value=float(d["dose_ppm"]),
                            step=1.0,
                            key=f"{key_prefix}_dppm"
                        )
 
                        daily_qt = ppm_to_qt_day(dose, water_for_dose)
 
                    st.metric(
                        "= Daily Dose",
                        f"{daily_qt:.2f} quarts/day",
                        f"at {water_for_dose:.0f} bbl/d water"
                    )
 
                    st.markdown("**💊 Residual Inhibitor — quarts per day (in system) — THIS drives protection**")
 
                    residual_qt = st.number_input(
                        "Residual Inhibitor (quarts/day)",
                        min_value=0.0,
                        max_value=500.0,
                        value=round(daily_qt, 2),
                        step=0.5,
                        key=f"{key_prefix}_rqt"
                    )
 
                    residual_ppm = (
                        residual_qt * 0.946 * 1e6 / (water_for_dose * 159)
                        if water_for_dose > 0
                        else 0.0
                    )
 
                    st.metric(
                        "= Residual",
                        f"{residual_ppm:.0f} ppm",
                        f"{residual_qt:.2f} qt/day in system"
                    )
 
                    st.caption(
                        f"💡 {residual_qt:.2f} qt/day ÷ {water_for_dose:.0f} bbl/d water "
                        f"= {residual_ppm:.0f} ppm residual. "
                        f"20+ ppm residual → 10-12+ yr life."
                    )
 
                if st.button(
                    f"🔮 Calculate — {tab_name}",
                    type="primary",
                    use_container_width=True,
                    key=f"{key_prefix}_calc"
                ):
                    v = dict(
                        fluid=fluid,
                        well_name=well_name,
                        pressure=pressure,
                        temp=temp,
                        wt=wt,
                        min_required_wall=min_required_wall,
                        material=material,
                        coating_condition=coating_condition,
                        cp_on=cp_on,
                        co2=co2,
                        h2s=h2s,
                        ph=ph,
                        dose=dose,
                        itype=itype,
                        residual_ppm=residual_ppm,
                        residual_qt=residual_qt,
                        daily_qt=daily_qt,
                        velocity=velocity,
                        water_cut=auto_water_cut,
                        chlorides=chlorides,
                        dose_water_bbl=water_for_dose,
                        service_years=service_years,
                        diameter=pipe_dia,
                        oil_rate=oil_bbl,
                        water_rate=water_bbl,
                        gas_rate=gas_mmscfd,
                        gas_sg=gas_sg
                    )
 
                    rate, life, life_raw, risk, emoji, steps = full_calculation(v)
                    st.session_state[f"{key_prefix}_result"] = (
                        v, rate, life, life_raw, risk, emoji, steps
                    )
                    audit_log("calculate", detail=f"{tab_name} | {well_name}")
 
                if f"{key_prefix}_result" in st.session_state:
                    v, rate, life, life_raw, risk, emoji, steps = st.session_state[f"{key_prefix}_result"]
 
                    st.caption("📍 **Result applies to this specific well/plant only.**")
                    st.subheader("📊 Integrity Assessment")
 
                    if v.get("well_name", "").strip():
                        st.info(f"📍 **Report for:** {v.get('well_name', '').strip()}")
 
                    m1, m2, m3, m4 = st.columns(4)
 
                    m1.metric("Corrosion Rate", f"{rate:.3f} mm/yr")
                    m2.metric(
                        "Remaining Life",
                        f"≥ {DESIGN_LIFE} yrs" if life_raw > DESIGN_LIFE else f"{life:.1f} yrs"
                    )
                    m3.metric("Service Years", f"{v.get('service_years', 0):.0f} yr")
                    m4.metric("Risk", f"{emoji} {risk}")
 
                    if v.get("h2s", 0) > 500:
                        st.warning("🔴 **NACE MR0175: Sour Service** — H₂S > 500 ppm")
                    elif rate > 0.5:
                        st.warning(
                            "🟡 **Corrosion rate is above 0.5 mm/yr.** "
                            "Review inhibitor residual, corrosion monitoring, and protection performance."
                        )
                    else:
                        st.success("✅ NACE MR0175: No compliance flags")
 
                    if chlorides > 50000:
                        st.warning("⚠️ **High Chlorides** — pitting risk. Consider alloy upgrade.")
 
                    if st.button("🤖 Try ML Prediction", key=f"{key_prefix}_ml"):
                        model = joblib.load("corrosion_model.pkl")
                        cols = joblib.load("feature_cols.pkl")
 
                        ml_input = {
                            "Fluid_Type": v.get("fluid", "Crude Oil"),
                            "Material_Grade": v.get("material", "API 5L X52"),
                            "Coating_Condition": v.get("coating_condition", "Good"),
                            "CP_Status": v.get("cp_on", "Yes"),
                            "Pipe_Diameter_in": v.get("diameter", 6.0),
                            "Wall_Thickness_mm": v.get("wt", 8.0),
                            "Service_Years": v.get("service_years", 10.0),
                            "Operating_Pressure_psi": v.get("pressure", 1000),
                            "Operating_Temperature_F": v.get("temp", 150),
                            "CO2_percent": v.get("co2", 2.0),
                            "H2S_ppm": v.get("h2s", 100),
                            "pH": v.get("ph", 6.5),
                            "Chlorides_mg_L": v.get("chlorides", 10000),
                            "Oil_Rate_bbl_d": v.get("oil_rate", 1000),
                            "Water_Rate_bbl_d": v.get("water_rate", 500),
                            "Gas_Rate_MMscfd": v.get("gas_rate", 5.0),
                            "Gas_SG": v.get("gas_sg", 0.7),
                            "Velocity_m_s": v.get("velocity", 2.0),
                            "Water_Cut_percent": v.get("water_cut", 50),
                            "Inhibitor_Type": v.get("itype", "Film-Forming Amine"),
                            "Inhibitor_Dose_ppm": v.get("dose", 0),
                            "Inhibitor_Dose_qt_day": v.get("daily_qt", 0),
                            "Residual_Inhibitor_ppm": v.get("residual_ppm", 0),
                            "Target_Remaining_Life_yr": 10.0,
                            "Fe_Count_mg_L": 50.0,
                        }
 
                        df_in = pd.DataFrame([ml_input])
                        df_in = add_features(df_in)
 
                        for c in cols:
                            if c not in df_in.columns:
                                df_in[c] = 0
 
                        df_in = df_in[cols]
                        df_in["Velocityms"] = min(max(v["velocity"], 0.5), 6.0)
 
                        st.write(
                            f"ML input velocity being used: "
                            f"{df_in['Velocityms'].iloc[0]:.3f} m/s"
                        )
 
                        ml_rate_raw = float(model.predict(df_in)[0])
 
                        try:
                            calib = joblib.load("ml_calibration.pkl")
                            ml_rate = ml_rate_raw * calib.get(v["fluid"], 1.0)
                        except Exception:
                            ml_rate = ml_rate_raw
 
                        available_wall = get_available_wall(v)
                        ml_life = min(
                            available_wall / ml_rate if ml_rate > 0 else 999,
                            life
                        )
 
                        st.success(
                            f"🤖 **ML Cross-Check:** Remaining Life: {ml_life:.1f} yrs "
                            f"(conservatively bounded by physics)"
                        )
 
                        st.caption(
                            f"🤖 ML rate estimate: {ml_rate:.3f} mm/yr | "
                            f"Physics (primary): {rate:.3f} mm/yr | "
                            f"Remaining life: {ml_life:.1f} yrs (physics-bounded)"
                        )
 
                    st.subheader("💊 Dose Optimizer — Daily & Continuous")
 
                    target_life = st.number_input(
                        "Target Remaining Life (years)",
                        min_value=1.0,
                        max_value=30.0,
                        value=10.0,
                        step=0.5,
                        key=f"{key_prefix}_tlife"
                    )
 
                    price_per_gallon = st.number_input(
                        "Inhibitor Price ($/gallon)",
                        min_value=0.5,
                        max_value=200.0,
                        value=3.0,
                        step=0.5,
                        key=f"{key_prefix}_price"
                    )
 
                    available_wall = get_available_wall(v)
 
                    target_rate = (
                        available_wall / target_life
                        if target_life > 0 and available_wall > 0
                        else 0.0
                    )
 
                    st.caption(
                        f"Target corrosion rate for {target_life:.0f} yr life: "
                        f"{target_rate:.3f} mm/yr"
                    )
 
                    opt_dose, opt_rate, opt_status = find_optimal_dose(v, target_rate)
                    water_bbl_opt = v.get("dose_water_bbl", 100)
                    opt_qt_day = ppm_to_qt_day(opt_dose, water_bbl_opt)
                    opt_ml_min = opt_qt_day * 946 / 1440
                    available_wall = get_available_wall(v)
 
                    est_life = (
                        min(available_wall / opt_rate, DESIGN_LIFE)
                        if opt_rate > 0 and available_wall > 0
                        else 0.0
                    )
 
                    if opt_status == "severe_overdose":
                        col_alert, col_table = st.columns([2, 1])
 
                        with col_alert:
                            st.error(
                                f"🚨 **SEVERE OVERDOSE** — Residual: "
                                f"{v.get('residual_ppm', 0):.0f} ppm (>300 ppm). "
                                f"Overdose penalty ×1.25 active — corrosion is worse, not better.\n\n"
                                f"**Action:** Cut dose immediately to ~50 ppm residual. "
                                f"Current rate: {opt_rate:.3f} mm/yr."
                            )
 
                        with col_table:
                            with st.container(border=True):
                                st.markdown("""
**⚠️ Why the penalty applies**
 
| Residual | Penalty |
|---|---|
| 0–40 ppm | ×1.00 |
| 41–100 ppm | ×1.05 |
| 101–300 ppm | ×1.15 |
| >300 ppm | ×1.25 |
 
Beyond ~40 ppm, more inhibitor gives no benefit and can disrupt the protective film.
""")
 
                    elif opt_status == "overdose":
                        vd_check = {
                            **v,
                            "dose": 50,
                            "residual_ppm": 50,
                            "residual_qt": (
                                50 * water_bbl_opt * 159 / 0.946e6
                                if water_bbl_opt > 0
                                else 0
                            )
                        }
 
                        r_check, _, _, _, _, _ = full_calculation(vd_check)
 
                        if r_check <= target_rate * 1.005:
                            st.success(
                                f"✅ **OVERDOSE — REDUCE DOSE** — Residual: "
                                f"{v.get('residual_ppm', 0):.0f} ppm is above optimal. "
                                f"Reducing to 50 ppm residual still meets your target "
                                f"({r_check:.3f} mm/yr ≤ {target_rate:.3f} mm/yr).\n\n"
                                f"**Action:** Reduce daily dose to "
                                f"{50 * water_bbl_opt * 0.00035:.2f} quarts/day "
                                f"to save chemical without losing protection."
                            )
                        else:
                            st.warning(
                                f"⚠️ **OVERDOSE** — Residual: "
                                f"{v.get('residual_ppm', 0):.0f} ppm (101-300 ppm). "
                                f"Overdose penalty ×1.15 active, but even at 50 ppm "
                                f"the target cannot be met with inhibitor alone "
                                f"({r_check:.3f} mm/yr vs {target_rate:.3f} mm/yr).\n\n"
                                f"**Action:** Reduce dose to 50 ppm AND consider "
                                f"wall upgrade, CP improvement, or coating repair."
                            )
 
                    elif opt_status == "at_target":
                        st.success(
                            f"✅ **OPTIMAL RESIDUAL: {opt_dose:.0f} ppm** — Literature-calibrated "
                            f"protection zone (20-50 ppm = 95%+ inhibition efficiency).\n\n"
                            f"Current rate: {opt_rate:.3f} mm/yr | Target: {target_rate:.3f} mm/yr | "
                            f"Life target achieved. Maintain current dosing."
                        )
 
                    elif opt_status == "optimal":
                        st.success(
                            f"✅ **OPTIMIZED DOSE: {opt_dose:.0f} ppm** | "
                            f"Daily: {opt_qt_day:.2f} quarts/day | "
                            f"Continuous: {opt_ml_min:.2f} mL/min pump setting | "
                            f"Achieved rate: {opt_rate:.3f} mm/yr"
                        )
 
                    else:
                        st.error(
                            f"❌ Inhibitor alone cannot reach target. "
                            f"Best realistic dose: {opt_dose:.0f} ppm gives "
                            f"{opt_rate:.3f} mm/yr (target: {target_rate:.3f} mm/yr).\n\n"
                            f"**Action:** Wall upgrade, CP improvement, or coating repair needed."
                        )
 
                    st.info(
                        f"**Daily basis:** {opt_qt_day:.2f} quarts/day | "
                        f"**Continuous form:** {opt_ml_min:.2f} mL/min | "
                        f"Predicted rate: {opt_rate:.3f} mm/yr | "
                        f"Estimated life: {est_life:.1f} yr"
                    )
 
                    st.markdown("### 🧭 What Do My Inputs Mean? & 📋 Ideal / Optimal Ranges")
 
                    col_notes, col_table = st.columns([1, 1], gap="large")
 
                    with col_notes:
                        st.markdown("#### 🔍 Interpretation of Your Inputs")
 
                        for note in interpret_inputs(v):
                            if isinstance(note, tuple):
                                st.markdown(f"{note[0]} {note[1]}")
                            else:
                                st.markdown(note)
 
                    with col_table:
                        st.markdown("#### 📋 Ideal / Optimal Ranges")
 
                        def status_icon(ok):
                            return "✅ OK" if ok else "⚠️ Review"
 
                        table_data = [
                            {"Input": "CO₂", "Ideal Range": "< 5%", "Your Value": f"{v.get('co2', 0):.1f}%", "Status": status_icon(v.get('co2', 0) < 5)},
                            {"Input": "Temperature", "Ideal Range": "< 200°F", "Your Value": f"{v.get('temp', 0):.0f}°F", "Status": status_icon(v.get('temp', 0) < 200)},
                            {"Input": "pH", "Ideal Range": "6.0–7.0", "Your Value": f"{v.get('ph', 0):.1f}", "Status": status_icon(6.0 <= v.get('ph', 0) <= 7.0)},
                            {"Input": "H₂S", "Ideal Range": "< 50 ppm", "Your Value": f"{v.get('h2s', 0):.0f} ppm", "Status": status_icon(v.get('h2s', 0) < 50)},
                            {"Input": "Chlorides", "Ideal Range": "< 10,000 mg/L", "Your Value": f"{v.get('chlorides', 0):,.0f} mg/L", "Status": status_icon(v.get('chlorides', 0) < 10000)},
                            {"Input": "Residual Inhibitor", "Ideal Range": "20–50 ppm", "Your Value": f"{v.get('residual_ppm', 0):.0f} ppm", "Status": status_icon(20 <= v.get('residual_ppm', 0) <= 50)},
                            {"Input": "Velocity", "Ideal Range": "≤ 3.0 m/s", "Your Value": f"{v.get('velocity', 0):.2f} m/s", "Status": status_icon(v.get('velocity', 0) <= 3.0)},
                            {"Input": "Water Cut", "Ideal Range": "< 40%", "Your Value": f"{v.get('water_cut', 0):.0f}%", "Status": status_icon(v.get('water_cut', 0) < 40)},
                        ]
 
                        ideal_df = pd.DataFrame(table_data)
 
                        st.dataframe(
                            ideal_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Input": st.column_config.TextColumn("Input", width="medium"),
                                "Ideal Range": st.column_config.TextColumn("Ideal Range", width="medium"),
                                "Your Value": st.column_config.TextColumn("Your Value", width="medium"),
                                "Status": st.column_config.TextColumn("Status", width="small"),
                            }
                        )
 
                    st.subheader("📋 Is My Setup Correct?")
                    inh, wall, overall = evaluate_inputs(v, rate, life, target_life)
 
                    st.info(f"{inh[0]} {inh[1]}")
                    st.info(f"{wall[0]} {wall[1]}")
 
                    if overall[0] == "✅":
                        st.success(f"{overall[0]} {overall[1]}")
                    else:
                        st.warning(f"{overall[0]} {overall[1]}")
 
                    st.markdown("### 🎯 Smart Recommendation")
 
                    if v.get("well_name", "").strip():
                        st.markdown(
                            f"📍 **Well / Stream:** {v.get('well_name', '').strip()}"
                        )
 
                    rec = smart_recommendation(
                        v, rate, life, target_life, price_per_gallon
                    )
 
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
 
                    st.subheader("📊 Life Projection — Year by Year")
 
                    import plotly.graph_objects as go
 
                    max_year = int(min(life + 5, 60))
                    proj_rows = []
 
                    for yr in range(0, max_year + 1):
                        loss = rate * yr
                        remaining = max(0, wt - loss)
 
                        if remaining <= 0:
                            status = "🔴 Failed — Wall Fully Consumed"
                            color = "#e74c3c"
                        elif remaining <= wt * 0.25:
                            status = "🟠 Critical — Plan Replacement"
                            color = "#f39c12"
                        else:
                            status = "🟢 Healthy"
                            color = "#2ecc71"
 
                        proj_rows.append({
                            "Year": yr,
                            "Metal Loss (mm)": round(loss, 2),
                            "Remaining Wall (mm)": round(remaining, 2),
                            "Status": status,
                            "Color": color
                        })
 
                    projdf = pd.DataFrame(proj_rows)
 
                    fig = go.Figure()
 
                    fig.add_trace(go.Bar(
                        x=projdf["Year"],
                        y=projdf["Remaining Wall (mm)"],
                        marker_color=projdf["Color"],
                        text=projdf["Remaining Wall (mm)"],
                        textposition="outside",
                        name="Remaining Wall (mm)",
                        hovertemplate=(
                            "<b>Year %{x}</b><br>"
                            "Remaining Wall: %{y:.2f} mm<br>"
                            "Metal Loss: %{customdata[0]:.2f} mm<br>"
                            "Status: %{customdata[1]}<extra></extra>"
                        ),
                        customdata=list(zip(
                            projdf["Metal Loss (mm)"],
                            projdf["Status"]
                        ))
                    ))
 
                    critical_wall = wt * 0.25
 
                    fig.add_hline(
                        y=critical_wall,
                        line_dash="dash",
                        line_color="#f39c12",
                        annotation_text=f"⚠️ Critical Threshold ({critical_wall:.1f} mm)",
                        annotation_position="top right",
                        annotation_font=dict(color="#f39c12", size=12)
                    )
 
                    life_end_year = int(life) if life < 60 else 60
 
                    fig.add_vline(
                        x=life_end_year,
                        line_dash="dash",
                        line_color="#e74c3c",
                        annotation_text=f"🔴 Life Ends (~{life:.0f} yrs)",
                        annotation_position="top left",
                        annotation_font=dict(color="#e74c3c", size=12)
                    )
 
                    fig.update_layout(
                        title={
                            "text": "Pipeline Life Projection — Year by Year",
                            "font": dict(size=20, color="#ecf0f1")
                        },
                        xaxis=dict(
                            title="Year",
                            gridcolor="#34495e",
                            zerolinecolor="#34495e"
                        ),
                        yaxis=dict(
                            title="Remaining Wall (mm)",
                            gridcolor="#34495e",
                            zerolinecolor="#34495e"
                        ),
                        hovermode="x unified",
                        template="plotly_dark",
                        paper_bgcolor="#1a2530",
                        plot_bgcolor="#1a2530",
                        font=dict(color="#ecf0f1"),
                        showlegend=False,
                        margin=dict(l=50, r=50, t=70, b=50),
                        height=450
                    )
 
                    st.plotly_chart(fig, use_container_width=True)
 
                    st.markdown("#### 📋 Year-by-Year Wall Projection")
 
                    def color_status(val):
                        if "Failed" in val:
                            return "background-color: #c0392b; color: white; font-weight: bold;"
                        elif "Critical" in val:
                            return "background-color: #e67e22; color: white; font-weight: bold;"
                        else:
                            return "background-color: #27ae60; color: white; font-weight: bold;"
 
                    styled_table = projdf[
                        ["Year", "Metal Loss (mm)", "Remaining Wall (mm)", "Status"]
                    ].style.map(
                        color_status,
                        subset=["Status"]
                    ).set_properties(**{
                        "text-align": "center",
                        "font-size": "14px"
                    }).hide(axis="index")
 
                    st.dataframe(styled_table, use_container_width=True)
 
                    # ═══ ADVANCED ANALYTICS (additive) — confidence, cross-check, field data, defects ═══
                    import plotly.graph_objects as go
                    from datetime import date, timedelta
 
                    extras = {"run_id": run_id(v)}
                    st.markdown("---")
                    st.subheader("🧠 Advanced Analytics — Confidence, Cross-Check & Field Data")
                    st.caption(f"Engine v{ENGINE_VERSION} | Run ID: **{extras['run_id']}** | "
                               f"{datetime.now().strftime('%Y-%m-%d %H:%M')} — quote the Run ID to reproduce this result.")
 
                    # 1) Confidence range
                    st.markdown("#### 🎯 Confidence Range")
                    conf = confidence_range(rate, get_available_wall(v), target_life)
                    extras["conf"] = conf
 
                    def _fmt_life(x):
                        return f"≥ {DESIGN_LIFE}" if x > DESIGN_LIFE else f"{x:.1f}"
 
                    cf1, cf2, cf3 = st.columns(3)
                    cf1.metric("Corrosion rate — likely range",
                               f"{conf['rate_lo']:.2f} – {conf['rate_hi']:.2f} mm/yr")
                    cf2.metric("Remaining life — likely range",
                               f"{_fmt_life(conf['life_worst'])} – {_fmt_life(conf['life_best'])} yrs")
                    cf3.metric(f"Chance of reaching {target_life:.0f}-yr target",
                               f"{conf['p_meet'] * 100:.0f}%")
                    if conf["p_meet"] >= 0.90:
                        st.success(f"✅ High confidence: about {conf['p_meet'] * 100:.0f}% chance the line "
                                   f"reaches the {target_life:.0f}-year target.")
                    elif conf["p_meet"] >= 0.50:
                        st.warning(f"⚠️ Moderate confidence: about {conf['p_meet'] * 100:.0f}% chance the line "
                                   f"reaches the {target_life:.0f}-year target.")
                    else:
                        st.error(f"🔴 Low confidence: only about {conf['p_meet'] * 100:.0f}% chance the line "
                                 f"reaches the {target_life:.0f}-year target — improve protection or confirm "
                                 f"with measured data.")
                    st.caption("The range assumes the model is accurate to about ±38% (typical for CO₂ "
                               "corrosion correlations). Enter UT readings below to check it against reality.")
 
                    # 2) Measured wall loss (UT)
                    with st.expander("📏 Field Data — Measured Wall Loss (UT readings → actual corrosion rate)", expanded=False):
                        ut_on = st.checkbox("I have two UT thickness readings for this line", key=f"{key_prefix}_adv_ut_on")
                        if ut_on:
                            ut_c1, ut_c2 = st.columns(2)
                            with ut_c1:
                                ut_t1 = st.number_input("Earlier UT thickness (mm)", min_value=0.5, max_value=60.0,
                                                        value=float(v["wt"]), step=0.1, key=f"{key_prefix}_adv_ut_t1")
                                ut_d1 = st.date_input("Earlier reading date",
                                                      value=date.today() - timedelta(days=3 * 365),
                                                      key=f"{key_prefix}_adv_ut_d1")
                            with ut_c2:
                                ut_t2 = st.number_input("Latest UT thickness (mm)", min_value=0.5, max_value=60.0,
                                                        value=float(v["wt"]), step=0.1, key=f"{key_prefix}_adv_ut_t2")
                                ut_d2 = st.date_input("Latest reading date", value=date.today(),
                                                      key=f"{key_prefix}_adv_ut_d2")
                            ut_years = (ut_d2 - ut_d1).days / 365.25
                            ut_loss = ut_t1 - ut_t2
                            if ut_years <= 0:
                                st.error("The latest reading date must be after the earlier reading date.")
                            elif ut_loss < 0:
                                st.warning("The latest reading is thicker than the earlier one — this is measurement scatter. "
                                           "Use more readings or a longer interval.")
                            else:
                                rate_m = ut_loss / ut_years
                                min_wall_ut = float(v.get("min_required_wall", 3.0))
                                avail_ut = max(0.0, ut_t2 - min_wall_ut)
                                gov_rate = max(rate, rate_m)
                                life_gov = avail_ut / gov_rate if gov_rate > 0 else 0.0
                                interval = min(life_gov / 2.0, 10.0)
                                calib = rate_m / rate if rate > 0 else float("nan")
 
                                m_a, m_b, m_c = st.columns(3)
                                m_a.metric("Measured rate", f"{rate_m:.3f} mm/yr")
                                m_b.metric("Model rate", f"{rate:.3f} mm/yr")
                                m_c.metric("Measured ÷ model", f"{calib:.2f}")
                                m_d, m_e = st.columns(2)
                                m_d.metric("Life at governing rate", f"{life_gov:.1f} yrs",
                                           "governing = the higher of model and measured")
                                m_e.metric("Suggested next inspection", f"{interval:.1f} yrs",
                                           "API 570 style: half the remaining life, max 10 yr")
 
                                if ut_years < 1.0 or ut_loss < 0.3:
                                    st.info("ℹ️ Short interval or small metal loss — UT scatter (about ±0.1–0.2 mm) can dominate "
                                            "this rate. Treat it as indicative and repeat the measurement.")
                                if avail_ut <= 0:
                                    st.error("🔴 The latest thickness is at or below the minimum required wall — "
                                             "fitness-for-service review required.")
                                if calib < 0.8:
                                    st.success(f"✅ The model is conservative here: it predicts {(1 / calib - 1) * 100:.0f}% "
                                               f"above the measured rate." if calib > 0 else
                                               "✅ No measurable wall loss — the model is conservative.")
                                elif calib <= 1.25:
                                    st.success("✅ The model agrees with the measured rate within ±25%.")
                                else:
                                    st.error(f"🔴 The model UNDER-predicts by {(calib - 1) * 100:.0f}% — plan with the measured rate "
                                             f"and review the inhibition program.")
 
                                extras["ut"] = {"t1": float(ut_t1), "t2": float(ut_t2), "years": float(ut_years),
                                                "rate_m": float(rate_m), "rate_model": float(rate), "calib": float(calib),
                                                "gov_rate": float(gov_rate), "life_gov": float(life_gov),
                                                "interval": float(interval)}
 
                    # 3) Defect assessment (modified B31G)
                    with st.expander("🔩 Pit / Metal-Loss Defect Assessment (modified B31G)", expanded=False):
                        bg_on = st.checkbox("Assess a measured pit / corrosion defect", key=f"{key_prefix}_adv_b31g_on")
                        if bg_on:
                            if v.get("material") in ("Stainless 316L", "Duplex 2205"):
                                st.warning("B31G applies to carbon / low-alloy line pipe. Results are not valid for "
                                           "stainless or duplex.")
                            bgc1, bgc2, bgc3 = st.columns(3)
                            bg_od = bgc1.number_input("Pipe outside diameter (mm)", min_value=20.0, max_value=2000.0,
                                                      value=float(round(v.get("diameter", 6.0) * 25.4 + 2 * v["wt"], 1)),
                                                      step=1.0, key=f"{key_prefix}_adv_bg_od")
                            bg_smys = bgc2.number_input("SMYS (MPa)", min_value=100.0, max_value=800.0,
                                                        value=float(SMYS_MPA.get(v.get("material"), 359)),
                                                        step=1.0, key=f"{key_prefix}_adv_bg_smys")
                            bg_maop = bgc3.number_input("MAOP / operating pressure (psi)", min_value=1.0, max_value=20000.0,
                                                        value=float(max(v.get("pressure", 1000), 1)),
                                                        step=10.0, key=f"{key_prefix}_adv_bg_maop")
                            bgc4, bgc5, bgc6 = st.columns(3)
                            bg_d0 = bgc4.number_input("Defect depth (mm)", min_value=0.0, max_value=float(v["wt"]),
                                                      value=round(float(v["wt"]) * 0.2, 1), step=0.1,
                                                      key=f"{key_prefix}_adv_bg_d0")
                            bg_len = bgc5.number_input("Defect length (mm)", min_value=1.0, max_value=2000.0,
                                                       value=50.0, step=5.0, key=f"{key_prefix}_adv_bg_len")
                            bg_mult = bgc6.number_input("Pit growth ÷ uniform rate", min_value=0.5, max_value=10.0,
                                                        value=1.0, step=0.5, key=f"{key_prefix}_adv_bg_mult")
                            bg_req = st.number_input("Required safety margin (Pf ÷ MAOP)", min_value=1.0, max_value=3.0,
                                                     value=1.39, step=0.01, key=f"{key_prefix}_adv_bg_req")
                            st.caption("Uses the nominal wall from the Prediction inputs. Pitting often grows 2–5× faster than "
                                       "uniform corrosion — set the growth multiplier from inspection data when available. "
                                       "1.39 = 1 ÷ 0.72 design factor.")
 
                            bg_t = float(v["wt"])
                            pf0, M0, dt0 = b31g_modified_failure_pressure(bg_od, bg_t, bg_smys, bg_d0, bg_len)
                            maop_mpa = bg_maop * 0.00689476
                            margin0 = pf0 / maop_mpa if maop_mpa > 0 else float("inf")
                            rate_use = rate
                            if extras.get("ut"):
                                rate_use = max(rate, extras["ut"]["rate_m"])
                            growth = rate_use * bg_mult
                            bg_rows, bg_limit = b31g_projection(bg_od, bg_t, bg_smys, bg_d0, bg_len, bg_maop,
                                                                growth, bg_req)
 
                            q1, q2, q3, q4 = st.columns(4)
                            q1.metric("Defect depth ÷ wall", f"{dt0 * 100:.1f}%")
                            q2.metric("Failure pressure Pf", f"{pf0 / 0.00689476:,.0f} psi")
                            q3.metric("Safety margin Pf ÷ MAOP", f"{margin0:.2f}")
                            q4.metric("Years to limit", "> 40" if bg_limit is None else f"{bg_limit:.1f}")
 
                            if dt0 > 0.80:
                                st.error("🔴 Defect is deeper than 80% of the wall — outside B31G validity. Repair or replace.")
                            elif margin0 < bg_req:
                                st.error(f"🔴 Safety margin {margin0:.2f} is below the required {bg_req:.2f} — "
                                         f"reduce pressure or repair now.")
                            elif bg_limit is not None and bg_limit <= 5:
                                st.warning(f"⚠️ The defect reaches the limit in about {bg_limit:.1f} years at "
                                           f"{growth:.2f} mm/yr growth — plan repair or re-inspection.")
                            else:
                                st.success("✅ The defect is acceptable now, and stays acceptable for the projected period "
                                           "at the assumed growth rate." if bg_limit is None else
                                           f"✅ Acceptable now; reaches the limit in about {bg_limit:.1f} years.")
 
                            bg_df = pd.DataFrame(bg_rows)
                            fig_bg = go.Figure()
                            fig_bg.add_trace(go.Scatter(x=bg_df["Year"], y=bg_df["Margin"], mode="lines",
                                                        line=dict(color="#2e86c1", width=3), name="Pf ÷ MAOP"))
                            fig_bg.add_hline(y=bg_req, line_dash="dash", line_color="#e74c3c",
                                             annotation_text=f"Required {bg_req:.2f}", annotation_position="top right")
                            fig_bg.update_layout(
                                title="Safety margin vs time (defect growth)",
                                xaxis_title="Year", yaxis_title="Pf ÷ MAOP",
                                template="plotly_dark", paper_bgcolor="#1a2530", plot_bgcolor="#1a2530",
                                showlegend=False, height=320, margin=dict(l=50, r=30, t=60, b=50)
                            )
                            st.plotly_chart(fig_bg, use_container_width=True)
 
                            extras["b31g"] = {"od": float(bg_od), "t": bg_t, "smys": float(bg_smys), "d0": float(bg_d0),
                                              "length": float(bg_len), "dt0": float(dt0), "pf_psi": float(pf0 / 0.00689476),
                                              "maop": float(bg_maop), "margin0": float(margin0), "required": float(bg_req),
                                              "limit_year": bg_limit}
 
                    # 4) Basis & validation status
                    with st.expander("📚 Model basis, assumptions & validation status", expanded=False):
                        st.markdown(f"""
                    **Engine version:** {ENGINE_VERSION} | **Run ID:** {extras['run_id']}
 
                    - **Base model:** de Waard-Milliams (1991) CO₂ correlation with fugacity correction, then pH, H₂S, inhibitor,
                      coating, CP, fluid, material, water-cut, velocity and chloride factors.
                                        - **Calibrated items:** FeCO₃ scale credit, residual-film synergy and the multiphase field factor are
                      engineering calibrations, not first-principles physics.
                    - **Validation status:** not yet validated against measured field data for this line. Enter two UT readings
                      above to compare the model with reality.
                    - **Defect assessment:** modified B31G (0.85dL) — corrosion metal loss in carbon / low-alloy pipe only.
                    - **Use:** engineering decision-support. It does not replace a qualified corrosion engineer.
                    """)
 
 
                    sensdf = sensitivity_analysis(v, rate)
                    target_rate = v["wt"] / life if life > 0 else 0.2
                    optdose, optrate, *_ = find_optimal_dose(v, target_rate)
                    optqtday = optdose * v.get("dose_water_bbl", 100) * 0.00035
 
                    word_file = make_word_report_plus(
                        projdf, wt, rate, life, v, steps,
                        None, sensdf, optdose, optqtday, risk, extras
                    )
 
                    st.download_button(
                        "📄 Download Complete Word Report",
                        data=word_file,
                        file_name=f"Corzynth_{key_prefix}_Report.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        key=f"{key_prefix}_word"
                    )
 
                    st.subheader("💾 Save This Calculation")
 
                    if st.button(
                        "💾 Save Results to Database",
                        type="primary",
                        use_container_width=True,
                        key=f"{key_prefix}_save"
                    ):
                        saved_file = save_result_csv(v, rate, life_raw, risk)
                        st.success(f"✅ Saved to {saved_file}")
                   
        render_line_tab(tab_oil, "Oil Line", LINE_DEFAULTS["oil"], "oil")
        render_line_tab(tab_water, "Water Line", LINE_DEFAULTS["water"], "water")    
        render_line_tab(tab_gas, "Natural Gas Line", LINE_DEFAULTS["gas"], "gas")   
        render_line_tab(tab_multi, "Multiphase Line", LINE_DEFAULTS["multi"], "multi")
 
elif page == "🧪 Chemical Advisor":
        render_header()
        st.header("🧪 Chemical Advisor — Smart Buyer's Guide")
 
        chem_purpose = st.selectbox(
            "What is the dose needed for? (Chemical Purpose)",
            [
                "🛡️ Corrosion Inhibition",
                "💧 Water Treatment (Demulsifier/Deoiler)",
                "🦠 Biocide / Bactericide",
                "🧴 Scale Inhibition",
                "🫧 Antifoam / Defoamer",
                "⛽ Drag Reducing Agent",
                "⛽ Gas Treatment (H2S/Scavenger/Hydrate)"
            ]
        )
 
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
        st.caption(
            f"Selected purpose: {chem_purpose} → chemical category: {chem_type}"
        )
 
        c1, c2, c3, c4 = st.columns(4)
 
        t_wc = c1.number_input("Water Cut (%)", 0, 100, 50, 5)
        t_temp = c2.number_input("Temperature (°F)", 32, 400, 150, 5)
        t_salinity = c3.number_input("Salinity TDS (ppm)", 0, 200000, 50000, 5000)
        t_api = c4.number_input("Crude API Gravity", 10, 50, 30, 1)
 
        c5, c6, c7 = st.columns(3)
 
        t_ph = c5.number_input("Water pH", 3.0, 10.0, 7.0, 0.1)
        t_h2s = c6.number_input("H₂S (ppm)", 0, 20000, 500, 50)
        t_co2 = c7.number_input("CO₂ (%)", 0.0, 100.0, 3.0, 0.5)
 
        if st.button(
            "🔍 Analyze & Recommend",
            type="primary",
            use_container_width=True
        ):
            if chem_type == "💧 Demulsifier":
                rec_product = (
                    "Asphaltene-Dispersing EO/PO"
                    if (t_api < 20 and t_wc > 60)
                    else "High-MW EO/PO Block Copolymer"
                )
                rec_dose = f"{max(15, int(t_wc / 3))}-{max(30, int(t_wc / 2))} ppm"
                real_eff = "70-85% water removal in 30 min"
 
            elif chem_type == "🛡️ Corrosion Inhibitor":
                rec_product = (
                    "Imidazoline H₂S Inhibitor"
                    if t_h2s > 500
                    else "Film-Forming Amine"
                )
                rec_dose = (
                    f"{max(25, int(t_h2s / 100))} ppm"
                    if t_h2s > 100
                    else "25-75 ppm"
                )
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
 
            st.markdown(
                f"""
                <div style="background:linear-gradient(135deg,#1a5276,#2e86c1);padding:28px;
                     border-radius:16px;color:white;margin:20px 0;">
                    <h2 style="margin:0;color:#fff;">✅ USE THIS CHEMICAL</h2>
                    <h3 style="margin:12px 0 6px 0;color:#ffd700;">🏆 {rec_product}</h3>
                    <p style="margin:0;font-size:17px;">
                        Dose: <b>{rec_dose}</b> |
                        Real Efficiency: <b>{real_eff}</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
 
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
    st.info("Compare all your saved wells side by side — corrosion rate, remaining life, risk, and inhibitor use.")
 
    csv_file = user_csv_file()   # ← only THIS user's data
 
    # Restore previously saved (old shared) data for admin, one time
    legacy_file = "corzynth_results.csv"
    if (not os.path.exists(csv_file)
            and os.path.exists(legacy_file)
            and st.session_state.get("username") == "admin"):
        import shutil
        shutil.copy(legacy_file, csv_file)
 
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
            show_cols = [c for c in ["Well / Stream Name", "Saved At", "Fluid", "Corrosion Rate (mm/yr)",
                                     "Remaining Life (yrs)", "Risk",
                                     "Inhibitor (quarts/day)", "Velocity (m/s)",
                                     "Water Cut (%)"] if c in df.columns]           
            st.dataframe(df[show_cols], use_container_width=True)
            if "Corrosion Rate (mm/yr)" in df.columns:
                st.subheader("📊 Corrosion Rate by Well")
                import plotly.graph_objects as go
 
                chart_df = df.copy()
                if "Well / Stream Name" in chart_df.columns:
                    chart_df["Well"] = chart_df["Well / Stream Name"].fillna("Unnamed Well")
                else:
                    chart_df["Well"] = ["Well " + str(i + 1) for i in range(len(chart_df))]
 
                # Color each bar by risk level
                def risk_color(risk):
                    if risk == "HIGH":
                        return "#e74c3c"
                    elif risk == "MEDIUM":
                        return "#f39c12"
                    else:
                        return "#2ecc71"
 
                bar_colors = [risk_color(r) for r in chart_df["Risk"]]
 
                fig = go.Figure()
 
                fig.add_trace(go.Bar(
                    y=chart_df["Well"],
                    x=chart_df["Corrosion Rate (mm/yr)"],
                    orientation="h",
                    marker_color=bar_colors,
                    text=chart_df["Corrosion Rate (mm/yr)"].round(3),
                    textposition="outside",
                    name="Corrosion Rate",
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Corrosion Rate: %{x:.3f} mm/yr<br>"
                        "Risk: %{customdata[0]}<br>"
                        "Remaining Life: %{customdata[1]:.1f} yrs<extra></extra>"
                    ),
                    customdata=list(zip(
                        chart_df["Risk"],
                        chart_df["Remaining Life (yrs)"]
                    ))
                ))
 
                # High-corrosion threshold line (0.5 mm/yr)
                fig.add_vline(
                    x=0.5,
                    line_dash="dash",
                    line_color="#e74c3c",
                    annotation_text="⚠️ High Corrosion (0.5 mm/yr)",
                    annotation_position="top right",
                    annotation_font=dict(color="#e74c3c", size=12)
                )
 
                fig.update_layout(
                    title={
                        "text": "Corrosion Rate by Well / Stream",
                        "font": dict(size=20, color="#ecf0f1")
                    },
                    xaxis=dict(
                        title="Corrosion Rate (mm/yr)",
                        gridcolor="#34495e",
                        zerolinecolor="#34495e"
                    ),
                    yaxis=dict(
                        title="Well / Stream Name",
                        gridcolor="#34495e",
                        autorange="reversed"
                    ),
                    hovermode="y unified",
                    template="plotly_dark",
                    paper_bgcolor="#1a2530",
                    plot_bgcolor="#1a2530",
                    font=dict(color="#ecf0f1"),
                    showlegend=False,
                    margin=dict(l=150, r=50, t=70, b=50),
                    height=max(300, 40 * len(chart_df))
                )
 
                st.plotly_chart(fig, use_container_width=True)
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
                                   data=f, file_name=csv_file,
                                   mime="text/csv", key="dl_dash")
 
elif page == "🛡️ Audit Log":
    render_header()
    st.header("🛡️ Audit Log — Admin Only")
    if st.session_state.get("username") != "admin":
        st.error("Admin access only.")
    elif not os.path.exists(AUDIT_FILE):
        st.info("No activity recorded yet.")
    else:
        try:
            alog = pd.read_csv(AUDIT_FILE, dtype=str).fillna("")
        except Exception as e:
            st.error(f"❌ Could not read the audit log: {e}")
            alog = pd.DataFrame()
        if len(alog):
            ev_pick = st.multiselect("Filter by event", sorted(alog["Event"].unique()))
            usr_pick = st.multiselect("Filter by user", sorted(alog["User"].unique()))
            view = alog
            if ev_pick:
                view = view[view["Event"].isin(ev_pick)]
            if usr_pick:
                view = view[view["User"].isin(usr_pick)]
            a1, a2, a3 = st.columns(3)
            a1.metric("Events", len(view))
            a2.metric("Failed logins", int((view["Event"] == "login_failed").sum()))
            a3.metric("Lock-outs", int((view["Event"] == "login_locked").sum()))
            st.dataframe(view.iloc[::-1].head(500), use_container_width=True)
            st.download_button("📥 Download Audit Log (CSV)",
                               data=alog.to_csv(index=False).encode("utf-8"),
                               file_name="corzynth_audit_log.csv", mime="text/csv",
                               key="dl_audit")
            st.caption("Shows the latest 500 matching events. Passwords are never logged.")
 
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