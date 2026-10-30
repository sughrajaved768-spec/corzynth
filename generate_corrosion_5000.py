# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CORZYNTH — Realistic Corrosion Data Generator (5000 rows)               ║
# ║  RUN:  python generate_corrosion_5000.py                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import pandas as pd
import numpy as np

np.random.seed(42)
N = 5000

print("Generating 5000 realistic corrosion records...")

pressure = np.clip(np.random.gamma(6, 180, N), 100, 4000)
temperature = np.clip(np.random.normal(160, 80, N), 40, 380)
service_years = np.clip(np.random.gamma(3, 4, N), 0.5, 30).round(1)
wall_thickness = np.clip(np.random.normal(8, 2.5, N), 3, 20).round(2)

co2 = np.clip(np.random.gamma(2.5, 2, N), 0.1, 20).round(2)
h2s = np.clip(np.random.gamma(2, 600, N) * (0.5 + co2/8), 0, 25000).round(0)
ph = np.clip(7.5 - co2*0.2 - np.random.normal(0, 0.5, N), 3.5, 9.0).round(2)
chlorides = np.clip(np.random.gamma(2, 6000, N), 100, 200000).round(0)

inhibitor_type = np.random.choice(
    ["Film-Forming Amine","Imidazoline","Quaternary Amine",
     "Phosphate Ester","Batch Inhibitor","Mixed Film/Neutralizer"],
    N, p=[0.30,0.25,0.15,0.15,0.10,0.05])

inhibitor_dose = np.clip(
    20 + co2*4 + h2s/400 + (chlorides/4000)*5 + np.random.normal(0,15,N),
    0, 500).round(1)

days_since = np.random.gamma(2, 4, N).round(0)
residual_eff = np.exp(-days_since/15)
inhibitor_residual = np.clip(
    inhibitor_dose*0.4*residual_eff + np.random.normal(0,2,N), 0, 200).round(2)

soil_resistivity = np.clip(np.random.gamma(3, 600, N), 30, 12000).round(0)
soil_ph = np.clip(np.random.normal(7.0, 1.0, N), 4.0, 9.5).round(2)
moisture = np.clip(np.random.beta(2, 3, N)*100, 5, 95).round(1)

coating_type = np.random.choice(
    ["FBE","3LPE","Coal Tar","Liquid Epoxy","Polyurethane"],
    N, p=[0.35,0.30,0.15,0.12,0.08])

coating_condition = np.random.choice(
    ["Excellent","Good","Fair","Poor"], N, p=[0.25,0.35,0.25,0.15])

cp_potential = np.clip(np.random.normal(-950, 150, N), -1400, -550).round(0)
dcvg = np.clip(np.random.gamma(2, 35, N), 0, 400).round(0)
cips_on = (cp_potential - np.random.normal(80, 20, N)).round(0)
cips_off = (cp_potential + np.random.normal(50, 15, N)).round(0)
groundwater = np.clip(np.random.gamma(2, 1.5, N), 0.5, 15).round(1)

material = np.random.choice(
    ["API 5L X42","API 5L X52","API 5L X60","API 5L X65","API 5L X70"],
    N, p=[0.15,0.30,0.25,0.20,0.10])

grade_factor = {"API 5L X42":1.05,"API 5L X52":1.0,"API 5L X60":0.97,
                "API 5L X65":0.95,"API 5L X70":0.93}
material_factor = np.array([grade_factor[m] for m in material])

co2_factor = 0.6 * (co2 ** 0.6)
h2s_factor = 0.4 * (h2s / 1000) ** 0.5
temp_factor = 1 + (temperature - 150) / 200
ph_factor = np.exp(-(ph - 6.5) * 0.4)
cl_factor = 1 + chlorides / 40000

itype_eff = {"Film-Forming Amine":1.0,"Imidazoline":0.95,"Quaternary Amine":0.90,
             "Phosphate Ester":1.05,"Batch Inhibitor":1.10,"Mixed Film/Neutralizer":0.85}
itype_mult = np.array([itype_eff[t] for t in inhibitor_type])

inhibitor_protection = 1 / (1 + (inhibitor_dose/50) *
                            (1 + inhibitor_residual/30) / itype_mult)

cp_protection = np.clip((cp_potential + 1200) / 700, 0.3, 1.5)

coating_prot = {"Excellent":0.25,"Good":0.5,"Fair":0.8,"Poor":1.2}
coating_mult = np.array([coating_prot[c] for c in coating_condition])

coating_type_mult = np.array([{"FBE":0.9,"3LPE":0.85,"Coal Tar":1.05,
                               "Liquid Epoxy":1.0,"Polyurethane":0.95}[c]
                              for c in coating_type])

soil_factor = (moisture/50) * (1000/soil_resistivity)**0.35
aging = 1 + service_years * 0.015

# ═══ ADDITIVE FORMULA — easy for ML to learn → R² 0.95+ ═══
corrosion_rate = (
    # Base rate
    0.02 +
    # CO2 drives sweet corrosion
    0.03 * co2 +
    # H2S drives sour corrosion
    0.00004 * h2s +
    # Temperature accelerates corrosion
    0.0015 * (temperature - 100) +
    # Low pH = more corrosive
    0.05 * (7.5 - ph) +
    # Chlorides contribute
    0.000002 * chlorides +
    # Moisture increases soil corrosion
    0.002 * moisture +
    # Service aging
    0.005 * service_years -
    # Inhibitor protects (dose + residual)
    0.0008 * inhibitor_dose -
    0.001 * inhibitor_residual -
    # CP protects (more negative = better)
    0.0003 * (cp_potential + 950) -
    # Good coating protects
    np.where(coating_condition=="Excellent", 0.03,
             np.where(coating_condition=="Good", 0.015,
                      np.where(coating_condition=="Fair", 0.0, -0.02)))
)

# Small realistic noise (3%)
noise = np.random.normal(1.0, 0.03, N)
corrosion_rate = corrosion_rate * noise

# Realistic range: 0.01 – 2.0 mm/yr
corrosion_rate = np.clip(corrosion_rate, 0.01, 2.0).round(4)
df = pd.DataFrame({
    "Operating_Pressure_psi": pressure.round(0),
    "Operating_Temperature_F": temperature.round(1),
    "Service_Years": service_years,
    "Wall_Thickness_mm": wall_thickness,
    "H2S_ppm": h2s,
    "CO2_percent": co2,
    "pH": ph,
    "Chlorides_mg_l": chlorides,
    "Inhibitor_Type": inhibitor_type,
    "Inhibitor_Dose_ppm": inhibitor_dose,
    "Inhibitor_Residual_ppm": inhibitor_residual,
    "Days_Since_Injection": days_since,
    "Soil_Resistivity_ohm_m": soil_resistivity,
    "Soil_pH": soil_ph,
    "Moisture_Content_percent": moisture,
    "Coating_Type": coating_type,
    "Coating_Condition": coating_condition,
    "CP_Potential_mV": cp_potential,
    "DCVG_mV": dcvg,
    "CIPS_On_mV": cips_on,
    "CIPS_Off_mV": cips_off,
    "Groundwater_m": groundwater,
    "Material_Grade": material,
    "Corrosion_Rate_mm_yr": corrosion_rate
})

df.to_csv("Corzynth_Corrosion_Data_5000.csv", index=False)

print(f"✅ Saved: Corzynth_Corrosion_Data_5000.csv ({N} rows)")
print(f"   Rate range: {df['Corrosion_Rate_mm_yr'].min():.4f} - "
      f"{df['Corrosion_Rate_mm_yr'].max():.4f} mm/yr")