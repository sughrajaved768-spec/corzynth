import joblib
import pandas as pd
from app import full_calculation

# Exact values from your Oil Line tab (from the debug box)
v = dict(
    fluid="🛢️ Crude Oil",
    pressure=1000,
    temp=200,
    wt=8.0,
    material="Carbon Steel A106",
    coating_condition="Good",
    cp_on="Yes",
    co2=25.0,
    h2s=20,
    ph=6.5,
    dose=20.0,
    itype="Film-Forming Amine",
    residual_ppm=61.876729559748426,
    residual_qt=5.2,
    daily_qt=3.5,
    velocity=3.0,
    water_cut=20.0,
    chlorides=15000,
    dose_water_bbl=500,
    service_years=10,
    diameter=6.0,
    oil_rate=2000,
    water_rate=500,
    gas_rate=5.0,
    gas_sg=0.65,
)

# 1. What physics says RIGHT NOW
rate, life, life_raw, risk, emoji, steps = full_calculation(v)
print(f"Physics rate (app, right now): {rate:.3f} mm/yr")

# 2. What the ML model predicts for the same inputs
model = joblib.load("corrosion_model.pkl")
cols = joblib.load("feature_cols.pkl")
ml_input = {
    "Fluid_Type": v["fluid"],
    "Material_Grade": v["material"],
    "Coating_Condition": v["coating_condition"],
    "CP_Status": v["cp_on"],
    "Pipe_Diameter_in": v["diameter"],
    "Wall_Thickness_mm": v["wt"],
    "Service_Years": v["service_years"],
    "Operating_Pressure_psi": v["pressure"],
    "Operating_Temperature_F": v["temp"],
    "CO2_percent": v["co2"],
    "H2S_ppm": v["h2s"],
    "pH": v["ph"],
    "Chlorides_mg_L": v["chlorides"],
    "Oil_Rate_bbl_d": v["oil_rate"],
    "Water_Rate_bbl_d": v["water_rate"],
    "Gas_Rate_MMscfd": v["gas_rate"],
    "Gas_SG": v["gas_sg"],
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
print(f"ML model prediction: {ml_rate:.3f} mm/yr")

# 3. What the training CSV says for the closest matching row
df = pd.read_csv("Corzynth_Corrosion_Data_5000.csv")
mask = (
    (df["Fluid_Type"] == "🛢️ Crude Oil") &
    (df["Operating_Temperature_F"].between(195, 205)) &
    (df["CO2_percent"].between(24, 26)) &
    (df["Residual_Inhibitor_ppm"].between(55, 70)) &
    (df["Coating_Condition"] == "Good") &
    (df["CP_Status"] == "Yes")
)
matches = df[mask]
if len(matches) > 0:
    print(f"Training rows matching these conditions: {len(matches)}")
    print(f"Training Corrosion_Rate_mm_yr (mean): {matches['Corrosion_Rate_mm_yr'].mean():.3f} mm/yr")
    print(f"Training Corrosion_Rate_mm_yr (min/max): {matches['Corrosion_Rate_mm_yr'].min():.3f} / {matches['Corrosion_Rate_mm_yr'].max():.3f} mm/yr")
else:
    print("No training rows match these conditions — data coverage gap!")