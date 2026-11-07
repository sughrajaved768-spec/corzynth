import numpy as np
import pandas as pd
import random
import sys
sys.path.insert(0, ".")
from app import full_calculation

random.seed(42)
np.random.seed(42)

n = 20000
rows = []
for i in range(n):
    v = {
        "fluid": random.choice(["🛢️ Crude Oil", "⛽ Natural Gas", "💧 Water", "🫧 Multiphase (Oil+Gas+Water)"]),
        "material": random.choice(["API 5L X42", "API 5L X52", "Carbon Steel A106"]),
        "coating_condition": random.choice(["Excellent", "Good", "Fair", "Poor"]),
        "cp_on": random.choice(["Yes", "No"]),
        "pressure": random.uniform(100, 2500),
        "temp": random.uniform(80, 250),
        "co2": random.uniform(0.5, 30),
        "h2s": random.choice([0, 0, 0, 50, 150, 500, 1000]),
        "ph": random.uniform(4.5, 8.5),
        "chlorides": random.choice([1000, 5000, 15000, 30000, 60000, 100000]),
        "wt": random.uniform(4, 15),
        "service_years": random.uniform(1, 30),
        "dose": random.choice([0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150]),
        "residual_ppm": random.uniform(0, 400),
        "residual_qt": 0,
        "velocity": random.uniform(0.5, 6),
        "water_cut": random.uniform(0, 100),
        "diameter": random.choice([2, 3, 4, 6, 8, 10, 12, 16]),
        "oil_rate": random.uniform(100, 5000),
        "water_rate": random.uniform(50, 3000),
        "gas_rate": random.uniform(0.5, 50),
        "gas_sg": random.uniform(0.55, 0.85),
    }
    rate, life, life_raw, risk, emoji, steps = full_calculation(v)
    rows.append({**v, "Corrosion_Rate_mm_yr": rate})

df = pd.DataFrame(rows)

# Rename columns to match what train_model.py expects
rename_map = {
    "fluid": "Fluid_Type",
    "material": "Material_Grade",
    "coating_condition": "Coating_Condition",
    "cp_on": "CP_Status",
    "pressure": "Operating_Pressure_psi",
    "temp": "Operating_Temperature_F",
    "co2": "CO2_percent",
    "h2s": "H2S_ppm",
    "ph": "pH",
    "chlorides": "Chlorides_mg_L",
    "wt": "Wall_Thickness_mm",
    "service_years": "Service_Years",
    "dose": "Inhibitor_Dose_ppm",
    "residual_ppm": "Residual_Inhibitor_ppm",
    "residual_qt": "Inhibitor_Dose_qt_day",
    "velocity": "Velocity_m_s",
    "water_cut": "Water_Cut_percent",
    "diameter": "Pipe_Diameter_in",
    "oil_rate": "Oil_Rate_bbl_d",
    "water_rate": "Water_Rate_bbl_d",
    "gas_rate": "Gas_Rate_MMscfd",
    "gas_sg": "Gas_SG",
}
df = df.rename(columns=rename_map)

# Add missing columns that train_model.py expects
df["Inhibitor_Type"] = "Film-Forming Amine"
df["Target_Remaining_Life_yr"] = 10.0
df["Fe_Count_mg_L"] = 50.0

df.to_csv("Corzynth_Corrosion_Data_5000.csv", index=False)
print(f"Saved {len(df)} rows using YOUR app's physics chain")