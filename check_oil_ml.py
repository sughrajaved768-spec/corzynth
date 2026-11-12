import pandas as pd
import joblib
from corzynth_features import add_features

model = joblib.load("corrosion_model.pkl")
cols = joblib.load("feature_cols.pkl")

ml_input = {
    "Fluid_Type": "🛢️ Crude Oil",
    "Material_Grade": "Carbon Steel A106",
    "Coating_Condition": "Good",
    "CP_Status": "Yes",
    "Pipe_Diameter_in": 6.0,
    "Wall_Thickness_mm": 8.0,
    "Service_Years": 10.0,
    "Operating_Pressure_psi": 1000.0,
    "Operating_Temperature_F": 200.0,
    "CO2_percent": 25.0,
    "H2S_ppm": 10.0,
    "pH": 6.5,
    "Chlorides_mg_L": 15000.0,
    "Oil_Rate_bbl_d": 2000.0,
    "Water_Rate_bbl_d": 500.0,
    "Gas_Rate_MMscfd": 0.0,
    "Gas_SG": 0.65,
    "Velocity_m_s": 0.5,
    "Water_Cut_percent": 20.0,
    "Inhibitor_Type": "Film-Forming",
    "Inhibitor_Dose_ppm": 30.0,
    "Inhibitor_Dose_qt_day": 2.52,
    "Residual_Inhibitor_ppm": 30.0,
    "Target_Remaining_Life_yr": 10.0,
    "Fe_Count_mg_L": 50.0,
}

df_in = pd.DataFrame([ml_input])
df_in = add_features(df_in)
for c in cols:
    if c not in df_in.columns:
        df_in[c] = 0
df_in = df_in[cols]

pred = model.predict(df_in)[0]
print(f"Model prediction: {pred:.3f} mm/yr")
print(f"Physics target:   0.324 mm/yr")
print(f"Ratio: {0.324/pred:.2f}x")

# Find nearest training rows to see what the model 'should' have learned
df = pd.read_csv("Corzynth_Corrosion_Data_5000.csv")
oil = df[df["Fluid_Type"] == "🛢️ Crude Oil"]
near = oil[(oil["CO2_percent"] >= 20) & (oil["Operating_Temperature_F"] >= 180) &
           (oil["Residual_Inhibitor_ppm"].between(20, 40))]
print(f"\nNearby training rows (CO2>=20, temp>=180, residual 20-40): {len(near)}")
if len(near) > 0:
    print(near[["CO2_percent", "Operating_Temperature_F", "Residual_Inhibitor_ppm",
                "Velocity_m_s", "Corrosion_Rate_mm_yr"]].head(10).to_string())