import pandas as pd, joblib

model = joblib.load("corrosion_model.pkl")

row = {
    "Pipe_Diameter_in": 8, "Wall_Thickness_mm": 8.0, "Service_Years": 10.0,
    "Operating_Pressure_psi": 1000, "Operating_Temperature_F": 200,
    "CO2_percent": 25, "H2S_ppm": 10, "pH": 6.5, "Chlorides_mg_L": 15000,
    "Oil_Rate_bbl_d": 2000, "Water_Rate_bbl_d": 500, "Gas_Rate_MMscfd": 5, "Gas_SG": 0.7,
    "Velocity_m_s": 3.4, "Water_Cut_percent": 20.0,
    "Inhibitor_Dose_ppm": 35, "Inhibitor_Dose_qt_day": 2.94, "Residual_Inhibitor_ppm": 24,
    "Target_Remaining_Life_yr": 10.0, "Fe_Count_mg_L": 50.0,
    "Fluid_Type": "🛢️ Crude Oil", "Material_Grade": "API 5L X42",
    "Coating_Condition": "Good", "CP_Status": "Yes", "Inhibitor_Type": "Film-Forming Amine"
}

X = pd.DataFrame([row])
print("ML prediction:", model.predict(X)[0])