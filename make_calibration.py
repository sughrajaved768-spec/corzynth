import pandas as pd
import joblib
from corzynth_features import add_features, RAW_COLS
from app import full_calculation
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("Corzynth_Corrosion_Data_5000.csv")
model = joblib.load("corrosion_model.pkl")

df_feat = add_features(df)
X = df_feat[RAW_COLS]
pred = model.predict(X)
df["pred"] = pred
df["ratio"] = df["Corrosion_Rate_mm_yr"] / df["pred"]

calib = {}
for fluid in df["Fluid_Type"].unique():
    sub = df[df["Fluid_Type"] == fluid]
    calib[fluid] = round(sub["ratio"].median(), 3)

# Region-specific Oil calibration: high CO2 + high temp (conservative, 75th percentile)
oil_hi = df[(df["Fluid_Type"] == "🛢️ Crude Oil") &
            (df["CO2_percent"] >= 20) &
            (df["Operating_Temperature_F"] >= 180)]
if len(oil_hi) > 50:
    calib["🛢️ Crude Oil_highCO2_highTemp"] = round(oil_hi["ratio"].quantile(0.75), 3)
    print(f"Oil high-region: median={oil_hi['ratio'].median():.3f}, "
          f"75th pct={oil_hi['ratio'].quantile(0.75):.3f}, "
          f"90th pct={oil_hi['ratio'].quantile(0.90):.3f}, "
          f"n={len(oil_hi)}")

joblib.dump(calib, "ml_calibration.pkl")
print("Calibration factors:", calib)