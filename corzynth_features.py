"""Shared feature engineering for Corzynth. Keep this file next to train_model.py and app.py:
the saved corrosion_model.pkl imports add_features from here when it is loaded."""
import numpy as np
import pandas as pd

TARGET = "Corrosion_Rate_mm_yr"
COAT_COND = {"Excellent": 0, "Good": 1, "Fair": 2, "Poor": 3}
COAT_TYPE = {"FBE": 0, "3LPE": 1, "Coal Tar": 2, "Liquid Epoxy": 3, "Polyurethane": 4}
CATEGORICAL = ["Inhibitor_Type", "Material_Grade"]
RAW_NUM = ["Operating_Pressure_psi", "Operating_Temperature_F", "Service_Years", "H2S_ppm", "CO2_percent",
           "pH", "Chlorides_mg_l", "Inhibitor_Dose_ppm", "Inhibitor_Residual_ppm", "Days_Since_Injection",
           "Soil_Resistivity_ohm_m", "Soil_pH", "Moisture_Content_percent", "CP_Potential_mV", "DCVG_mV",
           "CIPS_On_mV", "CIPS_Off_mV", "Groundwater_m"]
RAW_MAPPED = ["Coating_Type", "Coating_Condition"]       # text in the CSV, numeric after add_features
RAW_COLS = RAW_NUM + RAW_MAPPED + CATEGORICAL
ENGINEERED = ["CO2_H2S_factor", "TP_factor", "inhibitor_eff", "soil_corrosivity", "acid_load",
              "chloride_acid", "cp_effectiveness", "protection_score"]
NUMERIC_ALL = RAW_NUM + RAW_MAPPED + ENGINEERED

def _map(s, mapping, default):
    m = s.map(mapping)
    return m.where(m.notna(), pd.to_numeric(s, errors="coerce")).fillna(default)

def add_features(df):
    d = df[RAW_COLS].copy()
    d["Coating_Condition"] = _map(d["Coating_Condition"], COAT_COND, 1)
    d["Coating_Type"] = _map(d["Coating_Type"], COAT_TYPE, 0)
    for c in RAW_NUM:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["CO2_H2S_factor"] = d.CO2_percent * d.H2S_ppm / 1000
    d["TP_factor"] = d.Operating_Temperature_F * d.Operating_Pressure_psi / 1000
    d["inhibitor_eff"] = d.Inhibitor_Residual_ppm / (d.Inhibitor_Dose_ppm + 1)
    d["soil_corrosivity"] = d.Moisture_Content_percent * (14 - d.Soil_pH) / d.Soil_Resistivity_ohm_m.clip(lower=1)
    d["acid_load"] = d.CO2_percent + d.H2S_ppm / 1000
    d["chloride_acid"] = d.Chlorides_mg_l / (d.pH + 1)
    d["cp_effectiveness"] = (d.CP_Potential_mV + 1200) / 700
    d["protection_score"] = d.cp_effectiveness * d.inhibitor_eff
    return d.replace([np.inf, -np.inf], np.nan)

def physics_rate(df):
    """Approximate de Waard-Milliams rate for a sanity baseline (no fluid factor; CP <= -850 mV halves rate)."""
    tk = (df.Operating_Temperature_F - 32) * 5 / 9 + 273.15
    p = np.minimum(df.Operating_Pressure_psi * 0.0689476, 250)
    f = (10 ** (p * (0.0031 - 1.4 / tk)) * df.CO2_percent / 100 * p).clip(lower=1e-9)
    v = 10 ** (5.8 - 1710 / tk + 0.67 * np.log10(f))
    v *= np.where(df.pH < 4.5, 1.3, np.where(df.pH > 6.5, 0.6, 1.0))
    dose = df.Inhibitor_Dose_ppm
    v *= np.where(dose > 0, 1 - np.minimum(0.95, 0.5 + dose / 100), 1.0)
    cond = _map(df.Coating_Condition, COAT_COND, 1)
    v *= cond.map({0: 0.3, 1: 0.5, 2: 0.8, 3: 1.0}).fillna(0.5)
    v *= np.where(df.CP_Potential_mV <= -850, 0.5, 1.0)
    return v
