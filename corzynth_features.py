"""Shared feature engineering for Corzynth. Keep this file next to train_model.py and app.py:
the saved corrosion_model.pkl imports add_features from here when it is loaded."""
import numpy as np
import pandas as pd




TARGET = "Corrosion_Rate_mm_yr"



# ---- Column groups (match Corzynth_Corrosion_Model_Dataset_5000.csv) ----
CATEGORICAL = ["Fluid_Type", "Material_Grade", "Coating_Condition", "CP_Status", "Inhibitor_Type"]



RAW_NUM = [
    "Pipe_Diameter_in", "Wall_Thickness_mm", "Service_Years",
    "Operating_Pressure_psi", "Operating_Temperature_F",
    "CO2_percent", "H2S_ppm", "pH", "Chlorides_mg_L",
    "Oil_Rate_bbl_d", "Water_Rate_bbl_d", "Gas_Rate_MMscfd", "Gas_SG",
    "Velocity_m_s", "Water_Cut_percent",
    "Inhibitor_Dose_ppm", "Inhibitor_Dose_qt_day", "Residual_Inhibitor_ppm",
    "Target_Remaining_Life_yr", "Fe_Count_mg_L"
]



RAW_COLS = RAW_NUM + CATEGORICAL



ENGINEERED = [
    "CO2_H2S_factor", "TP_factor", "inhibitor_eff", "overdose_factor",
    "acid_load", "chloride_acid", "cp_effectiveness", "protection_score",
    "feco3_credit", "flow_severity",
    "Oil_x_LowWaterCut", "Multiphase_x_HighWaterCut",
    "Residual_x_Temp", "CO2_x_pH",
    "Oil_x_Velocity", "Oil_x_CO2", "Oil_x_Temp", "Oil_x_Residual"
]



NUMERIC_ALL = RAW_NUM + ENGINEERED



# Columns excluded from features (used only for grouping/calibration)
EXCLUDED = ["Pipeline_ID", "Segment_ID", "Date", "Notes",
            "Measured_Rate_Coupon_mm_yr", "UT_Wall_Thickness_mm",
            "Measured_Wall_Loss_mm", "Calibration_Factor"]




def _num(s, default=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(default)




def add_features(df):
    d = df[RAW_COLS].copy()
    for c in RAW_NUM:
        d[c] = _num(d[c])



    # ---- Engineered features ----
    d["CO2_H2S_factor"] = d.CO2_percent * (d.H2S_ppm / 1000.0)
    d["TP_factor"] = d.Operating_Temperature_F * d.Operating_Pressure_psi / 1000.0



    # Residual-driven inhibitor efficiency (matches app logic)
    resid = d.Residual_Inhibitor_ppm
    eff = np.where(resid > 0,
                   np.minimum(0.95, 0.75 + resid / 100.0),
                   0.75)
    eff = np.where((resid > 0) & (resid < 10), eff * 0.5, eff)
    d["inhibitor_eff"] = eff



    # Overdose penalty
    d["overdose_factor"] = np.where(resid > 300, 1.25,
                            np.where(resid > 100, 1.15,
                              np.where(resid > 40, 1.05, 1.0)))



    d["acid_load"] = d.CO2_percent + d.H2S_ppm / 1000.0
    d["chloride_acid"] = d.Chlorides_mg_L / (d.pH.clip(lower=0.1) + 1.0)



    # CP effectiveness proxy (higher = more protective)
    d["cp_effectiveness"] = np.where(d.CP_Status == "Yes", 1.0, 0.0)



    # Combined protection score
    d["protection_score"] = d.cp_effectiveness * d.inhibitor_eff



    # FeCO3 scale credit (matches app logic)
    d["feco3_credit"] = np.where(
        (d.Operating_Temperature_F > 150) & (d.pH >= 6.0), 0.74,
        np.where(d.pH >= 5.0, 0.84, 0.94))



    # Flow severity (velocity + water cut interaction)
    d["flow_severity"] = d.Velocity_m_s * (d.Water_Cut_percent / 100.0)



    # Fluid-specific interaction features
    d["Oil_x_LowWaterCut"] = (
        (d["Fluid_Type"] == "🛢️ Crude Oil") & (d["Water_Cut_percent"] < 30)
    ).astype(int)

    d["Multiphase_x_HighWaterCut"] = (
        (d["Fluid_Type"] == "🫧 Multiphase (Oil+Gas+Water)") & (d["Water_Cut_percent"] > 50)
    ).astype(int)

    d["Residual_x_Temp"] = d["Residual_Inhibitor_ppm"] * d["Operating_Temperature_F"]

    d["CO2_x_pH"] = d["CO2_percent"] * d["pH"]

    # Oil-specific interaction features (help model learn Oil correction chain)
    d["Oil_x_Velocity"] = (
        (d["Fluid_Type"] == "🛢️ Crude Oil") * d["Velocity_m_s"]
    )

    d["Oil_x_CO2"] = (
        (d["Fluid_Type"] == "🛢️ Crude Oil") * d["CO2_percent"]
    )

    d["Oil_x_Temp"] = (
        (d["Fluid_Type"] == "🛢️ Crude Oil") * d["Operating_Temperature_F"]
    )

    d["Oil_x_Residual"] = (
        (d["Fluid_Type"] == "🛢️ Crude Oil") * d["Residual_Inhibitor_ppm"]
    )



    return d.replace([np.inf, -np.inf], np.nan)




def physics_rate(df):
    """Simplified de Waard-Milliams baseline with correct units."""
    tk = (df.Operating_Temperature_F - 32) * 5.0 / 9.0 + 273.15
    p_atm = df.Operating_Pressure_psi * 0.068046
    pco2 = ((df.CO2_percent / 100.0) * p_atm).clip(lower=1e-6)



    # Correct de Waard-Milliams form: rate depends on pCO2 and temperature
    log_f = np.log10(pco2) - 0.0031 * pco2 - 1.4 / tk
    v = 10 ** (5.8 - 1710.0 / tk + 0.67 * log_f)



    # pH correction
    v *= np.where(df.pH < 4.5, 1.3, np.where(df.pH > 6.5, 0.6, 1.0))



    # Inhibitor (residual-driven)
    resid = df.Residual_Inhibitor_ppm
    eff = np.where(resid > 0, np.minimum(0.95, 0.75 + resid / 100.0), 0.0)
    eff = np.where((resid > 0) & (resid < 10), eff * 0.5, eff)
    v *= (1.0 - eff)



    # Coating
    coat = df.Coating_Condition.map({"Good": 0.5, "Fair": 0.75, "Poor": 1.0}).fillna(0.75)
    v *= coat



    # CP
    v *= np.where(df.CP_Status == "Yes", 0.5, 1.0)



    # Fluid factor
    v *= np.where(df.Fluid_Type == "Multiphase", 1.15,
           np.where(df.Fluid_Type == "Gas", 0.7, 1.0))



    # Water cut / velocity / chloride / H2S / FeCO3
    v *= np.where((df.Water_Cut_percent >= 20) & (df.Water_Cut_percent <= 40), 0.7, 1.0)
    v *= np.where(df.Velocity_m_s > 3, 1.15, 1.0)
    v *= np.where(df.Chlorides_mg_L > 10000, 1.1, 1.0)
    v *= np.where(df.H2S_ppm > 500, 1.3, 1.0)
    v *= np.where((df.Operating_Temperature_F > 150) & (df.pH >= 6.0), 0.74,
           np.where(df.pH >= 5.0, 0.84, 0.94))



    return v.clip(lower=1e-6, upper=50.0)