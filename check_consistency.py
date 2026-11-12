import pandas as pd
from app import full_calculation

df = pd.read_csv("Corzynth_Corrosion_Data_5000.csv").head(5)

for _, r in df.iterrows():
    v = dict(
        fluid=r["Fluid_Type"], pressure=r["Operating_Pressure_psi"],
        temp=r["Operating_Temperature_F"], wt=r["Wall_Thickness_mm"],
        min_required_wall=3.0, material=r["Material_Grade"],
        coating_condition=r["Coating_Condition"], cp_on=r["CP_Status"],
        co2=r["CO2_percent"], h2s=r["H2S_ppm"], ph=r["pH"],
        dose=r["Inhibitor_Dose_ppm"], itype=r["Inhibitor_Type"],
        residual_ppm=r["Residual_Inhibitor_ppm"],
        residual_qt=r["Inhibitor_Dose_qt_day"],
        daily_qt=r["Inhibitor_Dose_qt_day"],
        velocity=r["Velocity_m_s"], water_cut=r["Water_Cut_percent"],
        chlorides=r["Chlorides_mg_L"], dose_water_bbl=r["Water_Rate_bbl_d"],
        service_years=r["Service_Years"], diameter=r["Pipe_Diameter_in"],
        oil_rate=r["Oil_Rate_bbl_d"], water_rate=r["Water_Rate_bbl_d"],
        gas_rate=r["Gas_Rate_MMscfd"], gas_sg=r["Gas_SG"],
    )
    rate, *_ = full_calculation(v)
    csv_target = r["Corrosion_Rate_mm_yr"]
    match = "✅" if abs(rate - csv_target) / max(csv_target, 1e-6) < 0.05 else "❌ MISMATCH"
    print(f"CSV target: {csv_target:.3f} | App physics now: {rate:.3f} | {match}")