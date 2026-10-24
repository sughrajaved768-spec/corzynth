import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("Synthetic-Corrosion-Data-300-Data-Only.csv")

features = [
    "H2S_ppm", "CO2_percent", "Operating_Pressure_bar",
    "Operating_Temperature_C", "pH", "Chlorides_mg_L", "TDS_mg_L",
    "Soil_Resistivity_ohm_m", "Soil_pH", "Moisture_Content_percent",
    "Coating_Type", "Coating_Condition", "Cathodic_Protection",
    "CP_Potential_mV", "Inhibitor_Dose_ppm", "Service_Years",
    "Pipe_Wall_Thickness_mm", "Material_Grade"
]

target = "Corrosion_Rate_mm_year"

X = df[features]
y = df[target]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Average error:", mean_absolute_error(y_test, predictions))
print("R2 score:", r2_score(y_test, predictions))

joblib.dump(model, "corrosion_model.joblib")
print("Model saved successfully.")