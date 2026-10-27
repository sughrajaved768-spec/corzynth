import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ================= LOAD DATA =================
df = pd.read_csv("Synthetic-Corrosion-Data-300-Data-Only.csv")
print("Columns found:", list(df.columns))

# ================= FEATURES & TARGET =================
FEATURE_COLS = [
    "pipe_diameter_in",
    "wall_thickness_in",
    "operating_pressure_psig",
    "operating_temperature_f",
    "co2_ppm",
    "h2s_ppm",
    "soil_resistivity_ohm_cm",
    "ph"
]

TARGET_COL = "corrosion_rate_mpy"

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# ================= TRAIN/TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ================= MODEL =================
model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ================= EVALUATE =================
preds = model.predict(X_test)
print("\n--- Model Performance ---")
print("MAE (mpy):", round(mean_absolute_error(y_test, preds), 3))
print("R² Score:", round(r2_score(y_test, preds), 3))

# Feature importance
importance = pd.Series(model.feature_importances_, index=FEATURE_COLS)
print("\n--- Feature Importance ---")
print(importance.sort_values(ascending=False).round(3))

# ================= SAVE MODEL =================
joblib.dump(model, "corrosion_model.joblib")
print("\n✅ Model saved: corrosion_model.joblib")
print("✅ Trained on industry units: inches, psig, °F, mpy")