# CORZYNTH - TRAIN MODEL v2     run:  python train_model.py
# Honest evaluation: model chosen by cross-validation on the TRAIN split only, test set scored once,
# preprocessing saved inside the model, physics baseline reported, real-data files loaded and checked.
import json, os, sys, platform
import numpy as np, pandas as pd, joblib, sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler
from corzynth_features import (TARGET, RAW_COLS, CATEGORICAL, NUMERIC_ALL, add_features, physics_rate)

SEED = 42
GROUP_CANDIDATES = ["Pipeline_ID", "Segment_ID", "Pipeline", "Segment", "Line_ID"]

def need(path):
    if not os.path.exists(path):
        sys.exit(f"ERROR: required file not found: {path}")
    return pd.read_csv(path)

def optional(path):
    if os.path.exists(path):
        d = pd.read_csv(path); print(f"OK   {path}: {d.shape}"); return d
    print(f"WARN {path} not found, skipped"); return None

print("Loading data files...")
df = need("Corzynth_Corrosion_Data_5000.csv"); print(f"OK   corrosion data: {df.shape}")
geo = optional("Corzynth_Geophysical_Data_5000.csv")
chem = optional("Corzynth_Chemical_Training_Database_1700.csv")
integ = optional("Corzynth_Field_Integrity_Training_Data.csv")
inj = optional("Corzynth_Injection_Points.csv")            # REAL: inhibitor 11392 dosing table
kpi = optional("Corzynth_Integrity_KPI_Aug2026.csv")       # REAL: Aug 2026 integrity KPI report
calib = optional("Corzynth_Field_Calibration.csv")         # optional: real measured rates (same columns as training)

missing = [c for c in RAW_COLS + [TARGET] if c not in df.columns]
if missing:
    sys.exit(f"ERROR: corrosion data is missing columns: {missing}")

# ---- clean target ----
n0 = len(df)
df = df.drop_duplicates()
df = df[pd.to_numeric(df[TARGET], errors="coerce").notna()].copy()
df[TARGET] = df[TARGET].astype(float)
df = df[df[TARGET] >= 0].copy()
df[TARGET] = df[TARGET].clip(lower=1e-4)           # log-safe
print(f"Rows after cleaning: {len(df)} (dropped {n0 - len(df)})")

X, y = df[RAW_COLS], df[TARGET]
group_col = next((g for g in GROUP_CANDIDATES if g in df.columns), None)
groups = df[group_col] if group_col else None
if group_col:
    print(f"Group split by '{group_col}' (no pipeline appears in both train and test)")
    tr, te = next(GroupShuffleSplit(test_size=0.2, random_state=SEED).split(X, y, groups))
    cv = GroupKFold(n_splits=5)
else:
    print("WARN no pipeline/segment ID column: using a random split, so scores may be optimistic for neighbouring segments")
    tr, te = next(ShuffleSplit(test_size=0.2, random_state=SEED).split(X, y))
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
gtr = groups.iloc[tr] if group_col else None

def make(model):
    pre = ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), CATEGORICAL),
        ("num", SimpleImputer(strategy="median"), NUMERIC_ALL)])
    inner = Pipeline([("feat", FunctionTransformer(add_features)), ("pre", pre), ("model", model)])
    return TransformedTargetRegressor(regressor=inner, func=np.log, inverse_func=np.exp)

candidates = {
    "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
    "Extra Trees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
    "Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=6, random_state=SEED),
}
print("\nCross-validation on TRAIN split only (R2 on mm/yr scale)...")
cvres = {}
for name, m in candidates.items():
    sc = cross_val_score(make(m), Xtr, ytr, cv=cv, groups=gtr, scoring="r2", n_jobs=1)
    cvres[name] = (sc.mean(), sc.std()); print(f"  {name}: {sc.mean():.3f} +/- {sc.std():.3f}")
best = max(cvres, key=lambda k: cvres[k][0])
print(f"Selected by CV: {best}")

final = make(candidates[best]).fit(Xtr, ytr)
pred = final.predict(Xte)
test_r2, test_mae = r2_score(yte, pred), mean_absolute_error(yte, pred)
phys = physics_rate(Xte)
phys_r2 = r2_score(yte, phys)
phys_corr = float(np.corrcoef(yte, phys)[0, 1])
print(f"\nTEST (scored once): R2 = {test_r2:.3f} | MAE = {test_mae:.4f} mm/yr")
print(f"Physics-only baseline on same test rows: R2 = {phys_r2:.3f} | correlation = {phys_corr:.3f}")
print("Note: the physics baseline is the approximate de Waard-Milliams chain (no fluid factor).")

calib_res = None
if calib is not None and all(c in calib.columns for c in RAW_COLS + [TARGET]):
    cp = final.predict(calib[RAW_COLS])
    calib_res = dict(n=len(calib), r2=float(r2_score(calib[TARGET], cp)), mae=float(mean_absolute_error(calib[TARGET], cp)))
    print(f"REAL-DATA CALIBRATION CHECK: n={calib_res['n']} R2={calib_res['r2']:.3f} MAE={calib_res['mae']:.4f}")

print("\nPermutation importance on test set...")
pi = permutation_importance(final, Xte, yte, n_repeats=5, random_state=SEED, n_jobs=-1, scoring="r2")
imp = pd.Series(pi.importances_mean, index=RAW_COLS).sort_values(ascending=False)
print(imp.head(10).round(4))

# ---- refit on ALL cleaned data for deployment ----
deploy = make(candidates[best]).fit(X, y)
engineered = add_features(X)
scaler = StandardScaler().fit(engineered[NUMERIC_ALL].fillna(engineered[NUMERIC_ALL].median()))

joblib.dump(deploy, "corrosion_model.pkl")          # full pipeline: features + encoders + imputers + model + log transform
joblib.dump(RAW_COLS, "feature_cols.pkl")           # RAW input columns the pipeline expects
joblib.dump(scaler, "scaler.pkl")                   # kept for compatibility; trees do not need scaling
open("model_r2.txt", "w").write(f"{test_r2:.4f}")   # honest held-out TEST score
pd.concat([engineered, y], axis=1).to_csv("Corzynth_Unified_Training_Data_5000.csv", index=False)
imp.round(5).to_csv("Corzynth_Feature_Importance.csv", index_label="Feature")

# ---- real-data files: validated and summarised (they have no corrosion-rate labels, so they are NOT fitted) ----
real = {}
if inj is not None:
    frac = inj.Blend_Inhibitor_Parts / (inj.Blend_Inhibitor_Parts + inj.Blend_Diesel_Parts)
    vol = inj.Recommended_Dose * inj.Gas_MMscfd * frac * 0.946353     # assumes US quarts per MMscf
    real["injection"] = dict(points=len(inj), neat_inhibitor_L_per_day=round(float(vol.sum()), 1),
                             missing_gas_rate=int(inj.Gas_MMscfd.isna().sum()),
                             dose_changed=int((inj.Recommended_Dose != inj.Incumbent_Dose).sum()))
if kpi is not None:
    pc = kpi[kpi.Type == "percent"]
    real["kpi"] = dict(percent_kpis=len(pc), below_80=int((pc.Overall < 80).sum()),
                       worst=pc.nsmallest(5, "Overall")[["KPI", "Metric", "Overall"]].to_dict("records"))
real["dataset_context"] = dict(mean_residual_to_dose_ratio=float((X.Inhibitor_Residual_ppm / (X.Inhibitor_Dose_ppm + 1)).mean()))
for name, d in [("geophysical", geo), ("chemical", chem), ("integrity", integ)]:
    if d is not None:
        d.to_csv(f"Corzynth_{name.capitalize()}_Data_Active.csv", index=False)
        real[name] = dict(rows=len(d), columns=len(d.columns))
json.dump(real, open("real_data_summary.json", "w"), indent=2)

json.dump(dict(selected=best, cv={k: dict(mean=v[0], std=v[1]) for k, v in cvres.items()},
               test_r2=test_r2, test_mae_mm_yr=test_mae, physics_baseline_r2=phys_r2,
               physics_baseline_corr=phys_corr, rows=len(df), grouped_split=bool(group_col),
               calibration=calib_res, sklearn=sklearn.__version__, python=platform.python_version()),
          open("model_card.json", "w"), indent=2)
print(f"\nDONE  model={best}  test R2={test_r2:.3f}  MAE={test_mae:.4f}  (see model_card.json, real_data_summary.json)")
