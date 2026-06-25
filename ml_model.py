"""
EHR Workup Time Predictor — Training Script (XGBoost)
======================================================
Dataset: ehr_age7_ar1.xlsx
- Has header row
- Arrival hour column: 1-hour slots (08:00-09:00 … 16:00-17:00)
- Age category: 7 bins (0-12 … 76-90)
- TOTAL_WORKUP_TIME: datetime.time objects
Output: workup_model_age7_ar1.pkl
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

# ══════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════
print("\n[1/7] Loading data...")
df = pd.read_excel("ehr_age7_ar1.xlsx", header=0)
print(f"     Loaded {len(df):,} rows")

# ══════════════════════════════════════════════════════════════════
# 2. CLEAN
# ══════════════════════════════════════════════════════════════════
print("[2/7] Cleaning...")

df["Gender"]            = df["Gender"].replace({"Fema": "Female", "TBU": "Unknown"})
df["Patient Flow type"] = df["Patient Flow type"].str.strip()
df["Specialty"]         = df["Specialty"].str.strip()

df["TARGET"] = df["TOTAL_WORKUP_TIME"].apply(
    lambda x: x.hour * 60 + x.minute if hasattr(x, "hour") else np.nan
)
df = df[(df["TARGET"] >= 5) & (df["TARGET"] <= 300)].copy()
print(f"     {len(df):,} rows after outlier removal (5–300 min)")

# ══════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING  (intake-only)
# ══════════════════════════════════════════════════════════════════
print("[3/7] Engineering features...")

# Arrival hour: 1-hour slots → numeric start hour
arrival_map = {
    "08:00 - 09:00": 8.0,
    "09:00 - 10:00": 9.0,
    "10:00 - 11:00": 10.0,
    "11:00 - 12:00": 11.0,
    "12:00 - 13:00": 12.0,
    "13:00 - 14:00": 13.0,
    "14:00 - 15:00": 14.0,
    "15:00 - 16:00": 15.0,
    "16:00 - 17:00": 16.0,
}
df["Arrival_Hour_Num"] = df["Arrival hour"].map(arrival_map)

# Calendar
df["Month"]      = df["VISITDATE"].dt.month
df["Quarter"]    = df["VISITDATE"].dt.quarter
df["Is_Weekend"] = df["Day"].isin(["Sat", "Sun"]).astype(int)

# Specialty volume
df["Specialty_Volume"] = df["Specialty"].map(df["Specialty"].value_counts())

# Interactions
df["Flow_Session"]      = df["Patient Flow type"] + "_" + df["Session"]
df["Specialty_Flow"]    = df["Specialty"]         + "_" + df["Patient Flow type"]
df["Session_Desig"]     = df["Session"]           + "_" + df["Consultant Designation"]
df["Specialty_Session"] = df["Specialty"]         + "_" + df["Session"]
df["Flow_Desig"]        = df["Patient Flow type"] + "_" + df["Consultant Designation"]

# ══════════════════════════════════════════════════════════════════
# 4. FEATURE LIST
# ══════════════════════════════════════════════════════════════════
BASE_FEATURES = [
    "Age category", "Gender",
    "Patient visit category", "Patient Flow type", "Specialty",
    "Consultant Name", "Consultant Designation",
    "Day", "Session",
    "Arrival_Hour_Num", "Is_Weekend",
    "Month", "Quarter",
    "Specialty_Volume",
    "Flow_Session", "Specialty_Flow", "Session_Desig",
    "Specialty_Session", "Flow_Desig",
]

X     = df[BASE_FEATURES].copy()
y     = df["TARGET"].copy()
y_log = np.log1p(y)

X = X.fillna(X.mean(numeric_only=True))
for col in X.select_dtypes(include="object").columns:
    X[col] = X[col].fillna(X[col].mode()[0])

# ══════════════════════════════════════════════════════════════════
# 5. SPLIT
# ══════════════════════════════════════════════════════════════════
print("[4/7] Splitting data...")
X_train, X_test, y_train_log, y_test_log, y_train, y_test = train_test_split(
    X, y_log, y, test_size=0.2, random_state=42
)
print(f"     Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ══════════════════════════════════════════════════════════════════
# 6. TARGET ENCODING  (train only — no leakage)
# ══════════════════════════════════════════════════════════════════
global_mean = y_train.mean()

def target_encode(train_col, test_col, train_y, gm):
    enc = pd.Series(train_y.values, index=train_col.index).groupby(train_col.values).mean()
    return train_col.map(enc).fillna(gm), test_col.map(enc).fillna(gm), enc.to_dict()

X_train = X_train.copy()
X_test  = X_test.copy()

X_train["Consultant_AvgTime"], X_test["Consultant_AvgTime"], consultant_mean = \
    target_encode(X_train["Consultant Name"], X_test["Consultant Name"], y_train, global_mean)

X_train["Specialty_AvgTime"], X_test["Specialty_AvgTime"], spec_mean = \
    target_encode(X_train["Specialty"], X_test["Specialty"], y_train, global_mean)

X_train["Flow_AvgTime"], X_test["Flow_AvgTime"], flow_mean = \
    target_encode(X_train["Patient Flow type"], X_test["Patient Flow type"], y_train, global_mean)

ENCODING_MAPS = {
    "consultant_mean": consultant_mean,
    "spec_mean":       spec_mean,
    "flow_mean":       flow_mean,
    "global_mean":     global_mean,
    "arrival_map":     arrival_map,
    "spec_volume":     df.set_index("Specialty")["Specialty_Volume"].to_dict(),
}

X_train = pd.get_dummies(X_train, drop_first=True)
X_test  = pd.get_dummies(X_test,  drop_first=True)
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

FEATURE_COLUMNS = X_train.columns.tolist()
print(f"     Final feature count: {len(FEATURE_COLUMNS)}")

# ══════════════════════════════════════════════════════════════════
# 7. TRAIN
# ══════════════════════════════════════════════════════════════════
print("[5/7] Training XGBoost... (15–30 min)")

model = XGBRegressor(
    n_estimators          = 4000,
    max_depth             = 5,
    learning_rate         = 0.008,
    subsample             = 0.85,
    colsample_bytree      = 0.75,
    colsample_bylevel     = 0.75,
    colsample_bynode      = 0.8,
    min_child_weight      = 8,
    gamma                 = 0.15,
    reg_alpha             = 0.1,
    reg_lambda            = 1.5,
    early_stopping_rounds = 150,
    eval_metric           = "mae",
    random_state          = 42,
    n_jobs                = -1,
)

model.fit(
    X_train, y_train_log,
    eval_set=[(X_test, y_test_log)],
    verbose=200,
)

# ══════════════════════════════════════════════════════════════════
# 8. EVALUATE
# ══════════════════════════════════════════════════════════════════
y_pred = np.expm1(model.predict(X_test))

print("\n[6/7] Evaluation")
print(f"     MAE:      {mean_absolute_error(y_test, y_pred):.2f} minutes")
print(f"     R² Score: {r2_score(y_test, y_pred):.4f}")

for bucket, mask in zip(
    ["0-30", "30-60", "60-90", "90-120", "120+"],
    [y_test <= 30,
     (y_test > 30)  & (y_test <= 60),
     (y_test > 60)  & (y_test <= 90),
     (y_test > 90)  & (y_test <= 120),
     y_test > 120]
):
    if mask.sum() > 0:
        print(f"     R² [{bucket} min]: {r2_score(y_test[mask], y_pred[mask]):.3f}  (n={mask.sum():,})")

# ══════════════════════════════════════════════════════════════════
# 9. SAVE
# ══════════════════════════════════════════════════════════════════
print("\n[7/7] Saving...")
with open("workup_model_age7_ar1.pkl", "wb") as f:
    pickle.dump({
        "model":           model,
        "feature_columns": FEATURE_COLUMNS,
        "encoding_maps":   ENCODING_MAPS,
    }, f)

print("     ✓ Saved workup_model_age7_ar1.pkl")
print("\nDone! Run predict_app_age7_ar1.py to predict.")