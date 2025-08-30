# train_model.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ---------- Step 1: Load data ----------
df_raw = pd.read_csv("sample_aqi.csv")

def parse_dt(s):
    return pd.to_datetime(s, errors="coerce", format="%d-%m-%Y %H:%M:%S")

df_raw["timestamp"] = df_raw["last_update"].apply(parse_dt)

df = df_raw[["state","city","latitude","longitude",
             "pollutant_id","pollutant_avg","timestamp"]].copy()

# ---------- Step 2: Pivot pollutants ----------
pivot = df.pivot_table(
    index=["state","city","latitude","longitude","timestamp"],
    columns="pollutant_id",
    values="pollutant_avg",
    aggfunc="mean"
).reset_index()

# Ensure expected pollutants
for col in ["PM2.5","PM10","NO2","SO2","CO","OZONE","NH3"]:
    if col not in pivot.columns:
        pivot[col] = np.nan

# ---------- Step 3: Add calendar features ----------
pivot["month"] = pivot["timestamp"].dt.month
pivot["hour"] = pivot["timestamp"].dt.hour
pivot["dayofweek"] = pivot["timestamp"].dt.dayofweek

# ---------- Step 4: Define AQI target ----------
def sub_index_pm25(x):
    if x <= 30: return x*50/30
    elif x <= 60: return 50+(x-30)*50/30
    elif x <= 90: return 100+(x-60)*100/30
    elif x <= 120: return 200+(x-90)*100/30
    elif x <= 250: return 300+(x-120)*100/130
    else: return 400+(x-250)*100/130

def sub_index_pm10(x):
    if x <= 50: return x
    elif x <= 100: return 50+(x-50)*50/50
    elif x <= 250: return 100+(x-100)*100/150
    elif x <= 350: return 200+(x-250)*100/100
    elif x <= 430: return 300+(x-350)*100/80
    else: return 400+(x-430)*100/80

pivot["sub_PM2.5"] = pivot["PM2.5"].apply(lambda x: sub_index_pm25(x) if pd.notnull(x) else np.nan)
pivot["sub_PM10"]  = pivot["PM10"].apply(lambda x: sub_index_pm10(x) if pd.notnull(x) else np.nan)

pivot["AQI"] = pivot[["sub_PM2.5","sub_PM10"]].max(axis=1)

# ---------- Step 4.5: Cleaning ----------
# Drop rows where AQI target is missing
pivot = pivot.dropna(subset=["AQI"])

# Fill missing pollutant values with 0
for col in ["PM2.5","PM10","NO2","SO2","CO","OZONE","NH3"]:
    pivot[col] = pivot[col].fillna(0)

# ---------- Step 5: Features ----------
feature_cols_numeric = ["PM2.5","PM10","NO2","SO2","CO","OZONE","NH3",
                        "latitude","longitude","month","hour","dayofweek"]
feature_cols_categorical = ["state","city"]

X = pivot[feature_cols_numeric + feature_cols_categorical]
y = pivot["AQI"]

# ---------- Step 6: Preprocessing + Model ----------
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="mean"), feature_cols_numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_categorical)
])

pipe = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(X_train, y_train)

print("Model trained, R²:", r2_score(y_test, pipe.predict(X_test)))

# ---------- Step 7: Save artifacts ----------
os.makedirs("artifacts", exist_ok=True)
joblib.dump(pipe, "artifacts/aqi_model.pkl")
joblib.dump({
    "feature_cols_numeric": feature_cols_numeric,
    "feature_cols_categorical": feature_cols_categorical
}, "artifacts/meta.joblib")

print("Artifacts saved in /artifacts")
