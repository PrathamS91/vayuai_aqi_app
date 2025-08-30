# streamlit_app.py
import streamlit as st
import joblib
import pandas as pd
import datetime

# Load model + meta
pipe = joblib.load("artifacts/aqi_model.pkl")
meta = joblib.load("artifacts/meta.joblib")

feature_cols_numeric = meta["feature_cols_numeric"]
feature_cols_categorical = meta["feature_cols_categorical"]

# Config
st.set_page_config(page_title="VayuAI AQI Predictor", layout="wide")
st.title("🌍 VayuAI – Air Quality Prediction")

# Inputs
city = st.text_input("Enter City Name")
date = st.date_input("Select Date", datetime.date.today())

st.subheader("Enter pollutant values (µg/m³):")
pollutants = {}
for col in ["PM2.5","PM10","NO2","SO2","CO","OZONE","NH3"]:
    pollutants[col] = st.number_input(f"{col}", min_value=0.0, step=1.0)

# Predict
if st.button("Predict AQI"):
    df_input = pd.DataFrame([pollutants])
    df_input["latitude"] = 0.0
    df_input["longitude"] = 0.0
    df_input["month"] = date.month
    df_input["hour"] = 12
    df_input["dayofweek"] = date.weekday()
    df_input["state"] = "Unknown"
    df_input["city"] = city

    X = df_input[feature_cols_numeric + feature_cols_categorical]
    pred = pipe.predict(X)[0]

    # Category
    def cat(aqi):
        if aqi <= 25: return "Healthy"
        elif aqi <= 50: return "Good"
        elif aqi <= 100: return "Moderate"
        elif aqi <= 200: return "Poor"
        return "Worst"
    category = cat(pred)

    st.success(f"Predicted AQI: {pred:.1f} → {category}")

    advisory = {
        "Healthy":"Air is clean.",
        "Good":"Air quality is acceptable.",
        "Moderate":"Sensitive groups should limit outdoor activity.",
        "Poor":"Avoid outdoor exertion.",
        "Worst":"Health alert: everyone may be affected."
    }
    st.info(advisory[category])
