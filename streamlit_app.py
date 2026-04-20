import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/"

# =========================
# VALIDASI FILE
# =========================
required_files = [
    "autoencoder_soil.keras",
    "encoder_soil.keras",
    "isolation_forest_soil.pkl",
    "scaler_soil.pkl",
    "insight_soil.pkl"
]

missing = [
    f for f in required_files
    if not os.path.exists(MODEL_PATH + f)
]

if missing:
    st.error(f"File tidak ditemukan: {missing}")
    st.stop()

# =========================
# LOAD MODEL
# =========================
autoencoder = tf.keras.models.load_model(
    MODEL_PATH + "autoencoder_soil.keras",
    compile=False
)

encoder = tf.keras.models.load_model(
    MODEL_PATH + "encoder_soil.keras",
    compile=False
)

iso = joblib.load(
    MODEL_PATH + "isolation_forest_soil.pkl"
)

scaler = joblib.load(
    MODEL_PATH + "scaler_soil.pkl"
)

insight_data = joblib.load(
    MODEL_PATH + "insight_soil.pkl"
)

feature_cols = insight_data["feature_cols"]
mean_normal = insight_data["mean_normal"]
std_normal = insight_data["std_normal"]

# =========================
# INSIGHT ENGINE
# =========================
def generate_status(row):

    values = row[feature_cols].values.astype(float)
    zero_count = np.sum(values == 0)

    if zero_count == len(feature_cols):
        return "Power supply inactive or monitoring device offline"

    elif zero_count > 0:

        zero_params = []

        for col in feature_cols:
            if row[col] == 0:
                zero_params.append(col)

        return f"Sensor reading failure detected on parameter: {', '.join(zero_params)}"

    elif row["anomaly_flag"] == 0:
        return "Normal operating condition"

    else:

        z_scores = {}

        for col in feature_cols:
            z_scores[col] = (
                row[col] - mean_normal[col]
            ) / (std_normal[col] + 1e-9)

        main_cause = max(
            z_scores,
            key=lambda x: abs(z_scores[x])
        )

        direction = "high" if z_scores[main_cause] >= 0 else "low"

        mapping = {
            'hu': {
                'high': 'Abnormally high soil moisture detected',
                'low':  'Low soil moisture detected'
            },
            'ta': {
                'high': 'High soil temperature detected',
                'low':  'Low soil temperature detected'
            },
            'ec': {
                'high': 'High electrical conductivity detected',
                'low':  'Low electrical conductivity detected'
            },
            'ph': {
                'high': 'Soil pH excessively alkaline',
                'low':  'Soil pH excessively acidic'
            },
            'n': {
                'high': 'Nitrogen level abnormally high',
                'low':  'Nitrogen deficiency detected'
            },
            'p': {
                'high': 'Phosphorus level abnormally high',
                'low':  'Phosphorus deficiency detected'
            },
            'k': {
                'high': 'Potassium level abnormally high',
                'low':  'Potassium deficiency detected'
            }
        }

        return mapping[main_cause][direction]

# =========================
# UI
# =========================
st.set_page_config(
    page_title="Soil Anomaly Detector",
    page_icon="🌱",
    layout="centered"
)

st.title("Soil Sensor Anomaly Detection")
st.write("Metode: Autoencoder + Isolation Forest")

col1, col2, col3 = st.columns(3)

hu = col1.number_input("Humidity (hu)", value=33.5)
ta = col2.number_input("Temperature (ta)", value=25.6)
ec = col3.number_input("EC", value=650.0)

ph = col1.number_input("pH", value=5.0)
n  = col2.number_input("Nitrogen (N)", value=108.0)
p  = col3.number_input("Phosphorus (P)", value=295.0)

k = st.number_input("Potassium (K)", value=288.0)

# =========================
# DETEKSI
# =========================
if st.button("Deteksi"):

    X = np.array([[hu, ta, ec, ph, n, p, k]])

    # scaling
    X_scaled = scaler.transform(X)

    # reconstruction
    X_pred = autoencoder.predict(
        X_scaled,
        verbose=0
    )

    mse = np.mean(
        np.square(X_scaled - X_pred),
        axis=1
    )[0]

    mae = np.mean(
        np.abs(X_scaled - X_pred),
        axis=1
    )[0]

    # latent
    latent = encoder.predict(
        X_scaled,
        verbose=0
    )

    # isolation forest
    pred_label = iso.predict(latent)[0]
    anomaly_score = -iso.decision_function(latent)[0]

    anomaly_flag = 1 if pred_label == -1 else 0

    # insight
    temp_row = pd.Series(
        [hu, ta, ec, ph, n, p, k],
        index=feature_cols
    )

    temp_row["anomaly_flag"] = anomaly_flag

    insight = generate_status(temp_row)

    # =====================
    # OUTPUT
    # =====================
    st.subheader("Hasil Analisis")

    st.write(f"Reconstruction MSE : {mse:.6f}")
    st.write(f"Reconstruction MAE : {mae:.6f}")
    st.write(f"Anomaly Score      : {anomaly_score:.6f}")

    if anomaly_flag == 1:
        st.error("ANOMALI TERDETEKSI")
    else:
        st.success("DATA NORMAL")

    st.info(f"Insight: {insight}")