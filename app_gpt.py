import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import sklearn

# Pastikan versi scikit-learn yang benar digunakan
required_version = "0.24.2"  # ganti dengan versi yang sesuai
if sklearn.__version__ != required_version:
    raise ImportError(
        f"Expected scikit-learn version {required_version}, but found {sklearn.__version__}."
    )

# Load data
route_list = pd.read_excel("data/freshwater_dataset.xlsx", sheet_name=1)
FEATURES = ["SHIP-ROUTE", "AVG_SPEED", "DISTANCE"]


# Load models and encoders
def load_model(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


try:
    route_encoder = load_model("models/route_encoder.pkl")
    distance_scaler = load_model("models/distance_scaler.pkl")
    speed_scaler = load_model("models/speed_scaler.pkl")
    predictor = joblib.load("models/regressor.pkl")
except AttributeError as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# Logo and Navigation
logo_link = "image/spil_logo.svg"
st.sidebar.image(logo_link, width=100)
st.sidebar.title("Salam Pacific Indonesia Lines")

st.title("Prediksi Konsumsi Fresh Water untuk KM ASJ")

# Form inputs
rute = st.selectbox("Pilih rute perjalanan kapal:", route_list["ROUTE"].unique())

avg_velo = st.number_input(
    "Estimasi kecepatan rata-rata kapal (KNOTS):", min_value=0.0, value=9.3
)

# Prediction logic
if st.button("Prediksi"):
    try:
        rute_code = route_encoder.transform([rute])[0]
        velo_data = speed_scaler.transform(np.array([avg_velo]).reshape(1, -1))[0][0]
        data_jarak = route_list["DISTANCE"].loc[route_list["ROUTE"] == rute].values[0]
        jarak_data = distance_scaler.transform(np.array([data_jarak]).reshape(1, -1))[
            0
        ][0]

        ship_time = (data_jarak.astype(float) / avg_velo) / 24

        data_demo = pd.DataFrame([[rute_code, velo_data, jarak_data]], columns=FEATURES)

        # Predict freshwater consumption
        freshwater_consumption = predictor.predict(data_demo)[0]

        st.subheader("Hasil Prediksi")
        st.write(f"Rute: {rute}")
        st.write(f"Kecepatan Rata-rata: {avg_velo} KNOTS")
        st.write(f"Jarak Perjalanan: {data_jarak} Nautical Miles")
        st.write(f"Estimasi Durasi Perjalanan: {ship_time:.2f} Hari")
        st.write(
            f"Jumlah Freshwater yang dibutuhkan: {freshwater_consumption:.2f} Liter"
        )

        atsea, atport = st.columns(2)
        with atsea:
            st.header("At Sea")
            st.write(f"Estimasi Durasi Perjalanan: {ship_time:.2f} Hari")
            st.write(
                f"Jumlah Freshwater yang dibutuhkan: {freshwater_consumption:.2f} Liter"
            )

        with atport:
            st.header("At Port")
            st.write(f"Estimasi Durasi Perjalanan: {ship_time:.2f} Hari")
            st.write(
                f"Jumlah Freshwater yang dibutuhkan: {freshwater_consumption:.2f} Liter"
            )

    except Exception as e:
        st.error(f"Prediction error: {e}")
