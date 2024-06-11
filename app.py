import streamlit as st
from streamlit_navigation_bar import st_navbar
import pandas as pd
import numpy as np
import pickle
import joblib
import sklearn
import time
import openpyxl

# Load the route list from the Excel file
route_list = pd.read_excel("data/freshwater_dataset.xlsx", sheet_name=1)
FEATURES = ["SHIP-ROUTE", "AVG_SPEED", "DISTANCE"]


# Function to load model
def load_model(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


# Try to load models and scalers
try:
    route_encoder = load_model("models/route_encoder.pkl")
    distance_scaler = load_model("models/distance_scaler.pkl")
    speed_scaler = load_model("models/speed_scaler.pkl")
    knn_regressor = joblib.load("models/regressor.pkl")
except AttributeError as e:
    st.error("Model loading error")
    st.stop()

# Navbar setup
logo_link = "image/spil_logo.svg"
nav_styles = {
    "nav": {
        "background-color": "#389045",
        "height": "2.875rem",
    },
    "img": {
        "padding-right": "14px",
    },
    "span": {
        "color": "white",
        "padding": "14px",
        "font-weight": "bold",
        "justify-text": "left",
    },
    "active": {
        "background-color": "white",
        "color": "var(--text-color)",
        "font-weight": "normal",
        "padding": "14px",
    },
}

navbar_options = {
    "show_sidebar": False,
}

page = st_navbar(
    ["Salam Pacific Indonesia Lines".upper()],
    logo_path=logo_link,
    styles=nav_styles,
    options=navbar_options,
)

st.title("Prediksi Konsumsi Fresh Water untuk KM ASJ")

rute = st.selectbox(
    "Pilih rute perjalanan kapal:",
    route_list["ROUTE"].unique(),
    index=None,
    placeholder="Masukkan rute perjalanan",
)

avg_velo = st.number_input(
    "Estimasi kecepatan rata-rata kapal (KNOTS):", min_value=0.0, value=9.3
)

if st.button("Make Prediction", type="primary"):
    kode_rute = route_encoder.transform([rute])[0]
    velo_scaler = speed_scaler.transform(np.array([avg_velo]).reshape(1, -1))[0][0]
    dist_find = route_list["DISTANCE"].loc[route_list["ROUTE"] == rute].values[0]
    dist_scaler = distance_scaler.transform(np.array([dist_find]).reshape(1, -1))[0][0]
    ship_time = ((dist_find.astype(float)) / avg_velo) / 24
    input_data = pd.DataFrame(
        [[kode_rute, velo_scaler, dist_scaler]],
        columns=FEATURES,
    )
    prediction = knn_regressor.predict(input_data)
    atsea_freshwater = prediction[0]

    with st.spinner("Calculating....."):
        time.sleep(3)

    atsea, atport = st.columns(2)
    with atsea:
        container = st.container()
        container.header("At sea")
        container.write("Estimasi Durasi Perjalanan: ")
        perkiraan_durasi = f"{ship_time:.1f} Hari ({ship_time*24:.1f} Jam)"
        container.code(perkiraan_durasi, language="markdown")
        container.write("Jumlah freshwater yang dibutuhkan:")
        container.code(f"{atsea_freshwater:.2f} liter", language="markdown")

    with atport:
        container = st.container()
        container.header("At Port")
        bongkar_time = container.number_input(
            "Estimasi durasi bongkar muat: (dalam Jam)", min_value=0.0, value=16.0
        )
        st.warning(
            "Fitur masih dalam tahap pengembangan, data yang disajikan adalah dummy, prediksi belum dapat ditampilkan"
        )
        container.write("Jumlah freshwater yang dibutuhkan:")
        container.code("---- liter", language="markdown")
