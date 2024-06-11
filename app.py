import streamlit as st
from streamlit_navigation_bar import st_navbar
import pandas as pd
import numpy as np
import pickle
import joblib
import sklearn
import time
import openpyxl

# required_version = "1.2.2"
# if sklearn.__version__ != required_version:
#     raise ImportError(
#         f"Excpected scikit-learn version {required_version}, but found {sklearn.__version__}."
#     )

route_list = pd.read_excel("data/freshwater_dataset.xlsx", 1)
FEATURES = ["SHIP-ROUTE", "AVG_SPEED", "DISTANCE"]


def load_model(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


try:
    route_encoder = load_model("models/route_encoder.pkl")
    distance_scaler = load_model("models/distance_scaler.pkl")
    speed_scaler = load_model("models/speed_scaler.pkl")
    knn_regressor = joblib.load("models/regressor.pkl")
except AttributeError as e:
    st.error(f"Model loading error")
    st.stop()


# logo_link = "https://www.spil.co.id/img/spil_logo.581f4306.svg"
logo_link = "image/spil_logo.svg"
nav_styles = {
    "nav": {
        "background-color": "#389045",
        # "justify-content": "space-between",
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
    # "show_menu": False,
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
    (route_list),
    index=1,
    placeholder="Masukkan rute perjalanan",
)

avg_velo = st.number_input(
    "Estimasi kecepatan rata-rata kapal (KNOTS):", min_value=0.0, value=9.3
)

if st.button("Make Prediction", type="primary"):
    kode_rute = route_encoder.transform([rute])[0]
    velo_scaler = speed_scaler.transform(np.array([avg_velo]).reshape(1, -1))[0][0]
    dist_find = route_list["DISTANCE"].loc[route_list["ROUTE"] == rute]
    dist_scaler = distance_scaler.transform(np.array([dist_find]).reshape(1, -1))[0][0]
    ship_time = ((dist_find.astype(float)) / avg_velo) / 24
    input_data = pd.DataFrame(
        [
            [
                kode_rute,
                velo_scaler,
                dist_scaler,
            ]
        ],
        columns=FEATURES,
    )
    prediction = knn_regressor.predict(input_data)
    atsea_freshwater = prediction[0]

    # Tes hitung
    # st.dataframe(data=input_data)
    # st.write(atsea_freshwater)

    with st.spinner("Calculating....."):
        time.sleep(3)

    atsea, atport = st.columns(2)
    with atsea:
        container = st.container(border=True)
        container.header("At sea", divider="red")
        container.write("Estimasi Durasi Perjalanan: ")
        container.code(
            f"{ship_time[0]:.1f} Hari ({ship_time[0]*24:.1f} Jam)", language="markdown"
        )
        container.write("Jumlah freshwater yag dibutuhkan")
        container.code(f"{atsea_freshwater} liter", language="markdown")

    with atport:
        container = st.container(border=True)
        container.header("At Port", divider="red")
        bongkar_time = container.number_input(
            "Estimasi durasi bongkar muat: (dalam Jam)", min_value=0.0, value=16.0
        )
        st.warning(
            "Fitur masih dalam tahap pengembangan, data yang disajikan adalah dummy, prediksi belum dapat ditampilkan"
        )
        container.write("Jumlah freshwater yag dibutuhkan")
        container.code("---- liter", language="markdown")

# st.header("Test")
# st.write(rute)
# rute_code = route_encoder.transform([rute])[0]
# st.write(rute_code)
# st.write(avg_velo)
# # velo_data = pd.DataFrame(data=[avg_velo])
# velo_data = speed_scaler.transform(np.array([avg_velo]).reshape(1, -1))[0][0]
# st.write(velo_data)
# data_jarak = route_list["DISTANCE"].loc[route_list["ROUTE"] == rute]
# jarak_data = distance_scaler.transform(np.array([data_jarak]).reshape(1, -1))[0][0]

# st.write(data_jarak)
# st.write(jarak_data)

# ship_time = (data_jarak.astype(float) / avg_velo) / 24
# st.write(ship_time)

# data_demo = pd.DataFrame([[rute_code, velo_data, jarak_data]], columns=FEATURES)
# st.dataframe(data=data_demo)
# prediction = knn_regressor.predict(data_demo)
# st.header(f"{prediction[0]} Liter")
