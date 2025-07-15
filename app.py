import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Load model
with open('model_xgb.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Prediksi Harga Mobil 🚗💰 (dengan Depresiasi)")

year = st.number_input("Tahun", min_value=1990, max_value=2025, value=2018)
engine_hp = st.number_input("Engine HP", min_value=50, max_value=1500, value=250)
engine_cyl = st.number_input("Engine Cylinders", min_value=2, max_value=16, value=6)
market_category = st.selectbox("Market Category", ['Luxury', 'Crossover', 'Other', 'Unknown'])
make = st.selectbox("Make", ['BMW', 'Audi', 'Toyota', 'Other'])
vehicle_style = st.selectbox("Vehicle Style", ['Sedan', 'SUV', 'Other'])
is_collector = st.checkbox("Mobil Kolektor / Limited Edition", value=False)

if st.button("Prediksi Harga"):
    data = {
        'Year': [year],
        'Engine HP': [engine_hp],
        'Engine Cylinders': [engine_cyl],
        'Market Category_Luxury': [1 if market_category == 'Luxury' else 0],
        'Market Category_Other': [1 if market_category == 'Other' else 0],
        'Market Category_Unknown': [1 if market_category == 'Unknown' else 0],
        'Make_BMW': [1 if make == 'BMW' else 0],
        'Make_Other': [1 if make == 'Other' else 0],
        'Make_Toyota': [1 if make == 'Toyota' else 0],
        'Vehicle Style_SUV': [1 if vehicle_style == 'SUV' else 0],
        'Vehicle Style_Other': [1 if vehicle_style == 'Other' else 0],
    }
    df = pd.DataFrame(data)

    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0

    df = df[model.feature_names_in_]

    pred_log = model.predict(df)[0]
    pred_price = np.expm1(pred_log)

    # Konversi ke Rupiah
    kurs = 16000
    pred_rupiah = pred_price * kurs

    # Simulasi depresiasi
    current_year = datetime.now().year
    umur_mobil = max(current_year - year, 0)
    depreciation_rate = 0.10  # 10% per tahun
    harga_bekas = pred_rupiah * ((1 - depreciation_rate) ** umur_mobil)

    # Final price logic
    if is_collector:
        final_price = pred_rupiah * 1.10  # kolektor naik 10% dari MSRP
        keterangan = "Mobil kolektor → harga naik 10% dari MSRP"
    else:
        final_price = harga_bekas
        keterangan = f"Mobil normal → harga setelah depresiasi {umur_mobil} tahun"

    formatted_msrp = f"Rp {pred_rupiah:,.0f}".replace(",", ".")
    formatted_final = f"Rp {final_price:,.0f}".replace(",", ".")

    st.write(f"💰 **Perkiraan Harga Baru (MSRP):** {formatted_msrp}")
    st.success(f"💸 **Perkiraan Harga Akhir:** {formatted_final}")
    st.caption(keterangan)
