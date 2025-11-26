import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import matplotlib.pyplot as plt

# ----------------- Load model -----------------
with open('model_xgb.pkl', 'rb') as file:
    model = pickle.load(file)
    

# ----------------- App title / layout -----------------
st.set_page_config(page_title="SPK Prediksi Harga Mobil", layout="wide")
st.title("🚗 Prediksi Harga Mobil + SPK (Weighted Scoring, Risk, Visualisasi)")

col1, col2 = st.columns([2, 1])

print("Model: ", model.feature_names_in_)

with col1:
    st.header("Input Spesifikasi Mobil")
    year = st.number_input("Tahun", min_value=1990, max_value=2025, value=2018)
    engine_hp = st.number_input("Engine HP", min_value=50, max_value=1500, value=250)
    engine_cyl = st.number_input("Engine Cylinders", min_value=2, max_value=16, value=6)
    market_category = st.selectbox("Market Category", ['Luxury', 'Crossover', 'Other', 'Unknown'])
    make = st.selectbox("Make", ['BMW', 'Audi', 'Toyota', 'Other'])
    vehicle_style = st.selectbox("Vehicle Style", ['Sedan', 'SUV', 'Other'])
    is_collector = st.checkbox("Mobil Kolektor / Limited Edition", value=False)
    st.markdown("---")
    actual_market_price = st.number_input("Harga Pasar yang Ditawarkan (opsional, dalam Rupiah)", min_value=0, value=0, step=1000000, format="%d")

with col2:
    st.header("Pengaturan & Info")
    st.write("Tanggal:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    st.caption("Kurs yang digunakan: Rp 16.000 / USD (default). Depresiasi: 10% per tahun (default).")

# ----------------- SPK helper functions -----------------
def safe_build_input_df(year, hp, cyl, market_category, make, vehicle_style, model):
    data = {
        'Year': [year],
        'Engine HP': [hp],
        'Engine Cylinders': [cyl],
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
    # Ensure columns align with model
    try:
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)
    except Exception:
        # fallback: add missing cols, keep order from model if possible
        for col in getattr(model, "feature_names_in_", []):
            if col not in df.columns:
                df[col] = 0
        df = df[[c for c in getattr(model, "feature_names_in_", df.columns)]]
    return df

def decision_engine(pred_price, year, hp, cylinders, market_cat, vehicle_style, is_collector, actual_price=None):
    rules = []
    current_year = datetime.now().year
    age = current_year - year

    # Age rules
    if age > 15:
        rules.append("Usia mobil > 15 tahun → risiko kerusakan & biaya perbaikan tinggi.")
    elif age > 10:
        rules.append("Usia mobil > 10 tahun → depresiasi tinggi & biaya perawatan meningkat.")
    elif age > 5:
        rules.append("Usia mobil > 5 tahun → mulai ada depresiasi sedang.")

    # HP rules
    if hp > 350:
        rules.append("Engine HP sangat tinggi → pajak & konsumsi BBM besar.")
    elif hp > 250:
        rules.append("Engine HP tinggi → konsumsi BBM lebih mahal.")

    # Cylinders rules
    if cylinders > 8:
        rules.append("Mesin banyak silinder → boros & biaya servis lebih mahal.")
    elif cylinders > 6:
        rules.append("Silinder > 6 → biaya perawatan meningkat.")

    # Market category
    if market_cat == "Luxury":
        rules.append("Kategori luxury → biaya servis & sparepart mahal.")

    # Vehicle style
    if vehicle_style == "SUV":
        rules.append("SUV → konsumsi BBM lebih besar dan footprint perawatan tinggi.")

    # Collector
    if is_collector:
        rules.append("Kolektor → nilai cenderung naik, cocok untuk investasi jangka panjang.")

    # Price comparison
    price_note = None
    if actual_price and actual_price > 0:
        if actual_price < pred_price * 0.9:
            rules.append("Harga penawaran jauh lebih murah dari prediksi → peluang beli bagus.")
            price_note = "much_lower"
        elif actual_price <= pred_price * 1.1:
            rules.append("Harga penawaran wajar sesuai pasar.")
            price_note = "fair"
        else:
            rules.append("Harga penawaran lebih mahal dari prediksi → tidak disarankan.")
            price_note = "higher"

    return rules, age, price_note

# ----------------- Weighted scoring (lengkap) -----------------
def compute_weighted_score(age, hp, cylinders, market_cat, vehicle_style, is_collector, price_note):
    # Define raw score functions (higher better)
    # Age: newer => higher score (0..30)
    if age <= 2:
        age_score = 30
    elif age <= 5:
        age_score = 25
    elif age <= 10:
        age_score = 15
    elif age <= 15:
        age_score = 8
    else:
        age_score = 0

    # HP: moderate HP preferred (not too high). Max 15
    if hp <= 150:
        hp_score = 12
    elif hp <= 250:
        hp_score = 15
    elif hp <= 350:
        hp_score = 10
    else:
        hp_score = 4

    # Cylinders: fewer better for economy. Max 10
    if cylinders <= 4:
        cyl_score = 10
    elif cylinders <= 6:
        cyl_score = 8
    elif cylinders <= 8:
        cyl_score = 5
    else:
        cyl_score = 2

    # Market category: Other better for maintenance cost (max 10)
    if market_cat == "Luxury":
        market_score = 2
    elif market_cat == "Crossover":
        market_score = 6
    else:
        market_score = 10

    # Vehicle style: Sedan better economy than SUV (max 5)
    if vehicle_style == "SUV":
        vehicle_score = 2
    else:
        vehicle_score = 5

    # Collector: adds premium if collector (max 10)
    collector_score = 10 if is_collector else 5

    # Price comparison: most important (max 30)
    if price_note == "much_lower":
        price_score = 30
    elif price_note == "fair":
        price_score = 20
    elif price_note == "higher":
        price_score = 0
    else:
        # no market price provided: neutral mid-score
        price_score = 15

    # Raw totals (scale: age 0-30, hp 0-15, cyl 0-10, market 0-10, vehicle 0-5, collector 0-10, price 0-30)
    raw_total = age_score + hp_score + cyl_score + market_score + vehicle_score + collector_score + price_score
    max_total = 30 + 15 + 10 + 10 + 5 + 10 + 30  # 110

    # Normalize to 0-100
    normalized_score = (raw_total / max_total) * 100

    # Create breakdown dict
    breakdown = {
        "Age": (age_score, 30),
        "Engine HP": (hp_score, 15),
        "Cylinders": (cyl_score, 10),
        "Market Category": (market_score, 10),
        "Vehicle Style": (vehicle_score, 5),
        "Collector": (collector_score, 10),
        "Price Comparison": (price_score, 30),
        "Raw Total": (raw_total, max_total),
        "Score (%)": normalized_score
    }

    return normalized_score, breakdown

def risk_assessment(score, age, hp, cylinders, market_cat, is_collector, price_note):
    # Risk level by score thresholds
    if score >= 70:
        level = "Low"
        emoji = "🟢"
    elif score >= 45:
        level = "Medium"
        emoji = "🟡"
    else:
        level = "High"
        emoji = "🔴"

    # Additional risk flags
    flags = []
    if age > 15:
        flags.append("Condition Risk: usia > 15 tahun")
    if hp > 350:
        flags.append("Operational Cost Risk: HP > 350")
    if cylinders > 8:
        flags.append("Maintenance Risk: silinder banyak")
    if market_cat == "Luxury":
        flags.append("Cost Risk: kategori luxury (sparepart & servis mahal)")
    if not is_collector and price_note == "higher":
        flags.append("Financial Risk: penawaran lebih tinggi dari prediksi")

    return level, emoji, flags

# ----------------- Main prediction + UI -----------------
if st.button("Prediksi & Jalankan SPK"):
    # Build model input
    X = safe_build_input_df(year, engine_hp, engine_cyl, market_category, make, vehicle_style, model)

    # Predict (log-scale model)
    try:
        pred_log = model.predict(X)[0]
    except Exception as e:
        st.error(f"Model predict error: {e}")
        raise

    pred_price_usd = np.expm1(pred_log)  # asumsi model dilatih pada log(MSRP)
    kurs = 16000
    pred_rupiah = pred_price_usd * kurs

    # Depresiasi
    
    current_year = datetime.now().year
    umur_mobil = max(current_year - year, 0)
    depreciation_rate = 0.10
    harga_bekas = pred_rupiah * ((1 - depreciation_rate) ** umur_mobil)

    # Collector logic
    if is_collector:
        final_price = pred_rupiah * 1.10
        keterangan = "Mobil kolektor → nilai naik 10% dari MSRP"
    else:
        final_price = harga_bekas
        keterangan = f"Mobil normal → harga setelah depresiasi {umur_mobil} tahun"

    # Format
    formatted_msrp = f"Rp {pred_rupiah:,.0f}".replace(",", ".")
    formatted_final = f"Rp {final_price:,.0f}".replace(",", ".")

    # Decision engine (rules)
    rules, age, price_note = decision_engine(
        pred_price=final_price,
        year=year,
        hp=engine_hp,
        cylinders=engine_cyl,
        market_cat=market_category,
        vehicle_style=vehicle_style,
        is_collector=is_collector,
        actual_price=actual_market_price if actual_market_price > 0 else None
    )

    # Weighted scoring
    score, breakdown = compute_weighted_score(age=age, hp=engine_hp, cylinders=engine_cyl,
                                             market_cat=market_category, vehicle_style=vehicle_style,
                                             is_collector=is_collector, price_note=price_note)

    # Risk assessment
    risk_level, risk_emoji, risk_flags = risk_assessment(score, age, engine_hp, engine_cyl, market_category, is_collector, price_note)

    # Decide recommendation text from score thresholds (alternative to price-only rule)
    if score >= 75:
        reco_text = "Rekomendasi BELI"
    elif score >= 50:
        reco_text = "Boleh Dipertimbangkan"
    else:
        reco_text = "TIDAK Direkomendasikan"

    # ----------------- DISPLAY (Top metrics) -----------------
    st.markdown("## Hasil Prediksi & SPK")
    m1, m2, m3 = st.columns(3)
    m1.metric("Perkiraan MSRP (Rp)", formatted_msrp)
    m2.metric("Perkiraan Harga Akhir (Rp)", formatted_final)
    m3.metric("Skor SPK", f"{score:.1f} / 100", delta=None)

    st.markdown(f"**Keputusan Rekomendasi (gabungan):** `{reco_text}` {risk_emoji}  **Risk Level:** {risk_level}")
    st.caption(keterangan)

    # ----------------- Breakdown (expandable) -----------------
    with st.expander("🔎 Detail Analisa & Rule-based Notes"):
        st.write("**Rules inference:**")
        for r in rules:
            st.write("- ", r)

        st.write("**Breakdown skor faktor (raw / max):**")
        br_df = pd.DataFrame({
            "Factor": ["Age", "Engine HP", "Cylinders", "Market Category", "Vehicle Style", "Collector", "Price Comparison"],
            "Score": [breakdown["Age"][0], breakdown["Engine HP"][0], breakdown["Cylinders"][0], breakdown["Market Category"][0], breakdown["Vehicle Style"][0], breakdown["Collector"][0], breakdown["Price Comparison"][0]],
            "Max": [breakdown["Age"][1], breakdown["Engine HP"][1], breakdown["Cylinders"][1], breakdown["Market Category"][1], breakdown["Vehicle Style"][1], breakdown["Collector"][1], breakdown["Price Comparison"][1]]
        })
        st.table(br_df.style.format({"Score": "{:.0f}", "Max": "{:.0f}"}))

        st.write(f"**Total (raw):** {breakdown['Raw Total'][0]:.1f} / {breakdown['Raw Total'][1]}")
        st.write(f"**Skor Normalisasi:** {score:.2f} / 100")

        st.write("**Risk Flags:**")
        if risk_flags:
            for f in risk_flags:
                st.write("- ", f)
        else:
            st.write("- Tidak ada flag risiko signifikan.")

    # ----------------- Visualisasi (C) dua-duanya -----------------
    st.markdown("## Visualisasi")

    # (1) Depresiasi plot
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    years = np.arange(0, max(umur_mobil, 5) + 1)
    price_over_time = pred_rupiah * ((1 - depreciation_rate) ** years)
    ax1.plot(years, price_over_time)
    ax1.set_xlabel("Tahun sejak pembelian")
    ax1.set_ylabel("Perkiraan Harga (Rp)")
    ax1.set_title("Simulasi Depresiasi Harga (asumsi 10%/tahun)")
    ax1.grid(True)
    st.pyplot(fig1)

    # (2) Prediksi vs Harga Pasar
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    # Plot predicted final price
    ax2.scatter([0], [final_price], label="Prediksi Harga Akhir (model)", marker='o')
    if actual_market_price > 0:
        ax2.scatter([1], [actual_market_price], label="Harga Pasar (input)", marker='s')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Prediksi", "Harga Pasar"])
    ax2.set_ylabel("Harga (Rp)")
    ax2.set_title("Perbandingan Prediksi vs Harga Pasar")
    ax2.grid(axis='y')
    ax2.legend()
    st.pyplot(fig2)



    st.success("Analisa SPK selesai. Gunakan hasil sebagai referensi — pertimbangkan inspeksi fisik & dokumen saat membeli!")

# ----------------- Footer -----------------
st.markdown("---")
st.caption("Catatan: SPK ini membantu pengambilan keputusan berdasarkan model & aturan. Keputusan akhir tetap memerlukan verifikasi lapangan.")
