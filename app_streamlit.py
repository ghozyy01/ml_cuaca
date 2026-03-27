import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL
# =========================
model = joblib.load("model_cuaca.joblib")
encoders = joblib.load("encoders.joblib")

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="AI Prediksi Cuaca",
    page_icon="🌦️",
    layout="centered"
)

# =========================
# CUSTOM CSS (BIAR KEREN)
# =========================
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #74ebd5, #ACB6E5);
}
.main {
    background-color: #ffffffcc;
    padding: 20px;
    border-radius: 15px;
}
.stButton>button {
    background: linear-gradient(to right, #36D1DC, #5B86E5);
    color: white;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
}
.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
    <h1 style='text-align: center;'>🌦️ AI Prediksi Cuaca</h1>
    <p style='text-align: center;'>Masukkan kondisi untuk melihat prediksi cuaca</p>
""", unsafe_allow_html=True)

st.divider()

# =========================
# INPUT FORM
# =========================
st.subheader("📥 Input Kondisi Cuaca")

input_data = {}

col1, col2 = st.columns(2)

for i, col in enumerate(model.feature_names_in_):
    with (col1 if i % 2 == 0 else col2):
        if col in encoders:
            options = list(encoders[col].classes_)
            input_data[col] = st.selectbox(f"{col}", options)
        else:
            input_data[col] = st.number_input(f"{col}", value=0)

st.divider()

# =========================
# BUTTON
# =========================
if st.button("🔮 Prediksi Sekarang"):
    
    df_input = pd.DataFrame([input_data])

    # Encoding
    for col in df_input.columns:
        if col in encoders:
            df_input[col] = encoders[col].transform(df_input[col])

    hasil = model.predict(df_input)

    # Decode
    try:
        hasil_label = encoders['target'].inverse_transform(hasil)[0]
    except:
        hasil_label = hasil[0]

    # =========================
    # EMOJI CUACA
    # =========================
    if "hujan" in str(hasil_label).lower():
        emoji = "🌧️"
        warna = "#3498db"
    elif "cerah" in str(hasil_label).lower():
        emoji = "☀️"
        warna = "#f1c40f"
    elif "berawan" in str(hasil_label).lower():
        emoji = "☁️"
        warna = "#95a5a6"
    else:
        emoji = "🌈"
        warna = "#9b59b6"

    # =========================
    # HASIL
    # =========================
    st.markdown(f"""
        <div class="result-box" style="background-color:{warna}; color:white;">
            {emoji} Hasil Prediksi: {hasil_label}
        </div>
    """, unsafe_allow_html=True)

    st.balloons()

    # =========================
    # DETAIL INPUT
    # =========================
    with st.expander("📊 Lihat Detail Input"):
        st.write(input_data)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("📌 Tentang Aplikasi")
    st.write("""
    Aplikasi ini menggunakan Machine Learning untuk memprediksi cuaca berdasarkan data yang dimasukkan.

    🔹 Model: Decision Tree  
    🔹 Library: Scikit-learn  
    🔹 UI: Streamlit  
    """)

    st.divider()
    st.write("💡 Tips:")
    st.write("Masukkan data sesuai kondisi nyata agar hasil lebih akurat")

# =========================
# FOOTER
# =========================
st.divider()
st.markdown(
    "<p style='text-align:center;'>Dibuat dengan ❤️ oleh kamu 😎</p>",
    unsafe_allow_html=True
)