import streamlit as st
import pandas as pd
import joblib
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Student Mental Health Tracker",
    layout="centered"
)

# --- CSS KUSTOM UNTUK UI/UX MINIMALIS & WARM ---
st.markdown("""
    <style>
    .stApp {
        background-color: #FFF9F3;
        color: #4A4A4A;
    }

    .stTextInput, .stSelectbox, .stSlider {
        background-color: #FFFFFF;
        border-radius: 10px;
    }

    .stButton>button {
        background-color: #D4A373;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 25px;
        font-weight: bold;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #BC8A5F;
        border: none;
        color: white;
    }

    h1 {
        color: #8B5E3C;
        font-family: 'Helvetica Neue', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model_path = 'notebooks/model/student_depression_gb_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# --- HEADER ---
st.title("Student Depression Predictor")
st.markdown("""
Aplikasi ini dirancang untuk mendeteksi kondisi depresi pada mahasiswa berdasarkan faktor akademik dan gaya hidup.
Silakan isi data di bawah ini dengan jujur.
""")
st.divider()

# --- FORM INPUT (Minimalis) ---
if model is not None:
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Data Akademik")
            academic_pressure = st.slider("Tekanan Akademik (0-5)", 0, 5, 3)
            cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
            study_sat = st.slider("Kepuasan Belajar (0-5)", 0, 5, 3)

        with col2:
            st.subheader("Gaya Hidup")
            sleep_duration = st.selectbox("Durasi Tidur", ["<5 hours", "5-6 hours", "7-8 hours", ">8 hours"])
            dietary = st.selectbox("Pola Makan", ["Healthy", "Moderate", "Unhealthy"])
            work_hours = st.number_input("Jam Belajar per Hari", 0, 24, 6)

    st.divider()

    # Fitur tambahan
    st.subheader("Kondisi Tambahan")
    suicidal = st.selectbox("Pernah memiliki pikiran bunuh diri?", ["No", "Yes"])
    family_hist = st.selectbox("Riwayat gangguan mental di keluarga?", ["No", "Yes"])
    financial_stress = st.slider("Tingkat Stres Finansial (1-5)", 1, 5, 2)

    # --- PREDIKSI ---
    if st.button("Analisis Status Kesehatan Mental"):
        # Menyusun data sesuai format saat training
        input_data = pd.DataFrame({
            'Age': [20],  # Default atau tambahkan input
            'Gender': ['Male'], # Default atau tambahkan input
            'City': ['Semarang'], # Default
            'CGPA': [cgpa],
            'Sleep Duration': [sleep_duration],
            'Profession': ['Student'],
            'Work Pressure': [0],
            'Academic Pressure': [academic_pressure],
            'Study Satisfaction': [study_sat],
            'Job Satisfaction': [0],
            'Dietary Habits': [dietary],
            'Degree': ['Undergraduate'],
            'Have you ever had suicidal thoughts ?': [suicidal],
            'Work/Study Hours': [work_hours],
            'Financial Stress': [financial_stress],
            'Family History of Mental Illness': [family_hist]
        })

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.divider()
        if prediction == 1:
            st.error(f"### Hasil: Terdeteksi Depresi")
            st.write(f"Tingkat Keyakinan: {probability*100:.2f}%")
            st.warning("Pesan: Jangan ragu untuk mencari bantuan profesional atau berbicara dengan orang terdekat.")
        else:
            st.success(f"### Hasil: Tidak Terdeteksi Depresi")
            st.write(f"Tingkat Keyakinan: {(1-probability)*100:.2f}%")
            st.info("Pesan: Tetap jaga pola hidup sehat dan manajemen waktu yang baik!")
else:
    st.error("Model tidak ditemukan. Pastikan file model .pkl sudah ada di folder 'model/'.")

# --- FOOTER ---
st.markdown("---")
st.caption("Proyek Bengkel Koding - Universitas Dian Nuswantoro")
