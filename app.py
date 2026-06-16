import streamlit as st
import pandas as pd
import pickle

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Prediksi VO2 Max", page_icon="🏃‍♂️", layout="wide")
st.title("🏃‍♂️ Kalkulator Prediksi VO2 Max")
st.markdown("""
Aplikasi web ini memprediksi nilai kapasitas maksimal oksigen (VO2 Max) Anda menggunakan algoritma **XGBoost**. 
Silakan masukkan data demografi, metrik kebugaran, dan status kesehatan harian Anda di bawah ini.
""")

# 2. Memuat Artefak Model (.pkl)
@st.cache_resource
def load_artifacts():
    with open('vo2max_xgboost_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('vo2max_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('feature_columns.pkl', 'rb') as file:
        features = pickle.load(file)
    return model, scaler, features

# Panggil fungsi load
model, scaler, feature_columns = load_artifacts()

# 3. Form Input Pengguna
st.header("📊 Masukkan Data Anda")

# Membagi layar menjadi 3 kolom untuk kerapian
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demografi Fisik")
    age = st.number_input("Usia (tahun)", min_value=10, max_value=100, value=25)
    weight_kg = st.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0, value=65.0)
    height_cm = st.number_input("Tinggi Badan (cm)", min_value=100.0, max_value=250.0, value=170.0)
    
with col2:
    st.subheader("Metrik Olahraga")
    workout_type = st.selectbox("Jenis Olahraga", ["Cardio", "HIIT", "Strength", "Yoga"])
    workout_duration_min = st.number_input("Durasi Olahraga (menit)", min_value=0, max_value=300, value=45)
    calories_burned = st.number_input("Kalori Terbakar (kcal)", min_value=0, max_value=2000, value=300)
    avg_heart_rate = st.number_input("Rata-rata Detak Jantung (bpm)", min_value=40, max_value=220, value=120)

with col3:
    st.subheader("Kesehatan Harian")
    sleep_hours = st.number_input("Durasi Tidur (jam)", min_value=0.0, max_value=24.0, value=7.5)
    steps = st.number_input("Jumlah Langkah Harian", min_value=0, max_value=50000, value=8000)
    active_minutes = st.number_input("Menit Aktif", min_value=0, max_value=1440, value=60)
    resting_heart_rate = st.number_input("Detak Jantung Istirahat (bpm)", min_value=30, max_value=120, value=65)
    stress_level = st.slider("Tingkat Stres (1-10)", min_value=1, max_value=10, value=5)
    hydration_liters = st.number_input("Konsumsi Air (Liter)", min_value=0.0, max_value=10.0, value=2.0)
    recovery_score = st.slider("Skor Pemulihan (0-100)", min_value=0, max_value=100, value=75)

# 4. Proses Prediksi
st.markdown("---")
if st.button("🚀 Prediksi VO2 Max Sekarang", use_container_width=True):
    
    # Membuat DataFrame dummy dengan semua 17 kolom bernilai 0 sesuai urutan di feature_columns.pkl
    input_df = pd.DataFrame(columns=feature_columns)
    input_df.loc[0] = 0  
    
    # Memasukkan input numerik
    input_df.at[0, 'age'] = age
    input_df.at[0, 'weight_kg'] = weight_kg
    input_df.at[0, 'height_cm'] = height_cm
    input_df.at[0, 'workout_duration_min'] = workout_duration_min
    input_df.at[0, 'calories_burned'] = calories_burned
    input_df.at[0, 'avg_heart_rate'] = avg_heart_rate
    input_df.at[0, 'sleep_hours'] = sleep_hours
    input_df.at[0, 'steps'] = steps
    input_df.at[0, 'active_minutes'] = active_minutes
    input_df.at[0, 'resting_heart_rate'] = resting_heart_rate
    input_df.at[0, 'stress_level'] = stress_level
    input_df.at[0, 'hydration_liters'] = hydration_liters
    input_df.at[0, 'recovery_score'] = recovery_score
            
    # Mengisi nilai untuk fitur kategorikal sesuai input SelectBox
    if workout_type == "Cardio":
        input_df.at[0, 'workout_type_Cardio'] = 1
    elif workout_type == "HIIT":
        input_df.at[0, 'workout_type_HIIT'] = 1
    elif workout_type == "Strength":
        input_df.at[0, 'workout_type_Strength'] = 1
    elif workout_type == "Yoga":
        input_df.at[0, 'workout_type_Yoga'] = 1

    # Melakukan penyesuaian skala (Scaling) menggunakan scaler dari pickle
    scaled_input = scaler.transform(input_df)
    
    # Melakukan Prediksi
    prediction = model.predict(scaled_input)[0]
    
    # Menampilkan Hasil
    st.success(f"### 🎉 Estimasi VO2 Max Anda adalah: **{prediction:.2f} mL/kg/min**")
    st.info("Catatan: Nilai ini adalah estimasi dari model Machine Learning berdasarkan data yang Anda berikan.")