---
title: Prediksi VO2 Max
emoji: 🏃‍♂️
colorFrom: yellow
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
license: apache-2.0
---

# Project Overview

Proyek ini bertujuan memprediksi nilai kapasitas maksimal oksigen (VO2 Max) seseorang berdasarkan data demografi fisik, kebiasaan olahraga, dan kondisi kesehatan harian menggunakan algoritma *machine learning* (XGBoost) dengan metodologi CRISP-DM.

🔗 **Live Demo:** [Prediksi VO2 Max App]: https://huggingface.co/spaces/RikzaAidil/Prediksi-VO2-Max
📓 **Notebook**: https://colab.research.google.com/drive/1WIuRh2PUXX1NTwkUeqbAQb2iT1WeSBfz?usp=sharing#scrollTo=LO1Qpy-f5jxL 

👥 **Pengembang**
MUHAMMAD ADITYA GENTA ANDHARA (2330511092)
Rikza Aidil Rifqi	(2330511084)

📂 **Struktur Repositori**
text
├── app.py                      # Aplikasi antarmuka web Streamlit
├── PROYEK_PREDIKSI_VO2_MAX.ipynb # Notebook pemodelan (Eksplorasi & Pelatihan)
├── vo2max_xgboost_model.pkl    # Model XGBoost hasil training
├── vo2max_scaler.pkl           # Scaler untuk standardisasi data (StandardScaler)
├── feature_columns.pkl         # Daftar 17 urutan kolom fitur
├── requirements.txt            # Dependency library
└── README.md                   # Laporan proyek

1. Business Understanding
**Latar Belakang**
VO2 Max adalah metrik utama untuk mengukur tingkat kebugaran kardiovaskular dan daya tahan aerobik seseorang. Pengukuran VO2 Max secara langsung umumnya membutuhkan peralatan laboratorium khusus. Dengan memanfaatkan data gaya hidup harian yang mudah dilacak (seperti detak jantung, durasi latihan, dan kualitas tidur), kita dapat memberikan estimasi VO2 Max yang lebih praktis.

**Problem Statement **
Dapatkah kita memprediksi estimasi nilai VO2 Max seseorang berdasarkan data demografi, riwayat aktivitas olahraga, dan metrik kesehatan harian menggunakan algoritma regresi?

Goals * Membangun model regresi XGBoost untuk mengestimasi nilai VO2 Max.
•	Mengidentifikasi fitur gaya hidup yang memiliki pengaruh terhadap tingkat kebugaran.
•	Men-deploy model ke dalam bentuk aplikasi web interaktif.

Solution Statement * Model: XGBoost Regressor
•	Metrik Evaluasi: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), dan R-Squared (R²).

2. Data Understanding
Sumber Data Dataset berisikan metrik kebugaran harian pengguna, terdiri dari data numerik dan kategorikal.
•	Jumlah data: 3800 baris
•	Jumlah fitur: 15 fitur prediktif & 1 target target target (VO2 Max)

Deskripsi Fitur Utama
•	Demografi: age (Usia), weight_kg (Berat Badan), height_cm (Tinggi Badan).
•	Olahraga: workout_type (Cardio, HIIT, Strength, Yoga), workout_duration_min, calories_burned, avg_heart_rate.
•	Kesehatan Harian: sleep_hours, steps, active_minutes, resting_heart_rate, stress_level, hydration_liters, recovery_score.

3. Data Preparation
Tahapan prapemrosesan data yang dilakukan meliputi:
•	Feature Selection: Menghapus kolom user_id karena tidak memiliki korelasi prediktif terhadap target.
•	One-Hot Encoding: Mengonversi kolom kategorikal workout_type menjadi 4 kolom biner eksplisit (workout_type_Cardio, workout_type_HIIT, workout_type_Strength, workout_type_Yoga) menggunakan pd.get_dummies().
•	Data Splitting: Dataset dibagi menggunakan rasio 80% untuk Training (2400+ baris) dan 20% untuk Testing (760 baris) dengan random_state=42.
•	Standardisasi: Menyesuaikan skala seluruh fitur menggunakan StandardScaler untuk mengoptimalkan konvergensi algoritma gradient boosting. Scaler ini kemudian di-eksport menjadi pickle object.

4. Modeling
Pemodelan dilakukan menggunakan XGBoost Regressor dengan konfigurasi hiperparameter awal sebagai berikut:
•	n_estimators: 200
•	learning_rate: 0.05
•	max_depth: 5
•	random_state: 42

5. Evaluation
Hasil pengujian model terhadap data testing memberikan nilai evaluasi sebagai berikut:
Metrik Evaluasi	Nilai
RMSE (Root Mean Squared Error)	8.9887
MAE (Mean Absolute Error)	7.7538
R-Squared (R²)	-0.0901
Analisis Hasil (Insight):
Model menghasilkan tingkat kesalahan absolut rata-rata (MAE) sebesar ~7.75 mL/kg/min dari nilai VO2 Max sebenarnya. Namun, nilai R-Squared yang bernilai negatif (-0.0901) mengindikasikan bahwa model gagal menangkap pola linear/non-linear dari data secara optimal. Dalam tahap pengembangan selanjutnya, hasil ini dapat diperbaiki dengan melakukan Hyperparameter Tuning (misalnya menggunakan GridSearchCV), rekayasa fitur (Feature Engineering) lanjutan, atau mengidentifikasi keberadaan outlier yang mungkin mendistorsi proses pelatihan.
6. Deployment
Model telah di-deploy menggunakan Streamlit di environment Hugging Face Spaces. Aplikasi menerima input pengguna secara real-time, melakukan transformasi data di latar belakang menggunakan vo2max_scaler.pkl, dan langsung memberikan output estimasi VO2 Max.
⚠️ Disclaimer: Aplikasi web ini dibuat semata-mata untuk tujuan demonstrasi akademis dan proyek portfolio (UAS Machine Learning). Hasil estimasi VO2 Max tidak dapat menggantikan pengukuran uji klinis dan tidak ditujukan untuk diagnosis atau saran medis.


