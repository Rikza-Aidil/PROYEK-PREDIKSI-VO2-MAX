# 🏃‍♂️ Prediksi VO2 Max

Proyek ini bertujuan memprediksi nilai kapasitas maksimal oksigen (VO2 Max) seseorang berdasarkan data demografi fisik, kebiasaan olahraga, dan kondisi kesehatan harian menggunakan algoritma *machine learning* (XGBoost) dengan metodologi **CRISP-DM**.

🔗 **Live Demo:** [Prediksi VO2 Max App](https://huggingface.co/spaces/RikzaAidil/Prediksi-VO2-Max)
📓 **Notebook:** [Google Colab Notebook](https://colab.research.google.com/drive/1WIuRh2PUXX1NTwkUeqbAQb2iT1WeSBfz?usp=sharing)

---

## 👥 Tim Pengembang

| Nama | NIM |
|------|-----|
| Muhammad Aditya Genta Andhara | 2330511092 |
| Rikza Aidil Rifqi | 2330511084 |

---

## 📂 Struktur Repositori

```
├── app.py                          # Aplikasi antarmuka web Streamlit
├── PROYEK_PREDIKSI_VO2_MAX.ipynb   # Notebook pemodelan (Eksplorasi & Pelatihan)
├── vo2max_xgboost_model.pkl        # Model XGBoost hasil training
├── vo2max_scaler.pkl               # Scaler untuk standardisasi data (StandardScaler)
├── feature_columns.pkl             # Daftar 17 urutan kolom fitur
├── requirements.txt                # Dependency library
└── README.md                       # Laporan proyek
```

---

## 1. Business Understanding

### Latar Belakang

VO2 Max adalah metrik utama untuk mengukur tingkat kebugaran kardiovaskular dan daya tahan aerobik seseorang. Pengukuran VO2 Max secara langsung umumnya membutuhkan peralatan laboratorium khusus yang tidak mudah diakses masyarakat umum. Dengan memanfaatkan data gaya hidup harian yang mudah dilacak — seperti detak jantung, durasi latihan, dan kualitas tidur — kita dapat memberikan estimasi VO2 Max yang lebih praktis dan terjangkau.

### Problem Statement

Dapatkah kita memprediksi estimasi nilai VO2 Max seseorang berdasarkan data demografi, riwayat aktivitas olahraga, dan metrik kesehatan harian menggunakan algoritma regresi?

### Goals

- Membangun model regresi XGBoost untuk mengestimasi nilai VO2 Max.
- Mengidentifikasi fitur gaya hidup yang memiliki pengaruh paling signifikan terhadap tingkat kebugaran.
- Men-deploy model ke dalam bentuk aplikasi web interaktif yang mudah digunakan.

### Solution Statement

- **Model:** XGBoost Regressor
- **Metrik Evaluasi:** Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), dan R-Squared (R²)

---

## 2. Data Understanding

### Sumber Data

Dataset berisikan metrik kebugaran harian pengguna, terdiri dari data numerik dan kategorikal.

- **Nama file:** `fitness_workout_dataset_3800_rows.csv`
- **Jumlah data:** 3.800 baris
- **Jumlah fitur:** 15 fitur prediktif & 1 target (`vo2_max`)

### Deskripsi Fitur

| Kategori | Fitur | Tipe | Deskripsi |
|----------|-------|------|-----------|
| Identifikasi | `user_id` | numerik | ID unik pengguna (dihapus saat modeling) |
| Demografi | `age` | numerik | Usia dalam tahun |
| Demografi | `weight_kg` | numerik | Berat badan dalam kilogram |
| Demografi | `height_cm` | numerik | Tinggi badan dalam sentimeter |
| Olahraga | `workout_type` | kategorikal | Jenis olahraga (Cardio, HIIT, Strength, Yoga) |
| Olahraga | `workout_duration_min` | numerik | Durasi olahraga per sesi (menit) |
| Olahraga | `calories_burned` | numerik | Kalori yang terbakar per sesi |
| Olahraga | `avg_heart_rate` | numerik | Rata-rata detak jantung saat berolahraga (bpm) |
| Kesehatan Harian | `sleep_hours` | numerik | Durasi tidur per malam (jam) |
| Kesehatan Harian | `steps` | numerik | Jumlah langkah per hari |
| Kesehatan Harian | `active_minutes` | numerik | Menit aktif per hari |
| Kesehatan Harian | `resting_heart_rate` | numerik | Detak jantung istirahat (bpm) |
| Kesehatan Harian | `stress_level` | numerik | Tingkat stres skala 1–10 |
| Kesehatan Harian | `hydration_liters` | numerik | Konsumsi air per hari (liter) |
| Kesehatan Harian | `recovery_score` | numerik | Skor pemulihan tubuh |
| **Target** | `vo2_max` | numerik | **Kapasitas maksimal oksigen (mL/kg/min)** |

### EDA Findings

> 📸 *[Tambahkan screenshot visualisasi EDA dari Colab di sini, misalnya: distribusi VO2 Max, heatmap korelasi, boxplot per workout_type]*

Beberapa temuan utama dari eksplorasi data:

- Fitur `avg_heart_rate`, `calories_burned`, dan `active_minutes` menunjukkan korelasi yang cukup signifikan terhadap nilai `vo2_max`.
- Jenis olahraga (`workout_type`) HIIT dan Cardio cenderung menghasilkan nilai VO2 Max yang lebih tinggi dibandingkan Yoga dan Strength.
- Distribusi `vo2_max` relatif normal, memudahkan proses regresi.

---

## 3. Data Preparation

Tahapan prapemrosesan data yang dilakukan:

**1. Feature Selection**
Kolom `user_id` dihapus karena tidak memiliki korelasi prediktif terhadap target.

```python
df_clean = df.drop(columns=['user_id'])
```

**2. One-Hot Encoding**
Kolom kategorikal `workout_type` dikonversi menjadi 4 kolom biner eksplisit menggunakan `pd.get_dummies()` dengan `drop_first=False` agar semua kategori terwakili saat deployment.

```python
X = pd.get_dummies(X, columns=['workout_type'], drop_first=False)
# Menghasilkan: workout_type_Cardio, workout_type_HIIT, workout_type_Strength, workout_type_Yoga
```

**3. Data Splitting**
Dataset dibagi dengan rasio **80% Training** dan **20% Testing**.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**4. Standardisasi**
Seluruh fitur numerik distandarisasi menggunakan `StandardScaler`. Scaler di-fit **hanya pada data training** untuk menghindari data leakage, lalu di-transform ke kedua split. Scaler dan daftar kolom fitur kemudian diekspor sebagai file `.pkl` untuk keperluan deployment.

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit + transform
X_test_scaled  = scaler.transform(X_test)        # Transform saja (hindari data leakage)
```

---

## 4. Modeling

Pemodelan dilakukan menggunakan **XGBoost Regressor** dengan konfigurasi hiperparameter awal:

```python
xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
xgb_model.fit(X_train_scaled, y_train)
```

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| `n_estimators` | 200 | Jumlah pohon keputusan (boosting rounds) |
| `learning_rate` | 0.05 | Laju pembelajaran; kecil = konvergensi lebih stabil |
| `max_depth` | 5 | Kedalaman maksimum tiap pohon |
| `random_state` | 42 | Seed untuk reprodusibilitas hasil |

---

## 5. Evaluation

Hasil pengujian model terhadap data testing:

| Metrik | Nilai |
|--------|-------|
| RMSE (Root Mean Squared Error) | 8.9887 |
| MAE (Mean Absolute Error) | 7.7538 |
| R-Squared (R²) | -0.0901 |

### Feature Importance

>  <img width="557" height="336" alt="image" src="https://github.com/user-attachments/assets/7c4035bb-f18e-483d-99c6-29f62f4dd0ca" />


### Analisis Hasil

Model menghasilkan tingkat kesalahan absolut rata-rata (MAE) sebesar **~7.75 mL/kg/min** dari nilai VO2 Max sebenarnya. Namun, nilai R² yang bernilai negatif (-0.0901) mengindikasikan bahwa model belum optimal dalam menangkap pola dari data — performa model masih di bawah baseline sederhana (mean prediction).

Beberapa langkah perbaikan yang dapat dilakukan pada pengembangan selanjutnya:

- **Hyperparameter Tuning** menggunakan `GridSearchCV` atau `Optuna`
- **Feature Engineering** lanjutan untuk menambah fitur turunan yang lebih informatif
- Identifikasi dan penanganan **outlier** yang mungkin mendistorsi proses pelatihan
- Eksplorasi model alternatif seperti **Random Forest Regressor** atau **LightGBM**

---

## 6. Deployment

Model telah di-deploy menggunakan **Streamlit** di environment **Hugging Face Spaces**. Aplikasi menerima input pengguna secara real-time, melakukan transformasi data di latar belakang menggunakan `vo2max_scaler.pkl` dan `feature_columns.pkl`, lalu langsung memberikan output estimasi VO2 Max.

> 📸 *[Tambahkan screenshot tampilan aplikasi Streamlit di sini]*

### Cara Menjalankan Lokal

```bash
# Install dependensi
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py
```

---

> ⚠️ **Disclaimer:** Aplikasi web ini dibuat semata-mata untuk tujuan demonstrasi akademis dan proyek portfolio (UAS Machine Learning). Hasil estimasi VO2 Max tidak dapat menggantikan pengukuran uji klinis dan tidak ditujukan untuk diagnosis atau saran medis.
