# Laporan Proyek Machine Learning - Sahra Zulqaidah
## Domain Proyek

### Prediksi Biaya Asuransi Kesehatan Menggunakan Machine Learning

Layanan asuransi kesehatan merupakan salah satu kebutuhan penting dalam sistem kesehatan modern. Biaya premi asuransi biasanya disesuaikan dengan risiko kesehatan individu, yang dipengaruhi oleh berbagai faktor seperti usia, indeks massa tubuh (BMI), kebiasaan merokok, dan kondisi demografis lainnya. Namun, penetapan biaya ini sering kali bersifat kompleks dan rentan terhadap bias subjektif jika hanya mengandalkan pendekatan manual atau heuristik tradisional.

Menurut Centers for Medicare & Medicaid Services (CMS), pengeluaran per kapita untuk layanan kesehatan di Amerika Serikat meningkat dari tahun ke tahun dan diperkirakan mencapai lebih dari $14.000 pada tahun 2023 [1]. Peningkatan ini menunjukkan pentingnya efisiensi dalam pengelolaan dan prediksi biaya layanan kesehatan, termasuk premi asuransi.

Dengan memanfaatkan teknik *machine learning*, kita dapat membangun model prediktif berbasis data historis untuk memperkirakan biaya asuransi secara lebih objektif, efisien, dan akurat. Model ini akan membantu perusahaan asuransi dalam menyesuaikan premi sesuai profil risiko pengguna, serta mendukung kebijakan pricing yang lebih adil bagi masyarakat.

### Mengapa Masalah Ini Penting untuk Diselesaikan?

- **Bagi perusahaan asuransi**: prediksi premi yang akurat membantu dalam pengelolaan risiko keuangan dan pengambilan keputusan underwriting.
- **Bagi pelanggan**: transparansi dalam perhitungan premi akan meningkatkan kepercayaan dan kepuasan terhadap layanan.
- **Secara sistemik**: pengelolaan premi yang efisien akan mendorong keberlanjutan sistem asuransi kesehatan.

### Bagaimana Masalah Ini Diselesaikan?

Masalah ini diselesaikan dengan membangun model regresi machine learning yang mempelajari hubungan antara fitur demografis pengguna dan tagihan medis (charges). Model kemudian dievaluasi untuk memilih pendekatan yang paling akurat dan dapat diandalkan.

### Referensi

[1] Centers for Medicare & Medicaid Services (CMS), “National Health Expenditure Projections 2022–2031,” *U.S. Department of Health and Human Services*, 2022. [Online]. Available: https://www.cms.gov

[2] Mirichoi0218, “Medical Cost Personal Dataset,” *Kaggle*, 2020. [Online]. Available: https://www.kaggle.com/datasets/mirichoi0218/insurance

[3] T. Hastie, R. Tibshirani, J. Friedman, *The Elements of Statistical Learning*, 2nd ed., Springer, 2009.

[4] S. Raschka and V. Mirjalili, *Python Machine Learning*, 3rd ed., Packt Publishing, 2019.
## Business Understanding
### Problem Statements

1. **Bagaimana cara memprediksi biaya asuransi kesehatan (`charges`) berdasarkan data pengguna seperti usia, jenis kelamin, status merokok, dan wilayah tempat tinggal?**

2. **Fitur mana yang paling relevan dan berpengaruh terhadap biaya asuransi sehingga perlu dipertahankan dalam pemodelan?**

3. **Algoritma regresi mana yang memberikan hasil terbaik dalam memprediksi biaya asuransi?**


### Goals

1. **Menghasilkan model prediksi biaya asuransi berdasarkan variabel input pengguna**  
   → Dengan menggunakan data pengguna, model dapat secara otomatis menghitung estimasi biaya medis (`charges`) untuk membantu proses penetapan premi asuransi.

2. **Mengidentifikasi fitur-fitur yang signifikan terhadap `charges` dan menghapus fitur yang tidak relevan**  
   → Hal ini bertujuan menyederhanakan model, mempercepat proses komputasi, dan menghindari overfitting.

3. **Mengevaluasi performa dari beberapa algoritma regresi dan memilih model terbaik**  
   → Model terbaik akan dipilih berdasarkan metrik evaluasi Mean Squared Error (MSE) terendah pada data uji.


### Solution Statements

Untuk mencapai tujuan di atas, berikut adalah solusi yang diterapkan:

- **Solusi 1: Menerapkan beberapa algoritma regresi**  
  Menggunakan tiga model regresi populer untuk regresi tabular:
  - **K-Nearest Neighbors Regressor (KNN)**: algoritma berbasis kedekatan nilai.
  - **Random Forest Regressor**: model ensambel berbasis decision tree.
  - **AdaBoost Regressor**: model boosting yang berfokus memperbaiki kesalahan model sebelumnya.

- **Solusi 2: Melakukan preprocessing dan feature selection**  
  - Menghapus outlier pada kolom `bmi` menggunakan metode IQR.
  - Menghilangkan fitur `bmi` dan `children` karena memiliki korelasi sangat rendah dengan target `charges`.
  - Melakukan one-hot encoding pada fitur kategori (`sex`, `smoker`, `region`).
  - Melakukan standardisasi terhadap fitur numerik (`age`).

- **Solusi 3: Evaluasi performa model secara kuantitatif**  
  - Menggunakan metrik **Mean Squared Error (MSE)** pada data pelatihan dan pengujian.
  - Membandingkan nilai MSE antar model untuk memilih yang paling akurat.

Setiap solusi diukur secara kuantitatif dan obyektif menggunakan MSE, untuk memastikan model yang dipilih benar-benar memiliki performa terbaik dalam prediksi biaya asuransi.

## Data Understanding
### Sumber Data

Dataset yang digunakan dalam proyek ini merupakan dataset biaya asuransi kesehatan individu yang tersedia secara publik dan banyak digunakan dalam studi machine learning. Dataset ini dapat diunduh melalui tautan berikut:

- 📁 Kaggle - *Medical Cost Personal Dataset*  
  [https://www.kaggle.com/datasets/mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance)

Dataset ini terdiri dari **1.338 baris** data sebelum pembersihan dan **1.329 baris** setelah penghapusan outlier. Setiap baris merepresentasikan satu individu dengan data demografis serta estimasi tagihan medis tahunan (`charges`).


### Penjelasan Variabel

Berikut adalah deskripsi masing-masing fitur dalam dataset:

| Nama Fitur | Tipe Data | Deskripsi |
|------------|-----------|-----------|
| `age`      | Numerik   | Usia tertanggung (dalam tahun) |
| `sex`      | Kategori  | Jenis kelamin (`male` atau `female`) |
| `bmi`      | Numerik   | Body Mass Index (Indeks Massa Tubuh) |
| `children` | Numerik   | Jumlah anak yang menjadi tanggungan |
| `smoker`   | Kategori  | Status merokok (`yes` atau `no`) |
| `region`   | Kategori  | Wilayah tempat tinggal (`northeast`, `northwest`, `southeast`, `southwest`) |
| `charges`  | Numerik   | Biaya medis tahunan (target/output model) |


### Exploratory Data Analysis (EDA)

Beberapa teknik eksplorasi yang dilakukan untuk memahami struktur dan pola dalam data:

#### 1. **Tipe Data dan Statistik Deskriptif**
- Melalui fungsi `df.info()` dan `df.describe()`, ditemukan bahwa:
- Terdapat 3 fitur kategori (`sex`, `smoker`, `region`)
- Fitur numerik terdiri dari `age`, `bmi`, `children`, dan `charges`

#### 2. **Visualisasi Outlier**
- Digunakan **boxplot** untuk mendeteksi outlier pada kolom `age`, `bmi`, `children`, dan `charges`.
- Outlier dihapus menggunakan metode IQR agar tidak mempengaruhi hasil prediksi.

#### 3. **Distribusi Data Kategori**
- Visualisasi dengan **barplot** untuk fitur `sex`, `smoker`, dan `region`.
- Distribusi relatif seimbang, meskipun terdapat lebih banyak non-perokok dibandingkan perokok.

#### 4. **Analisis Korelasi**
- Digunakan **heatmap** dan **pairplot** untuk mengevaluasi korelasi antara fitur numerik.
- `age` dan `smoker` menunjukkan korelasi yang cukup terhadap `charges`, sedangkan `bmi` dan `children` sangat lemah → **di-drop pada tahap persiapan data**.

#### 5. **Distribusi Data**
- Histograms digunakan untuk melihat distribusi setiap fitur numerik.
- Pairplot dengan `hue="smoker"` memberikan wawasan bahwa perokok cenderung memiliki `charges` yang jauh lebih tinggi.


### Insight Penting dari EDA:
- Fitur `smoker` memiliki pengaruh paling signifikan terhadap biaya medis (`charges`).
- Individu dengan status merokok dikenakan biaya lebih tinggi dibanding non-perokok.
- Penghapusan fitur non-signifikan membantu menyederhanakan model tanpa mengorbankan akurasi.

## Data Preparation
### Tahapan dan Teknik Data Preparation

Berikut adalah langkah-langkah data preparation yang dilakukan secara berurutan sesuai notebook:

1. **Outlier Removal (bmi)**  
   - Metode: IQR (Interquartile Range)  
   - Alasan: Menghapus nilai ekstrem pada `bmi` agar model tidak terdistorsi oleh anomali.

2. **Feature Elimination berdasarkan Korelasi**  
   - Fitur `bmi` dan `children` dihapus karena memiliki korelasi sangat rendah terhadap `charges`.  
   - Alasan: Menyederhanakan model dan mengurangi noise.

3. **One-Hot Encoding pada Fitur Kategori**  
   - Fitur `sex`, `smoker`, dan `region` dikonversi ke bentuk numerik dengan `pd.get_dummies()`  
   - Alasan: Algoritma ML tidak dapat memproses data kategori secara langsung.
   
4. **Train-Test Split (80:20)**  
   - Data dibagi menjadi data latih dan uji menggunakan `train_test_split`.  
   - Alasan: Untuk mengevaluasi generalisasi model terhadap data yang belum pernah dilihat.
   
5. **Standard Scaling pada Fitur Numerik**  
   - Fitur `age` distandarisasi menggunakan `StandardScaler`.  
   - Alasan: Model seperti KNN dan AdaBoost sensitif terhadap skala fitur.

 
## Modeling
### Algoritma yang Digunakan

#### 1. **K-Nearest Neighbors Regressor (KNN)**
- **Parameter:** `n_neighbors=10`
- **Cara Kerja:** KNN merupakan algoritma non-parametrik berbasis kedekatan. Untuk regresi, KNN memprediksi nilai target dengan mengambil rata-rata dari `k` tetangga terdekat di data latih. Kedekatan biasanya diukur menggunakan jarak Euclidean. KNN termasuk lazy learner karena tidak membentuk model secara eksplisit, melainkan menghitung saat prediksi.
- **Implementasi di Notebook:** Menggunakan `KNeighborsRegressor` dari `sklearn.neighbors` dengan `n_neighbors=10`.
- **Kelebihan:** Sederhana, intuitif, tidak memerlukan pelatihan eksplisit.
- **Kekurangan:** Kurang efisien untuk data besar, sensitif terhadap fitur tidak relevan dan outlier.

#### 2. **Random Forest Regressor**
- **Parameter:** `n_estimators=50`, `max_depth=16`, `random_state=55`
- **Cara Kerja:** Random Forest merupakan algoritma ensemble yang membangun banyak decision tree pada subset data acak (bagging). Setiap pohon hanya mempertimbangkan subset fitur acak pada setiap pemisahan, sehingga mengurangi overfitting dan meningkatkan generalisasi.
- **Implementasi di Notebook:** Menggunakan `RandomForestRegressor` dari `sklearn.ensemble`.
- **Kelebihan:** Akurat, stabil, dan tahan terhadap overfitting.
- **Kekurangan:** Kurang interpretatif dan memerlukan sumber daya komputasi yang lebih besar.

#### 3. **AdaBoost Regressor**
- **Parameter:** `learning_rate=0.05`, `random_state=55`
- **Cara Kerja:** AdaBoost membangun model secara iteratif, dengan setiap model baru lebih fokus pada data yang sebelumnya diprediksi salah. Biasanya digunakan model lemah seperti decision stump (pohon kedalaman 1), dan hasil akhir merupakan kombinasi berbobot dari seluruh model.
- **Implementasi di Notebook:** Menggunakan `AdaBoostRegressor` dari `sklearn.ensemble` dengan parameter default weak learner.
- **Kelebihan:** Dapat meningkatkan akurasi model sederhana dan fokus pada kesalahan sulit.
- **Kekurangan:** Sensitif terhadap noise dan outlier.

---

### Pemilihan Model Terbaik

Model terbaik dipilih berdasarkan nilai **Mean Squared Error (MSE)** terkecil pada data **test**. Berdasarkan hasil, model **Random Forest Regressor** memberikan performa terbaik dengan generalisasi paling baik di antara ketiga algoritma.

---

## Evaluation

### 📏 Metrik Evaluasi: Mean Squared Error (MSE)

\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

- MSE mengukur rata-rata kesalahan kuadrat antara nilai aktual dan prediksi.
- Semakin kecil nilai MSE, semakin baik kemampuan prediktif model.

---

### Hasil Evaluasi Model

| Model         | Train MSE     | Test MSE      |
|---------------|---------------|---------------|
| **KNN**        | 37,385.01     | 114,480.02    |
| **RandomForest** | 18,614.80     | 89,983.53     |
| **AdaBoost**    | 39,826.34     | 71,642.70  |

> *Catatan: MSE dalam tabel telah disederhanakan (dibagi 1000) untuk keterbacaan.*

---

### 🔍 Analisis & Interpretasi

- AdaBoost** menunjukkan kinerja terbaik dengan **Test MSE terendah**, menandakan generalisasi yang kuat terhadap data baru.
- **Random Forest** memiliki Train MSE paling rendah, tetapi selisih Train–Test cukup besar, menandakan sedikit **overfitting**.
- *KNN** menunjukkan performa buruk pada data uji, dengan Test MSE yang paling tinggi, sehingga tidak cocok digunakan untuk kasus ini.

---

## Evaluasi Terhadap Business Understanding

### ✅ Problem Statement 1  
**Bagaimana cara memprediksi `charges` berdasarkan data pengguna?**

- **Status**: Tercapai
- **Penjelasan**: Model AdaBoost mampu mengestimasi biaya premi berdasarkan input pengguna seperti `age`, `sex`, `smoker`, dan `region`.
- **Dampak Bisnis**: Dapat digunakan sebagai alat estimasi biaya otomatis untuk penyesuaian premi secara personal.

---

### ✅ Problem Statement 2  
**Fitur mana yang paling relevan dan berpengaruh terhadap `charges`?**

- **Status**: Tercapai
- **Penjelasan**: Fitur `bmi` dan `children` dihapus karena kontribusinya sangat rendah terhadap `charges`.
- **Dampak Bisnis**: Menyederhanakan model, mengurangi overfitting, dan mempercepat waktu pelatihan serta prediksi.

---

### ✅ Problem Statement 3  
**Algoritma regresi mana yang paling akurat?**

- **Status**: Tercapai
- **Penjelasan**: Model **AdaBoost Regressor** dipilih berdasarkan MSE terendah pada data uji.
- **Dampak Bisnis**: Memberikan dasar kuat untuk memilih model terbaik dalam implementasi sistem estimasi biaya premi.

---

## Evaluasi Goals

| Goals | Status | Penjelasan |
|-------|--------|------------|
| **Model prediksi biaya premi** | ✅ Tercapai | Model mampu memprediksi `charges` secara akurat berdasarkan data pengguna. |
| **Identifikasi & seleksi fitur signifikan** | ✅ Tercapai | Fitur yang tidak relevan dihapus untuk menyederhanakan dan meningkatkan performa model. |
| **Evaluasi dan seleksi model terbaik** | ✅ Tercapai | AdaBoost dipilih berdasarkan performa MSE terbaik pada data uji. |

---

## Evaluasi Solusi yang Diterapkan

| Solusi | Dampak |
|--------|--------|
| **Regresi multialgoritma (KNN, RF, AdaBoost)** | Memberikan perbandingan yang adil untuk memilih model paling akurat. |
| **Preprocessing dan feature selection** | Meningkatkan efisiensi model dan mengurangi risiko overfitting. |
| **Evaluasi kuantitatif dengan MSE** | Menyediakan dasar obyektif dan konsisten dalam pemilihan model terbaik. |

---

## Kesimpulan Evaluasi

Model prediktif yang dibangun berhasil menjawab **seluruh problem statement**, memenuhi **semua goals bisnis**, dan solusi yang dirancang berdampak positif terhadap efisiensi dan akurasi.

**AdaBoost Regressor** direkomendasikan sebagai model akhir untuk diterapkan dalam sistem estimasi biaya premi.

---
