# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding
Jaya Jaya Institut merupakan institusi pendidikan tinggi yang telah berdiri sejak tahun 2000 dan berkomitmen mencetak lulusan unggul. Namun, institusi menghadapi tantangan serius berupa tingginya angka mahasiswa yang mengalami dropout sebelum menyelesaikan studi. Fenomena ini berdampak pada berbagai aspek penting, mulai dari reputasi institusi, efisiensi operasional, hingga pendapatan. Selain itu, tingginya tingkat dropout juga berkontribusi menurunkan rasio kelulusan yang menjadi salah satu indikator utama dalam proses akreditasi lembaga pendidikan.

Untuk mengatasi permasalahan ini, institusi perlu memahami secara menyeluruh pola dan faktor-faktor yang mendorong mahasiswa keluar dari studi. Dengan begitu, pihak manajemen dapat mengambil langkah antisipatif dan memberikan dukungan yang sesuai secara lebih dini. Proyek ini hadir sebagai upaya untuk menjawab kebutuhan tersebut melalui analisis data mahasiswa dan pengembangan sistem yang mampu membantu proses identifikasi risiko dropout secara sistematis dan terukur.

### Permasalahan Bisnis
Tingginya angka dropout menjadi perhatian utama bagi Jaya Jaya Institut. Untuk itu, beberapa pertanyaan kunci yang ingin dijawab dalam proyek ini meliputi:
1. Apa saja faktor-faktor yang berkontribusi terhadap risiko mahasiswa mengalami dropout?
2. Bagaimana cara mengidentifikasi mahasiswa yang berisiko tinggi dropout secara lebih dini?
3. Bagaimana menyajikan hasil analisis ini dalam bentuk yang mudah dipahami dan digunakan oleh pihak manajemen institusi?

### Cakupan Proyek
#### 1. Pengembangan Model Prediktif
Membangun model machine learning untuk mengklasifikasikan status mahasiswa menjadi tiga kategori: Dropout, Graduate, dan Enrolled. Proses ini melibatkan eksplorasi fitur, pemilihan algoritma, evaluasi performa model, serta penyimpanan model dan pipeline agar dapat digunakan ulang.

#### 2. Dashboard Interaktif Monitoring Mahasiswa
Melakukan eksplorasi data untuk memahami distribusi dan karakteristik mahasiswa berdasarkan status akademik, demografi, dan sosial ekonomi. Hasil analisis divisualisasikan dalam bentuk dashboard interaktif menggunakan Tableau untuk menyampaikan insight seperti rasio dropout, distribusi nilai, dan tingkat kehadiran dengan cara yang mudah dipahami oleh pemangku kebijakan.

#### 3. Integrasi Aplikasi Prediksi
Mengembangkan antarmuka input interaktif di aplikasi Streamlit yang memungkinkan pengguna mengisi data mahasiswa secara manual untuk mendapatkan prediksi status secara langsung. Fitur ini berguna untuk skrining awal terhadap risiko dropout.

#### 4. Penyusunan Insight dan Arahan Tindak Lanjut
Menarik kesimpulan dari hasil analisis dan prediksi yang diperoleh, serta menyusun rekomendasi intervensi yang dapat diterapkan terhadap mahasiswa berisiko. Temuan ini diharapkan mendukung kebijakan internal dalam rangka peningkatan angka kelulusan dan efisiensi pembelajaran.

### Persiapan

Sumber data: Dataset yang digunakan ialah <a href="https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance">Dataset Students Performance</a> dari Dicoding yang berisi data demografis, akademik, serta kondisi sosial ekonomi mahasiswa. 

Setup environment:

Agar dapat menjalankan project ini dengan baik, ikuti langkah-langkah berikut untuk mengatur environment dan menginstall dependency.

1. Membuat Lingkungan Kerja dengan Anaconda
```
conda create --name Proyek-ds2 python=3.11.7
conda activate Proyek-ds2
pip install -r requirements.txt
```
2. Menyimpan dan Memuat Model Terbaik untuk digunakan kembali pada prediksi mendatang
```
import joblib
import os

os. makedirs('model', exist_ok=True)
joblib.dump(xgb_best, 'model/model_xgb.pkl')
joblib.dump(scaler, 'model/robust_scaler.pkl')
joblib.dump(le_status, 'model/label_encoder.pkl')
feature_names = X_train.columns.tolist()
joblib.dump(feature_names, "model/feature_names.pkl")
```
3. Menjalankan Prediksi
```
streamlit run app.py
```

## Business Dashboard

![Dashboard](https://raw.githubusercontent.com/ftriaa/students_performance/main/ftriaanggra23-dashboard/Dashboard_Dropout.png)

Dashboard Student Performance Jaya Jaya Institute dibangun menggunakan Tableau yang dapat diakses melalui <a href="https://public.tableau.com/views/StudentsPerformanceDashboard_17495632449370/Dashboard1?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link">Link berikut ini</a>. Terdapat dua filter interaktif pada dashbord ini, yaitu Status dan Scholarship Holder, yang memungkinkan pengguna menyaring data berdasarkan status akademik mahasiswa dan apakah mereka menerima beasiswa atau tidak. Saat ini, seluruh data pada dashboard difokuskan untuk mahasiswa berstatus dropout.

Di bagian atas, lima metrik utama ditampilkan sebagai indikator kinerja:
- Total Students: 2.842 mahasiswa dropout,
- Average Age: rata-rata usia 26,07 tahun,
- Average GPA: nilai IPK rata-rata sebesar 6,58,
- UKT Overdue: sebanyak 32,16% mahasiswa memiliki tunggakan UKT,
- Avg Previous Qualification: rata-rata nilai kualifikasi akademik sebelumnya sebesar 131,1.

Visualisasi pertama, Students by Course, menunjukkan bahwa jurusan dengan jumlah dropout terbanyak adalah Management Evening (272 mahasiswa), diikuti Management (268), dan Nursing (236). Jurusan dengan dropout paling sedikit adalah Biofuel Production Technologies (16 mahasiswa), mengindikasikan variasi tingkat risiko dropout antar jurusan.

Visualisasi Students by Age Group dalam bentuk bubble chart memperlihatkan bahwa dropout paling banyak terjadi pada kelompok usia 15–20 tahun, diikuti oleh usia 21–25 tahun. Ini menunjukkan bahwa mahasiswa usia muda lebih rentan mengalami dropout dibandingkan yang lebih tua.

Diagram batang Semester 1 & 2 Grades by Scholarship membandingkan jumlah unit mata kuliah semester 1 dan 2 berdasarkan status beasiswa. Mahasiswa tanpa beasiswa (No) memiliki jumlah unit yang jauh lebih tinggi di kedua semester dibandingkan yang menerima beasiswa (Yes), yakni 1.287 unit vs 134 unit per semester. Namun, pola penurunan dari semester 1 ke 2 serupa di kedua kelompok, menunjukkan banyak mahasiswa dropout tidak melanjutkan atau menyelesaikan semester kedua, tanpa memandang status beasiswa.

Terakhir, grafik horizontal Academic Status by Admission Grade and Gender menunjukkan bahwa mahasiswa perempuan memiliki nilai masuk (admission grade) lebih tinggi dibanding laki-laki. Ini mengindikasikan bahwa meskipun memiliki modal akademik awal yang lebih baik, perempuan tetap mengalami dropout, sehingga faktor penyebabnya kemungkinan bukan hanya aspek akademik.

Secara keseluruhan, dashboard ini memberikan wawasan menyeluruh mengenai mahasiswa dropout, baik dari sisi akademik, demografi, maupun program studi. Data ini dapat menjadi landasan penting bagi institusi untuk menyusun strategi intervensi dan pencegahan dropout yang lebih efektif dan tepat sasaran.

## Menjalankan Sistem Machine Learning

Langkah-langkah untuk menggunakan sistem prediksi status mahasiswa melalui Streamlit adalah sebagai berikut:

### 1. Akses Dashboard Streamlit

Buka aplikasi melalui tautan Streamlit yang telah disediakan pada <a href="https://students-performance-predict.streamlit.app/">Link berikut ini</a>. Tunggu beberapa saat hingga aplikasi termuat sepenuhnya.

### 2. Navigasi Menu

Setelah aplikasi terbuka, terlihat **dua menu utama** pada bagian atas:

* 📖 **Latar Belakang**

  Menjelaskan konteks, tujuan, dan manfaat dari aplikasi prediksi dropout mahasiswa.

* 🔍 **Prediksi Dropout**

  Berisi form interaktif untuk memasukkan data mahasiswa dan mendapatkan hasil prediksi.

### 3. Melakukan Prediksi

Beralih ke tab **Prediksi Dropout** untuk mulai melakukan prediksi:

#### a. Isi Formulir

Masukkan informasi lengkap tentang mahasiswa yang ingin diprediksi, termasuk:

* Nama, jenis kelamin, usia
* Nilai akademik (rata-rata nilai mata kuliah, nilai evaluasi, dll.)
* Data ekonomi keluarga (pekerjaan dan pendidikan orang tua)
* Faktor eksternal (GDP negara asal, tingkat inflasi, pengangguran, dll.)
* Status pembayaran, beasiswa, program studi, dll.

#### b. Tekan Tombol "Prediksi Status Mahasiswa"

Setelah semua data diisi, klik tombol untuk melakukan prediksi. Sistem akan menampilkan:

* **Hasil Prediksi**: Apakah mahasiswa akan Dropout, Graduate, atau tetap Enrolled.
* **Keyakinan Model**: Persentase kemungkinan berdasarkan model machine learning yang telah dilatih.

### 4. Interpretasi Hasil

Hasil prediksi akan ditampilkan secara jelas, misalnya:

> **Status mahasiswa bernama <Nama_Mahasiswa> diprediksi sebagai: Dropout**

> **Kemungkinan mahasiswa mengalami dropout adalah: 91.10%**

Hal ini dapat digunakan sebagai bahan evaluasi dan tindakan preventif dari pihak institusi.

## Conclusion

Melalui proyek ini, Jaya Jaya Institut kini memiliki alat prediktif yang mampu mengidentifikasi mahasiswa berisiko tinggi mengalami dropout secara lebih dini. Model machine learning yang dikembangkan menunjukkan performa yang baik dalam mengklasifikasikan status mahasiswa menjadi Dropout, Enrolled, dan Graduate.

Selain itu, dashboard analitik yang dibangun dengan Tableau memberikan wawasan menyeluruh terkait faktor-faktor yang berkontribusi terhadap risiko dropout, termasuk performa akademik, status ekonomi, program studi, dan demografi mahasiswa.

Aplikasi interaktif berbasis Streamlit memudahkan pengguna non-teknis untuk memanfaatkan hasil model prediksi secara praktis dalam pengambilan keputusan. Secara keseluruhan, sistem ini dapat berperan sebagai alat bantu yang strategis dalam upaya meningkatkan retensi dan kualitas pendidikan di lingkungan Jaya Jaya Institut.

### Rekomendasi Action Items

Berdasarkan hasil analisis dan prediksi, berikut adalah beberapa rekomendasi strategis untuk institusi:

**1. Monitoring Proaktif Terhadap Mahasiswa Berisiko**

  Terapkan sistem notifikasi internal berbasis data untuk mendeteksi mahasiswa dengan nilai akademik menurun, tunggakan UKT, atau faktor risiko lainnya sejak awal semester.

**2. Intervensi Dini Berbasis Data**

  Lakukan bimbingan akademik atau konseling terhadap mahasiswa yang teridentifikasi berisiko dropout berdasarkan prediksi model, terutama yang berada di jurusan dengan tingkat dropout tinggi.

**3. Optimalisasi Program Beasiswa**

  Tinjau dan sesuaikan kriteria penerima beasiswa berdasarkan data dropout, guna memastikan bantuan tepat sasaran dan berdampak dalam meningkatkan retensi mahasiswa.

**4. Pengembangan Fitur Monitoring Lanjutan**

  Integrasikan sistem prediksi ke dalam sistem informasi akademik kampus (SIAKAD) agar dapat digunakan secara real-time oleh dosen wali atau bagian kemahasiswaan.

**5. Evaluasi Kurikulum dan Beban Studi**

  Tinjau ulang beban studi awal semester di jurusan dengan dropout tinggi untuk menghindari overload yang bisa memicu mahasiswa berhenti di tengah jalan.
