# ✨ KNN Patch Denoiser Studio

Aplikasi web interaktif modern untuk restorasi citra digital berbasis **K-Nearest Neighbors (KNN) Patch Similarity** dengan pembobotan adaptif **Non-Local Means (NLM)**, dibangun menggunakan **Streamlit** dengan antarmuka yang bersih, cepat, dan *straightforward*.

---

## 🌟 Fitur Utama

- **Antarmuka Ramping & Intuitif**:
  - **Profil Presets Siap Pakai**:
    - ⚡ *Ringan & Cepat*: Untuk grain tipis dan foto low-light ringan.
    - ⚖️ *Seimbang (Rekomendasi)*: Optimal untuk mempertahankan detail tekstur & garis tepi.
    - 🛡️ *Halus Maksimal*: Untuk noise parah / bintik kasar.
    - 🛠️ *Kustom*: Akses parameter internal ($K$, ukuran patch, jendela cari, parameter $h$) untuk riset lanjutan.
  - **Slider Kekuatan Tunggal**: Atur intensitas pembersihan (20% - 150%) tanpa perlu menghitung rumus matematis manual.
- **Akselerasi Integral Image (Box Sum)**:
  - Kompleksitas pencarian jarak patch direduksi menjadi $O(1)$ per jendela, menghasilkan waktu pemrosesan instan (<1 detik).
- **Inspeksi Visual Langsung (Before vs After)**:
  - Perbandingan berdampingan citra bernoise dan hasil restorasi dalam kartu visual yang bersih.
  - Opsi perbandingan 3 citra jika menggunakan mode simulasi *Ground Truth*.
- **Evaluasi Kualitas Gambar Lengkap**:
  - **Mode Citra Riil (Tanpa Simulasi Noise / No-Reference)**:
    - 🌟 **NIQE (Natural Image Quality Evaluator)**: Mengukur kealamian citra berbasis *Natural Scene Statistics* tanpa butuh *ground truth* (skor lebih rendah = lebih bersih).
    - 🛡️ **MAD Noise Level Estimator ($\sigma$)**: Mengestimasi penurunan deviasi noise residual frekuensi tinggi.
  - **Mode Simulasi (*Ground Truth*)**:
    - 📈 **PSNR & PSNR Gain**: Mengukur rasio kekuatan sinyal citra terhadap noise dalam dB.
    - 🔍 **SSIM (Structural Similarity Index)**: Menilai pelestarian struktur, tekstur, dan garis tepi.
- **Unduh Cepat**:
  - Unduh citra bersih resolusi penuh (PNG).
  - Unduh kolase perbandingan visual langsung.

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Dependensi
Pastikan Python 3.9+ sudah terpasang.

### 2. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi Streamlit
Jalankan perintah berikut pada terminal:
```bash
streamlit run app.py
```
Atau di Windows, cukup klik dua kali berkas **`run_app.bat`**.

Buka browser pada alamat: `http://localhost:8501`

---

## 📁 Struktur Direktori

```
KNN NLM Denoiser Streamlit/
│
├── .streamlit/
│   └── config.toml             # Konfigurasi tema 
│
├── assets/
│   └── sample_images/          # Sample gambar awal
│       ├── 1_geometric_shapes.png
│       ├── 2_nature_landscape.png
│       ├── 3_radial_textures.png
│       └── 4_medical_phantom.png
│
├── core/
│   ├── __init__.py
│   ├── denoiser.py             # Main denoiser dengan KNN dan NLM
│   ├── metrics.py              # Evaluasi metrik: MSE, PSNR, SSIM, MAD Estimator
│   └── sample_generator.py     # Pembuat sampel benchmark
│
├── app.py                      # Streamlit utama (UI ringkas & responsif)
├── style.css                   # Custom styling modern & glassmorphism
├── requirements.txt            # Daftar pustaka python
├── run_app.bat                 # Launcher serta mengeek dependensi buat Windows
└── README.md                   # Dokumentasi proyek
```
