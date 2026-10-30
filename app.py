"""
KNN Patch Denoiser Studio
Aplikasi Interaktif Restorasi Citra Digital berbasis K-Nearest Neighbors (KNN) Patch Similarity
dengan Pembobotan Adaptif Non-Local Means (NLM).
Antarmuka Sederhana, Bersih, dan Berfokus pada Perbandingan Sebelum vs Sesudah.
"""

import os
import io
import time
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

# Import core modules
from core.denoiser import (
    knn_patch_denoise,
    add_gaussian_noise,
    add_salt_and_pepper_noise,
    add_poisson_noise
)
from core.metrics import (
    mse,
    psnr,
    compute_ssim
)
from core.sample_generator import generate_sample_images

# =====================================================================
# Konfigurasi Halaman & CSS
# =====================================================================

st.set_page_config(
    page_title="Image Denoiser",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Pastikan folder dan gambar sampel bawaan tersedia
sample_dir = os.path.join(os.path.dirname(__file__), "assets", "sample_images")
if not os.path.exists(sample_dir) or len(os.listdir(sample_dir)) == 0:
    generate_sample_images(sample_dir)


# =====================================================================
# Fungsi Pembantu (Helpers)
# =====================================================================

def load_image_as_rgb_array(image_file_or_path, max_dim=None):
    """Membaca citra dan mengonversinya ke float32 RGB (H, W, 3) rentang 0..255."""
    img = Image.open(image_file_or_path).convert("RGB")
    if max_dim is not None and max_dim > 0:
        w, h = img.size
        if max(w, h) > max_dim:
            if w > h:
                new_w = max_dim
                new_h = int(h * (max_dim / w))
            else:
                new_h = max_dim
                new_w = int(w * (max_dim / h))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    return arr, img


def array_to_pil_bytes(arr, format="PNG"):
    """Mengonversi ndarray float32 (H, W, C) ke bytes gambar untuk diunduh."""
    clipped = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(clipped)
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    return buf.getvalue(), pil_img


def create_comparison_collage(clean_arr, noisy_arr, denoised_arr):
    """Membuat kolase gabungan berdampingan untuk dokumentasi."""
    has_gt = clean_arr is not None
    n_cols = 3 if has_gt else 2
    
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 5.0), dpi=180)
    fig.patch.set_facecolor('#081119')

    if has_gt:
        images = [clean_arr, noisy_arr, denoised_arr]
        titles = ["1. Citra Asli (Ground Truth)", "2. Citra Bernoise (Input)", "3. Hasil Pembersihan (KNN NLM)"]
    else:
        images = [noisy_arr, denoised_arr]
        titles = ["1. Citra Input (Bernoise)", "2. Hasil Pembersihan (KNN NLM)"]

    for ax, img, title in zip(axes, images, titles):
        ax.set_facecolor('#04090e')
        ax.imshow(np.clip(img / 255.0, 0, 1))
        ax.set_title(title, color='#f1f5f9', fontsize=11, fontweight='bold', pad=10)
        ax.axis('off')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    return buf.getvalue()


# =====================================================================
# Preset Profil Denoise Cerdas
# =====================================================================

PRESET_CONFIGS = {
    "⚡ Ringan & Cepat": {
        "k": 6,
        "patch_size": 3,
        "search_size": 9,
        "base_h": 16.0,
        "weight_mode": "NLM",
        "desc": "Untuk noise tipis / grain halus. Pemrosesan instan dengan fokus ketajaman garis."
    },
    "⚖️ Seimbang (Rekomendasi)": {
        "k": 10,
        "patch_size": 5,
        "search_size": 11,
        "base_h": 26.0,
        "weight_mode": "NLM",
        "desc": "Diatas dari Preset Ringan, namun cukup powerfull mengatasi noise yang lebih banyak"
    },
    "🛡️ Halus Maksimal (Noise Berat)": {
        "k": 18,
        "patch_size": 5,
        "search_size": 15,
        "base_h": 38.0,
        "weight_mode": "NLM",
        "desc": "Paling berat dalam mengatasi noise ekstrim terutama di pencahayaan"
    },
    "🛠️ Kustom (Atur Sendiri)": {
        "k": 10,
        "patch_size": 5,
        "search_size": 11,
        "base_h": 25.0,
        "weight_mode": "NLM",
        "desc": "Bebas mengatur parameter internal algoritma KNN sesuai kebutuhan riset."
    }
}


# =====================================================================
# Sidebar: Antarmuka Kontrol Bersih & Terarah
# =====================================================================

with st.sidebar:
    st.markdown('<div class="sidebar-header">✨ <b>Image Denoiser</b></div>', unsafe_allow_html=True)
    st.caption("Pilih option image dan tentukan preset cleaning paling pas")

    # 1. PILIH CITRA
    st.markdown("##### 📁 Image Source")
    source_type = st.radio(
        "Pilih Sumber:",
        options=["🖼️ Sample", "📤 Upload Image"],
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )

    clean_img_arr = None
    pil_preview = None
    is_sample = ("Sample" in source_type or "Sampel" in source_type)
    sample_noise_level = 25.0

    if is_sample:
        sample_map = {
            "Bentuk Geometris (Tepi Kontras)": "1_geometric_shapes.png",
            "Lanskap Alam (Gradien Halus)": "2_nature_landscape.png",
            "Tekstur Frekuensi Tinggi": "3_radial_textures.png",
            "Phantom Medis": "4_medical_phantom.png"
        }
        selected_sample_label = st.selectbox("Pilih Sampel Preset:", list(sample_map.keys()), index=0)
        sample_path = os.path.join(sample_dir, sample_map[selected_sample_label])
        if os.path.exists(sample_path):
            clean_img_arr, pil_preview = load_image_as_rgb_array(sample_path)

        sample_noise_level = st.slider(
            "Tingkat Noise pada Sampel:",
            min_value=10,
            max_value=60,
            value=25,
            step=5,
            help="Menyimulasikan noise pada image/sampel agar terlihat proses pembersihannya secara jelas."
        )
    else:
        uploaded_file = st.file_uploader(
            "Pilih file citra (JPG, PNG, WEBP):",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            clean_img_arr, pil_preview = load_image_as_rgb_array(uploaded_file)
        else:
            st.info("Silakan pilih atau unggah citra.")

        add_sim_noise = st.checkbox(
            "Tambahkan Noise Buatan (Simulasi)",
            value=False,
            help="Centang jika Anda ingin menambahkan noise untuk diuji."
        )
        if add_sim_noise:
            sample_noise_level = st.slider("Level Noise Tambahan:", 10, 60, 25, step=5)

    st.markdown("---")

    # 2. PROFIL & KEKUATAN DENOISE
    st.markdown("##### ⚙️ 2. Profil Pembersihan")
    selected_preset = st.selectbox(
        "Pilih Profil Restorasi:",
        options=list(PRESET_CONFIGS.keys()),
        index=1,
        help="Pilih preset yang paling sesuai dengan noise pada citra Anda."
    )
    
    preset_data = PRESET_CONFIGS[selected_preset]
    st.caption(f"ℹ️ {preset_data['desc']}")

    # Slider Kekuatan Sederhana (Denoising Strength)
    denoise_strength = st.slider(
        "Kekuatan Denoise:",
        min_value=20,
        max_value=150,
        value=100,
        step=5,
        format="%d%%",
        help="100% adalah nilai standar preset. Naikkan jika image masih berbintik, turunkan jika tepi tampak terlalu halus."
    )

    # Inisialisasi parameter kerja dari preset & slider kekuatan
    factor = denoise_strength / 100.0
    final_k = preset_data["k"]
    final_patch = preset_data["patch_size"]
    final_search = preset_data["search_size"]
    final_h = preset_data["base_h"] * factor
    final_weight_mode = preset_data["weight_mode"]

    # Parameter Lanjutan (Hanya jika memilih Kustom)
    if selected_preset == "🛠️ Kustom (Atur Sendiri)":
        with st.expander("🛠️ Parameter Internal KNN", expanded=True):
            col_k1, col_k2 = st.columns(2)
            with col_k1:
                final_patch = st.selectbox("Ukuran Patch:", [3, 5, 7], index=1, format_func=lambda x: f"{x}×{x}")
            with col_k2:
                final_search = st.selectbox("Jendela Cari:", [7, 9, 11, 15, 21], index=2, format_func=lambda x: f"{x}×{x}")

            final_k = st.slider("Jumlah Tetangga (K):", 1, min(35, final_search * final_search), 10)
            
            weight_choice = st.radio("Skema Bobot:", ["Adaptif NLM (Eksponensial)", "Seragam (Rata-rata Rata)"], index=0)
            final_weight_mode = "NLM" if "NLM" in weight_choice else "Uniform"

            if final_weight_mode == "NLM":
                final_h = st.slider("Parameter Smoothing (h):", 5.0, 60.0, float(preset_data["base_h"] * factor), step=1.0)
            else:
                final_h = None

    st.markdown("---")

    # 3. OPTIMASI UKURAN GAMBAR (AGAR RESPONSIF)
    max_res_option = st.selectbox(
        "Maksimum Ukuran Pemrosesan:",
        options=["Cepat (Maks 280 px)", "Standar (Maks 360 px)", "Resolusi Penuh (Asli)"],
        index=1,
        help="Menjaga pemrosesan tetap instan (<3 detik). Cocok untuk preview cepat."
    )
    dim_map = {
        "Cepat (Maks 280 px)": 280,
        "Standar (Maks 360 px)": 360,
        "Resolusi Penuh (Asli)": None
    }
    target_dim = dim_map[max_res_option]

    # Tombol Eksekusi
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("✨ Jalankan Denoise Sekarang", type="primary", use_container_width=True)


# =====================================================================
# Main Header (Minimalis, Modern, Tidak Memenuhi Layar)
# =====================================================================

st.markdown("""
<div class="compact-header">
    <div class="brand-badge">✨ KNN & NLM STUDIO</div>
    <h1 class="compact-title">Restorasi Citra Digital Cerdas</h1>
    <p class="compact-subtitle">
        Pembersihan noise berbasis <b>K-Nearest Neighbors Patch Similarity</b> dengan pembobotan eksponensial Gauss (NLM).
        Menghilangkan grain & noise tanpa merusak ketajaman garis objek.
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================================
# Session State Management
# =====================================================================

if "has_processed" not in st.session_state:
    st.session_state.has_processed = False
if "clean_arr" not in st.session_state:
    st.session_state.clean_arr = None
if "noisy_arr" not in st.session_state:
    st.session_state.noisy_arr = None
if "denoised_arr" not in st.session_state:
    st.session_state.denoised_arr = None
if "exec_time" not in st.session_state:
    st.session_state.exec_time = 0.0
if "metrics_data" not in st.session_state:
    st.session_state.metrics_data = {}


# =====================================================================
# Eksekusi Pemrosesan Denoising
# =====================================================================

should_run = (clean_img_arr is not None) and (run_btn or not st.session_state.has_processed)

if should_run:
    img_work = clean_img_arr.copy()

    # Penskalaan resolusi jika diperlukan
    if target_dim is not None and max(img_work.shape[0], img_work.shape[1]) > target_dim:
        pil_tmp = Image.fromarray(np.clip(img_work, 0, 255).astype(np.uint8))
        w, h = pil_tmp.size
        if w > h:
            new_w = target_dim
            new_h = int(h * (target_dim / w))
        else:
            new_h = target_dim
            new_w = int(w * (target_dim / h))
        pil_tmp = pil_tmp.resize((new_w, new_h), Image.Resampling.LANCZOS)
        img_work = np.array(pil_tmp, dtype=np.float32)

    # Persiapan input & noise
    # Jika menggunakan sampel atau mencentang tambah noise buatan:
    need_synthetic_noise = is_sample or (not is_sample and 'add_sim_noise' in locals() and add_sim_noise)

    if need_synthetic_noise:
        ground_truth = img_work.copy()
        noisy_input = add_gaussian_noise(ground_truth, sigma=sample_noise_level, seed=42)
        sigma_for_filter = sample_noise_level
    else:
        ground_truth = None
        noisy_input = img_work.copy()
        sigma_for_filter = 25.0

    # Progress bar interaktif
    progress_slot = st.empty()
    pbar = progress_slot.progress(0, text="Menginisialisasi pencarian patch KNN...")

    def update_progress(pct, current_strip, total_strips):
        pbar.progress(
            min(1.0, max(0.0, pct)),
            text=f"On processing {current_strip}/{total_strips} ({int(pct * 100)}%)..."
        )

    t_start = time.time()
    h_value = final_h if final_weight_mode == "NLM" else None
    
    denoised_result = knn_patch_denoise(
        noisy_input,
        k=final_k,
        patch_size=final_patch,
        search_size=final_search,
        h=h_value,
        sigma=sigma_for_filter if h_value is not None else None,
        progress_callback=update_progress
    )
    t_elapsed = time.time() - t_start
    progress_slot.empty()

    # Perhitungan metrik ringkas
    m_data = {}
    if ground_truth is not None:
        p_noisy = psnr(ground_truth, noisy_input)
        p_denoised = psnr(ground_truth, denoised_result)
        m_data = {
            "has_gt": True,
            "psnr_gain": p_denoised - p_noisy,
            "psnr_denoised": p_denoised
        }
    else:
        m_data = {"has_gt": False}

    # Simpan ke session state
    st.session_state.clean_arr = ground_truth
    st.session_state.noisy_arr = noisy_input
    st.session_state.denoised_arr = denoised_result
    st.session_state.exec_time = t_elapsed
    st.session_state.metrics_data = m_data
    st.session_state.has_processed = True


# =====================================================================
# Tampilan Hasil (Langsung, Bersih, Sederhana)
# =====================================================================

if st.session_state.has_processed and st.session_state.denoised_arr is not None:
    gt_arr = st.session_state.clean_arr
    noisy_arr = st.session_state.noisy_arr
    denoised_arr = st.session_state.denoised_arr
    m_data = st.session_state.metrics_data
    elapsed_time = st.session_state.exec_time
    h_img, w_img, _ = noisy_arr.shape

    # Notifikasi status sukses yang simpel dan informatif
    if m_data.get("has_gt", False):
        st.success(f"✨ Pembersihan selesai dalam **{elapsed_time:.2f} detik**! Kualitas citra meningkat **+{m_data['psnr_gain']:.2f} dB**.")
    else:
        st.success(f"✨ Pembersihan selesai dalam **{elapsed_time:.2f} detik**! Resolusi: **{w_img} × {h_img} px**.")

    # Pilihan tata letak kolom jika memiliki citra asli
    if gt_arr is not None:
        view_mode = st.radio(
            "Pilihan Tampilan:",
            options=["Bandingkan 2 Citra (Sebelum vs Sesudah)", "Bandingkan 3 Citra (+ Citra Asli)"],
            index=0,
            horizontal=True
        )
    else:
        view_mode = "Bandingkan 2 Citra (Sebelum vs Sesudah)"

    # Perbandingan Gambar Utama
    if "3 Citra" in view_mode and gt_arr is not None:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="img-header"><span class="badge blue">Asli</span> Citra Bersih (Ground Truth)</div>', unsafe_allow_html=True)
            st.image(np.clip(gt_arr / 255.0, 0, 1), use_container_width=True)
        with c2:
            st.markdown('<div class="img-header"><span class="badge red">Sebelum</span> Citra Bernoise (Input)</div>', unsafe_allow_html=True)
            st.image(np.clip(noisy_arr / 255.0, 0, 1), use_container_width=True)
        with c3:
            st.markdown('<div class="img-header"><span class="badge green">Sesudah</span> Hasil Pembersihan (KNN NLM)</div>', unsafe_allow_html=True)
            st.image(np.clip(denoised_arr / 255.0, 0, 1), use_container_width=True)
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="img-header"><span class="badge red">Sebelum</span> Citra Bernoise (Input)</div>', unsafe_allow_html=True)
            st.image(np.clip(noisy_arr / 255.0, 0, 1), use_container_width=True)
        with c2:
            st.markdown('<div class="img-header"><span class="badge green">Sesudah</span> Hasil Pembersihan (KNN NLM)</div>', unsafe_allow_html=True)
            st.image(np.clip(denoised_arr / 255.0, 0, 1), use_container_width=True)

    # Tombol Unduh Langsung
    st.markdown("<br>", unsafe_allow_html=True)
    denoised_bytes, _ = array_to_pil_bytes(denoised_arr)
    collage_bytes = create_comparison_collage(gt_arr, noisy_arr, denoised_arr)

    dl_col1, dl_col2, _ = st.columns([1.2, 1.2, 1.6])
    with dl_col1:
        st.download_button(
            label="📥 Unduh Citra Bersih (PNG)",
            data=denoised_bytes,
            file_name="citra_denoised_knn.png",
            mime="image/png",
            use_container_width=True
        )
    with dl_col2:
        st.download_button(
            label="🖼️ Unduh Kolase Perbandingan",
            data=collage_bytes,
            file_name="perbandingan_denoise.png",
            mime="image/png",
            use_container_width=True
        )

else:
    st.info("Pilih citra di sidebar lalu klik tombol **✨ Jalankan Denoise Sekarang**.")
