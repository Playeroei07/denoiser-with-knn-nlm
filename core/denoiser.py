"""
Algoritma KNN berbasis Patch untuk pengurangan noise pada citra digital.
Mendukung pembobotan eksponensial (NLM kernel), mode RGB & Grayscale,
serta pelaporan progres striping untuk antarmuka pengguna interaktif.
"""

import numpy as np
import time
import matplotlib.pyplot as plt


def _box_sum_valid(a, k):
    """
    Menghitung jumlah nilai pada jendela k x k (mode 'valid')
    menggunakan teknik integral image 2D (O(1) per jendela).
    """
    ii = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = a.cumsum(axis=0, dtype=np.float64).cumsum(axis=1)
    return (ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]).astype(np.float32)


def knn_patch_denoise(img, k=10, patch_size=5, search_size=11, h=None, sigma=None,
                      max_chunk_elems=4_000_000, progress_callback=None):
    """
    Denoising citra digital berbasis K-Nearest Neighbors (KNN) Patch Similarity.

    Parameter
    ---------
    img             : ndarray (H, W, C) float, rentang 0..255 (C = 1 atau 3)
    k               : jumlah patch tetangga paling mirip yang dirata-ratakan
    patch_size      : ukuran patch (harus ganjil, misal 3, 5, 7)
    search_size     : ukuran jendela pencarian lokal (harus ganjil, misal 7, 11, 15, 21)
    h               : parameter filtering smoothing. Jika None atau <= 0, menggunakan rata-rata seragam.
                      Jika > 0, menggunakan pembobotan eksponensial exp(-max(d - 2*sigma^2, 0) / h^2)
    sigma           : estimasi simpangan baku noise (hanya dipakai bila h > 0)
    max_chunk_elems : batas jumlah elemen array jarak per strip untuk efisiensi memori
    progress_callback: fungsi opsional callback(progress: float [0..1], current_strip: int, total_strips: int)

    Returns
    -------
    out : ndarray float32 (H, W, C), nilai intensitas piksel hasil denoising [0..255]
    """
    if patch_size % 2 == 0 or search_size % 2 == 0:
        raise ValueError("patch_size dan search_size harus bilangan ganjil.")
    if img.ndim != 3:
        raise ValueError("img harus berdimensi 3 dengan format (H, W, C).")

    img = img.astype(np.float32, copy=False)
    H, W, C = img.shape
    pr, sr = patch_size // 2, search_size // 2
    n_offsets = search_size * search_size
    k = int(max(1, min(k, n_offsets)))
    pad = pr + sr
    norm = float(patch_size * patch_size * C)

    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    out = np.empty((H, W, C), dtype=np.float32)

    rows_per_strip = max(1, min(H, max_chunk_elems // max(1, W * n_offsets)))
    s2 = 0.0 if sigma is None else 2.0 * float(sigma) ** 2

    total_strips = (H + rows_per_strip - 1) // rows_per_strip
    strip_idx = 0

    for ys in range(0, H, rows_per_strip):
        ye = min(H, ys + rows_per_strip)
        rows = ye - ys

        # Area referensi: baris ys..ye ditambah margin patch di semua sisi
        ref = padded[sr + ys: sr + ye + 2 * pr, sr: sr + W + 2 * pr, :]

        dist = np.empty((n_offsets, rows, W), dtype=np.float32)
        cand = np.empty((n_offsets, rows, W, C), dtype=np.float32)

        idx = 0
        for dy in range(-sr, sr + 1):
            for dx in range(-sr, sr + 1):
                shifted = padded[sr + ys + dy: sr + ye + 2 * pr + dy,
                                 sr + dx: sr + W + 2 * pr + dx, :]
                sq = ((shifted - ref) ** 2).sum(axis=2)
                dist[idx] = _box_sum_valid(sq, patch_size) / norm
                cand[idx] = padded[pad + ys + dy: pad + ye + dy,
                                   pad + dx: pad + dx + W, :]
                idx += 1

        nn = np.argpartition(dist, k - 1, axis=0)[:k]               # (k, rows, W)
        nn_dist = np.take_along_axis(dist, nn, axis=0)              # (k, rows, W)
        nn_val = np.take_along_axis(cand, nn[..., None], axis=0)    # (k, rows, W, C)

        if h is None or h <= 0:
            w = np.ones_like(nn_dist)
        else:
            w = np.exp(-np.maximum(nn_dist - s2, 0.0) / (float(h) ** 2))
        w = w[..., None]
        
        sum_w = w.sum(axis=0)
        # Cegah pembagian dengan nol jika bobot sangat kecil
        sum_w = np.where(sum_w == 0, 1e-6, sum_w)
        out[ys:ye] = (w * nn_val).sum(axis=0) / sum_w

        strip_idx += 1
        if progress_callback:
            progress_callback(strip_idx / total_strips, strip_idx, total_strips)

    return np.clip(out, 0.0, 255.0).astype(np.float32)


# =====================================================================
# Noise Generators
# =====================================================================

def add_gaussian_noise(img, sigma, seed=None):
    """
    Menambahkan noise Gaussian aditif (mean=0, std=sigma).
    """
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float32) + rng.normal(0.0, float(sigma), img.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 255.0).astype(np.float32)


def add_salt_and_pepper_noise(img, amount=0.05, s_vs_p=0.5, seed=None):
    """
    Menambahkan noise impulsif Salt & Pepper.
    """
    rng = np.random.default_rng(seed)
    noisy = img.copy().astype(np.float32)
    num_salt = np.ceil(amount * img.size * s_vs_p)
    num_pepper = np.ceil(amount * img.size * (1.0 - s_vs_p))

    # Salt
    coords = [rng.integers(0, i - 1, int(num_salt)) for i in img.shape]
    noisy[tuple(coords)] = 255.0

    # Pepper
    coords = [rng.integers(0, i - 1, int(num_pepper)) for i in img.shape]
    noisy[tuple(coords)] = 0.0

    return np.clip(noisy, 0.0, 255.0).astype(np.float32)


def add_poisson_noise(img, factor=1.0, seed=None):
    """
    Menambahkan noise Poisson (photon noise / shot noise).
    """
    rng = np.random.default_rng(seed)
    # Skala intensitas
    scaled = np.maximum(img, 0.0) * factor
    noisy = rng.poisson(scaled).astype(np.float32) / max(1e-6, factor)
    return np.clip(noisy, 0.0, 255.0).astype(np.float32)


# =====================================================================
# Residual & Difference Map
# =====================================================================

def compute_residual_map(img_a, img_b, colormap='viridis', amplify=2.0):
    """
    Menghitung peta residual selisih absolut |img_a - img_b|
    lalu memetakannya dengan colormap untuk visualisasi noise yang terangkat.
    """
    diff = np.abs(img_a.astype(np.float32) - img_b.astype(np.float32))
    if diff.ndim == 3:
        diff_mag = diff.mean(axis=2)
    else:
        diff_mag = diff

    # Amplifikasi kontras untuk menonjolkan noise halus
    norm_diff = np.clip(diff_mag * amplify / 255.0, 0.0, 1.0)
    
    cmap = plt.get_cmap(colormap)
    colored_residual = cmap(norm_diff)[:, :, :3]  # (H, W, 3) float [0..1]
    return (colored_residual * 255.0).astype(np.uint8), diff_mag
