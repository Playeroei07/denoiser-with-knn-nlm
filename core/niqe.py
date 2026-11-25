"""
Natural Image Quality Evaluator (NIQE)
No-Reference Image Quality Assessment (NR-IQA) berbasis Natural Scene Statistics (NSS).
Mengukur seberapa alami dan bersih citra tanpa memerlukan citra referensi asli (Ground Truth).
Semakin rendah skor NIQE, semakin baik kualitas citra.
"""

import math
import numpy as np
from PIL import Image
from scipy.ndimage import convolve
from scipy.special import gamma

from core.niqe_params import MU_PRIS_PARAM, COV_PRIS_PARAM, GAUSSIAN_WINDOW

# Precompute gamma look-up table sekali saat import untuk akselerasi 20x lipat
_GAM = np.arange(0.2, 10.001, 0.001)
_GAM_RECIPROCAL = np.reciprocal(_GAM)
_R_GAM = np.square(gamma(_GAM_RECIPROCAL * 2.0)) / (
    gamma(_GAM_RECIPROCAL) * gamma(_GAM_RECIPROCAL * 3.0)
)
_MU_PRIS = MU_PRIS_PARAM.squeeze()
_COV_PRIS = COV_PRIS_PARAM
_GAUSS_WIN = GAUSSIAN_WINDOW


def estimate_aggd_param(block):
    """
    Estimasi parameter Asymmetric Generalized Gaussian Distribution (AGGD).
    Mengembalikan (alpha, beta_l, beta_r).
    """
    block = block.flatten()
    left_mask = block < 0
    right_mask = block > 0

    if not np.any(left_mask) or not np.any(right_mask):
        return 1.0, 0.1, 0.1

    left_std = np.sqrt(np.mean(block[left_mask] ** 2))
    right_std = np.sqrt(np.mean(block[right_mask] ** 2))

    if right_std <= 1e-7:
        return 1.0, float(left_std), float(right_std)

    gammahat = left_std / right_std
    mean_abs = np.mean(np.abs(block))
    mean_sq = np.mean(block ** 2)

    if mean_sq <= 1e-7:
        return 1.0, float(left_std), float(right_std)

    rhat = (mean_abs ** 2) / mean_sq
    denom = (gammahat ** 2 + 1.0) ** 2
    if denom <= 1e-7:
        return 1.0, float(left_std), float(right_std)

    rhatnorm = (rhat * (gammahat ** 3 + 1.0) * (gammahat + 1.0)) / denom
    pos = np.argmin((_R_GAM - rhatnorm) ** 2)

    alpha = _GAM[pos]
    scale_fac = np.sqrt(gamma(1.0 / alpha) / gamma(3.0 / alpha))
    beta_l = float(left_std * scale_fac)
    beta_r = float(right_std * scale_fac)

    return float(alpha), beta_l, beta_r


def compute_patch_features(block):
    """
    Mengekstrak 18 fitur statistik spasial dari satu blok citra ternormalisasi (MSCN).
    """
    feat = []
    alpha, beta_l, beta_r = estimate_aggd_param(block)
    feat.extend([alpha, (beta_l + beta_r) / 2.0])

    # Pergeseran 4 orientasi: horizontal, vertical, diagonal utama, diagonal sekunder
    shifts = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for shift in shifts:
        shifted = np.roll(block, shift, axis=(0, 1))
        a, bl, br = estimate_aggd_param(block * shifted)
        g_ratio = gamma(2.0 / a) / gamma(1.0 / a)
        mean_val = (br - bl) * g_ratio
        feat.extend([a, mean_val, bl, br])

    return feat


def calculate_niqe(img, block_size=96):
    """
    Menghitung skor NIQE dari citra input (RGB atau Grayscale).
    Semakin rendah skornya, kualitas citra semakin alami & minim noise.
    
    Args:
        img (np.ndarray): Citra input dalam rentang 0..255 (Grayscale atau RGB).
        block_size (int): Ukuran blok patch analisis (standar = 96x96).
        
    Returns:
        float: Skor NIQE.
    """
    img_arr = np.asarray(img, dtype=np.float32)

    # Konversi citra ke Y-channel (Luminansi Grayscale 0..255)
    if img_arr.ndim == 3 and img_arr.shape[2] >= 3:
        # Rumus ITU-R BT.601 Y channel
        y_img = 0.299 * img_arr[:, :, 0] + 0.587 * img_arr[:, :, 1] + 0.114 * img_arr[:, :, 2]
    elif img_arr.ndim == 3:
        y_img = img_arr[:, :, 0]
    else:
        y_img = img_arr.squeeze()

    h, w = y_img.shape

    # Adaptasi ukuran blok jika citra lebih kecil dari 96x96
    bs_h = min(block_size, max(16, h // 2))
    bs_w = min(block_size, max(16, w // 2))

    num_block_h = max(1, math.floor(h / bs_h))
    num_block_w = max(1, math.floor(w / bs_w))

    curr_img = y_img[:num_block_h * bs_h, :num_block_w * bs_w].copy()

    distparam = []

    # Analisis Multiskala: Skala 1 (Asli) dan Skala 2 (Downsample 0.5x)
    for scale in (1, 2):
        cur_h, cur_w = curr_img.shape
        b_h = max(8, bs_h // scale)
        b_w = max(8, bs_w // scale)

        n_bh = max(1, math.floor(cur_h / b_h))
        n_bw = max(1, math.floor(cur_w / b_w))

        # MSCN Normalization (Mean Subtracted Contrast Normalized)
        mu = convolve(curr_img, _GAUSS_WIN, mode='nearest')
        sigma = np.sqrt(np.abs(convolve(curr_img ** 2, _GAUSS_WIN, mode='nearest') - (mu ** 2)))
        norm_img = (curr_img - mu) / (sigma + 1.0)

        scale_feats = []
        for ibw in range(n_bw):
            for ibh in range(n_bh):
                patch = norm_img[ibh * b_h:(ibh + 1) * b_h, ibw * b_w:(ibw + 1) * b_w]
                scale_feats.append(compute_patch_features(patch))

        distparam.append(np.array(scale_feats, dtype=np.float64))

        if scale == 1:
            # Resize ke 0.5 skala menggunakan Bicubic antialiasing
            pil_scaled = Image.fromarray(curr_img).resize(
                (max(16, cur_w // 2), max(16, cur_h // 2)),
                resample=Image.Resampling.BICUBIC
            )
            curr_img = np.asarray(pil_scaled, dtype=np.float32)

    # Gabungkan fitur multiskala (total 36 parameter per blok)
    min_len = min(len(distparam[0]), len(distparam[1]))
    if min_len == 0:
        return 0.0

    features = np.concatenate([distparam[0][:min_len], distparam[1][:min_len]], axis=1)

    # Bersihkan baris yang mengandung NaN jika ada
    valid_mask = ~np.isnan(features).any(axis=1)
    if not np.any(valid_mask):
        return 0.0

    valid_feats = features[valid_mask]

    # Model Multivariate Gaussian (MVG) untuk citra yang dievaluasi
    mu_dist = np.mean(valid_feats, axis=0)
    cov_dist = np.cov(valid_feats, rowvar=False) if len(valid_feats) > 1 else np.eye(36)

    # Hitung jarak statistik Mahalanobis antara Model Alami dan Model Citra
    cov_avg = (_COV_PRIS + cov_dist) / 2.0
    inv_cov = np.linalg.pinv(cov_avg)

    diff = _MU_PRIS - mu_dist
    quality = float(np.sqrt(np.maximum(0.0, np.dot(np.dot(diff, inv_cov), diff.T))))

    return float(round(quality, 2))
