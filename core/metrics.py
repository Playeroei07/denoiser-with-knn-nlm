"""
Metrik Evaluasi Kualitas Citra Digital:
- MSE (Mean Squared Error)
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- MAD Noise Level Estimator (tanpa ground truth)
"""

import numpy as np
from scipy.signal import convolve2d


def mse(a, b):
    """
    Menghitung Mean Squared Error (MSE) antara dua citra.
    Nilai b dibulatkan dan dipotong ke rentang 0..255 seperti citra 8-bit.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.clip(np.rint(np.asarray(b, dtype=np.float64)), 0, 255)
    return float(np.mean((a_arr - b_arr) ** 2))


def psnr(a, b):
    """
    Menghitung Peak Signal-to-Noise Ratio (PSNR) dalam desibel (dB).
    Nilai semakin tinggi menunjukkan kualitas rekonsiliasi yang semakin baik.
    """
    m = mse(a, b)
    if m <= 1e-10:
        return float("inf")
    return float(10.0 * np.log10(255.0 ** 2 / m))


def _gaussian_kernel(size=11, sigma=1.5):
    """Membuat kernel Gaussian 2D terstandarisasi untuk perhitungan SSIM lokal."""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)


def compute_ssim(img1, img2, win_size=11, sigma=1.5):
    """
    Menghitung Structural Similarity Index (SSIM) lokal berbasis window Gaussian.
    Mendukung citra Grayscale (H, W, 1) maupun RGB (H, W, 3).
    """
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)

    if img1.shape != img2.shape:
        raise ValueError("Dimensi kedua citra harus identik.")

    # Tetapan stabilitas menurut Wang et al. (2004)
    K1, K2, L = 0.01, 0.03, 255.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    kernel = _gaussian_kernel(win_size, sigma)

    def _ssim_single_channel(x, y):
        mu_x = convolve2d(x, kernel, mode='valid')
        mu_y = convolve2d(y, kernel, mode='valid')

        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x_sq = convolve2d(x * x, kernel, mode='valid') - mu_x_sq
        sigma_y_sq = convolve2d(y * y, kernel, mode='valid') - mu_y_sq
        sigma_xy = convolve2d(x * y, kernel, mode='valid') - mu_xy

        numerator = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
        denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
        ssim_map = numerator / np.maximum(denominator, 1e-10)
        return float(np.mean(ssim_map))

    if img1.ndim == 2 or (img1.ndim == 3 and img1.shape[2] == 1):
        x = img1.squeeze()
        y = img2.squeeze()
        return _ssim_single_channel(x, y)
    elif img1.ndim == 3:
        channel_ssims = [
            _ssim_single_channel(img1[:, :, c], img2[:, :, c])
            for c in range(img1.shape[2])
        ]
        return float(np.mean(channel_ssims))
    else:
        raise ValueError("Format array citra tidak valid.")


def estimate_noise_sigma_mad(img):
    """
    Estimasi tingkat noise sigma menggunakan Median Absolute Deviation (MAD)
    dari filter Laplacian / residual frekuensi tinggi (Donoho & Johnstone).
    Berguna bila citra asli (ground truth) tidak tersedia di dunia nyata.
    """
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 3:
        # Konversi cepat ke grayscale intensitas
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img.squeeze()

    H, W = gray.shape
    # Filter Laplacian sederhana untuk mengisolasi noise frekuensi tinggi
    laplacian = np.array([[1, -2, 1],
                          [-2, 4, -2],
                          [1, -2, 1]], dtype=np.float32)
    
    res = convolve2d(gray, laplacian, mode='valid')
    # Normalisasi standar MAD: sigma = median(|res|) / (0.6745 * sqrt(sum(mask^2)))
    # sum(mask^2) = 1+4+1+4+16+4+1+4+1 = 36 -> sqrt(36) = 6
    mad = np.median(np.abs(res))
    sigma_est = mad / (0.6745 * 6.0)
    return float(sigma_est)
