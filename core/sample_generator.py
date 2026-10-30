"""
Generator citra uji sampel (benchmark images) untuk eksplorasi denoiser instan.
Menghasilkan citra resolusi ideal (~256x256 atau 320x320) dengan tepi tajam, gradien halus,
dan tekstur kaya.
"""

import os
import numpy as np
from PIL import Image, ImageDraw


def generate_sample_images(output_dir="assets/sample_images"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Geometric & Architecture Pattern (280x280)
    w, h = 280, 280
    img1 = Image.new("RGB", (w, h), color="#0f172a")
    draw = ImageDraw.Draw(img1)
    # Background gradient
    for y in range(h):
        r = int(15 + (40 - 15) * (y / h))
        g = int(23 + (110 - 23) * (y / h))
        b = int(42 + (180 - 42) * (y / h))
        draw.line([(0, y), (w, y)], fill=(r, g, b))
    # Sharp shapes
    draw.rectangle([30, 40, 110, 120], fill="#10b981", outline="#34d399", width=3)
    draw.ellipse([140, 30, 240, 130], fill="#06b6d4", outline="#67e8f9", width=4)
    draw.polygon([(140, 240), (200, 150), (260, 240)], fill="#f59e0b", outline="#fde68a", width=3)
    draw.rectangle([40, 150, 120, 250], fill="#ec4899", outline="#f472b6", width=3)
    # Fine grid lines
    for i in range(10, w, 20):
        draw.line([(i, 0), (i, h)], fill=(255, 255, 255, 20), width=1)
    img1.save(os.path.join(output_dir, "1_geometric_shapes.png"))

    # 2. Rich Natural Scenery (280x280)
    img2 = Image.new("RGB", (w, h))
    draw2 = ImageDraw.Draw(img2)
    # Sky
    for y in range(160):
        t = y / 160.0
        r = int(56 + (186 - 56) * t)
        g = int(189 + (230 - 189) * t)
        b = int(248 + (253 - 248) * t)
        draw2.line([(0, y), (w, y)], fill=(r, g, b))
    # Sun
    draw2.ellipse([190, 30, 240, 80], fill="#fbbf24", outline="#f59e0b", width=2)
    # Mountain 1 (distant)
    draw2.polygon([(0, 160), (70, 70), (160, 160)], fill="#64748b")
    draw2.polygon([(110, 160), (200, 90), (280, 160)], fill="#475569")
    # Mountain 2 (near)
    draw2.polygon([(30, 160), (120, 110), (220, 160)], fill="#334155")
    # Rolling green hills
    for y in range(160, h):
        t = (y - 160) / (h - 160)
        r = int(16 + (5 - 16) * t)
        g = int(185 + (150 - 185) * t)
        b = int(129 + (105 - 129) * t)
        draw2.line([(0, y), (w, y)], fill=(r, g, b))
    # Trees
    for tx, ty in [(40, 180), (70, 210), (180, 190), (230, 220), (130, 235)]:
        draw2.rectangle([tx - 2, ty, tx + 2, ty + 20], fill="#78350f")
        draw2.polygon([(tx - 12, ty), (tx, ty - 25), (tx + 12, ty)], fill="#047857")
        draw2.polygon([(tx - 9, ty - 12), (tx, ty - 32), (tx + 9, ty - 12)], fill="#10b981")
    img2.save(os.path.join(output_dir, "2_nature_landscape.png"))

    # 3. Concentric Radial Details / Textures (280x280)
    xx, yy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    r = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)
    pattern1 = 128.0 + 120.0 * np.sin(r * 32.0)
    pattern2 = 128.0 + 120.0 * np.cos(theta * 12.0)
    pattern3 = 128.0 + 120.0 * np.sin(r * 16.0 + theta * 6.0)
    radial_arr = np.stack([pattern1, pattern2, pattern3], axis=-1)
    radial_arr = np.clip(radial_arr, 0, 255).astype(np.uint8)
    Image.fromarray(radial_arr).save(os.path.join(output_dir, "3_radial_textures.png"))

    # 4. Medical Diagnostic Phantom (280x280 Grayscale / RGB)
    phantom = np.zeros((h, w, 3), dtype=np.uint8)
    p_img = Image.fromarray(phantom)
    p_draw = ImageDraw.Draw(p_img)
    # Skull
    p_draw.ellipse([30, 20, 250, 260], fill=(210, 210, 210))
    # Brain tissue
    p_draw.ellipse([45, 35, 235, 245], fill=(70, 70, 70))
    # Ventricles
    p_draw.ellipse([80, 80, 130, 160], fill=(20, 20, 20))
    p_draw.ellipse([150, 80, 200, 160], fill=(20, 20, 20))
    # Tumors / small lesions (testing subtle detail preservation)
    p_draw.ellipse([100, 180, 125, 205], fill=(180, 180, 180))
    p_draw.ellipse([170, 190, 185, 205], fill=(240, 240, 240))
    p_draw.ellipse([70, 120, 80, 130], fill=(230, 230, 230))
    p_img.save(os.path.join(output_dir, "4_medical_phantom.png"))

    print(f"Sample images successfully created in {output_dir}")


if __name__ == "__main__":
    generate_sample_images()
