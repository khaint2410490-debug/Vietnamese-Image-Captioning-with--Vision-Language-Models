"""
preprocessing/noise_generator.py
Sinh dữ liệu ảnh nhiễu Gaussian trắng (AWGN) có kiểm soát.
"""

import csv
from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image

from project_paths import CAPTIONS_PATH, IMAGES_DIR, PROCESSED_DATA_DIR, NOISY_IMAGE_MAP_PATH


def add_gaussian_noise(image: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    """Cộng nhiễu Gaussian N(0, sigma^2) vào ảnh RGB trên miền [0, 255]."""
    arr = np.asarray(image, dtype=np.float32)
    noise = rng.normal(loc=0.0, scale=sigma, size=arr.shape).astype(np.float32)
    noisy_arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(noisy_arr, mode="RGB")


def generate_noisy_dataset(
    captions_csv: Path = CAPTIONS_PATH,
    clean_images_dir: Path = IMAGES_DIR,
    noise_level: Union[int, float, str] = 25,
    seed: int = 42,
) -> Path:
    rng = np.random.default_rng(seed)
    suffix = f"sigma{noise_level}"
    output_dir = PROCESSED_DATA_DIR / f"NoisyImages_{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    unique_images = []
    seen = set()
    with open(captions_csv, mode="r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not {"image", "caption"} <= set(reader.fieldnames):
            raise ValueError(
                "File caption cần có header với hai cột: image,caption"
            )
        for row in reader:
            img_name = (row.get("image") or "").strip()
            if img_name and img_name not in seen:
                seen.add(img_name)
                unique_images.append(img_name)

    total_images = len(unique_images)
    print(
        f"[NoiseGenerator] Bắt đầu tạo ảnh nhiễu cho {total_images} ảnh "
        f"tại: {output_dir}",
        flush=True,
    )
    map_rows = []
    for index, img_name in enumerate(unique_images, start=1):
        clean_path = clean_images_dir / img_name
        if clean_path.is_file():
            sigma = (
                float(rng.uniform(0.0, 55.0))
                if str(noise_level).lower() == "random"
                else float(noise_level)
            )
            with Image.open(clean_path) as img:
                rgb_img = img.convert("RGB")
                noisy_img = add_gaussian_noise(rgb_img, sigma=sigma, rng=rng)
                noisy_path = output_dir / img_name
                noisy_img.save(noisy_path, format="JPEG", quality=95)

            map_rows.append({
                "image": img_name,
                "clean_path": str(clean_path),
                "noisy_path": str(noisy_path),
                "sigma": round(sigma, 2),
            })
        if index % 100 == 0 or index == total_images:
            print(
                f"[NoiseGenerator] Tiến độ: {index}/{total_images} ảnh",
                flush=True,
            )

    with open(NOISY_IMAGE_MAP_PATH, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "clean_path", "noisy_path", "sigma"])
        writer.writeheader()
        writer.writerows(map_rows)

    print(f"[NoiseGenerator] Đã tạo {len(map_rows)} ảnh nhiễu tại: {output_dir}")
    return output_dir
