"""
noise_generator.py

Tạo tập ảnh nhiễu Gaussian trắng (AWGN) từ ảnh gốc Flickr8k:
    I_noisy = Clip(I_clean + N(0, sigma^2), 0, 255)

Có 2 kiểu thêm nhiễu:
    1. Mức cố định (Fixed) : dùng chung một sigma (15, 25 hoặc 50) cho cả tập.
    2. Ngẫu nhiên (Random): mỗi ảnh lấy ngẫu nhiên một giá trị sigma trong [0, 55].

Kết quả xuất ra:
    - Thư mục ảnh nhiễu : data/processed/NoisyImages_sigma{level}/
    - File bảng ánh xạ  : data/processed/noisy_image_map.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_paths import (
    CAPTIONS_PATH,
    IMAGES_DIR,
    NOISY_IMAGE_MAP_PATH,
    NOISY_IMAGES_DIR,
    PROJECT_ROOT,
    get_noisy_images_dir,
)

DEFAULT_FIXED_LEVELS = (15, 25, 50)
DEFAULT_RANDOM_RANGE = (0.0, 55.0)
DEFAULT_RESIZE_SIZE = (224, 224)
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAP_FIELDNAMES = ["image", "clean_path", "noisy_path", "sigma", "noise_mode"]


def _format_level_label(level: float) -> str:
    """Chuyển giá trị số thành nhãn ngắn gọn (25.0 -> '25', 12.5 -> '12.5')."""
    return str(int(level)) if float(level).is_integer() else f"{level:g}"


def _parse_noise_level(noise_level: int | float | str) -> tuple[str, float | None, str]:
    """
    Chuẩn hóa tham số noise_level về (mode, sigma_value, dir_suffix).
    """
    if isinstance(noise_level, str):
        cleaned = noise_level.strip().rstrip(".").lower()
        if cleaned == "random":
            return "random", None, "random"
        try:
            value = float(cleaned)
        except ValueError as error:
            raise ValueError(
                f"noise_level must be a number >= 0 or 'random', got: {noise_level!r}"
            ) from error
    else:
        value = float(noise_level)

    if value < 0:
        raise ValueError(f"noise_level (sigma) must be >= 0, got: {value}")
    return "fixed", value, _format_level_label(value)


def _to_project_path_str(path: Path) -> str:
    """Ưu tiên lưu đường dẫn tương đối theo PROJECT_ROOT để dễ chia sẻ giữa các máy."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def read_image_list_from_captions(
    captions_path: str | Path = CAPTIONS_PATH,
    images_dir: str | Path = IMAGES_DIR,
) -> list[str]:
    """
    Đọc danh sách tên ảnh duy nhất từ captions.csv và lọc các ảnh thực sự tồn tại
    trong thư mục `images_dir`.
    """
    captions_file = Path(captions_path)
    img_dir = Path(images_dir)
    if not captions_file.is_file():
        raise FileNotFoundError(f"Captions file not found: {captions_file}")
    if not img_dir.is_dir():
        raise FileNotFoundError(
            f"Clean images directory not found: {img_dir}. "
            "Please place the Flickr8k images in data/raw/Images/."
        )

    unique_names: list[str] = []
    seen: set[str] = set()
    with open(captions_file, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or "image" not in reader.fieldnames:
            raise ValueError(f"File {captions_file} is missing the 'image' column.")
        for row in reader:
            name = (row["image"] or "").strip()
            if name and name not in seen:
                seen.add(name)
                unique_names.append(name)

    existing = [name for name in unique_names if (img_dir / name).is_file()]
    missing_count = len(unique_names) - len(existing)
    if missing_count > 0:
        print(
            f"Warning: {missing_count}/{len(unique_names)} images in {captions_file.name} "
            f"are missing in {img_dir} (processing {len(existing)} available images)."
        )
    if not existing:
        raise RuntimeError(
            f"No valid images found in {img_dir} matching {captions_file}."
        )
    return existing


def add_awgn_to_array(
    clean_array: np.ndarray,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Áp dụng nhiễu Gaussian trắng cộng tính (AWGN) trên mảng ảnh [0, 255]:
        I_noisy = Clip(I_clean + N(0, sigma^2), 0, 255)
    Trả về mảng uint8 shape (H, W, C).
    """
    if sigma < 0:
        raise ValueError(f"Standard deviation sigma must be >= 0, got: {sigma}")

    clean_float = np.asarray(clean_array, dtype=np.float32)
    if sigma == 0:
        return np.clip(np.rint(clean_float), 0, 255).astype(np.uint8)

    generator = rng if rng is not None else np.random.default_rng()
    noise = generator.normal(loc=0.0, scale=float(sigma), size=clean_float.shape).astype(
        np.float32
    )
    noisy_float = np.clip(clean_float + noise, 0.0, 255.0)
    return np.rint(noisy_float).astype(np.uint8)


def generate_noisy_image(
    clean_image_path: str | Path,
    output_image_path: str | Path,
    sigma: float,
    rng: np.random.Generator | None = None,
    jpeg_quality: int = 95,
    resize_size: tuple[int, int] | None = None,
) -> Path:
    """
    Đọc một ảnh sạch, tùy chọn resize trước, cộng nhiễu AWGN với độ lệch chuẩn `sigma`
    và lưu ra file ảnh đích.
    """
    src_path = Path(clean_image_path)
    dst_path = Path(output_image_path)
    if not src_path.is_file():
        raise FileNotFoundError(f"Clean image not found: {src_path}")
    if src_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {src_path.suffix}")

    with Image.open(src_path) as image:
        rgb_image = image.convert("RGB")
        if resize_size is not None:
            rgb_image = rgb_image.resize(resize_size, Image.Resampling.BILINEAR)
        clean_array = np.asarray(rgb_image, dtype=np.float32)

    noisy_array = add_awgn_to_array(clean_array, sigma=sigma, rng=rng)
    noisy_image = Image.fromarray(noisy_array, mode="RGB")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = (
        {"quality": jpeg_quality, "subsampling": 0}
        if dst_path.suffix.lower() in {".jpg", ".jpeg"}
        else {}
    )
    noisy_image.save(dst_path, **save_kwargs)
    return dst_path


def _sync_file_to_default_dir(src_file: Path, default_file: Path) -> None:
    """Đồng bộ file ảnh sang thư mục NoisyImages/ mặc định (ưu tiên hardlink để tiết kiệm ổ đĩa)."""
    default_file.parent.mkdir(parents=True, exist_ok=True)
    if default_file.exists():
        default_file.unlink()
    try:
        os.link(src_file, default_file)
    except OSError:
        shutil.copy2(src_file, default_file)


def _write_noise_map(
    map_path: Path,
    rows: list[dict[str, str]],
    append: bool = False,
) -> None:
    """Ghi hoặc cập nhật file CSV ánh xạ ảnh sạch - ảnh nhiễu."""
    map_path.parent.mkdir(parents=True, exist_ok=True)

    merged_rows: list[dict[str, str]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    if append and map_path.is_file():
        with open(map_path, "r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for existing in reader:
                key = (
                    existing.get("image", ""),
                    existing.get("noise_mode", ""),
                    existing.get("noisy_path", ""),
                )
                if key not in seen_keys:
                    seen_keys.add(key)
                    merged_rows.append(
                        {field: existing.get(field, "") for field in MAP_FIELDNAMES}
                    )

    for row in rows:
        key = (row["image"], row["noise_mode"], row["noisy_path"])
        if key in seen_keys:
            merged_rows = [
                r
                for r in merged_rows
                if (r["image"], r["noise_mode"], r["noisy_path"]) != key
            ]
        seen_keys.add(key)
        merged_rows.append(row)

    with open(map_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MAP_FIELDNAMES, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(merged_rows)


def _process_single_image(
    image_name: str,
    img_dir: Path,
    target_dir: Path,
    noise_mode: str,
    fixed_sigma: float | None,
    low_sigma: float,
    high_sigma: float,
    rng: np.random.Generator,
    resize_size: tuple[int, int] | None,
    sync_default_dir: bool,
) -> dict[str, str]:
    """Sinh một ảnh nhiễu AWGN, đồng bộ sang NoisyImages/ nếu cần và trả về dòng CSV ánh xạ."""
    clean_path = img_dir / image_name
    noisy_path = target_dir / image_name

    sigma = (
        float(rng.uniform(low_sigma, high_sigma))
        if noise_mode == "random"
        else float(fixed_sigma)  # type: ignore[arg-type]
    )
    generate_noisy_image(
        clean_path,
        noisy_path,
        sigma=sigma,
        rng=rng,
        resize_size=resize_size,
    )

    if sync_default_dir and target_dir.resolve() != NOISY_IMAGES_DIR.resolve():
        _sync_file_to_default_dir(noisy_path, NOISY_IMAGES_DIR / image_name)

    return {
        "image": image_name,
        "clean_path": _to_project_path_str(clean_path),
        "noisy_path": _to_project_path_str(noisy_path),
        "sigma": f"{sigma:.4f}" if noise_mode == "random" else _format_level_label(sigma),
        "noise_mode": noise_mode,
    }


def generate_noisy_dataset(
    noise_level: int | float | str = 25,
    captions_path: str | Path = CAPTIONS_PATH,
    images_dir: str | Path = IMAGES_DIR,
    output_dir: str | Path | None = None,
    map_path: str | Path = NOISY_IMAGE_MAP_PATH,
    random_range: tuple[float, float] = DEFAULT_RANDOM_RANGE,
    seed: int = 42,
    sync_default_dir: bool = False,
    append_map: bool = False,
    log_every: int = 500,
    noiseL: int | float | str | None = None,
    num_workers: int = 1,
    resize_size: tuple[int, int] | None = None,
) -> Path:
    """
    Sinh toàn bộ tập ảnh nhiễu và xuất file noisy_image_map.csv.

    Args:
        noise_level     : Mức nhiễu cố định (ví dụ 15, 25, 50) hoặc "random" trong [0, 55].
        captions_path   : Đường dẫn tới captions.csv gốc.
        images_dir      : Thư mục chứa ảnh sạch gốc.
        output_dir      : Thư mục lưu ảnh nhiễu (mặc định NoisyImages_sigma{level}).
        map_path        : Đường dẫn xuất file CSV ánh xạ (noisy_image_map.csv).
        random_range    : Khoảng (min_sigma, max_sigma) dùng khi noise_level='random'.
        seed            : Seed cố định để tái lập kết quả sinh nhiễu.
        sync_default_dir: Nếu True, tạo liên kết/bản sao vào data/processed/NoisyImages/.
        append_map      : Nếu True, giữ các dòng cũ trong noisy_image_map.csv và cập nhật thêm.
        log_every       : Tần suất in tiến độ theo số ảnh.
        noiseL          : Tên gọi khác của `noise_level` để khớp với bản thiết kế ban đầu.
        num_workers     : Số luồng chạy song song (>1 dùng SeedSequence riêng cho từng ảnh).
        resize_size     : Nếu truyền (width, height), resize ảnh sạch trước khi cộng nhiễu.
    Returns:
        Đường dẫn tới file CSV ánh xạ vừa xuất.
    """
    effective_level = noiseL if noiseL is not None else noise_level
    noise_mode, fixed_sigma, suffix = _parse_noise_level(effective_level)
    low_sigma, high_sigma = random_range
    if noise_mode == "random" and (low_sigma < 0 or high_sigma < low_sigma):
        raise ValueError(f"Invalid random_range: {random_range}")
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got: {num_workers}")

    img_dir = Path(images_dir)
    target_dir = Path(output_dir) if output_dir is not None else get_noisy_images_dir(suffix)
    target_dir.mkdir(parents=True, exist_ok=True)
    if sync_default_dir:
        NOISY_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    csv_map_path = Path(map_path)

    image_names = read_image_list_from_captions(captions_path=captions_path, images_dir=img_dir)
    total = len(image_names)
    print(
        f"Generating noisy images | mode={noise_mode} | level={suffix} | "
        f"images={total} | workers={num_workers} | resize={resize_size} | output={target_dir}"
    )

    rows: list[dict[str, str]] = []
    if num_workers == 1:
        rng = np.random.default_rng(seed)
        for idx, image_name in enumerate(image_names, start=1):
            rows.append(
                _process_single_image(
                    image_name=image_name,
                    img_dir=img_dir,
                    target_dir=target_dir,
                    noise_mode=noise_mode,
                    fixed_sigma=fixed_sigma,
                    low_sigma=low_sigma,
                    high_sigma=high_sigma,
                    rng=rng,
                    resize_size=resize_size,
                    sync_default_dir=sync_default_dir,
                )
            )
            if log_every > 0 and (idx % log_every == 0 or idx == total):
                print(f"  Processed {idx}/{total} images...")
    else:
        child_seeds = np.random.SeedSequence(seed).spawn(total)

        def _worker(task: tuple[str, np.random.SeedSequence]) -> dict[str, str]:
            name, child_seed = task
            return _process_single_image(
                image_name=name,
                img_dir=img_dir,
                target_dir=target_dir,
                noise_mode=noise_mode,
                fixed_sigma=fixed_sigma,
                low_sigma=low_sigma,
                high_sigma=high_sigma,
                rng=np.random.default_rng(child_seed),
                resize_size=resize_size,
                sync_default_dir=sync_default_dir,
            )

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for idx, row in enumerate(executor.map(_worker, zip(image_names, child_seeds)), start=1):
                rows.append(row)
                if log_every > 0 and (idx % log_every == 0 or idx == total):
                    print(f"  Processed {idx}/{total} images...")

    _write_noise_map(csv_map_path, rows, append=append_map)
    print(f"Completed! Saved mapping file to: {csv_map_path}")
    return csv_map_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate AWGN noisy images for the Noisy Image Captioning project."
    )
    parser.add_argument(
        "--noise_level",
        "--noiseL",
        dest="noise_level",
        default="25",
        help="Sigma noise level (e.g., 15, 25, 50) or 'random' for range [0, 55].",
    )
    parser.add_argument(
        "--all_levels",
        action="store_true",
        help="Generate all standard levels sequentially: sigma = 15, 25, 50, and random.",
    )
    parser.add_argument("--captions_path", type=Path, default=CAPTIONS_PATH)
    parser.add_argument("--images_dir", type=Path, default=IMAGES_DIR)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--map_path", type=Path, default=NOISY_IMAGE_MAP_PATH)
    parser.add_argument("--min_sigma", type=float, default=DEFAULT_RANDOM_RANGE[0])
    parser.add_argument("--max_sigma", type=float, default=DEFAULT_RANDOM_RANGE[1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker threads (>1 uses per-image SeedSequence).",
    )
    parser.add_argument(
        "--resize_first",
        action="store_true",
        help="Resize clean images to (224, 224) before adding AWGN to preserve noise variance.",
    )
    parser.add_argument(
        "--sync_default",
        action="store_true",
        help="Also link/copy noisy images into data/processed/NoisyImages/.",
    )
    parser.add_argument(
        "--append_map",
        action="store_true",
        help="Merge rows into existing noisy_image_map.csv instead of overwriting.",
    )
    args = parser.parse_args()
    resize_target = DEFAULT_RESIZE_SIZE if args.resize_first else None

    if args.all_levels:
        levels: list[int | str] = [*DEFAULT_FIXED_LEVELS, "random"]
        for i, level in enumerate(levels):
            generate_noisy_dataset(
                noise_level=level,
                captions_path=args.captions_path,
                images_dir=args.images_dir,
                output_dir=None,
                map_path=args.map_path,
                random_range=(args.min_sigma, args.max_sigma),
                seed=args.seed + i,
                sync_default_dir=(args.sync_default and str(level) == "25"),
                append_map=(i > 0 or args.append_map),
                num_workers=args.num_workers,
                resize_size=resize_target,
            )
    else:
        generate_noisy_dataset(
            noise_level=args.noise_level,
            captions_path=args.captions_path,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            map_path=args.map_path,
            random_range=(args.min_sigma, args.max_sigma),
            seed=args.seed,
            sync_default_dir=args.sync_default,
            append_map=args.append_map,
            num_workers=args.num_workers,
            resize_size=resize_target,
        )


if __name__ == "__main__":
    main()
