"""Tạo ảnh nhiễu Gaussian và bảng ánh xạ giữa ảnh sạch với ảnh nhiễu."""

from __future__ import annotations

import csv
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from project_paths import (
    CAPTIONS_PATH,
    IMAGES_DIR,
    NOISY_IMAGES_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    get_noisy_image_map_path,
)

DEFAULT_RANDOM_RANGE = (0.0, 55.0)
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAP_FIELDNAMES = ["image", "clean_path", "noisy_path", "sigma", "noise_mode"]


def _format_level_label(level: float) -> str:
    return str(int(level)) if level.is_integer() else f"{level:g}"


def _parse_noise_level(
    noise_level: int | float | str,
) -> tuple[str, float | None, str]:
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
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"noise_level (sigma) must be finite and >= 0, got: {value}")
    return "fixed", value, _format_level_label(value)


def _to_project_path_str(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def read_image_list_from_captions(
    captions_path: str | Path = CAPTIONS_PATH,
    images_dir: str | Path = IMAGES_DIR,
) -> list[str]:
    """Trả về tên ảnh duy nhất trong CSV caption và có file thực tế trên đĩa."""
    captions_file = Path(captions_path)
    image_dir = Path(images_dir)
    if not captions_file.is_file():
        raise FileNotFoundError(f"Captions file not found: {captions_file}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Clean images directory not found: {image_dir}")

    names: list[str] = []
    seen: set[str] = set()
    with captions_file.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or "image" not in reader.fieldnames:
            raise ValueError(f"File {captions_file} is missing the 'image' column.")
        for row in reader:
            name = (row.get("image") or "").strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)

    existing = [name for name in names if (image_dir / name).is_file()]
    missing_count = len(names) - len(existing)
    if missing_count:
        print(
            f"Warning: {missing_count}/{len(names)} images are missing from "
            f"{image_dir}; processing {len(existing)} available images.",
            flush=True,
        )
    if not existing:
        raise RuntimeError(
            f"No valid images found in {image_dir} matching {captions_file}."
        )
    return existing


def add_awgn_to_array(
    clean_array: np.ndarray,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Thêm nhiễu Gaussian vào mảng ảnh có giá trị pixel trong [0, 255]."""
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError(f"Standard deviation sigma must be finite and >= 0, got: {sigma}")
    clean_float = np.asarray(clean_array, dtype=np.float32)
    if not np.isfinite(clean_float).all():
        raise ValueError("Image array must contain only finite pixel values.")
    if sigma == 0:
        return np.clip(np.rint(clean_float), 0, 255).astype(np.uint8)

    generator = rng if rng is not None else np.random.default_rng()
    noise = generator.normal(0.0, float(sigma), size=clean_float.shape).astype(
        np.float32
    )
    return np.clip(np.rint(clean_float + noise), 0, 255).astype(np.uint8)


def add_gaussian_noise(
    image: Image.Image,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> Image.Image:
    """Thêm nhiễu Gaussian trắng cộng tính (AWGN) vào ảnh RGB dạng PIL."""
    noisy_array = add_awgn_to_array(np.asarray(image.convert("RGB")), sigma, rng)
    return Image.fromarray(noisy_array, mode="RGB")


def generate_noisy_image(
    clean_image_path: str | Path,
    output_image_path: str | Path,
    sigma: float,
    rng: np.random.Generator | None = None,
    jpeg_quality: int = 95,
    resize_size: tuple[int, int] | None = None,
) -> Path:
    """Tạo và lưu một ảnh nhiễu, có thể đổi kích thước ảnh trước khi thêm nhiễu."""
    source = Path(clean_image_path)
    destination = Path(output_image_path)
    if not source.is_file():
        raise FileNotFoundError(f"Clean image not found: {source}")
    if source.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {source.suffix}")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100.")
    if resize_size is not None and (
        len(resize_size) != 2 or any(size < 1 for size in resize_size)
    ):
        raise ValueError("resize_size must contain two positive dimensions.")

    with Image.open(source) as image:
        rgb_image = image.convert("RGB")
        if resize_size is not None:
            rgb_image = rgb_image.resize(
                resize_size, Image.Resampling.BILINEAR
            )
        clean_array = np.asarray(rgb_image)
    noisy_image = Image.fromarray(add_awgn_to_array(clean_array, sigma, rng), mode="RGB")

    destination.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = (
        {"quality": jpeg_quality, "subsampling": 0}
        if destination.suffix.lower() in {".jpg", ".jpeg"}
        else {}
    )
    noisy_image.save(destination, **save_kwargs)
    return destination


def _sync_file_to_default_dir(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _write_noise_map(
    map_path: Path,
    rows: list[dict[str, str]],
    append: bool = False,
) -> None:
    map_path.parent.mkdir(parents=True, exist_ok=True)
    merged: list[dict[str, str]] = []
    if append and map_path.is_file():
        with map_path.open("r", encoding="utf-8-sig", newline="") as file:
            for existing in csv.DictReader(file):
                merged.append(
                    {field: existing.get(field, "") or "" for field in MAP_FIELDNAMES}
                )

    by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in merged + rows:
        key = (row["image"], row["noise_mode"], row["noisy_path"])
        by_key[key] = row

    with map_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=MAP_FIELDNAMES, quoting=csv.QUOTE_ALL
        )
        writer.writeheader()
        writer.writerows(by_key.values())


def _process_single_image(
    image_name: str,
    images_dir: Path,
    output_dir: Path,
    noise_mode: str,
    fixed_sigma: float | None,
    random_range: tuple[float, float],
    rng: np.random.Generator,
    resize_size: tuple[int, int] | None,
    jpeg_quality: int,
    sync_default_dir: bool,
) -> dict[str, str]:
    clean_path = images_dir / image_name
    noisy_path = output_dir / image_name
    if noise_mode == "random":
        sigma = float(rng.uniform(*random_range))
    else:
        if fixed_sigma is None:
            raise RuntimeError("Fixed noise mode requires a sigma value.")
        sigma = fixed_sigma

    generate_noisy_image(
        clean_path,
        noisy_path,
        sigma=sigma,
        rng=rng,
        jpeg_quality=jpeg_quality,
        resize_size=resize_size,
    )
    if sync_default_dir and output_dir.resolve() != NOISY_IMAGES_DIR.resolve():
        _sync_file_to_default_dir(noisy_path, NOISY_IMAGES_DIR / image_name)

    return {
        "image": image_name,
        "clean_path": _to_project_path_str(clean_path),
        "noisy_path": _to_project_path_str(noisy_path),
        "sigma": f"{sigma:.4f}" if noise_mode == "random" else _format_level_label(sigma),
        "noise_mode": noise_mode,
    }


def generate_noisy_dataset(
    captions_csv: str | Path = CAPTIONS_PATH,
    clean_images_dir: str | Path = IMAGES_DIR,
    noise_level: int | float | str = 25,
    seed: int = 42,
    *,
    captions_path: str | Path | None = None,
    images_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    map_path: str | Path | None = None,
    random_range: tuple[float, float] = DEFAULT_RANDOM_RANGE,
    sync_default_dir: bool = False,
    append_map: bool = False,
    log_every: int = 100,
    noiseL: int | float | str | None = None,
    num_workers: int = 1,
    resize_size: tuple[int, int] | None = None,
    jpeg_quality: int = 95,
) -> Path:
    """Tạo các bản ảnh nhiễu và trả về thư mục ảnh (API được main.py sử dụng)."""
    effective_level = noiseL if noiseL is not None else noise_level
    mode, fixed_sigma, suffix = _parse_noise_level(effective_level)
    low_sigma, high_sigma = random_range
    if (
        not np.isfinite(low_sigma)
        or not np.isfinite(high_sigma)
        or low_sigma < 0
        or high_sigma < low_sigma
    ):
        raise ValueError(f"Invalid random_range: {random_range}")
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got: {num_workers}")
    if log_every < 0:
        raise ValueError("log_every must be >= 0.")

    source_csv = Path(captions_path or captions_csv)
    source_images = Path(images_dir or clean_images_dir)
    target_dir = (
        Path(output_dir)
        if output_dir is not None
        else PROCESSED_DATA_DIR / f"NoisyImages_sigma{suffix}"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    if sync_default_dir:
        NOISY_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    image_names = read_image_list_from_captions(source_csv, source_images)
    total = len(image_names)
    print(
        f"[NoiseGenerator] Bắt đầu tạo ảnh nhiễu: mode={mode}, "
        f"sigma={suffix}, ảnh={total}, workers={num_workers}, "
        f"output={target_dir}",
        flush=True,
    )
    rows: list[dict[str, str]] = []
    if num_workers == 1:
        rng = np.random.default_rng(seed)
        for index, image_name in enumerate(image_names, start=1):
            rows.append(
                _process_single_image(
                    image_name,
                    source_images,
                    target_dir,
                    mode,
                    fixed_sigma,
                    random_range,
                    rng,
                    resize_size,
                    jpeg_quality,
                    sync_default_dir,
                )
            )
            if log_every and (index % log_every == 0 or index == total):
                print(
                    f"[NoiseGenerator] Tiến độ: {index}/{total} ảnh",
                    flush=True,
                )
    else:
        seeds = np.random.SeedSequence(seed).spawn(total)

        def process(task: tuple[str, np.random.SeedSequence]) -> dict[str, str]:
            image_name, child_seed = task
            return _process_single_image(
                image_name,
                source_images,
                target_dir,
                mode,
                fixed_sigma,
                random_range,
                np.random.default_rng(child_seed),
                resize_size,
                jpeg_quality,
                sync_default_dir,
            )

        tasks = zip(image_names, seeds)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for index, row in enumerate(executor.map(process, tasks), start=1):
                rows.append(row)
                if log_every and (index % log_every == 0 or index == total):
                    print(
                        f"[NoiseGenerator] Tiến độ: {index}/{total} ảnh",
                        flush=True,
                    )

    csv_map_path = (
        Path(map_path) if map_path is not None else get_noisy_image_map_path(suffix)
    )
    _write_noise_map(csv_map_path, rows, append=append_map)
    print(
        f"[NoiseGenerator] Đã tạo {len(rows)} ảnh nhiễu tại: {target_dir}; "
        f"map: {csv_map_path}",
        flush=True,
    )
    return target_dir
