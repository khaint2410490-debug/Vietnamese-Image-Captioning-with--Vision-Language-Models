"""Tạo ảnh nhiễu (AWGN, mờ, xước, nếp gấp, sáng tối) và file ánh xạ noisy_image_map.csv."""

from __future__ import annotations

import csv
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from project_paths import (
    CAPTIONS_PATH,
    IMAGES_DIR,
    NOISY_IMAGE_MAP_PATH,
    NOISY_IMAGES_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
)

ImageInput = Image.Image | np.ndarray | torch.Tensor

DEFAULT_RANDOM_RANGE = (0.0, 55.0)
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAP_FIELDNAMES = ["image", "clean_path", "noisy_path", "sigma", "noise_mode"]
SUPPORTED_DEGRADATION_MODES = (
    "clean",
    "none",
    "gaussian",
    "awgn",
    "blur",
    "gaussian_blur",
    "motion_blur",
    "scratch",
    "scratches",
    "fold",
    "fold_line",
    "contrast",
    "brightness",
    "contrast_brightness",
    "combined",
)


def _resolve_rng(
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.random.Generator:
    if rng is not None and seed is not None:
        raise ValueError("Specify either 'rng' or 'seed', not both.")
    if rng is not None:
        return rng
    return np.random.default_rng(seed) if seed is not None else np.random.default_rng()


def _check_range(
    bounds: tuple[float, float],
    name: str,
    low_min: float | None = None,
    high_max: float | None = None,
    *,
    integer: bool = False,
) -> tuple[Any, Any]:
    if len(bounds) != 2:
        raise ValueError(f"{name} must be a 2-element (low, high) tuple, got: {bounds!r}")
    low, high = bounds
    if integer:
        if (
            isinstance(low, bool)
            or isinstance(high, bool)
            or not isinstance(low, (int, np.integer))
            or not isinstance(high, (int, np.integer))
        ):
            raise ValueError(f"{name} bounds must be integers, got: {bounds!r}")
        low_v: Any = int(low)
        high_v: Any = int(high)
    else:
        low_v = float(low)
        high_v = float(high)
        if not np.isfinite(low_v) or not np.isfinite(high_v):
            raise ValueError(f"{name} bounds must be finite, got: {bounds!r}")

    if low_v > high_v:
        raise ValueError(f"{name} must satisfy low <= high, got: {bounds!r}")
    if low_min is not None and low_v < low_min:
        raise ValueError(f"{name} lower bound must be >= {low_min}, got: {low_v}")
    if high_max is not None and high_v > high_max:
        raise ValueError(f"{name} upper bound must be <= {high_max}, got: {high_v}")
    return low_v, high_v


def _check_odd_kernel(kernel_size: int) -> int:
    if (
        isinstance(kernel_size, bool)
        or not isinstance(kernel_size, (int, np.integer))
        or kernel_size < 1
        or kernel_size % 2 == 0
    ):
        raise ValueError(f"kernel_size must be a positive odd integer, got: {kernel_size!r}")
    return int(kernel_size)


def _to_hwc_frames(image: ImageInput) -> tuple[list[np.ndarray], dict[str, Any]]:
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
        if arr.size == 0:
            raise ValueError("Input PIL image must not be empty.")
        return [arr], {"kind": "pil"}

    if isinstance(image, np.ndarray):
        if image.size == 0:
            raise ValueError("Input numpy array must not be empty.")
        if image.dtype == np.bool_:
            raise ValueError("Boolean numpy arrays are not supported as images.")
        if image.ndim not in (2, 3):
            raise ValueError(
                f"Expected 2D (H, W) or 3D (H, W, C) numpy array, got shape {image.shape}."
            )
        arr = np.asarray(image, dtype=np.float32)
        if not np.isfinite(arr).all():
            raise ValueError("Image array must contain only finite pixel values.")
        was_2d = arr.ndim == 2
        if was_2d:
            arr = arr[..., np.newaxis]
        return [arr.copy()], {"kind": "numpy", "orig_dtype": image.dtype, "was_2d": was_2d}

    if isinstance(image, torch.Tensor):
        if image.numel() == 0:
            raise ValueError("Input tensor must not be empty.")
        if not image.is_floating_point():
            raise ValueError(
                f"Expected a floating-point torch.Tensor in [0.0, 1.0], got dtype {image.dtype}."
            )
        if image.ndim not in (3, 4):
            raise ValueError(
                f"Expected 3D (C, H, W) or 4D (B, C, H, W) tensor, got shape {tuple(image.shape)}."
            )
        if not torch.isfinite(image).all():
            raise ValueError("Input tensor must contain only finite values.")

        was_3d = image.ndim == 3
        batch = image.detach().to(device="cpu", dtype=torch.float32)
        if was_3d:
            batch = batch.unsqueeze(0)
        # Nhân 255 để dùng chung các tham số nhiễu viết theo thang [0, 255]
        batch_hwc = batch.permute(0, 2, 3, 1).contiguous().numpy() * 255.0
        frames = [batch_hwc[i].copy() for i in range(batch_hwc.shape[0])]
        return frames, {
            "kind": "tensor",
            "orig_device": image.device,
            "orig_dtype": image.dtype,
            "was_3d": was_3d,
        }

    raise TypeError(
        f"Unsupported image type {type(image).__name__}; expected PIL.Image.Image, "
        "numpy.ndarray, or torch.Tensor."
    )


def _from_hwc_frames(frames: list[np.ndarray], meta: dict[str, Any]) -> ImageInput:
    kind = meta["kind"]
    if kind == "pil":
        return Image.fromarray(np.clip(np.rint(frames[0]), 0.0, 255.0).astype(np.uint8), mode="RGB")

    if kind == "numpy":
        arr = np.clip(frames[0], 0.0, 255.0)
        orig_dtype = meta["orig_dtype"]
        if np.issubdtype(orig_dtype, np.integer):
            info = np.iinfo(orig_dtype)
            arr = np.clip(np.rint(arr), max(0, info.min), min(255, info.max)).astype(orig_dtype)
        else:
            arr = arr.astype(orig_dtype)
        return arr[..., 0] if meta["was_2d"] else arr

    if kind == "tensor":
        chw_list = [np.transpose(np.clip(f, 0.0, 255.0) / 255.0, (2, 0, 1)) for f in frames]
        raw = chw_list[0] if meta["was_3d"] else np.stack(chw_list, axis=0)
        out = torch.clamp(torch.from_numpy(np.ascontiguousarray(raw)), 0.0, 1.0)
        return out.to(device=meta["orig_device"], dtype=meta["orig_dtype"])

    raise RuntimeError(f"Unknown internal frame metadata kind: {kind!r}")


def _apply_to_image(image: ImageInput, transform_fn: Callable[[np.ndarray], np.ndarray]) -> ImageInput:
    frames, meta = _to_hwc_frames(image)
    return _from_hwc_frames([np.clip(transform_fn(f), 0.0, 255.0) for f in frames], meta)


def _convolve_2d_hwc(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    if kh == 1 and kw == 1:
        return np.clip(arr * float(kernel[0, 0]), 0.0, 255.0).astype(np.float32)

    height, width, channels = arr.shape
    pad_h, pad_w = kh // 2, kw // 2
    # Lật kernel hai trục để đúng phép tích chập (F.conv2d mặc định tính tương quan chéo)
    weight = torch.from_numpy(np.ascontiguousarray(kernel[::-1, ::-1], dtype=np.float32))
    weight = weight.view(1, 1, kh, kw).repeat(channels, 1, 1, 1)

    tensor = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
    # Ảnh nhỏ hơn phần đệm sẽ gây lỗi ở mode 'reflect', nên tự lùi về 'replicate'
    pad_mode = "reflect" if (height > pad_h and width > pad_w) else "replicate"
    padded = F.pad(tensor, (pad_w, pad_w, pad_h, pad_h), mode=pad_mode)
    out = F.conv2d(padded, weight, groups=channels).squeeze(0).permute(1, 2, 0).contiguous().numpy()
    return np.clip(out, 0.0, 255.0).astype(np.float32)


def _gaussian_noise_frame(frame: np.ndarray, sigma: float, generator: np.random.Generator) -> np.ndarray:
    if sigma == 0:
        return np.clip(frame, 0.0, 255.0).astype(np.float32)
    noise = generator.normal(0.0, float(sigma), size=frame.shape).astype(np.float32)
    return np.clip(frame + noise, 0.0, 255.0).astype(np.float32)


def _gaussian_blur_frame(frame: np.ndarray, kernel_size: int, sigma: float | None) -> np.ndarray:
    if kernel_size == 1:
        return np.clip(frame, 0.0, 255.0).astype(np.float32)
    eff_sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1.0) + 0.8 if sigma is None else float(sigma)
    radius = kernel_size // 2
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * (eff_sigma**2))).astype(np.float32)
    kernel /= float(kernel.sum())
    return _convolve_2d_hwc(frame, kernel)


def _motion_blur_frame(
    frame: np.ndarray,
    kernel_size: int,
    angle: float | None,
    generator: np.random.Generator,
) -> np.ndarray:
    if kernel_size == 1:
        return np.clip(frame, 0.0, 255.0).astype(np.float32)
    eff_angle = float(generator.uniform(0.0, 360.0)) if angle is None else float(angle)
    base_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    base_kernel[kernel_size // 2, :] = 1.0 / float(kernel_size)

    rotated = Image.fromarray(base_kernel, mode="F").rotate(
        eff_angle, resample=Image.Resampling.BILINEAR, expand=False
    )
    kernel = np.clip(np.asarray(rotated, dtype=np.float32), 0.0, None)
    total = float(kernel.sum())
    kernel = (kernel / total).astype(np.float32) if total > 1e-6 else base_kernel
    return _convolve_2d_hwc(frame, kernel)


def _random_scratches_frame(
    frame: np.ndarray,
    num_range: tuple[int, int],
    alpha_range: tuple[float, float],
    width_range: tuple[int, int],
    color_range: tuple[int, int],
    generator: np.random.Generator,
) -> np.ndarray:
    height, width, _ = frame.shape
    low_n, high_n = num_range
    count = low_n if low_n == high_n else int(generator.integers(low_n, high_n + 1))
    if count <= 0:
        return np.clip(frame, 0.0, 255.0).astype(np.float32)

    result = frame.copy()
    for _ in range(count):
        pts = [
            (int(generator.integers(0, width)), int(generator.integers(0, height))),
            (int(generator.integers(0, width)), int(generator.integers(0, height))),
        ]
        thickness = (
            width_range[0]
            if width_range[0] == width_range[1]
            else int(generator.integers(width_range[0], width_range[1] + 1))
        )
        alpha = (
            alpha_range[0]
            if alpha_range[0] == alpha_range[1]
            else float(generator.uniform(*alpha_range))
        )
        color = float(
            color_range[0]
            if color_range[0] == color_range[1]
            else generator.uniform(*color_range)
        )

        mask_img = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask_img).line(pts, fill=255, width=thickness)
        blend = alpha * (np.asarray(mask_img, dtype=np.float32) / 255.0)[..., np.newaxis]
        result = blend * color + (1.0 - blend) * result
    return np.clip(result, 0.0, 255.0).astype(np.float32)


def _fold_line_frame(
    frame: np.ndarray,
    orientation: str | None,
    band_width: int | None,
    position: int | None,
    dark_range: tuple[float, float],
    bright_range: tuple[float, float],
    generator: np.random.Generator,
) -> np.ndarray:
    height, width, _ = frame.shape
    eff_orientation = (
        ("vertical" if int(generator.integers(0, 2)) == 0 else "horizontal")
        if orientation is None
        else orientation
    )
    axis_len = width if eff_orientation == "vertical" else height
    eff_width = min(int(generator.integers(6, 11)) if band_width is None else int(band_width), axis_len)

    dark_w = eff_width // 2
    bright_w = eff_width - dark_w
    parts: list[np.ndarray] = []
    if dark_w > 0:
        parts.append(np.linspace(dark_range[1], dark_range[0], dark_w, dtype=np.float32))
    if bright_w > 0:
        parts.append(np.linspace(bright_range[1], bright_range[0], bright_w, dtype=np.float32))
    profile = np.concatenate(parts)
    if orientation is None and int(generator.integers(0, 2)) == 1:
        profile = profile[::-1].copy()

    max_start = max(0, axis_len - eff_width)
    if position is None:
        margin = int(0.1 * axis_len)
        low_pos = min(margin, max_start)
        high_pos = max(low_pos, max_start - margin)
        start = low_pos if low_pos == high_pos else int(generator.integers(low_pos, high_pos + 1))
    else:
        start = max(0, min(int(position) - eff_width // 2, max_start))

    result = frame.copy()
    if eff_orientation == "vertical":
        result[:, start : start + eff_width, :] *= profile[np.newaxis, :, np.newaxis]
    else:
        result[start : start + eff_width, :, :] *= profile[:, np.newaxis, np.newaxis]
    return np.clip(result, 0.0, 255.0).astype(np.float32)


def _contrast_brightness_frame(
    frame: np.ndarray,
    alpha: float | None,
    beta: float | None,
    alpha_range: tuple[float, float],
    beta_range: tuple[float, float],
    generator: np.random.Generator,
) -> np.ndarray:
    eff_alpha = float(generator.uniform(*alpha_range)) if alpha is None else float(alpha)
    eff_beta = float(generator.uniform(*beta_range)) if beta is None else float(beta)
    return np.clip(eff_alpha * frame + eff_beta, 0.0, 255.0).astype(np.float32)


def add_awgn_to_array(
    clean_array: np.ndarray,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Thêm nhiễu Gaussian vào mảng ảnh có giá trị pixel trong [0, 255]."""
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError(f"Standard deviation sigma must be finite and >= 0, got: {sigma}")
    raw_array = np.asarray(clean_array)
    if raw_array.dtype == np.bool_:
        raise ValueError("Boolean numpy arrays are not supported as images.")
    clean_float = raw_array.astype(np.float32)
    if clean_float.size == 0:
        raise ValueError("Image array must not be empty.")
    if not np.isfinite(clean_float).all():
        raise ValueError("Image array must contain only finite pixel values.")
    if sigma == 0:
        return np.clip(np.rint(clean_float), 0, 255).astype(np.uint8)

    generator = rng if rng is not None else np.random.default_rng()
    noise = generator.normal(0.0, float(sigma), size=clean_float.shape).astype(np.float32)
    return np.clip(np.rint(clean_float + noise), 0, 255).astype(np.uint8)


def add_gaussian_noise(
    image: ImageInput,
    sigma: float = 25.0,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> ImageInput:
    """Thêm nhiễu Gaussian (AWGN) với độ lệch chuẩn sigma cho ảnh PIL, NumPy hoặc Tensor."""
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError(f"Standard deviation sigma must be finite and >= 0, got: {sigma}")
    generator = _resolve_rng(rng, seed)
    return _apply_to_image(image, lambda f: _gaussian_noise_frame(f, float(sigma), generator))


def add_gaussian_blur(
    image: ImageInput,
    kernel_size: int = 5,
    sigma: float | None = None,
) -> ImageInput:
    """Làm mờ ảnh bằng bộ lọc Gaussian 2D kích thước lẻ kernel_size x kernel_size."""
    k_size = _check_odd_kernel(kernel_size)
    if sigma is not None and (not np.isfinite(sigma) or sigma <= 0):
        raise ValueError(f"Blur sigma must be finite and > 0, got: {sigma}")
    return _apply_to_image(image, lambda f: _gaussian_blur_frame(f, k_size, sigma))


def add_motion_blur(
    image: ImageInput,
    kernel_size: int = 9,
    angle: float | None = None,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> ImageInput:
    """Tạo mờ chuyển động theo góc angle (độ) với kernel kích thước kernel_size."""
    k_size = _check_odd_kernel(kernel_size)
    if angle is not None and not np.isfinite(angle):
        raise ValueError(f"Motion blur angle must be finite, got: {angle}")
    generator = _resolve_rng(rng, seed)
    return _apply_to_image(image, lambda f: _motion_blur_frame(f, k_size, angle, generator))


def add_random_scratches(
    image: ImageInput,
    num_scratches: int | tuple[int, int] = (3, 10),
    alpha_range: tuple[float, float] = (0.4, 0.8),
    width_range: tuple[int, int] = (1, 2),
    color_range: tuple[int, int] = (180, 255),
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> ImageInput:
    """Vẽ các vết xước ngẫu nhiên rồi pha trộn alpha blending lên ảnh."""
    if isinstance(num_scratches, tuple):
        num_bounds = _check_range(num_scratches, "num_scratches", low_min=0, integer=True)
    else:
        if (
            isinstance(num_scratches, bool)
            or not isinstance(num_scratches, (int, np.integer))
            or num_scratches < 0
        ):
            raise ValueError(f"num_scratches must be a non-negative integer, got: {num_scratches!r}")
        num_bounds = (int(num_scratches), int(num_scratches))

    eff_alpha = _check_range(alpha_range, "alpha_range", low_min=0.0, high_max=1.0)
    eff_width = _check_range(width_range, "width_range", low_min=1, integer=True)
    eff_color = _check_range(color_range, "color_range", low_min=0, high_max=255, integer=True)
    generator = _resolve_rng(rng, seed)

    return _apply_to_image(
        image,
        lambda f: _random_scratches_frame(f, num_bounds, eff_alpha, eff_width, eff_color, generator),
    )


def add_fold_line(
    image: ImageInput,
    orientation: str | None = None,
    band_width: int | None = None,
    position: int | None = None,
    dark_range: tuple[float, float] = (0.5, 0.9),
    bright_range: tuple[float, float] = (1.1, 1.4),
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> ImageInput:
    """Tạo vết gấp giấy gồm hai dải tối và sáng chạy song song theo chiều dọc hoặc ngang."""
    eff_orientation: str | None = None
    if orientation is not None:
        if not isinstance(orientation, str) or orientation.strip().lower() not in {"vertical", "horizontal"}:
            raise ValueError(
                f"orientation must be 'vertical', 'horizontal', or None, got: {orientation!r}"
            )
        eff_orientation = orientation.strip().lower()

    if band_width is not None and (
        isinstance(band_width, bool)
        or not isinstance(band_width, (int, np.integer))
        or band_width < 2
    ):
        raise ValueError(f"band_width must be an integer >= 2, got: {band_width!r}")
    if position is not None and (
        isinstance(position, bool) or not isinstance(position, (int, np.integer))
    ):
        raise ValueError(f"position must be an integer or None, got: {position!r}")

    eff_dark = _check_range(dark_range, "dark_range", low_min=0.0, high_max=1.0)
    if eff_dark[0] <= 0.0:
        raise ValueError(f"dark_range lower bound must be > 0.0, got: {eff_dark[0]}")
    eff_bright = _check_range(bright_range, "bright_range", low_min=1.0)
    generator = _resolve_rng(rng, seed)

    return _apply_to_image(
        image,
        lambda f: _fold_line_frame(
            f,
            eff_orientation,
            int(band_width) if band_width is not None else None,
            int(position) if position is not None else None,
            eff_dark,
            eff_bright,
            generator,
        ),
    )


def adjust_contrast_brightness(
    image: ImageInput,
    alpha: float | None = None,
    beta: float | None = None,
    alpha_range: tuple[float, float] = (0.5, 1.5),
    beta_range: tuple[float, float] = (-50.0, 50.0),
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> ImageInput:
    """Chỉnh độ tương phản và độ sáng tuyến tính theo công thức P_out = alpha * P_in + beta."""
    eff_alpha = _check_range(alpha_range, "alpha_range", low_min=0.0)
    eff_beta = _check_range(beta_range, "beta_range")
    if alpha is not None and (not np.isfinite(alpha) or alpha < 0):
        raise ValueError(f"alpha must be finite and >= 0, got: {alpha!r}")
    if beta is not None and not np.isfinite(beta):
        raise ValueError(f"beta must be finite, got: {beta!r}")
    generator = _resolve_rng(rng, seed)
    return _apply_to_image(
        image,
        lambda f: _contrast_brightness_frame(f, alpha, beta, eff_alpha, eff_beta, generator),
    )


def apply_degradation(
    image: ImageInput,
    mode: str = "gaussian",
    sigma: float | None = None,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    **kwargs: Any,
) -> ImageInput:
    """
    Hàm gọi chung cho các kiểu làm nhiễu và biến dạng ảnh.

    Các mode hỗ trợ: clean/none, gaussian/awgn, blur/gaussian_blur,
    motion_blur, scratch/scratches, fold/fold_line, contrast/brightness,
    contrast_brightness và combined.
    """
    if not isinstance(mode, str):
        raise ValueError(f"mode must be a string, got: {type(mode).__name__}")
    normalized_mode = mode.strip().lower()
    generator = _resolve_rng(rng, seed)

    if normalized_mode in {"clean", "none"}:
        if kwargs:
            raise TypeError(
                f"Unexpected keyword arguments for mode={normalized_mode!r}: "
                f"{', '.join(sorted(kwargs))}"
            )
        return _apply_to_image(image, lambda f: f.copy())
    if normalized_mode in {"gaussian", "awgn"}:
        return add_gaussian_noise(image, sigma=25.0 if sigma is None else sigma, rng=generator, **kwargs)
    if normalized_mode in {"blur", "gaussian_blur"}:
        return add_gaussian_blur(image, sigma=sigma, **kwargs)
    if normalized_mode == "motion_blur":
        return add_motion_blur(image, rng=generator, **kwargs)
    if normalized_mode in {"scratch", "scratches"}:
        return add_random_scratches(image, rng=generator, **kwargs)
    if normalized_mode in {"fold", "fold_line"}:
        return add_fold_line(image, rng=generator, **kwargs)
    if normalized_mode in {"contrast", "brightness", "contrast_brightness"}:
        return adjust_contrast_brightness(image, rng=generator, **kwargs)

    if normalized_mode == "combined":
        prob_defaults = {
            "gaussian_prob": 1.0,
            "scratch_prob": 0.4,
            "blur_prob": 0.3,
            "fold_prob": 0.15,
            "contrast_prob": 0.2,
        }
        probs: dict[str, float] = {}
        for key, default in prob_defaults.items():
            val = float(kwargs.pop(key, default))
            if not np.isfinite(val) or not 0.0 <= val <= 1.0:
                raise ValueError(f"{key} must be in [0.0, 1.0], got: {val!r}")
            probs[key] = val

        sigma_range = _check_range(kwargs.pop("sigma_range", (15.0, 50.0)), "sigma_range", low_min=0.0)
        if kwargs:
            raise TypeError(
                f"Unexpected keyword arguments for mode='combined': {', '.join(sorted(kwargs))}"
            )
        if sigma is not None and (not np.isfinite(sigma) or sigma < 0):
            raise ValueError(f"Standard deviation sigma must be finite and >= 0, got: {sigma}")

        def _combined_frame(frame: np.ndarray) -> np.ndarray:
            current = frame.copy()
            if float(generator.random()) < probs["contrast_prob"]:
                current = _contrast_brightness_frame(
                    current, None, None, (0.5, 1.5), (-50.0, 50.0), generator
                )
            if float(generator.random()) < probs["blur_prob"]:
                if int(generator.integers(0, 2)) == 0:
                    current = _gaussian_blur_frame(current, kernel_size=3, sigma=None)
                else:
                    current = _motion_blur_frame(current, kernel_size=5, angle=None, generator=generator)
            if float(generator.random()) < probs["fold_prob"]:
                current = _fold_line_frame(
                    current, None, None, None, (0.5, 0.9), (1.1, 1.4), generator
                )
            if float(generator.random()) < probs["scratch_prob"]:
                current = _random_scratches_frame(
                    current, (3, 10), (0.4, 0.8), (1, 2), (180, 255), generator
                )
            if float(generator.random()) < probs["gaussian_prob"]:
                eff_sigma = float(generator.uniform(*sigma_range)) if sigma is None else float(sigma)
                current = _gaussian_noise_frame(current, eff_sigma, generator)
            return current

        return _apply_to_image(image, _combined_frame)

    raise ValueError(
        f"Unsupported degradation mode {mode!r}. "
        f"Expected one of: {', '.join(SUPPORTED_DEGRADATION_MODES)}."
    )


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
            rgb_image = rgb_image.resize(resize_size, Image.Resampling.BILINEAR)
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
        f"[NoiseGenerator] Starting noisy dataset generation: mode={mode}, "
        f"sigma={suffix}, images={total}, workers={num_workers}, output={target_dir}",
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
                print(f"[NoiseGenerator] Progress: {index}/{total} images", flush=True)
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
                    print(f"[NoiseGenerator] Progress: {index}/{total} images", flush=True)

    csv_map_path = Path(map_path) if map_path is not None else NOISY_IMAGE_MAP_PATH
    _write_noise_map(csv_map_path, rows, append=append_map)
    print(
        f"[NoiseGenerator] Generated {len(rows)} noisy images in: {target_dir}; "
        f"mapping CSV: {csv_map_path}",
        flush=True,
    )
    return target_dir
