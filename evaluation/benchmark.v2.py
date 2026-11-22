"""So sánh baseline ảnh sạch, khử nhiễu bằng DnCNN và captioning trực tiếp ảnh nhiễu."""

import csv
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluation import corpus_bleu, tokenize_reference
from evaluation.generate_caption import load_model
from model.denoising.dncnn import DnCNN
from preprocessing.dataset import image_to_unit_tensor, normalize_imagenet_batch
from project_paths import (
    IMAGES_DIR,
    PROCESSED_DATA_DIR,
    TEST_CAPTIONS_PATH,
    VOCAB_PATH,
    WEIGHTS_PATH,
    get_dncnn_weights_path,
    get_end2end_weights_path,
)

NOISE_LEVEL = "25"
LIMIT = None
MAX_LENGTH = 20
DEVICE = None


def calculate_bleu_scores(
    reference_caption: str | Sequence[str], candidate_caption: str
) -> dict[str, float]:
    """Tính BLEU-1..4 với cách tách token tiếng Việt của dự án."""
    references = (
        [reference_caption]
        if isinstance(reference_caption, str)
        else list(reference_caption)
    )
    return corpus_bleu(
        [[tokenize_reference(caption) for caption in references]],
        [tokenize_reference(candidate_caption)],
    )


def _as_numpy(image: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    return np.asarray(image)


def _image_data_range(clean_img: np.ndarray, denoised_img: np.ndarray) -> float:
    minimum = min(float(clean_img.min()), float(denoised_img.min()))
    maximum = max(float(clean_img.max()), float(denoised_img.max()))
    if minimum < 0:
        raise ValueError("Ảnh phải có pixel trong miền [0, 1] hoặc [0, 255].")
    if np.issubdtype(clean_img.dtype, np.integer) or np.issubdtype(
        denoised_img.dtype, np.integer
    ):
        if maximum <= 255:
            return 255.0
        raise ValueError("Ảnh số nguyên phải có pixel trong miền [0, 255].")
    if maximum <= 1:
        return 1.0
    if maximum <= 255:
        return 255.0
    raise ValueError("Ảnh phải có pixel trong miền [0, 1] hoặc [0, 255].")


def calculate_image_quality(
    clean_img: np.ndarray | torch.Tensor,
    denoised_img: np.ndarray | torch.Tensor,
) -> tuple[float, float]:
    """Tính PSNR và SSIM cho hai ảnh cùng kích thước, có pixel trong [0,1] hoặc [0,255]."""
    clean = _as_numpy(clean_img)
    denoised = _as_numpy(denoised_img)
    if clean.shape != denoised.shape:
        raise ValueError(
            f"Kích thước ảnh không khớp: {clean.shape} và {denoised.shape}."
        )
    if clean.ndim not in (2, 3):
        raise ValueError("Ảnh phải có shape (H, W) hoặc (H, W, C).")
    if not np.isfinite(clean).all() or not np.isfinite(denoised).all():
        raise ValueError("Ảnh chứa giá trị NaN hoặc vô cực.")

    data_range = _image_data_range(clean, denoised)
    channel_axis = -1 if clean.ndim == 3 else None
    psnr_value = np.asarray(
        peak_signal_noise_ratio(clean, denoised, data_range=data_range)
    )
    ssim_value = np.asarray(
        structural_similarity(
            clean,
            denoised,
            data_range=data_range,
            channel_axis=channel_axis,
        )
    )
    if psnr_value.ndim != 0 or ssim_value.ndim != 0:
        raise ValueError("Chỉ chấp nhận kết quả metric dạng scalar.")
    return round(float(psnr_value.item()), 2), round(float(ssim_value.item()), 4)


def _read_test_samples(
    captions_path: Path,
    images_dir: Path,
    noisy_dir: Path,
    limit: int | None,
) -> list[tuple[str, Path, Path, list[str]]]:
    grouped: dict[str, list[str]] = {}
    with captions_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not {"image", "caption"} <= set(reader.fieldnames):
            raise ValueError("File test cần có header với hai cột: image,caption.")
        for row in reader:
            image_name = (row["image"] or "").strip()
            caption = (row["caption"] or "").strip()
            if image_name and caption:
                grouped.setdefault(image_name, []).append(caption)

    if not grouped:
        raise RuntimeError(f"Không có caption hợp lệ trong {captions_path}.")
    if limit is not None:
        if limit < 1:
            raise ValueError("LIMIT phải lớn hơn 0.")
        grouped = dict(list(grouped.items())[:limit])

    samples = []
    missing = []
    for image_name, captions in grouped.items():
        clean_path = images_dir / image_name
        noisy_path = noisy_dir / image_name
        if not clean_path.is_file() or not noisy_path.is_file():
            missing.append(image_name)
            continue
        samples.append((image_name, clean_path, noisy_path, captions))

    if missing:
        raise FileNotFoundError(
            f"Thiếu ảnh sạch hoặc ảnh nhiễu cho {len(missing)} ảnh; "
            f"ví dụ: {', '.join(missing[:5])}. Kiểm tra thư mục "
            f"{images_dir} và {noisy_dir}."
        )
    return samples


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_dncnn(weights_path: Path, device: torch.device) -> DnCNN:
    checkpoint = torch.load(weights_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint DnCNN không đúng định dạng: {weights_path}")
    model = DnCNN(num_of_layers=int(checkpoint.get("num_of_layers", 17))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _run_benchmark(
    captions_path: Path = TEST_CAPTIONS_PATH,
    images_dir: Path = IMAGES_DIR,
    noisy_dir: Path | None = None,
    baseline_weights: Path = WEIGHTS_PATH,
    dncnn_weights: Path | None = None,
    end2end_weights: Path | None = None,
    vocab_path: Path = VOCAB_PATH,
    output_path: Path = PROCESSED_DATA_DIR / "benchmark_comparison.csv",
    noise_level: str = "25",
    limit: int | None = None,
    max_length: int = 20,
    device: torch.device | None = None,
) -> list[dict]:
    noise_level = str(noise_level).lower()
    if noise_level not in {"15", "25", "50", "random"}:
        raise ValueError("noise_level phải là 15, 25, 50 hoặc random.")
    if limit is not None and limit < 1:
        raise ValueError("limit phải lớn hơn 0.")
    if max_length < 2:
        raise ValueError("max_length phải lớn hơn hoặc bằng 2.")

    captions_path = Path(captions_path)
    images_dir = Path(images_dir)
    baseline_weights = Path(baseline_weights)
    dncnn_weights = (
        Path(dncnn_weights)
        if dncnn_weights is not None
        else get_dncnn_weights_path(noise_level)
    )
    vocab_path = Path(vocab_path)
    output_path = Path(output_path)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noisy_dir = (
        Path(noisy_dir)
        if noisy_dir is not None
        else PROCESSED_DATA_DIR / f"NoisyImages_sigma{noise_level}"
    )
    default_end2end = get_end2end_weights_path(noise_level)
    requested_end2end = end2end_weights is not None
    end2end_weights = (
        Path(end2end_weights) if end2end_weights is not None else default_end2end
    )

    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA đã được chọn nhưng không khả dụng.")
    if images_dir.resolve() == noisy_dir.resolve():
        raise ValueError("images_dir và noisy_dir không được trùng nhau.")
    protected_paths = (
        captions_path,
        baseline_weights,
        dncnn_weights,
        end2end_weights,
        vocab_path,
    )
    if any(output_path.resolve() == path.resolve() for path in protected_paths):
        raise ValueError("output không được trùng với file dữ liệu hoặc checkpoint đầu vào.")
    resolved_output = output_path.resolve()
    for directory in (images_dir, noisy_dir):
        try:
            resolved_output.relative_to(directory.resolve())
        except ValueError:
            continue
        raise ValueError("output không được nằm trong thư mục ảnh đầu vào.")
    if output_path.exists() and output_path.is_dir():
        raise ValueError(f"output phải là đường dẫn file, không phải thư mục: {output_path}")

    for path, name in (
        (captions_path, "Tập test"),
        (baseline_weights, "Checkpoint baseline"),
        (dncnn_weights, "Checkpoint DnCNN"),
        (vocab_path, "Vocabulary"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} không tồn tại: {path}")
    if not noisy_dir.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục ảnh nhiễu: {noisy_dir}. "
            "Hãy chạy main.py để tạo dữ liệu nhiễu trước."
        )
    if requested_end2end and not end2end_weights.is_file():
        raise FileNotFoundError(f"Checkpoint end-to-end không tồn tại: {end2end_weights}")

    samples = _read_test_samples(captions_path, images_dir, noisy_dir, limit)
    if not samples:
        raise RuntimeError("Không còn ảnh hợp lệ để benchmark.")

    print(f"Thiết bị: {device}; số ảnh: {len(samples)}")
    baseline = load_model(
        weights_path=baseline_weights,
        vocab_path=str(vocab_path),
        device=device,
    )
    denoiser = _load_dncnn(dncnn_weights, device)
    end2end = (
        load_model(
            weights_path=end2end_weights,
            vocab_path=str(vocab_path),
            device=device,
        )
        if end2end_weights.is_file()
        else None
    )
    if end2end is None:
        print(
            f"Bỏ qua Pipeline 2: chưa có checkpoint {end2end_weights}. "
            "Hãy chạy main.py để train phương án 2 trước khi benchmark."
        )

    _, first_clean, first_noisy, _ = samples[0]
    baseline.generate_caption(first_clean, max_length=max_length)
    with Image.open(first_noisy) as image:
        first_noisy_tensor = image_to_unit_tensor(image).unsqueeze(0).to(device)
    denoised_warmup = denoiser.denoise(first_noisy_tensor)
    baseline.generate_caption(
        normalize_imagenet_batch(denoised_warmup[0]), max_length=max_length
    )
    if end2end is not None:
        end2end.generate_caption(
            normalize_imagenet_batch(first_noisy_tensor[0]), max_length=max_length
        )
    _synchronize(device)

    results = []
    references_by_image = []
    baseline_hypotheses, dncnn_hypotheses, end2end_hypotheses = [], [], []

    for image_name, clean_path, noisy_path, captions in samples:
        with Image.open(clean_path) as image:
            clean_tensor = image_to_unit_tensor(image)
        with Image.open(noisy_path) as image:
            noisy_tensor = image_to_unit_tensor(image).unsqueeze(0).to(device)

        _synchronize(device)
        started = time.perf_counter()
        baseline_tokens = baseline.generate_caption(clean_path, max_length=max_length)
        _synchronize(device)
        baseline_seconds = time.perf_counter() - started

        _synchronize(device)
        started = time.perf_counter()
        denoised_tensor = denoiser.denoise(noisy_tensor)
        dncnn_tokens = baseline.generate_caption(
            normalize_imagenet_batch(denoised_tensor[0]), max_length=max_length
        )
        _synchronize(device)
        dncnn_seconds = time.perf_counter() - started

        dncnn_psnr, dncnn_ssim = calculate_image_quality(
            clean_tensor.permute(1, 2, 0),
            denoised_tensor[0].permute(1, 2, 0),
        )

        end2end_tokens = None
        end2end_seconds = None
        if end2end is not None:
            _synchronize(device)
            started = time.perf_counter()
            end2end_tokens = end2end.generate_caption(
                normalize_imagenet_batch(noisy_tensor[0]), max_length=max_length
            )
            _synchronize(device)
            end2end_seconds = time.perf_counter() - started

        reference_tokens = [tokenize_reference(caption) for caption in captions]
        references_by_image.append(reference_tokens)
        baseline_hypotheses.append(baseline_tokens)
        dncnn_hypotheses.append(dncnn_tokens)
        if end2end_tokens is not None:
            end2end_hypotheses.append(end2end_tokens)

        row = {
            "Image_ID": image_name,
            "Reference_Captions": " || ".join(captions),
            "Base_Caption": baseline.tokens_to_text(baseline_tokens),
            "Base_Time_s": round(baseline_seconds, 4),
            "P1_Caption": baseline.tokens_to_text(dncnn_tokens),
            "P1_PSNR": dncnn_psnr,
            "P1_SSIM": dncnn_ssim,
            "P1_Time_s": round(dncnn_seconds, 4),
            "P2_Caption": (
                end2end.tokens_to_text(end2end_tokens)
                if end2end_tokens is not None and end2end is not None
                else ""
            ),
            "P2_Time_s": (
                round(end2end_seconds, 4) if end2end_seconds is not None else ""
            ),
        }
        for prefix, hypothesis in (
            ("Base", baseline_tokens),
            ("P1", dncnn_tokens),
        ):
            scores = calculate_bleu_scores(captions, baseline.tokens_to_text(hypothesis))
            row.update(
                {f"{prefix}_{name.replace('-', '')}": score for name, score in scores.items()}
            )
        if end2end_tokens is not None and end2end is not None:
            scores = calculate_bleu_scores(
                captions, end2end.tokens_to_text(end2end_tokens)
            )
            row.update(
                {f"P2_{name.replace('-', '')}": score for name, score in scores.items()}
            )
        else:
            row.update({f"P2_BLEU{number}": "" for number in range(1, 5)})
        results.append(row)

        if len(results) % 25 == 0:
            print(f"Đã xử lý {len(results)}/{len(samples)} ảnh.")

    fields = [
        "Image_ID",
        "Reference_Captions",
        "Base_Caption",
        "Base_BLEU1",
        "Base_BLEU2",
        "Base_BLEU3",
        "Base_BLEU4",
        "Base_Time_s",
        "P1_Caption",
        "P1_BLEU1",
        "P1_BLEU2",
        "P1_BLEU3",
        "P1_BLEU4",
        "P1_PSNR",
        "P1_SSIM",
        "P1_Time_s",
        "P2_Caption",
        "P2_BLEU1",
        "P2_BLEU2",
        "P2_BLEU3",
        "P2_BLEU4",
        "P2_Time_s",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"Đã lưu kết quả: {output_path}")
    for name, hypotheses in (
        ("Baseline", baseline_hypotheses),
        ("Pipeline 1", dncnn_hypotheses),
    ):
        scores = corpus_bleu(references_by_image, hypotheses)
        print(
            f"{name} BLEU: "
            + ", ".join(f"{key}={value:.4f}" for key, value in scores.items())
        )
    if end2end is not None:
        scores = corpus_bleu(references_by_image, end2end_hypotheses)
        print(
            "Pipeline 2 BLEU: "
            + ", ".join(f"{key}={value:.4f}" for key, value in scores.items())
        )
    mean_psnr = sum(float(row["P1_PSNR"]) for row in results) / len(results)
    mean_ssim = sum(float(row["P1_SSIM"]) for row in results) / len(results)
    print(
        "Pipeline 1 image quality: "
        f"PSNR={mean_psnr:.2f} dB, SSIM={mean_ssim:.4f}"
    )
    return results


def run_benchmark() -> list[dict]:
    """Chạy benchmark theo các hằng cấu hình ở đầu file."""
    return _run_benchmark(
        noise_level=NOISE_LEVEL,
        limit=LIMIT,
        max_length=MAX_LENGTH,
        device=DEVICE,
    )
