"""Tổng hợp các đường dẫn trong dự án."""

import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VOCABULARY_DIR = DATA_DIR / "vocabulary"

CAPTIONS_PATH = RAW_DATA_DIR / "captions.csv"
TRAIN_CAPTIONS_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_CAPTIONS_PATH = PROCESSED_DATA_DIR / "validation.csv"
TEST_CAPTIONS_PATH = PROCESSED_DATA_DIR / "test.csv"
EVALUATION_RESULTS_PATH = PROCESSED_DATA_DIR / "evaluation.csv"

IMAGES_DIR = RAW_DATA_DIR / "Images" / "Images"
VOCAB_PATH = VOCABULARY_DIR / "tokenizer.json"
WEIGHTS_PATH = PROCESSED_DATA_DIR / "caption_weights.pt"

# --- Đường dẫn mở rộng cho bài toán Noisy Image Captioning ---
NOISY_IMAGES_DIR = PROCESSED_DATA_DIR / "NoisyImages"
NOISY_IMAGE_MAP_PATH = PROCESSED_DATA_DIR / "noisy_image_map.csv"
DNCNN_WEIGHTS_PATH = PROCESSED_DATA_DIR / "dncnn_weights_noise_25.pt"
END2END_WEIGHTS_PATH = PROCESSED_DATA_DIR / "end2end_caption_weights_sigma25.pt"
BENCHMARK_REPORT_PATH = PROCESSED_DATA_DIR / "benchmark_comparison.csv"
BENCHMARK_SUMMARY_PATH = PROCESSED_DATA_DIR / "benchmark_summary.md"


def _noise_level_label(noise_level: int | float | str) -> str:
    if isinstance(noise_level, str):
        value_text = noise_level.strip().lower()
        if value_text == "random":
            return "random"
        try:
            value = float(value_text)
        except ValueError as error:
            raise ValueError(
                f"noise_level phải là số không âm hoặc 'random': {noise_level!r}"
            ) from error
    else:
        value = float(noise_level)

    if not math.isfinite(value) or value < 0:
        raise ValueError(f"noise_level phải là số hữu hạn không âm: {value}")
    return str(int(value)) if value.is_integer() else f"{value:g}"


def get_dncnn_weights_path(noise_level: int | float | str) -> Path:
    """Trả về đường dẫn checkpoint DnCNN tương ứng với mức nhiễu."""
    label = _noise_level_label(noise_level)
    return PROCESSED_DATA_DIR / f"dncnn_weights_noise_{label}.pt"


def get_end2end_weights_path(noise_level: int | float | str) -> Path:
    """Trả về đường dẫn checkpoint captioner tương ứng với mức nhiễu."""
    label = _noise_level_label(noise_level)
    return PROCESSED_DATA_DIR / f"end2end_caption_weights_sigma{label}.pt"


def get_noisy_image_map_path(noise_level: int | float | str) -> Path:
    """Trả về đường dẫn file map ảnh nhiễu tương ứng với mức nhiễu."""
    label = _noise_level_label(noise_level)
    return PROCESSED_DATA_DIR / f"noisy_image_map_noise_{label}.csv"
