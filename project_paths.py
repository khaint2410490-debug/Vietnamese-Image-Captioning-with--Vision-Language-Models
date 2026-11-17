"""Tổng hợp các đường dẫn trong dự án."""

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

IMAGES_DIR = RAW_DATA_DIR / "Images"
VOCAB_PATH = VOCABULARY_DIR / "tokenizer.json"
WEIGHTS_PATH = PROCESSED_DATA_DIR / "caption_weights.pt"

# --- Đường dẫn mở rộng cho bài toán Noisy Image Captioning ---
NOISY_IMAGES_DIR = PROCESSED_DATA_DIR / "NoisyImages"
NOISY_IMAGE_MAP_PATH = PROCESSED_DATA_DIR / "noisy_image_map.csv"
DNCNN_WEIGHTS_PATH = PROCESSED_DATA_DIR / "dncnn_weights.pt"
END2END_WEIGHTS_PATH = PROCESSED_DATA_DIR / "end2end_caption_weights.pt"
BENCHMARK_REPORT_PATH = PROCESSED_DATA_DIR / "benchmark_comparison.csv"
BENCHMARK_SUMMARY_PATH = PROCESSED_DATA_DIR / "benchmark_summary.md"
