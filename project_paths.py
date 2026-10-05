"""Tổng hợp các đường dẫn trong dự án."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VOCABULARY_DIR = DATA_DIR / "vocabulary"
CAPTIONS_PATH = RAW_DATA_DIR / "captions.csv"
TRAIN_CAPTIONS_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_CAPTIONS_PATH = PROCESSED_DATA_DIR / "test.csv"
EVALUATION_RESULTS_PATH = PROCESSED_DATA_DIR / "evaluation.csv"
IMAGES_DIR = RAW_DATA_DIR / "Images"
VOCAB_PATH = VOCABULARY_DIR / "tokenizer.json"
WEIGHTS_PATH = PROCESSED_DATA_DIR / "caption_weights.pt"
