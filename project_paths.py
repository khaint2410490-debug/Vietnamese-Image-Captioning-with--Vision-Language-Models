"""Tổng hợp các đường dẫn mặc định trong dự án."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VOCABULARY_DIR = DATA_DIR / "vocabulary"
CAPTIONS_PATH = RAW_DATA_DIR / "captions.csv"
TRAIN_CAPTIONS_PATH = PROCESSED_DATA_DIR / "train.csv"
VALIDATION_CAPTIONS_PATH = PROCESSED_DATA_DIR / "validation.csv"
TEST_CAPTIONS_PATH = PROCESSED_DATA_DIR / "test.csv"
EVALUATION_RESULTS_PATH = PROCESSED_DATA_DIR / "evaluation.csv"
IMAGES_DIR = RAW_DATA_DIR / "Images"
NOISY_IMAGES_DIR = PROCESSED_DATA_DIR / "NoisyImages"
NOISY_IMAGE_MAP_PATH = PROCESSED_DATA_DIR / "noisy_image_map.csv"
VOCAB_PATH = VOCABULARY_DIR / "tokenizer.json"
WEIGHTS_PATH = PROCESSED_DATA_DIR / "caption_weights.pt"


def get_noisy_images_dir(noise_level: int | float | str | None = None) -> Path:
    """Trả về thư mục ảnh nhiễu theo mức nhiễu sigma hoặc thư mục mặc định."""
    if noise_level is None:
        return NOISY_IMAGES_DIR
    if isinstance(noise_level, str) and not noise_level.strip():
        return NOISY_IMAGES_DIR
    if isinstance(noise_level, float) and noise_level.is_integer():
        noise_level = int(noise_level)
    return PROCESSED_DATA_DIR / f"NoisyImages_sigma{noise_level}"


def verify_paths() -> list[tuple[str, Path, bool]]:
    """Kiểm tra xem các thư mục và file dữ liệu Giai đoạn 1 đã có đủ chưa."""
    items = [
        ("CAPTIONS_PATH (Phase 1)", CAPTIONS_PATH),
        ("IMAGES_DIR (Phase 1)", IMAGES_DIR),
        ("TRAIN_CAPTIONS_PATH (Phase 1)", TRAIN_CAPTIONS_PATH),
        ("VALIDATION_CAPTIONS_PATH (Phase 1)", VALIDATION_CAPTIONS_PATH),
        ("TEST_CAPTIONS_PATH (Phase 1)", TEST_CAPTIONS_PATH),
        ("NOISY_IMAGES_DIR (Phase 1)", NOISY_IMAGES_DIR),
        ("NOISY_IMAGES_SIGMA15 (Phase 1)", get_noisy_images_dir(15)),
        ("NOISY_IMAGES_SIGMA25 (Phase 1)", get_noisy_images_dir(25)),
        ("NOISY_IMAGES_SIGMA50 (Phase 1)", get_noisy_images_dir(50)),
        ("NOISY_IMAGES_RANDOM (Phase 1)", get_noisy_images_dir("random")),
        ("NOISY_IMAGE_MAP_PATH (Phase 1)", NOISY_IMAGE_MAP_PATH),
        ("VOCAB_PATH (Phase 1)", VOCAB_PATH),
        ("WEIGHTS_PATH (Baseline)", WEIGHTS_PATH),
    ]
    return [(label, p, p.exists()) for label, p in items]


if __name__ == "__main__":
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    for label, path, exists in verify_paths():
        status = "OK" if exists else "MISSING"
        try:
            display_path = path.relative_to(PROJECT_ROOT)
        except ValueError:
            display_path = path
        print(f"  [{status:7s}] {label:33s} -> {display_path}")
