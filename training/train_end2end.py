"""Fine-tune the captioning model directly on noisy images."""

import random
from pathlib import Path

import torch

from model.captioning.image_captioning import ImageCaptioningModel
from preprocessing.tokenizer import Tokenizer
from project_paths import (
    IMAGES_DIR,
    PROCESSED_DATA_DIR,
    TRAIN_CAPTIONS_PATH,
    VOCAB_PATH,
    WEIGHTS_PATH,
)
from training.train import encode_captions, load_captions, train

VALIDATION_CAPTIONS_PATH = PROCESSED_DATA_DIR / "validation.csv"


def train_end2end(
    noise_level: int | float | str = 25,
    noisy_dir: Path | None = None,
    save_path: Path | None = None,
    init_weights: Path = WEIGHTS_PATH,
    vocab_path: Path = VOCAB_PATH,
    train_csv: Path = TRAIN_CAPTIONS_PATH,
    val_csv: Path = VALIDATION_CAPTIONS_PATH,
    epochs: int = 10,
    batch_size: int = 8,
    lr_decoder: float = 1e-4,
    lr_encoder: float = 1e-5,
    fine_tune_stages: int = 2,
    patience: int = 3,
    max_length: int = 20,
    limit: int | None = None,
    seed: int = 42,
    device: torch.device | None = None,
) -> list[dict]:
    noisy_dir = noisy_dir or PROCESSED_DATA_DIR / f"NoisyImages_sigma{noise_level}"
    save_path = (
        save_path
        or PROCESSED_DATA_DIR / f"end2end_caption_weights_sigma{noise_level}.pt"
    )
    if not noisy_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh nhiễu: {noisy_dir}")
    if noisy_dir.resolve() == Path(IMAGES_DIR).resolve():
        raise ValueError("noisy_dir không được trùng thư mục ảnh sạch.")
    if limit is not None and limit < 1:
        raise ValueError("limit phải lớn hơn 0.")
    if max_length < 2:
        raise ValueError("max_length phải lớn hơn hoặc bằng 2.")
    if not init_weights.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint baseline: {init_weights}")
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy vocabulary: {vocab_path}")

    random.seed(seed)
    torch.manual_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[End-to-end] Thiết bị: {device}")
    print(f"[End-to-end] Ảnh nhiễu: {noisy_dir}")
    print(f"[End-to-end] Checkpoint đầu vào: {init_weights}")
    print(f"[End-to-end] Checkpoint đầu ra: {save_path}")

    tokenizer = Tokenizer.load(str(vocab_path))
    train_pairs = load_captions(str(train_csv), str(noisy_dir))
    val_pairs = load_captions(str(val_csv), str(noisy_dir))
    if limit is not None:
        train_pairs = train_pairs[:limit]
        val_pairs = val_pairs[: max(1, limit // 8)]
    samples = encode_captions(train_pairs, tokenizer, max_length)
    val_samples = encode_captions(val_pairs, tokenizer, max_length)
    print(
        f"[End-to-end] Train: {len(samples)} cặp | "
        f"Validation: {len(val_samples)} cặp"
    )

    model = ImageCaptioningModel(
        tokenizer,
        feature_dim=256,
        word_embed_dim=256,
        pretrained=False,
        freeze_backbone=True,
    ).to(device)
    model.load_weights(str(init_weights))

    history = train(
        model=model,
        samples=samples,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr_decoder,
        device=device,
        weights_path=str(save_path),
        max_length=max_length,
        val_samples=val_samples,
        patience=patience,
        fine_tune_after_epoch=0,
        fine_tune_last_n_stages=fine_tune_stages,
        backbone_learning_rate=lr_encoder,
    )

    best = min(history, key=lambda record: record.get("val_loss", float("inf")))
    print(
        f"[End-to-end] Hoàn tất epoch {best['epoch']}, "
        f"val loss = {best.get('val_loss', float('nan')):.4f}"
    )
    return history
