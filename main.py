"""Train the baseline and both noisy-captioning approaches."""

import io
import random
import sys

import torch

from model.captioning.image_captioning import ImageCaptioningModel
from preprocessing.noise_generator import generate_noisy_dataset
from preprocessing.tokenizer import Tokenizer
from project_paths import (
    CAPTIONS_PATH,
    DNCNN_WEIGHTS_PATH,
    IMAGES_DIR,
    PROCESSED_DATA_DIR,
    VOCAB_PATH,
    WEIGHTS_PATH,
)
from split import split_dataset
from training.train import encode_captions, load_captions, train
from training.train_dncnn import train_dncnn
from training.train_end2end import train_end2end

# Adjust experiment settings here; no command-line arguments are required.
NOISE_LEVEL = 15
EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
BACKBONE_LEARNING_RATE = 1e-5
FINE_TUNE_AFTER_EPOCH = 3
FINE_TUNE_LAST_N_STAGES = 2
EARLY_STOPPING_PATIENCE = 3
MAX_LENGTH = 20
MIN_WORD_FREQUENCY = 1
FEATURE_DIM = 256
WORD_EMBEDDING_DIM = 256
TRAINING_LIMIT = None
RANDOM_SEED = 42

DNCNN_EPOCHS = 15
DNCNN_BATCH_SIZE = 8
DNCNN_LEARNING_RATE = 1e-3
DNCNN_PATIENCE = 5
DNCNN_PATCH_SIZE = 64

END2END_EPOCHS = 10
END2END_BATCH_SIZE = 8
END2END_DECODER_LEARNING_RATE = 1e-4
END2END_ENCODER_LEARNING_RATE = 1e-5
END2END_PATIENCE = 3
END2END_FINE_TUNE_STAGES = 2


def train_baseline(device: torch.device) -> None:
    print("\n[Baseline] Chia dữ liệu")
    train_path, validation_path, _ = split_dataset(
        captions_path=CAPTIONS_PATH,
        output_dir=PROCESSED_DATA_DIR,
        images_dir=IMAGES_DIR,
        seed=RANDOM_SEED,
    )

    print("\n[Baseline] Xây vocabulary và mã hóa caption")
    pairs = load_captions(str(train_path), str(IMAGES_DIR))
    validation_pairs = load_captions(str(validation_path), str(IMAGES_DIR))
    if TRAINING_LIMIT is not None:
        pairs = pairs[:TRAINING_LIMIT]
    tokenizer = Tokenizer()
    tokenizer.build_vocab(
        [caption for _, caption in pairs],
        min_freq=MIN_WORD_FREQUENCY,
    )
    tokenizer.save(str(VOCAB_PATH))
    samples = encode_captions(pairs, tokenizer, MAX_LENGTH)
    validation_samples = encode_captions(
        validation_pairs, tokenizer, MAX_LENGTH
    )

    print("\n[Baseline] Huấn luyện captioning trên ảnh sạch")
    model = ImageCaptioningModel(
        tokenizer,
        feature_dim=FEATURE_DIM,
        word_embed_dim=WORD_EMBEDDING_DIM,
        pretrained=True,
        freeze_backbone=True,
    ).to(device)
    train(
        model=model,
        samples=samples,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        device=device,
        weights_path=str(WEIGHTS_PATH),
        max_length=MAX_LENGTH,
        val_samples=validation_samples,
        patience=EARLY_STOPPING_PATIENCE,
        fine_tune_after_epoch=FINE_TUNE_AFTER_EPOCH,
        fine_tune_last_n_stages=FINE_TUNE_LAST_N_STAGES,
        backbone_learning_rate=BACKBONE_LEARNING_RATE,
    )
    print(f"[Baseline] Checkpoint: {WEIGHTS_PATH}")


def train_noisy_approaches(device: torch.device) -> None:
    print(f"\n[Tạo dữ liệu] Sinh ảnh nhiễu sigma={NOISE_LEVEL}")
    noisy_dir = generate_noisy_dataset(
        captions_csv=CAPTIONS_PATH,
        clean_images_dir=IMAGES_DIR,
        noise_level=NOISE_LEVEL,
        seed=RANDOM_SEED,
    )

    train_csv = PROCESSED_DATA_DIR / "train.csv"
    validation_csv = PROCESSED_DATA_DIR / "validation.csv"

    print("\n[Phương án 1] Huấn luyện DnCNN khử nhiễu")
    train_dncnn(
        train_csv=train_csv,
        val_csv=validation_csv,
        clean_images_dir=IMAGES_DIR,
        noisy_images_dir=noisy_dir,
        weights_save_path=DNCNN_WEIGHTS_PATH,
        noise_level=NOISE_LEVEL,
        epochs=DNCNN_EPOCHS,
        batch_size=DNCNN_BATCH_SIZE,
        learning_rate=DNCNN_LEARNING_RATE,
        patience=DNCNN_PATIENCE,
        patch_size=DNCNN_PATCH_SIZE,
        limit=TRAINING_LIMIT,
        device=str(device),
    )

    print("\n[Phương án 2] Train captioner trực tiếp trên ảnh nhiễu")
    train_end2end(
        noise_level=NOISE_LEVEL,
        noisy_dir=noisy_dir,
        vocab_path=VOCAB_PATH,
        train_csv=train_csv,
        val_csv=validation_csv,
        epochs=END2END_EPOCHS,
        batch_size=END2END_BATCH_SIZE,
        lr_decoder=END2END_DECODER_LEARNING_RATE,
        lr_encoder=END2END_ENCODER_LEARNING_RATE,
        fine_tune_stages=END2END_FINE_TUNE_STAGES,
        patience=END2END_PATIENCE,
        max_length=MAX_LENGTH,
        limit=TRAINING_LIMIT,
        seed=RANDOM_SEED,
        device=device,
    )


def main() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device}")

    train_baseline(device)
    train_noisy_approaches(device)

    print("\nĐã hoàn thành baseline và cả hai phương án noisy-captioning.")


if __name__ == "__main__":
    main()
