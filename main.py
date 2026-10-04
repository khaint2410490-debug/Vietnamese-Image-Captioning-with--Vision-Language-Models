"""Điều khiển pipeline huấn luyện image captioning."""

import random

import torch

from model.captioning.image_captioning import ImageCaptioningModel
from preprocessing.tokenizer import Tokenizer
from project_paths import (
    CAPTIONS_PATH,
    IMAGES_DIR,
    VOCAB_PATH,
    WEIGHTS_PATH,
    PROCESSED_DATA_DIR,
)
from split import split_dataset
from training.train import encode_captions, load_captions, train

# Chỉnh các giá trị này để thay đổi cấu hình chạy.
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

def main() -> None:
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device}")

    print("\nBước 1/4: Chia dữ liệu")
    train_path, validation_path, _ = split_dataset(
        captions_path=CAPTIONS_PATH,
        output_dir=PROCESSED_DATA_DIR,
        images_dir=IMAGES_DIR,
        seed=RANDOM_SEED,
    )

    print("\nBước 2/4: Đọc caption và xây từ điển")
    pairs = load_captions(str(train_path), str(IMAGES_DIR))
    validation_pairs = load_captions(
        str(validation_path), str(IMAGES_DIR)
    )
    if TRAINING_LIMIT is not None:
        pairs = pairs[:TRAINING_LIMIT]
    print(f"Số cặp ảnh-caption dùng để huấn luyện: {len(pairs)}")
    print(f"Số cặp ảnh-caption dùng để validation: {len(validation_pairs)}")

    tokenizer = Tokenizer()
    tokenizer.build_vocab(
        [caption for _, caption in pairs],
        min_freq=MIN_WORD_FREQUENCY,
    )
    tokenizer.save(str(VOCAB_PATH))
    print(f"Đã lưu từ điển gồm {tokenizer.vocab_size} token: {VOCAB_PATH}")
    samples = encode_captions(pairs, tokenizer, MAX_LENGTH)
    validation_samples = encode_captions(
        validation_pairs, tokenizer, MAX_LENGTH
    )

    print("\nBước 3/4: Khởi tạo mô hình")
    model = ImageCaptioningModel(
        tokenizer,
        feature_dim=FEATURE_DIM,
        word_embed_dim=WORD_EMBEDDING_DIM,
        pretrained=True,
        freeze_backbone=True,
    ).to(device)

    print("\nBước 4/4: Huấn luyện")
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
    print(f"Huấn luyện hoàn tất. Trọng số được lưu tại: {WEIGHTS_PATH}")
