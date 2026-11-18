"""
train_end2end.py  ->  đặt vào thư mục training/

Phương án 2 - End-to-End Noise-Robust Captioning (phụ trách: Hưng)

Ý tưởng: lấy mô hình Captioning đã train trên ẢNH SẠCH (caption_weights.pt),
rồi fine-tune tiếp trực tiếp trên ẢNH NHIỄU, không qua bước khử nhiễu.

    - Đóng băng : stem (conv1, bn1), layer1, layer2 của ResNet-18
    - Mở khóa   : layer3, layer4 (2 stage cuối)  -> lr = 1e-5
    - Decoder   : Attention + LSTM + projection  -> lr = 1e-4
    - Loss      : Cross-Entropy (Teacher Forcing) - dùng lại hàm train() có sẵn

File này KHÔNG viết lại vòng lặp huấn luyện, chỉ gọi training.train.train()
với ảnh nhiễu và fine_tune_after_epoch=0 (mở khóa ngay từ epoch 1).

Chạy (từ thư mục gốc của repo):
    python -m training.train_end2end --noise_level 25
    python -m training.train_end2end --noise_level 25 --limit 64 --epochs 1   # chạy thử nhanh

Yêu cầu: thư mục ảnh nhiễu (do Hiếu sinh) có CÙNG TÊN FILE .jpg với ảnh gốc.
"""

import argparse
import random
import sys
from pathlib import Path

import torch

if not __package__:  # cho phép chạy trực tiếp: python training/train_end2end.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phương án 2: fine-tune end-to-end trên ảnh nhiễu")
    p.add_argument("--noise_level", default="25", help="15 | 25 | 50 | random")
    p.add_argument(
        "--noisy_dir",
        default=None,
        help="Thư mục ảnh nhiễu. Mặc định: data/processed/NoisyImages_sigma{noise_level}",
    )
    p.add_argument(
        "--save_path",
        default=None,
        help="File trọng số đầu ra. Mặc định: data/processed/end2end_caption_weights_sigma{noise_level}.pt",
    )
    p.add_argument("--init_weights", default=str(WEIGHTS_PATH), help="Trọng số baseline (train trên ảnh sạch)")
    p.add_argument("--vocab_path", default=str(VOCAB_PATH))
    p.add_argument("--train_csv", default=str(TRAIN_CAPTIONS_PATH))
    p.add_argument("--val_csv", default=str(VALIDATION_CAPTIONS_PATH))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr_decoder", type=float, default=1e-4)
    p.add_argument("--lr_encoder", type=float, default=1e-5)
    p.add_argument("--fine_tune_stages", type=int, default=2, help="Số stage cuối ResNet được mở khóa (2 = layer3, layer4)")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--max_length", type=int, default=20)
    p.add_argument("--limit", type=int, default=None, help="Chỉ dùng N cặp ảnh-caption đầu tiên (để chạy thử)")
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = get_parser().parse_args()

    noisy_dir = Path(args.noisy_dir) if args.noisy_dir else PROCESSED_DATA_DIR / f"NoisyImages_sigma{args.noise_level}"
    save_path = (
        Path(args.save_path)
        if args.save_path
        else PROCESSED_DATA_DIR / f"end2end_caption_weights_sigma{args.noise_level}.pt"
    )
    if not noisy_dir.is_dir():
        raise FileNotFoundError(
            f"Chưa có thư mục ảnh nhiễu: {noisy_dir}\n"
            "Hãy chạy noise_generator.py của Hiếu trước, hoặc truyền --noisy_dir."
        )
    if noisy_dir.resolve() == Path(IMAGES_DIR).resolve():
        raise ValueError("noisy_dir đang trùng thư mục ảnh sạch - như vậy không phải huấn luyện trên ảnh nhiễu.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device}")
    print(f"Ảnh nhiễu : {noisy_dir}")
    print(f"Khởi tạo từ: {args.init_weights}")
    print(f"Lưu kết quả: {save_path}")

    # 1. Từ điển: PHẢI dùng đúng từ điển lúc train baseline
    tokenizer = Tokenizer.load(args.vocab_path)

    # 2. Dữ liệu: cặp (ảnh NHIỄU, caption) - caption giữ nguyên
    train_pairs = load_captions(args.train_csv, str(noisy_dir))
    val_pairs = load_captions(args.val_csv, str(noisy_dir))
    if args.limit:
        train_pairs = train_pairs[: args.limit]
        val_pairs = val_pairs[: max(1, args.limit // 8)]
    samples = encode_captions(train_pairs, tokenizer, args.max_length)
    val_samples = encode_captions(val_pairs, tokenizer, args.max_length)
    print(f"Train: {len(samples)} cặp | Validation: {len(val_samples)} cặp")

    # 3. Mô hình: cấu hình giống lúc train baseline, rồi nạp trọng số baseline
    model = ImageCaptioningModel(
        tokenizer,
        feature_dim=256,
        word_embed_dim=256,
        pretrained=False,  # trọng số ImageNet sẽ bị ghi đè bởi checkpoint
        freeze_backbone=True,
    ).to(device)
    model.load_weights(args.init_weights)

    # 4. Fine-tune: fine_tune_after_epoch=0 -> mở khóa layer3, layer4 ngay từ đầu
    history = train(
        model=model,
        samples=samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr_decoder,
        device=device,
        weights_path=str(save_path),
        max_length=args.max_length,
        val_samples=val_samples,
        patience=args.patience,
        fine_tune_after_epoch=0,
        fine_tune_last_n_stages=args.fine_tune_stages,
        backbone_learning_rate=args.lr_encoder,
    )

    best = min(history, key=lambda r: r.get("val_loss", float("inf")))
    print(f"\nHoàn tất. Epoch tốt nhất: {best['epoch']} (val loss = {best.get('val_loss', float('nan')):.4f})")
    print(f"Trọng số: {save_path}")


if __name__ == "__main__":
    main()
