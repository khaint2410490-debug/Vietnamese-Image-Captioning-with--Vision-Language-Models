"""
generate_caption.py

Dùng mô hình đã huấn luyện để tạo caption cho một ảnh mới:

    Image (.jpg) --> CNN --> Image feature --> sinh từng từ một --> ghép lại thành caption

Ngôn ngữ caption phụ thuộc vào dữ liệu đã dùng để huấn luyện; hàm này không dịch.
Lưu ý: feature_dim, word_embed_dim và từ điển phải khớp với lúc huấn luyện.
"""

import argparse
import sys
from pathlib import Path

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.captioning.image_captioning import ImageCaptioningModel
from preprocessing.tokenizer import DEFAULT_VOCAB_PATH, Tokenizer
from project_paths import CAPTIONS_PATH, WEIGHTS_PATH


def load_model(
    weights_path: str | Path = WEIGHTS_PATH,
    vocab_path: str | None = None,
    feature_dim: int = 256,
    word_embed_dim: int = 256,
    backbone: str = "resnet18",
    hidden_dim: int = 512,
    attn_dim: int = 256,
    device: torch.device | None = None,
) -> ImageCaptioningModel:
    """Tải từ điển và weight đã huấn luyện, trả về mô hình ở chế độ eval."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_file = Path(vocab_path) if vocab_path else DEFAULT_VOCAB_PATH
    weights_file = Path(weights_path)
    if not vocab_file.is_file():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")
    if not weights_file.is_file():
        raise FileNotFoundError(f"Weights checkpoint not found: {weights_file}")
    if CAPTIONS_PATH.is_file():
        latest_data_time = max(
            CAPTIONS_PATH.stat().st_mtime,
            vocab_file.stat().st_mtime,
        )
        if weights_file.stat().st_mtime < latest_data_time:
            raise RuntimeError(
                "Checkpoint is older than captions.csv or vocabulary. "
                "Re-run main.py to retrain the model on updated data."
            )

    tokenizer = Tokenizer.load(str(vocab_file))
    model = ImageCaptioningModel(
        tokenizer,
        feature_dim=feature_dim,
        word_embed_dim=word_embed_dim,
        backbone=backbone,
        pretrained=False,
        hidden_dim=hidden_dim,
        attn_dim=attn_dim,
    ).to(device)

    try:
        model.load_weights(str(weights_file))
    except RuntimeError as error:
        raise RuntimeError(
            "Checkpoint weights do not match model architecture. Verify vocabulary, "
            f"feature_dim, and word_embed_dim.\nDetails: {error}"
        ) from error

    model.eval()
    return model


def generate_caption(
    model: ImageCaptioningModel,
    image_path: str,
    max_length: int = 20,
    greedy: bool = True,
    temperature: float = 1.0,
    top_k: int | None = None,
    verbose: bool = False,
) -> str:
    """
    Tạo caption cho một ảnh .jpg.

    Args:
        model      : mô hình đã load weight (xem load_model).
        image_path : đường dẫn ảnh .jpg.
        max_length : độ dài tối đa của chuỗi token (tính cả <start>).
        greedy     : True chọn từ xác suất cao nhất, False lấy mẫu ngẫu nhiên.
        temperature, top_k: chỉ dùng khi greedy=False.
        verbose    : in ra từng từ được chọn cùng xác suất của nó.
    Returns:
        Caption dạng chuỗi văn bản.
    """
    return model.generate_caption_text(
        image_path,
        max_length=max_length,
        greedy=greedy,
        temperature=temperature,
        top_k=top_k,
        verbose=verbose,
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Generate a caption for one image using the trained model."
    )
    parser.add_argument("image_path", help="Path to a .jpg or .jpeg image")
    args = parser.parse_args()

    model = load_model()
    print(generate_caption(model, args.image_path))


if __name__ == "__main__":
    main()
