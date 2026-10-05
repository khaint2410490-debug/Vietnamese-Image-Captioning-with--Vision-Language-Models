"""
evaluation.py

Sinh caption cho từng ảnh trong tập test và đánh giá dự đoán với các caption tham chiếu.

Các chỉ số:
    - BLEU-1 .. BLEU-4: mức độ trùng khớp giữa caption dự đoán và các caption thật
      (corpus-level, nhiều caption tham chiếu cho mỗi ảnh, không dùng smoothing).
      Caption được tách theo âm tiết và dấu câu (cùng cách tách với text_preprocessing.py),
      nên điểm BLEU ở đây tính trên âm tiết.
    - Loss và Perplexity: loss Categorical Cross Entropy trung bình trên mỗi từ khi mô hình
      được cho các từ đúng đứng trước (giống lúc huấn luyện). Perplexity = exp(loss);
      càng thấp càng tốt.
    - BLEU-1 .. BLEU-4 cho từng ảnh, cùng caption dự đoán.
    - Loss và Perplexity tính theo teacher forcing trên các caption tham chiếu.

Chạy từ thư mục gốc dự án: python -m evaluation.evaluation
"""

import csv
import math
from collections import Counter
from pathlib import Path

import torch

from generate_caption import load_model
from model.captioning.image_captioning import ImageCaptioningModel
from preprocessing.text_preprocessing import preprocess_caption
from training.train import compute_batch_loss, encode_captions, load_captions
from project_paths import (
    IMAGES_DIR,
    EVALUATION_RESULTS_PATH,
    TEST_CAPTIONS_PATH,
    VOCAB_PATH,
    WEIGHTS_PATH,
)

MAX_LENGTH = 20
BATCH_SIZE = 8
FEATURE_DIM = 256
WORD_EMBEDDING_DIM = 256

# BLEU
def ngrams(tokens: list[str], n: int) -> Counter:
    """Đếm các n-gram (cụm n token liên tiếp) trong một câu."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

def corpus_bleu(
    references: list[list[list[str]]],
    hypotheses: list[list[str]],
    max_n: int = 4,
) -> dict[str, float]:
    """
    BLEU cấp corpus (Papineni et al., 2002) với trọng số đều cho các n-gram.

    Args:
        references: với mỗi ảnh, một list các caption tham chiếu (mỗi caption là list token).
        hypotheses: với mỗi ảnh, caption dự đoán (list token).
        max_n     : n-gram lớn nhất (4 cho BLEU-4).
    Returns:
        {"BLEU-1": ..., "BLEU-2": ..., ...}
    """
    clipped = [0] * max_n       # số n-gram dự đoán khớp với tham chiếu (đã cắt theo số lần xuất hiện tối đa)
    total = [0] * max_n         # tổng số n-gram trong các caption dự đoán
    hyp_len = ref_len = 0

    for refs, hyp in zip(references, hypotheses):
        hyp_len += len(hyp)
        # độ dài tham chiếu gần nhất với độ dài caption dự đoán (hòa thì lấy câu ngắn hơn)
        ref_len += min((abs(len(r) - len(hyp)), len(r)) for r in refs)[1]

        for n in range(1, max_n + 1):
            hyp_counts = ngrams(hyp, n)
            max_ref_counts: Counter = Counter()
            for ref in refs:
                for gram, count in ngrams(ref, n).items():
                    max_ref_counts[gram] = max(max_ref_counts[gram], count)
            clipped[n - 1] += sum(min(c, max_ref_counts[g]) for g, c in hyp_counts.items())
            total[n - 1] += sum(hyp_counts.values())

    if hyp_len == 0:
        return {f"BLEU-{k}": 0.0 for k in range(1, max_n + 1)}

    # Brevity penalty: phạt caption dự đoán ngắn hơn tham chiếu
    brevity_penalty = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)

    scores = {}
    for k in range(1, max_n + 1):
        precisions = [clipped[i] / total[i] if total[i] > 0 else 0.0 for i in range(k)]
        if min(precisions) == 0:
            scores[f"BLEU-{k}"] = 0.0
        else:
            scores[f"BLEU-{k}"] = brevity_penalty * math.exp(
                sum(math.log(p) for p in precisions) / k
            )
    return scores

# Loss / Perplexity
@torch.no_grad()
def compute_loss(
    model: ImageCaptioningModel,
    pairs: list[tuple[str, str]],
    max_length: int,
    batch_size: int,
) -> tuple[float, float]:
    """Trả về (loss trung bình mỗi từ, perplexity) trên các cặp (ảnh, caption)."""
    device = next(model.parameters()).device
    model.eval()
    samples = encode_captions(pairs, model.tokenizer, max_length)

    total_loss, total_words = 0.0, 0
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        loss, batch_words = compute_batch_loss(model, batch, device)
        total_loss += loss.item() * batch_words
        total_words += batch_words

    avg_loss = total_loss / max(total_words, 1)
    return avg_loss, math.exp(min(avg_loss, 50))         # min(...) tránh tràn số khi loss quá lớn

# Đánh giá
def tokenize_reference(caption: str) -> list[str]:
    """Caption thật --> list token (cùng cách tách với lúc huấn luyện, bỏ <start>/<end>)."""
    return preprocess_caption(caption)[1:-1]

def evaluate(
    model: ImageCaptioningModel,
    pairs: list[tuple[str, str]],
    max_length: int = 20,
    batch_size: int = 8,
) -> dict:
    # Gom các caption thật theo ảnh: {đường dẫn ảnh: [caption, ...]}
    grouped: dict[str, list[str]] = {}
    for image_path, caption in pairs:
        grouped.setdefault(image_path, []).append(caption)

    # 1. Sinh caption dự đoán cho từng ảnh (chọn từ xác suất cao nhất)
    image_paths = list(grouped)
    hypotheses, references = [], []
    for i, image_path in enumerate(image_paths, 1):
        hypotheses.append(model.generate_caption(image_path, max_length=max_length))
        references.append([tokenize_reference(c) for c in grouped[image_path]])
        if i % 100 == 0:
            print(f"  đã sinh caption cho {i}/{len(image_paths)} ảnh")

    # 2. BLEU-1..4 corpus-level và theo từng ảnh
    results: dict = corpus_bleu(references, hypotheses)
    results["per_image"] = [
        {
            "image": Path(image_path).name,
            "references": grouped[image_path],
            "prediction": ImageCaptioningModel.tokens_to_text(hypothesis),
            **corpus_bleu([image_references], [hypothesis]),
        }
        for image_path, image_references, hypothesis in zip(
            image_paths, references, hypotheses
        )
    ]

    # 3. Loss và perplexity
    loss, perplexity = compute_loss(model, pairs, max_length, batch_size)
    results["loss"] = loss
    results["perplexity"] = perplexity
    results["num_images"] = len(image_paths)
    results["num_captions"] = len(pairs)

    return results

def print_results(results: dict) -> None:
    print(f"\nKết quả trên {results['num_images']} ảnh ({results['num_captions']} caption):")
    for k in range(1, 5):
        print(f"  BLEU-{k}     : {results[f'BLEU-{k}']:.4f}")
    print(f"  Loss       : {results['loss']:.4f}")
    print(f"  Perplexity : {results['perplexity']:.2f}")

    print("\nĐiểm theo từng ảnh:")
    for item in results["per_image"]:
        print(
            f"  {item['image']}: "
            f"BLEU-1={item['BLEU-1']:.4f}, "
            f"BLEU-2={item['BLEU-2']:.4f}, "
            f"BLEU-3={item['BLEU-3']:.4f}, "
            f"BLEU-4={item['BLEU-4']:.4f}"
        )
        print(f"    Caption: {item['prediction'] or '(rỗng)'}")

def save_per_image_results(results: dict) -> None:
    """Ghi caption dự đoán và BLEU từng ảnh ra CSV, ghi đè kết quả cũ."""
    fieldnames = ["caption", "image", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]
    with EVALUATION_RESULTS_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for item in results["per_image"]:
            writer.writerow(
                {
                    "caption": item["prediction"],
                    "image": item["image"],
                    **{f"BLEU-{n}": item[f"BLEU-{n}"] for n in range(1, 5)},
                }
            )
    print(f"\nĐã lưu kết quả caption theo từng ảnh vào: {EVALUATION_RESULTS_PATH}")

def main() -> None:
    print(f"Thiết bị: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Checkpoint: {WEIGHTS_PATH}")
    print(f"Từ điển: {VOCAB_PATH}")
    print(f"Tập test: {TEST_CAPTIONS_PATH}")

    pairs = load_captions(str(TEST_CAPTIONS_PATH), str(IMAGES_DIR))
    model = load_model(
        weights_path=WEIGHTS_PATH,
        vocab_path=VOCAB_PATH,
        feature_dim=FEATURE_DIM,
        word_embed_dim=WORD_EMBEDDING_DIM,
    )
    results = evaluate(model, pairs, MAX_LENGTH, BATCH_SIZE)
    print_results(results)
    save_per_image_results(results)

if __name__ == "__main__":
    main()