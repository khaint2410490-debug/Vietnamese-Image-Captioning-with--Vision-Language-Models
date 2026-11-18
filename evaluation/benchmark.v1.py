import time
from pathlib import Path
import torch

from evaluation.generate_caption import load_model
from evaluation.evaluation import corpus_bleu, tokenize_reference
from model.captioning.image_captioning import ImageCaptioningModel
from training.train import load_captions
from project_paths import IMAGES_DIR, TEST_CAPTIONS_PATH

def run_inference_benchmark(
    model: ImageCaptioningModel | None = None,
    captions_path: Path = TEST_CAPTIONS_PATH,
    images_dir: Path = IMAGES_DIR,
    max_length: int = 20,
    greedy: bool = True,
    limit: int | None = None,  # Optional: test on just N images for quick checks
) -> dict:
    """
    Pure inference benchmark: measures speed (latency/FPS) and caption quality (BLEU-1..4)
    without teacher-forcing loss or perplexity.
    """
    model = model or load_model()
    device = next(model.parameters()).device
    pairs = load_captions(captions_path, images_dir)

    # Group reference captions by image
    grouped: dict[str, list[str]] = {}
    for image_path, caption in pairs:
        grouped.setdefault(image_path, []).append(caption)

    image_paths = list(grouped.keys())
    if limit:
        image_paths = image_paths[:limit]

    hypotheses, references = [], []
    latencies = []

    # Warm-up run (prevents first-image initialization overhead from skewing timing)
    if image_paths:
        model.generate_caption(image_paths[0], max_length=max_length, greedy=greedy)

    for image_path in image_paths:
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        # Pure inference generation
        tokens = model.generate_caption(
            image_path, max_length=max_length, greedy=greedy
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start_time)

        hypotheses.append(tokens)
        references.append([tokenize_reference(c) for c in grouped[image_path]])

    total_time = sum(latencies)
    num_images = len(image_paths)
    bleu_scores = corpus_bleu(references, hypotheses)

    return {
        "num_images": num_images,
        "device": str(device),
        "total_time_sec": round(total_time, 4),
        "avg_latency_ms": round((total_time / max(num_images, 1)) * 1000, 2),
        "throughput_img_per_sec": round(num_images / max(total_time, 1e-9), 2),
        **bleu_scores,
    }
