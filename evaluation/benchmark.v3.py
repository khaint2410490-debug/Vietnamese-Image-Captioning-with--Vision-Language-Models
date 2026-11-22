import time
from pathlib import Path
import pandas as pd
import torch

from evaluation.denoise_metrics import compute_psnr_ssim
from evaluation.evaluation import corpus_bleu, tokenize_reference
from evaluation.generate_caption import generate_caption, load_model
from project_paths import VOCAB_PATH, WEIGHTS_PATH


def run_baseline(caption_model, image_path: str):
    start = time.perf_counter()
    caption_text = generate_caption(caption_model, image_path)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return caption_text, elapsed_ms


def run_pipeline_dncnn(caption_model, noisy_image_path: str, clean_tensor=None):
    start = time.perf_counter()

    # TODO (Stub 2): Replace with real DnCNN model when ready
    denoised_tensor = clean_tensor  # temporary placeholder
    denoised_image_path = noisy_image_path  # temporary placeholder

    caption_text = generate_caption(caption_model, denoised_image_path)
    elapsed_ms = (time.perf_counter() - start) * 1000

    metrics = {"psnr": None, "ssim": None}
    if clean_tensor is not None and denoised_tensor is not None:
        metrics = compute_psnr_ssim(clean_tensor, denoised_tensor)

    return caption_text, metrics["psnr"], metrics["ssim"], elapsed_ms


def run_pipeline_end2end(noisy_image_path: str):
    start = time.perf_counter()

    # TODO (Stub 3): Replace with loaded End2End fine-tuned model when ready
    caption_text = "chú thích tạm thời cho end2end"

    elapsed_ms = (time.perf_counter() - start) * 1000
    return caption_text, elapsed_ms


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = load_model(weights_path=WEIGHTS_PATH, vocab_path=str(VOCAB_PATH), device=device)

    # Example with 1 dummy item:
    sample_ref = "Một con mèo đang ngồi trên ghế."
    sample_hyp, lat_ms = run_baseline(base_model, "data/raw/Images/example.jpg")

    ref_tokens = [[tokenize_reference(sample_ref)]]
    hyp_tokens = [sample_hyp.lower().split()]
    bleu = corpus_bleu(ref_tokens, hyp_tokens)

    print("Caption:", sample_hyp)
    print("Latency (ms):", round(lat_ms, 2))
    print("BLEU:", bleu)


if __name__ == "__main__":
    main()