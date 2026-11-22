import time
import torch

from evaluation.denoise_metrics import compute_psnr_ssim
from evaluation.evaluation import corpus_bleu, tokenize_reference
from evaluation.generate_caption import load_model
from project_paths import VOCAB_PATH, WEIGHTS_PATH


def run_baseline(caption_model, image_input):
    start = time.perf_counter()
    # image_input can be a file path OR a torch.Tensor
    caption_text = caption_model.generate_caption_text(image_input)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return caption_text, elapsed_ms


def run_pipeline_dncnn(caption_model, noisy_tensor: torch.Tensor, clean_tensor: torch.Tensor | None = None):
    start = time.perf_counter()

    # TODO (Stub 2): Replace with real DnCNN model when ready
    denoised_tensor = noisy_tensor.clone()

    # Pass tensor straight into caption_model—no temp file needed!
    caption_text = caption_model.generate_caption_text(denoised_tensor)
    elapsed_ms = (time.perf_counter() - start) * 1000

    psnr_val = None
    ssim_val = None
    if clean_tensor is not None:
        metrics = compute_psnr_ssim(clean_tensor, denoised_tensor)
        psnr_val = metrics["psnr"]
        ssim_val = metrics["ssim"]

    return caption_text, psnr_val, ssim_val, elapsed_ms


def run_pipeline_end2end(noisy_input):
    start = time.perf_counter()

    # TODO (Stub 3): Replace with loaded End2End fine-tuned model when ready
    caption_text = "chú thích tạm thời cho end2end"

    elapsed_ms = (time.perf_counter() - start) * 1000
    return caption_text, elapsed_ms