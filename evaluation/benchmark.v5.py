import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from evaluation.denoise_metrics import compute_psnr_ssim
from evaluation.evaluation import corpus_bleu, tokenize_reference
from evaluation.generate_caption import load_model
from model.captioning.image_captioning import ImageCaptioningModel
from model.denoising.dncnn import DnCNN
from preprocessing.noise_generator import add_gaussian_noise
from preprocessing.tokenizer import Tokenizer
from project_paths import IMAGES_DIR, VOCAB_PATH, WEIGHTS_PATH
from training.train import load_captions


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark captioning and denoising pipelines.")
    parser.add_argument("--test_csv", type=str, default="data/processed/test.csv")
    parser.add_argument("--sigmas", type=int, nargs="+", default=[0, 15, 25, 50, 75])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="data/processed/eval_results")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_image_tensor(image_path: str, device: torch.device) -> torch.Tensor:
    """Loads an image from disk as a float tensor in [0.0, 1.0] with shape (3, H, W)."""
    with Image.open(image_path) as img:
        tensor = to_tensor(img.convert("RGB"))
    return tensor.to(device)


def load_dncnn_model(weights_path: Path, device: torch.device) -> DnCNN:
    model = DnCNN().to(device) #[cite: 7]
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def load_end2end_model(weights_path: Path, device: torch.device) -> ImageCaptioningModel:
    tokenizer = Tokenizer.load(str(VOCAB_PATH))
    model = ImageCaptioningModel(
        tokenizer,
        feature_dim=256,
        word_embed_dim=256,
        pretrained=False,
        freeze_backbone=True,
    ).to(device) #[cite: 6]
    model.load_weights(str(weights_path)) #[cite: 6]
    model.eval()
    return model


def score_caption(ref_captions: list[str], predicted_caption: str) -> dict[str, float]:
    ref_tokens = [[tokenize_reference(c) for c in ref_captions]] #[cite: 3]
    hyp_tokens = [predicted_caption.lower().split()]
    return corpus_bleu(ref_tokens, hyp_tokens) #[cite: 3]


def save_csv_reports(rows: list[dict], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    per_image_path = output_dir / "per_image_results.csv" #
    df.to_csv(per_image_path, index=False, encoding="utf-8-sig")

    numeric_cols = ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "psnr", "ssim", "latency_ms"]
    summary_df = (
        df.groupby(["model_name", "sigma"], dropna=False)[numeric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_df.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary_df.columns
    ]

    summary_path = output_dir / "summary_metrics.csv" #[cite: 1]
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return per_image_path, summary_path


def main():
    args = parse_args()
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # 1. Load baseline captioning model once
    base_model = load_model(
        weights_path=WEIGHTS_PATH,
        vocab_path=str(VOCAB_PATH),
        device=device,
    ) #[cite: 3, 4]

    # 2. Load real test (image_path, caption) pairs from CSV and group captions per image
    pairs = load_captions(args.test_csv, str(IMAGES_DIR))
    grouped_refs: dict[str, list[str]] = {}
    for img_path, cap in pairs:
        grouped_refs.setdefault(img_path, []).append(cap)

    all_rows = []

    for sigma in args.sigmas:
        print(f"\n=== Evaluating sigma = {sigma} ===")

        # Optional per-sigma weights if present in data/processed
        dncnn_ckpt = Path(f"data/processed/dncnn_weights_noise_{sigma}.pt")
        e2e_ckpt = Path(f"data/processed/end2end_caption_weights_sigma{sigma}.pt")

        dncnn_model = load_dncnn_model(dncnn_ckpt, device) if dncnn_ckpt.is_file() else None
        e2e_model = load_end2end_model(e2e_ckpt, device) if e2e_ckpt.is_file() else None

        for idx, (img_path, ref_captions) in enumerate(grouped_refs.items()):
            clean_tensor = load_image_tensor(img_path, device)
            noisy_tensor = add_gaussian_noise(
                clean_tensor,
                sigma=float(sigma),
                seed=args.seed + idx + sigma * 100000,
            ) #[cite: 6]

            image_id = Path(img_path).name

            # A. Baseline
            t0 = time.perf_counter()
            base_cap = base_model.generate_caption_text(noisy_tensor) #[cite: 6]
            base_ms = (time.perf_counter() - t0) * 1000.0
            base_bleu = score_caption(ref_captions, base_cap)
            all_rows.append({
                "image_id": image_id,
                "model_name": "Baseline",
                "sigma": sigma,
                "caption": base_cap,
                "bleu_1": base_bleu["BLEU-1"],
                "bleu_2": base_bleu["BLEU-2"],
                "bleu_3": base_bleu["BLEU-3"],
                "bleu_4": base_bleu["BLEU-4"],
                "psnr": np.nan,
                "ssim": np.nan,
                "latency_ms": round(base_ms, 3),
            })

            # B. Pipeline 2-Stage (DnCNN -> Base Captioner)
            if dncnn_model is not None:
                t0 = time.perf_counter()
                denoised_tensor = dncnn_model.denoise(noisy_tensor) #[cite: 7]
                p1_cap = base_model.generate_caption_text(denoised_tensor) #[cite: 6]
                p1_ms = (time.perf_counter() - t0) * 1000.0
                denoise_scores = compute_psnr_ssim(clean_tensor, denoised_tensor) #[cite: 5]
                p1_bleu = score_caption(ref_captions, p1_cap)
                all_rows.append({
                    "image_id": image_id,
                    "model_name": "Pipeline_DnCNN",
                    "sigma": sigma,
                    "caption": p1_cap,
                    "bleu_1": p1_bleu["BLEU-1"],
                    "bleu_2": p1_bleu["BLEU-2"],
                    "bleu_3": p1_bleu["BLEU-3"],
                    "bleu_4": p1_bleu["BLEU-4"],
                    "psnr": denoise_scores["psnr"],
                    "ssim": denoise_scores["ssim"],
                    "latency_ms": round(p1_ms, 3),
                })

            # C. End-to-End Fine-Tuned
            if e2e_model is not None:
                t0 = time.perf_counter()
                p2_cap = e2e_model.generate_caption_text(noisy_tensor) #[cite: 6]
                p2_ms = (time.perf_counter() - t0) * 1000.0
                p2_bleu = score_caption(ref_captions, p2_cap)
                all_rows.append({
                    "image_id": image_id,
                    "model_name": "End2End_FineTuned",
                    "sigma": sigma,
                    "caption": p2_cap,
                    "bleu_1": p2_bleu["BLEU-1"],
                    "bleu_2": p2_bleu["BLEU-2"],
                    "bleu_3": p2_bleu["BLEU-3"],
                    "bleu_4": p2_bleu["BLEU-4"],
                    "psnr": np.nan,
                    "ssim": np.nan,
                    "latency_ms": round(p2_ms, 3),
                })

    per_img_csv, summary_csv = save_csv_reports(all_rows, Path(args.output_dir))
    print(f"\nSaved: {per_img_csv}")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()