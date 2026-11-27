import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
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


# =====================================================================
# 1. COMMAND-LINE ARGUMENTS
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark captioning and denoising pipelines.")
    parser.add_argument("--test_csv", type=str, default="data/processed/test.csv")
    parser.add_argument("--sigmas", type=int, nargs="+", default=[0, 15, 25, 50, 75])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="data/processed/eval_results")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# =====================================================================
# 2. IMAGE & MODEL LOADING HELPERS
# =====================================================================

def load_image_tensor(image_path: str, device: torch.device) -> torch.Tensor:
    """Loads an image from disk as a float tensor in [0.0, 1.0] with shape (3, H, W)."""
    with Image.open(image_path) as img:
        tensor = to_tensor(img.convert("RGB"))
    return tensor.to(device)


def load_dncnn_model(weights_path: Path, device: torch.device) -> DnCNN:
    """Builds DnCNN and loads its trained checkpoint weights."""
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        num_layers = checkpoint.get("num_of_layers", 17)
        model = DnCNN(num_of_layers=num_layers).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model = DnCNN().to(device)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model = DnCNN().to(device)
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def load_end2end_model(weights_path: Path, device: torch.device) -> ImageCaptioningModel:
    """Builds the fine-tuned ImageCaptioningModel and loads its checkpoint weights."""
    tokenizer = Tokenizer.load(str(VOCAB_PATH))
    model = ImageCaptioningModel(
        tokenizer,
        feature_dim=256,
        word_embed_dim=256,
        pretrained=False,
        freeze_backbone=True,
    ).to(device)
    model.load_weights(str(weights_path))
    model.eval()
    return model


# =====================================================================
# 3. SCORING, CSV EXPORT & CHART GENERATION
# =====================================================================

def score_caption(ref_captions: list[str], predicted_caption: str) -> dict[str, float]:
    """Computes BLEU-1..4 for a predicted caption against human reference captions."""
    ref_tokens = [[tokenize_reference(c) for c in ref_captions]]
    hyp_tokens = [predicted_caption.lower().split()]
    return corpus_bleu(ref_tokens, hyp_tokens)


def generate_charts(df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Generates the 4 comparison charts required by the benchmark spec."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_charts: list[Path] = []

    if df.empty or summary_df.empty:
        return saved_charts

    # Chart 1: BLEU comparison across models (at sigma=25 if available, else first sigma)
    target_sigma = 25 if 25 in summary_df["sigma"].values else summary_df["sigma"].iloc[0]
    sub_bleu = summary_df[summary_df["sigma"] == target_sigma]
    if not sub_bleu.empty:
        bleu_cols = ["bleu_1_mean", "bleu_2_mean", "bleu_3_mean", "bleu_4_mean"]
        ax = sub_bleu.set_index("model_name")[bleu_cols].plot(kind="bar", figsize=(8, 5))
        ax.set_title(f"BLEU Comparison Across Models (sigma={target_sigma})")
        ax.set_ylabel("Mean BLEU Score")
        ax.set_xlabel("Model")
        plt.xticks(rotation=0)
        plt.tight_layout()
        p1 = output_dir / "bleu_comparison.png"
        plt.savefig(p1, dpi=150)
        plt.close()
        saved_charts.append(p1)

    # Chart 2: Latency comparison across models (mean latency_ms across sigmas)
    lat_by_model = summary_df.groupby("model_name")["latency_ms_mean"].mean()
    if not lat_by_model.empty:
        ax = lat_by_model.plot(kind="bar", figsize=(7, 4.5))
        ax.set_title("Mean Inference Latency by Pipeline")
        ax.set_ylabel("Latency (ms)")
        ax.set_xlabel("Model")
        plt.xticks(rotation=0)
        plt.tight_layout()
        p2 = output_dir / "latency_comparison.png"
        plt.savefig(p2, dpi=150)
        plt.close()
        saved_charts.append(p2)

    # Chart 3: DnCNN restoration quality (PSNR & SSIM vs sigma)
    dncnn_df = summary_df[summary_df["model_name"] == "Pipeline_DnCNN"].sort_values("sigma")
    if not dncnn_df.empty and dncnn_df["psnr_mean"].notna().any():
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(dncnn_df["sigma"], dncnn_df["psnr_mean"], marker="o", label="PSNR (dB)")
        ax1.set_xlabel("Noise Level (sigma)")
        ax1.set_ylabel("Mean PSNR (dB)")
        ax1.tick_params(axis="y")

        if dncnn_df["ssim_mean"].notna().any():
            ax2 = ax1.twinx()
            ax2.plot(
                dncnn_df["sigma"],
                dncnn_df["ssim_mean"],
                marker="s",
                linestyle="--",
                label="SSIM",
            )
            ax2.set_ylabel("Mean SSIM")

        plt.title("DnCNN Restoration Quality Across Noise Levels")
        fig.tight_layout()
        p3 = output_dir / "dncnn_restoration_quality.png"
        plt.savefig(p3, dpi=150)
        plt.close()
        saved_charts.append(p3)

    # Chart 4: Quality vs sigma curves (BLEU-4 across all models)
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, grp in summary_df.groupby("model_name"):
        grp_sorted = grp.sort_values("sigma")
        ax.plot(grp_sorted["sigma"], grp_sorted["bleu_4_mean"], marker="o", label=model_name)
    ax.set_title("BLEU-4 vs Noise Level (sigma)")
    ax.set_xlabel("Noise Level (sigma)")
    ax.set_ylabel("Mean BLEU-4")
    ax.legend()
    plt.tight_layout()
    p4 = output_dir / "quality_vs_sigma_curves.png"
    plt.savefig(p4, dpi=150)
    plt.close()
    saved_charts.append(p4)

    return saved_charts


def save_reports_and_charts(rows: list[dict], output_dir: Path) -> None:
    """Exports per_image_results.csv, summary_metrics.csv, and the 4 PNG charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    per_image_path = output_dir / "per_image_results.csv"
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

    summary_path = output_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    charts = generate_charts(df, summary_df, output_dir)

    print(f"\nSaved per-image CSV : {per_image_path}")
    print(f"Saved summary CSV   : {summary_path}")
    for chart in charts:
        print(f"Saved chart         : {chart}")


# =====================================================================
# 4. MAIN BENCHMARK LOOP
# =====================================================================

def main():
    args = parse_args()
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Load baseline captioning model once
    base_model = load_model(
        weights_path=WEIGHTS_PATH,
        vocab_path=str(VOCAB_PATH),
        device=device,
    )

    # Load real test (image_path, caption) pairs and group multiple captions per image
    pairs = load_captions(args.test_csv, str(IMAGES_DIR))
    grouped_refs: dict[str, list[str]] = {}
    for img_path, cap in pairs:
        grouped_refs.setdefault(img_path, []).append(cap)

    all_rows = []

    for sigma in args.sigmas:
        print(f"\n=== Evaluating sigma = {sigma} ===")

        dncnn_ckpt = Path(f"data/processed/dncnn_weights_noise_{sigma}.pt")
        e2e_ckpt = Path(f"data/processed/end2end_caption_weights_sigma{sigma}.pt")

        dncnn_model = load_dncnn_model(dncnn_ckpt, device) if dncnn_ckpt.is_file() else None
        e2e_model = load_end2end_model(e2e_ckpt, device) if e2e_ckpt.is_file() else None

        if dncnn_model is None:
            print(f"  [Info] No DnCNN weights for sigma={sigma} ({dncnn_ckpt}), skipping Pipeline_DnCNN.")
        if e2e_model is None:
            print(f"  [Info] No End2End weights for sigma={sigma} ({e2e_ckpt}), skipping End2End_FineTuned.")

        for idx, (img_path, ref_captions) in enumerate(grouped_refs.items()):
            clean_tensor = load_image_tensor(img_path, device)
            noisy_tensor = add_gaussian_noise(
                clean_tensor,
                sigma=float(sigma),
                seed=args.seed + idx + sigma * 100000,
            )

            image_id = Path(img_path).name

            # --- Model 1: Baseline (noisy image -> base caption model) ---
            t0 = time.perf_counter()
            base_cap = base_model.generate_caption_text(noisy_tensor)
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

            # --- Model 2: Pipeline_DnCNN (noisy -> DnCNN denoise -> base caption model) ---
            if dncnn_model is not None:
                # --- Model 2: Pipeline_DnCNN (noisy -> DnCNN denoise -> base caption model) ---
            if dncnn_model is not None:
                t0 = time.perf_counter()
                was_3d = noisy_tensor.ndim == 3
                dncnn_input = noisy_tensor.unsqueeze(0) if was_3d else noisy_tensor
                denoised_tensor = dncnn_model.denoise(dncnn_input)
                if was_3d and denoised_tensor.ndim == 4 and denoised_tensor.size(0) == 1:
                    denoised_tensor = denoised_tensor.squeeze(0)
                p1_cap = base_model.generate_caption_text(denoised_tensor)
                p1_ms = (time.perf_counter() - t0) * 1000.0
                denoise_scores = compute_psnr_ssim(clean_tensor, denoised_tensor)
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

            # --- Model 3: End2End_FineTuned (noisy image -> fine-tuned caption model) ---
            if e2e_model is not None:
                t0 = time.perf_counter()
                p2_cap = e2e_model.generate_caption_text(noisy_tensor)
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

    save_reports_and_charts(all_rows, Path(args.output_dir))


if __name__ == "__main__":
    main()