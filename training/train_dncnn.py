"""
training/train_dncnn.py
Huấn luyện mạng khử nhiễu DnCNN với Residual MSE Loss, AdamW, Patch-training và Early Stopping.
"""

from pathlib import Path
from typing import Optional, Union
import torch
from torch.utils.data import DataLoader

from model.denoising.dncnn import DnCNN
from preprocessing.dataset import NoisyCaptionDataset
from evaluation.denoise_metrics import compute_psnr_ssim
from project_paths import (
    TRAIN_CAPTIONS_PATH,
    VAL_CAPTIONS_PATH,
    IMAGES_DIR,
    get_dncnn_weights_path,
)


def dncnn_residual_loss(predicted_noise: torch.Tensor, target_noise: torch.Tensor) -> torch.Tensor:
    batch_size = predicted_noise.size(0)
    diff = predicted_noise - target_noise
    return torch.sum(diff * diff) / (2.0 * batch_size)


def random_crop_pair(noisy: torch.Tensor, target_noise: torch.Tensor, patch_size: int = 64):
    _, _, h, w = noisy.shape
    if h <= patch_size or w <= patch_size:
        return noisy, target_noise
    top = torch.randint(0, h - patch_size + 1, (1,)).item()
    left = torch.randint(0, w - patch_size + 1, (1,)).item()
    return (
        noisy[:, :, top : top + patch_size, left : left + patch_size],
        target_noise[:, :, top : top + patch_size, left : left + patch_size],
    )


def train_dncnn(
    train_csv: Path = TRAIN_CAPTIONS_PATH,
    val_csv: Path = VAL_CAPTIONS_PATH,
    clean_images_dir: Path = IMAGES_DIR,
    noisy_images_dir: Optional[Path] = None,
    weights_save_path: Optional[Path] = None,
    noise_level: Union[int, float, str] = 25,
    num_of_layers: Optional[int] = None,
    epochs: int = 15,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    patience: int = 5,
    patch_size: int = 64,
    limit: Optional[int] = None,
    device: Optional[str] = None,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if weights_save_path is None:
        weights_save_path = get_dncnn_weights_path(noise_level)

    if num_of_layers is None:
        num_of_layers = 20 if str(noise_level).lower() == "random" else 17

    train_dataset = NoisyCaptionDataset(
        split_csv_path=train_csv,
        clean_images_dir=clean_images_dir,
        noisy_images_dir=noisy_images_dir,
        noise_level=noise_level,
        unique_images=True,
    )
    val_dataset = NoisyCaptionDataset(
        split_csv_path=val_csv,
        clean_images_dir=clean_images_dir,
        noisy_images_dir=noisy_images_dir,
        noise_level=noise_level,
        unique_images=True,
    )

    if limit is not None:
        if limit < 1:
            raise ValueError("limit phải lớn hơn 0.")
        train_dataset.samples = train_dataset.samples[:limit]
        val_dataset.samples = val_dataset.samples[: max(1, limit // 8)]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = DnCNN(num_of_layers=num_of_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_psnr = -1.0
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_psnr": [], "val_ssim": []}

    print(
        f"[Train DnCNN] Device: {device} | Layers: {num_of_layers} | "
        f"Train images: {len(train_dataset)} | Patch: {patch_size}x{patch_size}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            noisy = batch["noisy"].to(device)
            target_noise = batch["noise_target"].to(device)
            noisy_patch, target_patch = random_crop_pair(noisy, target_noise, patch_size=patch_size)

            optimizer.zero_grad(set_to_none=True)
            pred_noise = model(noisy_patch)
            loss = dncnn_residual_loss(pred_noise, target_patch)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * noisy.size(0)
            if step % 100 == 0 or step == len(train_loader):
                print(f"  [Epoch {epoch:02d}] Step {step}/{len(train_loader)} - Batch Loss: {loss.item():.4f}", end="\r")

        print()
        avg_train_loss = running_train_loss / max(1, len(train_dataset))

        model.eval()
        running_val_loss = 0.0
        psnr_scores, ssim_scores = [], []

        with torch.no_grad():
            for batch in val_loader:
                clean = batch["clean"].to(device)
                noisy = batch["noisy"].to(device)
                target_noise = batch["noise_target"].to(device)

                pred_noise = model(noisy)
                val_loss = dncnn_residual_loss(pred_noise, target_noise)
                running_val_loss += val_loss.item() * noisy.size(0)

                denoised = torch.clamp(noisy - pred_noise, 0.0, 1.0)
                metrics = compute_psnr_ssim(clean, denoised)
                psnr_scores.append(metrics["psnr"])
                ssim_scores.append(metrics["ssim"])

        avg_val_loss = running_val_loss / max(1, len(val_dataset))
        avg_psnr = sum(psnr_scores) / max(1, len(psnr_scores))
        avg_ssim = sum(ssim_scores) / max(1, len(ssim_scores))

        scheduler.step(avg_val_loss)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_psnr"].append(avg_psnr)
        history["val_ssim"].append(avg_ssim)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val PSNR: {avg_psnr:.2f} dB | Val SSIM: {avg_ssim:.4f}"
        )

        if avg_psnr > best_val_psnr:
            best_val_psnr = avg_psnr
            epochs_no_improve = 0
            weights_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_of_layers": num_of_layers,
                    "noise_level": noise_level,
                    "val_loss": avg_val_loss,
                    "val_psnr": avg_psnr,
                    "val_ssim": avg_ssim,
                },
                weights_save_path,
            )
            print(f"  -> Đã lưu checkpoint DnCNN tốt nhất tại: {weights_save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[Early Stopping] Dừng sớm tại epoch {epoch}.")
                break

    return history
