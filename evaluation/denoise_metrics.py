"""
evaluation/denoise_metrics.py
Đánh giá chất lượng khôi phục thị giác của DnCNN bằng chỉ số PSNR và SSIM.
"""

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _to_hwc_numpy(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().clamp(0.0, 1.0).numpy()
    return np.transpose(arr, (1, 2, 0))


def compute_psnr_ssim(clean_batch: torch.Tensor, denoised_batch: torch.Tensor) -> dict:
    if clean_batch.dim() == 3:
        clean_batch = clean_batch.unsqueeze(0)
        denoised_batch = denoised_batch.unsqueeze(0)

    psnr_list = []
    ssim_list = []

    for i in range(clean_batch.size(0)):
        clean_np = _to_hwc_numpy(clean_batch[i])
        denoised_np = _to_hwc_numpy(denoised_batch[i])

        psnr_val = peak_signal_noise_ratio(clean_np, denoised_np, data_range=1.0)
        ssim_val = structural_similarity(
            clean_np, denoised_np, data_range=1.0, channel_axis=-1
        )
        psnr_list.append(float(psnr_val))
        ssim_list.append(float(ssim_val))

    return {
        "psnr": float(np.mean(psnr_list)) if psnr_list else 0.0,
        "ssim": float(np.mean(ssim_list)) if ssim_list else 0.0,
    }
