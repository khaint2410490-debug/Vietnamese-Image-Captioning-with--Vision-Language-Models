"""
preprocessing/dataset.py
Quản lý NoisyCaptionDataset cho huấn luyện DnCNN và Image Captioning.
"""

import csv
from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from preprocessing.image_preprocessing import IMAGE_SIZE, MEAN, STD, resize_image
from project_paths import IMAGES_DIR


def image_to_unit_tensor(image: Image.Image, size=IMAGE_SIZE) -> torch.Tensor:
    """Resize ảnh và chuyển về Tensor float32 trong miền [0, 1], shape (3, H, W)."""
    resized = resize_image(image.convert("RGB"), size=size)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def normalize_imagenet_batch(tensor: torch.Tensor, mean=MEAN, std=STD) -> torch.Tensor:
    """Chuẩn hóa tensor ảnh miền [0, 1] sang miền ImageNet cho ResNet-18."""
    device = tensor.device
    dtype = tensor.dtype
    mean_t = torch.tensor(mean, dtype=dtype, device=device).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=dtype, device=device).view(-1, 1, 1)
    if tensor.dim() == 4:
        mean_t = mean_t.unsqueeze(0)
        std_t = std_t.unsqueeze(0)
    return (tensor - mean_t) / std_t


class NoisyCaptionDataset(Dataset):
    def __init__(
        self,
        split_csv_path: Union[str, Path],
        clean_images_dir: Union[str, Path] = IMAGES_DIR,
        noisy_images_dir: Optional[Union[str, Path]] = None,
        tokenizer=None,
        noise_level: Union[int, float, str] = 25,
        unique_images: bool = False,
        max_len: int = 40,
        seed: int = 42,
    ):
        self.split_csv_path = Path(split_csv_path)
        self.clean_images_dir = Path(clean_images_dir)
        self.noisy_images_dir = Path(noisy_images_dir) if noisy_images_dir else None
        self.tokenizer = tokenizer
        self.noise_level = noise_level
        self.unique_images = unique_images
        self.max_len = max_len
        self.seed = seed

        self.samples = []
        seen_images = set()

        with open(self.split_csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row["image"].strip()
                caption = row.get("caption", "").strip()
                if not (self.clean_images_dir / img_name).is_file():
                    continue
                if self.unique_images:
                    if img_name in seen_images:
                        continue
                    seen_images.add(img_name)
                self.samples.append((img_name, caption))

    def __len__(self) -> int:
        return len(self.samples)

    def _encode_caption(self, caption: str) -> torch.Tensor:
        if self.tokenizer is None or not caption:
            return torch.empty(0, dtype=torch.long)
        if hasattr(self.tokenizer, "encode_caption"):
            try:
                ids = self.tokenizer.encode_caption(caption, max_length=self.max_len, pad=True)
            except TypeError:
                ids = self.tokenizer.encode_caption(caption)
        elif hasattr(self.tokenizer, "encode"):
            ids = self.tokenizer.encode(caption)
        else:
            ids = []
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int) -> dict:
        img_name, caption = self.samples[idx]
        clean_path = self.clean_images_dir / img_name

        with Image.open(clean_path) as clean_pil:
            clean_tensor = image_to_unit_tensor(clean_pil)

        if self.noisy_images_dir and (self.noisy_images_dir / img_name).is_file():
            with Image.open(self.noisy_images_dir / img_name) as noisy_pil:
                noisy_tensor = image_to_unit_tensor(noisy_pil)
        else:
            rng = np.random.default_rng(self.seed + idx)
            sigma = (
                float(rng.uniform(0.0, 55.0))
                if str(self.noise_level).lower() == "random"
                else float(self.noise_level)
            )
            noise = torch.from_numpy(
                rng.normal(0.0, sigma / 255.0, size=clean_tensor.shape).astype(np.float32)
            )
            noisy_tensor = torch.clamp(clean_tensor + noise, 0.0, 1.0)

        noise_target = noisy_tensor - clean_tensor

        return {
            "image_name": img_name,
            "clean": clean_tensor,
            "noisy": noisy_tensor,
            "noise_target": noise_target,
            "caption": caption,
            "tokens": self._encode_caption(caption),
        }
