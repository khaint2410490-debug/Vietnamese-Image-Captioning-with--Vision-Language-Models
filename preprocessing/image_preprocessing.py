"""
image_preprocessing.py

Pipeline: Image (.jpg) --> Resize --> Normalize --> Tensor --> CNN

Input : đường dẫn tới file ảnh .jpg
Output: torch.Tensor có shape (1, 3, H, W), sẵn sàng đưa vào CNN

Yêu cầu: pip install torch pillow numpy
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Mặc định theo chuẩn ImageNet 
IMAGE_SIZE = (224, 224)  # (width, height)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Các bước tiền xử lý
def load_image(image_path: str) -> Image.Image:
    """Bước 0: Đọc ảnh .jpg và đưa về RGB (3 kênh)."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    if path.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError(f"Chỉ hỗ trợ ảnh .jpg/.jpeg, nhận được: {path.suffix}")
    return Image.open(path).convert("RGB")

# Tiến hành resizee ảnh
def resize_image(image: Image.Image, size=IMAGE_SIZE) -> Image.Image:
    """Bước 1: Resize ảnh về kích thước cố định."""
    return image.resize(size, Image.BILINEAR)

# Chuẩn hóa ảnh
def normalize_image(image: Image.Image, mean=MEAN, std=STD) -> np.ndarray:
    """
    Bước 2: Normalize.
    - Đưa pixel từ [0, 255] về [0, 1]
    - Chuẩn hóa theo từng kênh: (pixel - mean) / std
    Trả về mảng float32 shape (H, W, C).
    """
    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    return (array - mean) / std

# Chuyển thành tensor
def to_tensor(array: np.ndarray) -> torch.Tensor:
    """
    Bước 3: Chuyển sang Tensor.
    - Đổi layout (H, W, C) -> (C, H, W) theo chuẩn PyTorch
    - Thêm chiều batch -> (1, C, H, W)
    """
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.unsqueeze(0)

# def chính cần import để xử lí ảnh
def preprocess_image(image_path: str, size=IMAGE_SIZE) -> torch.Tensor:
    """Chạy toàn bộ pipeline: Image -> Resize -> Normalize -> Tensor."""
    image = load_image(image_path)
    image = resize_image(image, size)
    array = normalize_image(image)
    return to_tensor(array)
