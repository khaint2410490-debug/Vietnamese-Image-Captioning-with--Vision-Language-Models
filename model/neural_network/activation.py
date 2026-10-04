"""
activation.py

Các hàm kích hoạt (activation function) dùng trong neural network.

    sigmoid(z) = 1 / (1 + e^-z)            -> ép giá trị về khoảng (0, 1)
    relu(z)    = max(0, z)                 -> giữ phần dương, bỏ phần âm
    softmax(z) = e^z_i / sum_j(e^z_j)      -> biến vector điểm số thành phân phối
                                              xác suất (các giá trị >= 0, tổng = 1)
"""

import torch

def sigmoid(z: torch.Tensor) -> torch.Tensor:
    """Sigmoid: ép mỗi phần tử về khoảng (0, 1)."""
    return 1.0 / (1.0 + torch.exp(-z))

def relu(z: torch.Tensor) -> torch.Tensor:
    """ReLU: max(0, z)."""
    return torch.clamp(z, min=0.0)

def softmax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax theo chiều `dim` (mặc định là chiều cuối, tức chiều từ vựng).
    Trừ đi giá trị lớn nhất trước khi lấy mũ để tránh tràn số (numerical stability);
    kết quả không đổi về mặt toán học.
    """
    shifted = z - z.max(dim=dim, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / exp.sum(dim=dim, keepdim=True)

def linear(z: torch.Tensor) -> torch.Tensor:
    """Không kích hoạt (identity): dùng khi cần giữ nguyên tổng tuyến tính z."""
    return z

_ACTIVATIONS = {
    "sigmoid": sigmoid,
    "relu": relu,
    "softmax": softmax,
    "linear": linear,
}

def get_activation(name: str):
    """Lấy hàm kích hoạt theo tên: 'sigmoid', 'relu', 'softmax' hoặc 'linear'."""
    key = name.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(
            f"Activation '{name}' không được hỗ trợ. Chọn một trong: {list(_ACTIVATIONS)}"
        )
    return _ACTIVATIONS[key]