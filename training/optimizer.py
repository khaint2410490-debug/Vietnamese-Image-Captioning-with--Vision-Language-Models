"""
optimizer.py: Cập nhật weight bằng Gradient Descent:

    w_new = w_current - learning_rate * gradient

- gradient dương  --> w giảm
- gradient âm     --> w tăng
"""

import torch
import torch.nn as nn


def gradient_descent(
    weight: torch.Tensor, gradient: torch.Tensor, learning_rate: float
) -> torch.Tensor:
    """Công thức Gradient Descent cho một weight: w_new = w - lr * gradient."""
    if weight.shape != gradient.shape:
        raise ValueError(
            f"weight {tuple(weight.shape)} and gradient {tuple(gradient.shape)} must match in shape"
        )
    return weight - learning_rate * gradient


class GradientDescent:
    """
    Nhận current weight (từ network), gradient và learning rate rồi ghi weight mới
    lại vào network.

    Args:
        learning_rate: bước học (lr). Có thể chỉnh giữa các epoch qua `learning_rate`
                       để giảm dần tốc độ học.
    """

    def __init__(self, learning_rate: float = 0.01):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        self.learning_rate = learning_rate

    @torch.no_grad()
    def step(self, network: nn.Module, gradients: dict[str, torch.Tensor]) -> None:
        """
        Cập nhật mọi weight của network tại chỗ (in-place).

        Args:
            network  : mô hình cần cập nhật weight.
            gradients: dict {tên weight: gradient} trả về từ backpropagation().
        """
        for name, param in network.named_parameters():
            if name not in gradients:
                continue
            new_weight = gradient_descent(param, gradients[name], self.learning_rate)
            param.copy_(new_weight)
