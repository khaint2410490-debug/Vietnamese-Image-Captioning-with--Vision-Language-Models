"""
layers.py

Một lớp neuron (dense / fully-connected layer). Mỗi neuron thực hiện 2 bước:

    Bước 1 - Linear sum : z = w1*x1 + w2*x2 + ... + wn*xn + b
    Bước 2 - Activation : a = f(z)

Cả lớp tính song song cho mọi neuron và mọi mẫu trong batch bằng phép nhân ma trận:

    Z = X @ W + b          X: (B, in_features)   W: (in_features, out_features)
    A = f(Z)               Z, A: (B, out_features)
"""

import math

import torch
import torch.nn as nn

from .activation import get_activation

class DenseLayer(nn.Module):
    """
    Lớp gồm `out_features` neuron, mỗi neuron nhận toàn bộ `in_features` đầu vào.

    Args:
        in_features : số đầu vào của mỗi neuron.
        out_features: số neuron trong lớp.
        activation  : 'relu', 'sigmoid', 'softmax' hoặc 'linear'.
    """

    def __init__(self, in_features: int, out_features: int, activation: str = "relu"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_name = activation
        self.activation = get_activation(activation)

        # Mỗi cột của W là trọng số của một neuron; b là bias của từng neuron
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self._init_weights()

    def _init_weights(self) -> None:
        """He init cho ReLU, Xavier init cho các activation còn lại."""
        if self.activation_name.lower() == "relu":
            std = math.sqrt(2.0 / self.in_features)
        else:
            std = math.sqrt(2.0 / (self.in_features + self.out_features))
        with torch.no_grad():
            self.weight.normal_(0.0, std)
            self.bias.zero_()

    def linear_sum(self, x: torch.Tensor) -> torch.Tensor:
        """Bước 1: z = x @ W + b."""
        return x @ self.weight + self.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Bước 1 (linear sum) rồi Bước 2 (activation)."""
        z = self.linear_sum(x)
        return self.activation(z)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"activation={self.activation_name}"
        )