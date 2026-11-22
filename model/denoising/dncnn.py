"""
model/denoising/dncnn.py
Kiến trúc mạng DnCNN dựa trên Residual Learning.
"""

import torch
import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_of_layers: int = 17,
        features: int = 64,
    ):
        super().__init__()
        if num_of_layers < 3:
            raise ValueError("num_of_layers phải >= 3")

        layers = [
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        ]

        for _ in range(num_of_layers - 2):
            layers.extend([
                nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
            ])

        layers.append(
            nn.Conv2d(features, out_channels, kernel_size=3, padding=1, bias=True)
        )

        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dncnn(x)

    @torch.no_grad()
    def denoise(self, noisy_image: torch.Tensor) -> torch.Tensor:
        predicted_noise = self.forward(noisy_image)
        return torch.clamp(noisy_image - predicted_noise, 0.0, 1.0)
