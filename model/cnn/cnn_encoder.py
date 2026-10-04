"""
cnn_encoder.py

Pipeline:
    Image --> Pretrained backbone (ResNet / EfficientNet) --> Feature map
          --> Global avg pooling --> Linear projection --> Image feature vector

Backbone được lấy từ torchvision, mặc định nạp trọng số ImageNet (pretrained).
Cách dùng khuyến nghị gồm 2 giai đoạn:

    Giai đoạn 1 - đóng băng backbone, chỉ train lớp projection (head):
        enc = CNNEncoder(backbone="resnet18", pretrained=True, freeze_backbone=True)
        optimizer = torch.optim.AdamW(enc.get_param_groups(head_lr=1e-3))

    Giai đoạn 2 - mở khóa vài stage cuối rồi fine-tune với lr nhỏ:
        enc.unfreeze_backbone(last_n_stages=2)      # hoặc None để mở toàn bộ
        optimizer = torch.optim.AdamW(
            enc.get_param_groups(backbone_lr=1e-5, head_lr=1e-4)
        )

Ví dụ shape với ảnh 224 x 224 x 3 (resnet18):

    Input            : (B,   3, 224, 224)
    Feature map      : (B, 512,   7,   7)   <-- extract_feature_map
    Global avg pooling : (B, 512)
    Linear projection  : (B, feature_dim)   <-- image feature vector

LƯU Ý: backbone pretrained được huấn luyện với ảnh đã chuẩn hóa theo ImageNet
(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). Hãy chuẩn hóa như vậy
trong image_preprocessing.preprocess_image, hoặc đặt normalize_input=True ở đây
(khi đó input phải là ảnh trong khoảng [0, 1], chưa chuẩn hóa).

Input là output của image_preprocessing.preprocess_image: tensor (B, 3, H, W).
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# dùng backbone pretrained để encode ảnh
class CNNEncoder(nn.Module):
    """
    Encoder CNN pretrained: chuyển image tensor thành image feature vector.

    Args:
        backbone        : tên model torchvision, ví dụ "resnet18", "resnet34",
                          "resnet50", "efficientnet_b0", "efficientnet_b1", ...
        pretrained      : nạp trọng số ImageNet (True) hay khởi tạo ngẫu nhiên (False).
        freeze_backbone : đóng băng backbone ngay từ đầu (chỉ train projection).
        feature_dim     : độ dài của image feature vector đầu ra.
        dropout         : tỉ lệ dropout trước lớp Linear (0 để tắt).
        in_channels     : số kênh ảnh đầu vào (backbone pretrained yêu cầu 3 - RGB).
        normalize_input : tự chuẩn hóa ImageNet bên trong encoder (input trong [0, 1]).
        keep_bn_eval    : luôn để BatchNorm của backbone ở chế độ eval, kể cả khi
                          fine-tune (hữu ích khi batch size nhỏ).
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        feature_dim: int = 256,
        dropout: float = 0.0,
        in_channels: int = 3,
        normalize_input: bool = False,
        keep_bn_eval: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.backbone_name = backbone
        self.keep_bn_eval = keep_bn_eval

        stages, out_channels = self._build_backbone(backbone, pretrained)
        # backbone chia thành các stage theo thứ tự từ nông đến sâu,
        # để có thể mở khóa dần từ stage cuối (xem unfreeze_backbone)
        self.backbone = nn.Sequential(*stages)
        self.feature_map_channels = out_channels

        self.normalize_input = normalize_input
        if normalize_input:
            self.register_buffer(
                "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
            )
            self.register_buffer(
                "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
            )

        # Gộp feature map (C, H, W) thành vector C chiều bất kể kích thước ảnh
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.projection = nn.Linear(out_channels, feature_dim)

        if freeze_backbone:
            self.freeze_backbone()

    # build
    @staticmethod
    def _build_backbone(name: str, pretrained: bool):
        """Trả về (list các stage, số kênh của feature map cuối)."""
        weights = "DEFAULT" if pretrained else None
        model = models.get_model(name, weights=weights)

        if name.startswith("resnet"):
            stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
            stages = [stem, model.layer1, model.layer2, model.layer3, model.layer4]
            out_channels = model.fc.in_features
        elif name.startswith("efficientnet"):
            # model.features là Sequential các khối (stem, MBConv stages, conv cuối)
            stages = list(model.features.children())
            out_channels = model.classifier[-1].in_features
        else:
            raise ValueError(
                f"Backbone '{name}' chưa được hỗ trợ. Dùng resnet* hoặc efficientnet*."
            )
        return stages, out_channels

    # freeze / tune
    def freeze_backbone(self) -> None:
        """Đóng băng toàn bộ backbone; chỉ projection (và dropout) được train."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()  # BatchNorm không cập nhật running stats

    def unfreeze_backbone(self, last_n_stages: Optional[int] = None) -> None:
        """
        Mở khóa backbone để fine-tune.

        Args:
            last_n_stages: chỉ mở khóa n stage cuối (các stage nông giữ nguyên
                           vì chứa đặc trưng chung như cạnh, texture). None = mở toàn bộ.
        """
        stages = list(self.backbone.children())
        if last_n_stages is not None and last_n_stages < 1:
            raise ValueError("last_n_stages phải lớn hơn 0 hoặc là None.")
        n = len(stages) if last_n_stages is None else min(last_n_stages, len(stages))
        for i, stage in enumerate(stages):
            trainable = i >= len(stages) - n
            for p in stage.parameters():
                p.requires_grad = trainable
        self.train(self.training)  # áp lại chế độ train/eval cho từng stage

    def train(self, mode: bool = True):
        """Stage nào còn bị đóng băng (hoặc keep_bn_eval) thì giữ BatchNorm ở eval."""
        super().train(mode)
        if mode:
            for stage in self.backbone.children():
                if not any(p.requires_grad for p in stage.parameters()):
                    stage.eval()
            if self.keep_bn_eval:
                for m in self.backbone.modules():
                    if isinstance(m, nn.modules.batchnorm._BatchNorm):
                        m.eval()
        return self

    def get_param_groups(
        self,
        backbone_lr: Optional[float] = None,
        head_lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> list[dict]:
        """
        Tạo param groups cho optimizer với learning rate riêng cho backbone và head.
        Chỉ gồm các tham số đang requires_grad.
        """
        groups = []
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        if backbone_params:
            if backbone_lr is None:
                raise ValueError("Backbone đang được mở khóa: hãy truyền backbone_lr.")
            groups.append(
                {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay}
            )
        head_params = [p for p in self.projection.parameters() if p.requires_grad]
        groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
        return groups

    # forward
    def _check_input(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError(
                f"Input phải có shape (B, C, H, W), nhận được {tuple(x.shape)}"
            )
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Input phải có {self.in_channels} kênh, nhận được {x.size(1)}"
            )

    def extract_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Image --> backbone.
        Trả về feature map (B, C, H', W'), ví dụ (B, 512, 7, 7) với resnet18.
        Hữu ích nếu decoder dùng attention trên từng vùng của ảnh.
        """
        self._check_input(x)
        if self.normalize_input:
            x = (x - self.mean) / self.std
        return self.backbone(x)

    def forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """Return a projected feature map (B, feature_dim, H', W') for attention."""
        feature_map = self.extract_feature_map(x)
        batch_size, channels, height, width = feature_map.shape
        locations = feature_map.flatten(2).transpose(1, 2)
        locations = self.projection(self.dropout(locations))
        return locations.transpose(1, 2).reshape(
            batch_size, self.feature_dim, height, width
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Image tensor (B, 3, H, W) --> Image feature vector (B, feature_dim).
        """
        feature_map = self.extract_feature_map(x)          # (B, C, H', W')
        pooled = self.global_pool(feature_map)             # (B, C, 1, 1)
        flat = torch.flatten(pooled, start_dim=1)          # (B, C)
        return self.projection(self.dropout(flat))         # (B, feature_dim)