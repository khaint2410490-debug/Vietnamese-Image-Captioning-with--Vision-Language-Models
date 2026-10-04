"""
Xác định, với mỗi weight của mạng: nếu weight thay đổi một chút thì loss thay đổi thế nào.
Xác định bằng Gradient

Cách đọc gradient của một weight w:
    gradient = +0.3 : tăng w một chút thì loss TĂNG (xấu đi)  --> nên GIẢM w
    gradient = -0.3 : tăng w một chút thì loss GIẢM (tốt lên) --> nên TĂNG w
    gradient ≈ 0    : w gần như không ảnh hưởng đến loss
Độ lớn |gradient| cho biết loss nhạy với w đến mức nào.

Backpropagation: áp dụng quy tắc chuỗi (chain rule) đi ngược từ loss về đầu vào:

    Loss <-- Softmax <-- Dense (z = xW + b, a = f(z)) <-- ... <-- Embedding / CNN

Khi chạy xuôi (forward), PyTorch ghi lại mọi phép tính đã dùng để tạo ra loss. Khi gọi
loss.backward(), nó đi ngược qua từng phép tính đó và nhân các đạo hàm cục bộ lại với nhau
để ra gradient của từng weight. File này KHÔNG thay đổi weight, chỉ trả về gradient;
việc cập nhật weight là của optimizer.py.
"""

import torch
import torch.nn as nn

def zero_gradients(network: nn.Module) -> None:
    """Xóa gradient cũ (PyTorch cộng dồn gradient qua các lần backward)."""
    for param in network.parameters():
        param.grad = None

def backpropagation(loss: torch.Tensor, network: nn.Module) -> dict[str, torch.Tensor]:
    """
    Args:
        loss   : loss (scalar) được tính từ output của network.
        network: mô hình đã dùng để tính ra loss.
    Returns:
        dict {tên weight: gradient}, gradient có cùng shape với weight tương ứng.
        Ví dụ: {"decoder.network.output_layer.weight": tensor(...), ...}
    """
    if loss.dim() != 0:
        raise ValueError(f"loss phải là scalar, nhận được shape {tuple(loss.shape)}")
    if not loss.requires_grad:
        raise ValueError("loss không gắn với network (không có đồ thị tính toán để lan truyền ngược)")

    zero_gradients(network)
    loss.backward()

    gradients: dict[str, torch.Tensor] = {}
    for name, param in network.named_parameters():
        if not param.requires_grad:
            continue
        grad = param.grad if param.grad is not None else torch.zeros_like(param)
        gradients[name] = grad.detach().clone()
    return gradients

def weight_direction(gradients: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Hướng nên thay đổi của từng weight để giảm loss:
        +1 : nên tăng weight,  -1 : nên giảm weight,  0 : giữ nguyên.
    """
    return {name: -torch.sign(grad) for name, grad in gradients.items()}

def gradient_norm(gradients: dict[str, torch.Tensor]) -> float:
    """Độ lớn (chuẩn L2) của toàn bộ gradient, dùng để theo dõi quá trình huấn luyện."""
    total = sum(float((grad ** 2).sum()) for grad in gradients.values())
    return total ** 0.5