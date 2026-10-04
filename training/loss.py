"""
Categorical Cross Entropy (CCE): đo mô hình dự đoán sai đến mức nào.

Ví dụ: ảnh con chó, từ đúng là "chó"
    p("chó") = 0.72  -->  L = -ln(0.72) ≈ 0.33   (loss nhỏ, dự đoán tốt)
    p("chó") = 0.03  -->  L = -ln(0.03) ≈ 3.51   (loss lớn, dự đoán tệ)

Loss của cả batch là trung bình loss của từng mẫu.
"""

import torch

def _check_inputs(scores: torch.Tensor, targets: torch.Tensor) -> None:
    if scores.dim() != 2:
        raise ValueError(f"Cần tensor (B, vocab_size), nhận được {tuple(scores.shape)}")
    if targets.dim() != 1 or targets.size(0) != scores.size(0):
        raise ValueError(
            f"targets phải có shape ({scores.size(0)},), nhận được {tuple(targets.shape)}"
        )

def _split_ignored(targets: torch.Tensor, ignore_index: int | None):
    """Trả về (targets an toàn để gather, mask các mẫu hợp lệ hoặc None)."""
    if ignore_index is None:
        return targets, None
    ignored = targets == ignore_index
    return targets.masked_fill(ignored, 0), (~ignored).float()

def _mean_loss(loss_each: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
    if valid is None:
        return loss_each.mean()
    return (loss_each * valid).sum() / valid.sum().clamp(min=1.0)

def categorical_cross_entropy(
    probs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-12,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """
    CCE tính trực tiếp từ xác suất (đầu ra của softmax).

    Args:
        probs       : (B, vocab_size), mỗi hàng là phân phối xác suất (tổng = 1).
        targets     : (B,) token ID của từ đúng.
        eps         : chặn dưới của xác suất để log(0) không ra vô cực.
        ignore_index: token ID không tính vào loss (ví dụ <pad>); None nếu không cần.
    Returns:
        Scalar tensor: loss trung bình của batch.
    """
    _check_inputs(probs, targets)
    safe_targets, valid = _split_ignored(targets.long(), ignore_index)
    p_correct = probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)    # (B,)
    loss_each = -torch.log(p_correct.clamp(min=eps))
    return _mean_loss(loss_each, valid)

def categorical_cross_entropy_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """
    CCE tính từ logits (đầu ra TRƯỚC softmax). Cho kết quả giống hệt
    categorical_cross_entropy(softmax(logits), targets) nhưng ổn định số học hơn:

        log p_i = z_i - log( sum_j e^z_j )

    Nên dùng hàm này khi huấn luyện.
    """
    _check_inputs(logits, targets)
    safe_targets, valid = _split_ignored(targets.long(), ignore_index)
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)     # (B, V)
    loss_each = -log_probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
    return _mean_loss(loss_each, valid)