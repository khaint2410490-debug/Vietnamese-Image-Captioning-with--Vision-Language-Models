"""
train.py: Điều khiển quá trình huấn luyện mô hình image captioning.

Luồng xử lý chính:
    for epoch:
        for (image, caption):
            1. image_feature = CNN(image)
            2. prediction    = Decoder(image_feature, các từ trước đó)
            3. loss          = CategoricalCrossEntropy(prediction, từ đúng)
            4. gradients     = backpropagation(loss, model)
            5. optimizer.step(model, gradients)
        tính validation loss (không cập nhật trọng số)
        lưu checkpoint nếu val loss tốt hơn, dừng sớm nếu không cải thiện

Teacher forcing: với caption [<start>, w1, w2, ..., wn, <end>], mô hình học
dự đoán từng từ kế tiếp dựa vào các từ đúng đứng trước nó:
    [<start>]            --> w1
    [<start>, w1]        --> w2
    ...
    [<start>, ..., wn]   --> <end>
Tất cả các bước của một caption được gộp chung trong một lần tính, và ảnh chỉ đi qua CNN một lần.

Validation & early stopping:
    - Tách tập validation theo ẢNH (xem split_samples), không tách theo cặp (ảnh, caption),
      vì một ảnh có nhiều caption và dễ bị rò rỉ dữ liệu nếu tách theo cặp.
    - Sau mỗi epoch tính val loss theo đúng công thức lúc train. Nếu giảm hơn
      best_val - min_delta thì lưu vào weights_path; sau `patience` epoch không giảm thì dừng.

Định dạng CSV đầu vào: header `image,caption`, ảnh .jpg nằm trong images_dir.
"""

import csv
import json
import random
from pathlib import Path

import torch

from model.captioning.image_captioning import ImageCaptioningModel
from preprocessing.image_preprocessing import preprocess_image
from preprocessing.tokenizer import Tokenizer
from .loss import categorical_cross_entropy_with_logits


def load_captions(captions_path: str, images_dir: str) -> list[tuple[str, str]]:
    """Đọc file caption, trả về list (đường dẫn ảnh, caption), bỏ qua ảnh không tồn tại."""
    path = Path(captions_path)
    images_dir = Path(images_dir)

    pairs: list[tuple[str, str]] = []
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for name, captions in data.items():
            if isinstance(captions, str):
                captions = [captions]
            pairs.extend((name, c) for c in captions)
    else:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or not {"image", "caption"} <= set(reader.fieldnames):
                raise ValueError("Caption CSV file must contain headers: image,caption")
            pairs = [(row["image"].strip(), row["caption"]) for row in reader]

    found, missing = [], 0
    for name, caption in pairs:
        image_path = images_dir / name
        if image_path.is_file():
            found.append((str(image_path), caption))
        else:
            missing += 1
    if missing:
        print(f"Warning: skipped {missing} captions whose images are missing in {images_dir}")
    if not found:
        raise RuntimeError("No valid (image, caption) pairs available for training")
    return found


def encode_captions(
    pairs: list[tuple[str, str]], tokenizer: Tokenizer, max_length: int
) -> list[tuple[str, list[int]]]:
    """Chuyển caption sang list token ID, cắt ngắn nếu vượt quá max_length (vẫn giữ <end>)."""
    samples = []
    for image_path, caption in pairs:
        ids = tokenizer.encode(caption)
        if len(ids) > max_length:
            ids = ids[: max_length - 1] + [tokenizer.end_id]
        samples.append((image_path, ids))
    return samples


def split_samples(
    samples: list[tuple[str, list[int]]], val_ratio: float = 0.1, seed: int = 42
) -> tuple[list[tuple[str, list[int]]], list[tuple[str, list[int]]]]:
    """
    Tách samples thành (train, validation) THEO ẢNH để toàn bộ caption của cùng một ảnh
    luôn nằm chung một phía.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be strictly between 0 and 1")

    by_image: dict[str, list] = {}
    for sample in samples:
        by_image.setdefault(sample[0], []).append(sample)

    image_paths = sorted(by_image)
    random.Random(seed).shuffle(image_paths)

    n_val = max(1, round(len(image_paths) * val_ratio))
    if n_val >= len(image_paths):
        raise ValueError("Insufficient distinct images to split a validation set")

    val = [s for p in image_paths[:n_val] for s in by_image[p]]
    train = [s for p in image_paths[n_val:] for s in by_image[p]]
    return train, val


def make_batch(batch: list[tuple[str, list[int]]], pad_id: int, device: torch.device):
    """
    Gom một nhóm (ảnh, caption) thành các tensor teacher forcing.

    Trả về:
        images    : (B, 3, H, W)
        row_image : (R,)   chỉ số ảnh tương ứng của từng dòng dự đoán
        prev_ids  : (R, L) các token đứng trước, đệm <pad> cho đủ L
        targets   : (R,)   token đúng cần dự đoán
    """
    images = torch.cat([preprocess_image(path) for path, _ in batch], dim=0)

    row_image, prev_rows, targets = [], [], []
    for b, (_, ids) in enumerate(batch):
        for t in range(1, len(ids)):
            row_image.append(b)
            prev_rows.append(ids[:t])
            targets.append(ids[t])

    max_len = max(len(row) for row in prev_rows)
    prev_rows = [row + [pad_id] * (max_len - len(row)) for row in prev_rows]

    return (
        images.to(device),
        torch.tensor(row_image, dtype=torch.long, device=device),
        torch.tensor(prev_rows, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


def compute_batch_loss(
    model: ImageCaptioningModel,
    batch: list[tuple[str, list[int]]],
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """
    Bước 1-3: ảnh --> CNN --> decoder --> loss. Dùng chung cho train và validation.
    Trả về (loss trung bình theo dòng, số dòng dự đoán R).
    """
    images, row_image, prev_ids, targets = make_batch(batch, model.decoder.pad_id, device)

    # 1. image_feature = CNN(image)
    image_features = model.encode_image(images)
    image_features = image_features[row_image]

    # 2. prediction = Decoder(image_feature, previous words) (logits trước softmax)
    prediction = model.decoder(image_features, prev_ids, return_logits=True)

    # 3. loss = Categorical Cross Entropy(prediction, target)
    loss = categorical_cross_entropy_with_logits(prediction, targets)
    return loss, targets.numel()


@torch.no_grad()
def evaluate(
    model: ImageCaptioningModel,
    samples: list[tuple[str, list[int]]],
    batch_size: int,
    device: torch.device,
) -> float:
    """
    Tính validation loss (không cập nhật weight, không shuffle), trung bình theo số dòng dự đoán.
    """
    model.eval()

    total_loss, total_rows = 0.0, 0
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        loss, n_rows = compute_batch_loss(model, batch, device)
        if not torch.isfinite(loss):
            raise FloatingPointError("Validation loss is invalid (nan/inf).")
        total_loss += loss.item() * n_rows
        total_rows += n_rows

    if not total_rows:
        raise RuntimeError("No valid tokens available to compute validation loss.")
    return total_loss / total_rows


def train_one_epoch(
    model: ImageCaptioningModel,
    samples: list[tuple[str, list[int]]],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
    log_every: int = 50,
) -> float:
    """Chạy một epoch huấn luyện, trả về loss trung bình theo số dòng."""
    model.train()
    samples = samples[:]
    random.shuffle(samples)

    total_loss, total_rows, num_steps = 0.0, 0, 0
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]

        loss, n_rows = compute_batch_loss(model, batch, device)
        if not torch.isfinite(loss):
            print(
                "Warning: non-finite loss (nan/inf), skipping batch. "
                "Consider lowering learning_rate."
            )
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * n_rows
        total_rows += n_rows
        num_steps += 1
        if num_steps % log_every == 0:
            print(f"  step {num_steps}: loss = {loss.item():.4f}")

    if not total_rows:
        raise FloatingPointError("No valid training batches completed in this epoch.")
    return total_loss / total_rows


def train(
    model: ImageCaptioningModel,
    samples: list[tuple[str, list[int]]],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    weights_path: str,
    max_length: int,
    log_every: int = 50,
    val_samples: list[tuple[str, list[int]]] | None = None,
    patience: int = 3,
    min_delta: float = 1e-4,
    fine_tune_after_epoch: int | None = 3,
    fine_tune_last_n_stages: int = 2,
    backbone_learning_rate: float | None = None,
) -> list[dict]:
    """
    Huấn luyện mô hình.

    Args:
        val_samples            : tập validation (xem split_samples). None = lưu mỗi epoch.
        patience               : dừng sớm nếu val loss không giảm sau chừng này epoch (<= 0 để tắt).
        min_delta              : mức giảm tối thiểu của val loss để tính là có cải thiện.
        fine_tune_after_epoch  : mở khóa các stage cuối sau epoch này; None để giữ đóng băng.
        fine_tune_last_n_stages: số stage cuối của backbone được mở khóa.
        backbone_learning_rate : learning rate nhỏ hơn dùng cho phần backbone khi fine-tune.

    Khi có val_samples, weights_path giữ checkpoint có val loss thấp nhất.

    Returns:
        history: list dict {"epoch", "train_loss", "val_loss"} theo từng epoch.
    """
    if not samples:
        raise ValueError("No training samples provided.")
    if batch_size < 1 or epochs < 1 or log_every < 1:
        raise ValueError("epochs, batch_size, and log_every must be > 0.")
    if learning_rate <= 0 or (
        backbone_learning_rate is not None and backbone_learning_rate <= 0
    ):
        raise ValueError("learning_rate and backbone_learning_rate must be > 0.")
    if min_delta < 0:
        raise ValueError("min_delta must be >= 0.")
    if val_samples is not None and not val_samples:
        raise ValueError("val_samples was provided but is empty.")
    if fine_tune_after_epoch is not None and fine_tune_after_epoch < 0:
        raise ValueError("fine_tune_after_epoch must be >= 0 or None.")
    if (
        fine_tune_after_epoch is not None
        and fine_tune_after_epoch > epochs
    ):
        raise ValueError("fine_tune_after_epoch cannot exceed epochs.")
    if fine_tune_last_n_stages < 1:
        raise ValueError("fine_tune_last_n_stages must be > 0.")

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    if backbone_learning_rate is None:
        backbone_learning_rate = learning_rate * 0.1

    def add_backbone_parameters() -> int:
        optimized_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        parameters = [
            parameter
            for parameter in model.encoder.backbone.parameters()
            if parameter.requires_grad and id(parameter) not in optimized_ids
        ]
        if parameters:
            optimizer.add_param_group(
                {
                    "params": parameters,
                    "lr": backbone_learning_rate,
                    "weight_decay": 1e-4,
                }
            )
        return len(parameters)

    if fine_tune_after_epoch == 0:
        model.encoder.unfreeze_backbone(fine_tune_last_n_stages)
        add_backbone_parameters()

    sample_images = list(dict.fromkeys(image_path for image_path, _ in samples))

    use_val = bool(val_samples)
    best_val, best_epoch, bad_epochs = float("inf"), 0, 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_loss = train_one_epoch(model, samples, optimizer, batch_size, device, log_every)
        record = {"epoch": epoch, "train_loss": train_loss}

        if use_val:
            val_loss = evaluate(model, val_samples, batch_size, device)
            record["val_loss"] = val_loss
            print(
                f"Epoch {epoch}: train loss = {train_loss:.4f}, "
                f"val loss = {val_loss:.4f}"
            )
        else:
            print(f"Epoch {epoch}: mean train loss = {train_loss:.4f}")

        sample_image = random.choice(sample_images)
        caption = model.tokens_to_text(
            model.generate_caption(sample_image, max_length=max_length)
        )
        print(f"  sample caption for {Path(sample_image).name}: {caption}")
        history.append(record)

        if fine_tune_after_epoch == epoch:
            model.encoder.unfreeze_backbone(fine_tune_last_n_stages)
            new_parameters = add_backbone_parameters()
            print(
                f"  Unfroze last {fine_tune_last_n_stages} encoder stage(s) "
                f"({new_parameters} tensors added to optimizer, "
                f"learning rate = {backbone_learning_rate:g})"
            )

        if not use_val:
            model.save_weights(weights_path)
            continue

        if val_loss < best_val - min_delta:
            best_val, best_epoch, bad_epochs = val_loss, epoch, 0
            model.save_weights(weights_path)
            print(f"  val loss improved; saved best checkpoint to {weights_path}")
        else:
            bad_epochs += 1
            print(f"  val loss did not improve ({bad_epochs}/{patience})")
            if patience > 0 and bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    if use_val:
        model.load_weights(weights_path)
        print(
            f"Best checkpoint: epoch {best_epoch}, val loss = {best_val:.4f} "
            f"({weights_path})"
        )
    return history
