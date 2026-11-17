"""
dataset.py

Cung cấp lớp `NoisyCaptionDataset` đọc cùng lúc ảnh sạch, ảnh nhiễu và caption tiếng Việt
đã mã hóa thành token ID, dùng được cho cả bước khử nhiễu (DnCNN), fine-tune lẫn đánh giá.

Mỗi mẫu __getitem__(idx) trả về một dict gồm:
    {
        "clean_image"   : Tensor float32 (3, 224, 224) chuẩn ImageNet,
        "noisy_image"   : Tensor float32 (3, 224, 224) chuẩn ImageNet,
        "noise_target"  : Tensor float32 (3, 224, 224) = noisy_image - clean_image,
        "caption_tokens": Tensor int64   (T,) gồm <start> ... <end> <pad>...,
        "image_name"    : str,
        "caption"       : str,
        "clean_path"    : str,
        "noisy_path"    : str,
        "sigma"         : float,
    }
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.image_preprocessing import IMAGE_SIZE, MEAN, STD, denormalize_image
from preprocessing.noise_generator import (
    DEFAULT_RANDOM_RANGE,
    _parse_noise_level,
    add_awgn_to_array,
)
from preprocessing.tokenizer import Tokenizer
from project_paths import (
    IMAGES_DIR,
    NOISY_IMAGE_MAP_PATH,
    NOISY_IMAGES_DIR,
    PROJECT_ROOT,
    TEST_CAPTIONS_PATH,
    TRAIN_CAPTIONS_PATH,
    VALIDATION_CAPTIONS_PATH,
    VOCAB_PATH,
    get_noisy_images_dir,
)


def build_default_transform(
    image_size: tuple[int, int] = IMAGE_SIZE,
    mean: tuple[float, float, float] = MEAN,
    std: tuple[float, float, float] = STD,
) -> transforms.Compose:
    """Biến đổi chuẩn: Resize (height, width) -> ToTensor [0, 1] -> Normalize ImageNet."""
    # IMAGE_SIZE trong image_preprocessing.py là (width, height), torchvision Resize nhận (H, W)
    size_hw = (image_size[1], image_size[0])
    return transforms.Compose(
        [
            transforms.Resize(size_hw, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_tensor_transform(
    mean: tuple[float, float, float] = MEAN,
    std: tuple[float, float, float] = STD,
) -> transforms.Compose:
    """Biến đổi từ ảnh PIL đã đúng kích thước sang Tensor chuẩn hóa ImageNet (không resize lại)."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _resolve_project_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _load_noisy_map(
    map_path: Path,
    noise_level: int | float | str | None = None,
) -> dict[str, tuple[Path, float]]:
    """
    Đọc noisy_image_map.csv thành bảng tra {image_name: (noisy_path, sigma)}.
    Nếu file chứa nhiều mức nhiễu và `noise_level` được chỉ định, tự động lọc đúng mức đó.
    """
    if not map_path.is_file():
        return {}

    desired_mode: str | None = None
    desired_sigma: float | None = None
    if noise_level is not None and not (isinstance(noise_level, str) and not noise_level.strip()):
        desired_mode, desired_sigma, _ = _parse_noise_level(noise_level)

    mapping: dict[str, tuple[Path, float]] = {}
    with open(map_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_name = (row.get("image") or "").strip()
            raw_noisy = (row.get("noisy_path") or "").strip()
            if not image_name or not raw_noisy:
                continue

            row_mode = (row.get("noise_mode") or "").strip().lower()
            try:
                row_sigma = float(row.get("sigma", "-1"))
            except ValueError:
                row_sigma = -1.0

            if desired_mode is not None:
                if desired_mode == "random":
                    if row_mode and row_mode != "random":
                        continue
                else:
                    if row_mode == "random":
                        continue
                    if desired_sigma is not None and row_sigma >= 0:
                        if abs(row_sigma - desired_sigma) > 1e-4:
                            continue

            noisy_path = _resolve_project_path(raw_noisy)
            if noisy_path.is_file():
                mapping[image_name] = (noisy_path, row_sigma)

    return mapping


def _infer_level_from_dir(
    noisy_dir: Path,
    fallback: int | float | str | None = None,
) -> int | float | str | None:
    """Suy luận nhãn mức nhiễu từ tên thư mục (ví dụ NoisyImages_sigmarandom -> 'random')."""
    if noisy_dir.name.startswith("NoisyImages_sigma"):
        suffix = noisy_dir.name.removeprefix("NoisyImages_sigma").strip()
        if suffix:
            return suffix
    if fallback is not None and not (isinstance(fallback, str) and not fallback.strip()):
        return fallback
    return None


def _sigma_from_level_spec(level: int | float | str | None) -> float:
    """Chuyển nhãn mức nhiễu sang giá trị sigma số (trả về -1.0 nếu là 'random' hoặc None)."""
    if level is None or (isinstance(level, str) and not level.strip()):
        return -1.0
    try:
        mode, sigma_val, _ = _parse_noise_level(level)
    except ValueError:
        return -1.0
    if mode == "fixed" and sigma_val is not None:
        return float(sigma_val)
    return -1.0


class NoisyCaptionDataset(Dataset):
    """
    PyTorch Dataset quản lý bộ ba (ảnh sạch, ảnh nhiễu, caption tiếng Việt).
    """

    def __init__(
        self,
        split_csv_path: str | Path = TRAIN_CAPTIONS_PATH,
        images_dir: str | Path = IMAGES_DIR,
        noisy_images_dir: str | Path | None = None,
        noisy_map_path: str | Path | None = NOISY_IMAGE_MAP_PATH,
        tokenizer: Tokenizer | str | Path | None = None,
        max_length: int = 20,
        image_size: tuple[int, int] = IMAGE_SIZE,
        noise_level: int | float | str | None = None,
        on_the_fly: bool = False,
        random_range: tuple[float, float] = DEFAULT_RANDOM_RANGE,
        unique_images_only: bool = False,
        seed: int = 42,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        noiseL: int | float | str | None = None,
        noise_after_resize: bool = False,
    ):
        super().__init__()
        if noise_level is None and noiseL is not None:
            noise_level = noiseL
        self.split_csv_path = Path(split_csv_path)
        self.images_dir = Path(images_dir)
        self.max_length = max_length
        self.image_size = image_size
        self.on_the_fly = on_the_fly
        self.random_range = random_range
        self.unique_images_only = unique_images_only
        self.seed = seed
        self.noise_after_resize = noise_after_resize
        self.transform = transform or build_default_transform(image_size=image_size)
        self._post_resize_transform = transform or build_tensor_transform()

        if not self.split_csv_path.is_file():
            raise FileNotFoundError(f"Split CSV file not found: {self.split_csv_path}")
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Clean images directory not found: {self.images_dir}")

        if isinstance(tokenizer, Tokenizer):
            self.tokenizer = tokenizer
        else:
            vocab_path = Path(tokenizer) if tokenizer is not None else VOCAB_PATH
            self.tokenizer = Tokenizer.load(str(vocab_path))

        # Cấu hình thông số sinh nhiễu nếu dùng chế độ on_the_fly
        if self.on_the_fly:
            level_spec = "25" if noise_level is None else noise_level
            self._fly_mode, self._fly_sigma, _ = _parse_noise_level(level_spec)
            self._rng: np.random.Generator | None = np.random.default_rng(seed)
            self._rng_worker_id: int | None = -1
            self._noisy_map: dict[str, tuple[Path, float]] = {}
            self.noisy_images_dir: Path | None = None
            self._use_default_noisy_dir = False
            self._explicit_noisy_dir = False
            self._fallback_sigma = float(self._fly_sigma) if self._fly_sigma is not None else -1.0
        else:
            self._fly_mode, self._fly_sigma = "fixed", -1.0
            self._rng = None
            self._rng_worker_id = None
            map_file = Path(noisy_map_path) if noisy_map_path is not None else None
            default_noisy_resolved = NOISY_IMAGES_DIR.resolve()

            if noisy_images_dir is not None:
                self.noisy_images_dir = Path(noisy_images_dir)
                self._explicit_noisy_dir = True
                self._use_default_noisy_dir = (
                    self.noisy_images_dir.resolve() == default_noisy_resolved
                )
                inferred_level = _infer_level_from_dir(
                    self.noisy_images_dir,
                    fallback=25 if self._use_default_noisy_dir else noise_level,
                )
                should_load_map = map_file is not None and (
                    inferred_level is not None
                    or map_file.resolve() != NOISY_IMAGE_MAP_PATH.resolve()
                )
                self._noisy_map = (
                    _load_noisy_map(map_file, noise_level=inferred_level)
                    if should_load_map
                    else {}
                )
                self._fallback_sigma = _sigma_from_level_spec(inferred_level)
            else:
                self._explicit_noisy_dir = False
                is_default_level = noise_level is None or (
                    isinstance(noise_level, str) and not noise_level.strip()
                )
                self._use_default_noisy_dir = is_default_level
                self.noisy_images_dir = (
                    NOISY_IMAGES_DIR if is_default_level else get_noisy_images_dir(noise_level)
                )
                map_level: int | float | str = 25 if is_default_level else noise_level  # type: ignore[assignment]
                self._noisy_map = (
                    _load_noisy_map(map_file, noise_level=map_level)
                    if map_file is not None
                    else {}
                )
                self._fallback_sigma = _sigma_from_level_spec(map_level)

        self.samples = self._load_samples()

    def _get_rng(self) -> np.random.Generator:
        """Cấp RNG tách biệt theo từng DataLoader worker khi chạy on_the_fly với num_workers > 0."""
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else -1
        if self._rng is None or self._rng_worker_id != worker_id:
            rng_seed = self.seed if worker_id < 0 else (self.seed + (worker_id + 1) * 100_003)
            self._rng = np.random.default_rng(rng_seed)
            self._rng_worker_id = worker_id
        return self._rng

    def _resolve_noisy_path(self, image_name: str) -> tuple[Path | None, float]:
        if self.on_the_fly:
            return None, -1.0

        mapped = self._noisy_map.get(image_name)

        if self.noisy_images_dir is not None and (
            self._explicit_noisy_dir or self._use_default_noisy_dir
        ):
            candidate = self.noisy_images_dir / image_name
            if candidate.is_file():
                if mapped is not None and (
                    self._use_default_noisy_dir
                    or mapped[0].resolve() == candidate.resolve()
                ):
                    return candidate, mapped[1]
                sigma_val = _sigma_from_level_spec(_infer_level_from_dir(self.noisy_images_dir))
                if sigma_val < 0:
                    sigma_val = self._fallback_sigma
                return candidate, sigma_val
            if self._explicit_noisy_dir:
                return None, -1.0

        if mapped is not None:
            return mapped

        if self.noisy_images_dir is not None:
            candidate = self.noisy_images_dir / image_name
            if candidate.is_file():
                sigma_val = _sigma_from_level_spec(_infer_level_from_dir(self.noisy_images_dir))
                if sigma_val < 0:
                    sigma_val = self._fallback_sigma
                return candidate, sigma_val

        return None, -1.0

    def _load_samples(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        seen_images: set[str] = set()
        missing_clean = 0
        missing_noisy = 0

        with open(self.split_csv_path, "r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or not {"image", "caption"} <= set(reader.fieldnames):
                raise ValueError(
                    f"File {self.split_csv_path} must include 'image' and 'caption' headers."
                )
            for row in reader:
                image_name = (row["image"] or "").strip()
                caption = (row["caption"] or "").strip()
                if not image_name or not caption:
                    continue
                if self.unique_images_only and image_name in seen_images:
                    continue

                clean_path = self.images_dir / image_name
                if not clean_path.is_file():
                    missing_clean += 1
                    continue

                noisy_path, sigma_val = self._resolve_noisy_path(image_name)
                if not self.on_the_fly and noisy_path is None:
                    missing_noisy += 1
                    continue

                seen_images.add(image_name)
                samples.append(
                    {
                        "image_name": image_name,
                        "caption": caption,
                        "clean_path": clean_path,
                        "noisy_path": noisy_path,
                        "sigma": sigma_val,
                    }
                )

        if missing_clean > 0:
            print(
                f"Warning ({self.split_csv_path.name}): skipped {missing_clean} rows "
                f"missing clean images in {self.images_dir}."
            )
        if missing_noisy > 0:
            print(
                f"Warning ({self.split_csv_path.name}): skipped {missing_noisy} rows "
                f"missing noisy images (run noise_generator.py or set on_the_fly=True)."
            )
        if not samples:
            raise RuntimeError(
                f"No valid samples loaded from {self.split_csv_path}. "
                "Verify clean images directory or run preprocessing/noise_generator.py first."
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        with Image.open(item["clean_path"]) as img:
            clean_pil = img.convert("RGB")

        if self.noise_after_resize:
            clean_pil = clean_pil.resize(
                (self.image_size[0], self.image_size[1]),
                Image.Resampling.BILINEAR,
            )

        if self.on_the_fly or (self.noise_after_resize and item["sigma"] >= 0):
            if self.on_the_fly:
                rng = self._get_rng()
                sigma = (
                    float(rng.uniform(*self.random_range))
                    if self._fly_mode == "random"
                    else float(self._fly_sigma)
                )
            else:
                sigma = float(item["sigma"])
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, idx]))

            clean_array = np.asarray(clean_pil, dtype=np.float32)
            noisy_array = add_awgn_to_array(clean_array, sigma=sigma, rng=rng)
            noisy_pil = Image.fromarray(noisy_array, mode="RGB")
            noisy_path_str = ""
        else:
            sigma = float(item["sigma"])
            noisy_path = item["noisy_path"]
            with Image.open(noisy_path) as img:
                noisy_pil = img.convert("RGB")
            noisy_path_str = str(noisy_path)

        active_transform = self._post_resize_transform if self.noise_after_resize else self.transform
        clean_tensor = active_transform(clean_pil)
        noisy_tensor = active_transform(noisy_pil)
        noise_target = noisy_tensor - clean_tensor

        token_ids = self.tokenizer.encode_caption(item["caption"], max_length=self.max_length)
        caption_tokens = torch.tensor(token_ids, dtype=torch.long)

        return {
            "clean_image": clean_tensor,
            "noisy_image": noisy_tensor,
            "noise_target": noise_target,
            "caption_tokens": caption_tokens,
            "image_name": item["image_name"],
            "caption": item["caption"],
            "clean_path": str(item["clean_path"]),
            "noisy_path": noisy_path_str,
            "sigma": sigma,
        }


def collate_noisy_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Gom list các mẫu từ NoisyCaptionDataset thành một batch cho DataLoader."""
    return {
        "clean_image": torch.stack([b["clean_image"] for b in batch], dim=0),
        "noisy_image": torch.stack([b["noisy_image"] for b in batch], dim=0),
        "noise_target": torch.stack([b["noise_target"] for b in batch], dim=0),
        "caption_tokens": torch.stack([b["caption_tokens"] for b in batch], dim=0),
        "sigma": torch.tensor([b["sigma"] for b in batch], dtype=torch.float32),
        "image_name": [b["image_name"] for b in batch],
        "caption": [b["caption"] for b in batch],
        "clean_path": [b["clean_path"] for b in batch],
        "noisy_path": [b["noisy_path"] for b in batch],
    }


def build_teacher_forcing_targets(
    caption_tokens: torch.Tensor,
    pad_id: int = 0,
    end_id: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tách batch `caption_tokens` (B, T) thành (row_image, prev_ids, targets) cho decoder."""
    row_image: list[int] = []
    prev_rows: list[list[int]] = []
    targets: list[int] = []
    device = caption_tokens.device

    for batch_idx, seq in enumerate(caption_tokens.tolist()):
        # Cắt bỏ padding phía sau <end>
        valid_ids: list[int] = []
        for token_id in seq:
            if token_id == pad_id:
                break
            valid_ids.append(token_id)
            if token_id == end_id:
                break

        for step in range(1, len(valid_ids)):
            row_image.append(batch_idx)
            prev_rows.append(valid_ids[:step])
            targets.append(valid_ids[step])

    if not prev_rows:
        raise ValueError("Batch contains no valid tokens to build teacher forcing pairs.")

    max_prev_len = max(len(row) for row in prev_rows)
    padded_prev = [row + [pad_id] * (max_prev_len - len(row)) for row in prev_rows]

    return (
        torch.tensor(row_image, dtype=torch.long, device=device),
        torch.tensor(padded_prev, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


def create_dataloaders(
    batch_size: int = 8,
    noise_level: int | float | str | None = 25,
    on_the_fly: bool = False,
    unique_images_only: bool = False,
    max_length: int = 20,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    tokenizer: Tokenizer | str | Path | None = None,
    images_dir: str | Path = IMAGES_DIR,
    noisy_images_dir: str | Path | None = None,
    noisy_map_path: str | Path | None = NOISY_IMAGE_MAP_PATH,
    noiseL: int | float | str | None = None,
    noise_after_resize: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Tạo nhanh 3 DataLoader (train, validation, test)."""
    tok = (
        tokenizer
        if isinstance(tokenizer, Tokenizer)
        else Tokenizer.load(str(tokenizer or VOCAB_PATH))
    )
    effective_noise_level = noiseL if noiseL is not None else noise_level

    common_kwargs = {
        "images_dir": images_dir,
        "noisy_images_dir": noisy_images_dir,
        "noisy_map_path": noisy_map_path,
        "tokenizer": tok,
        "max_length": max_length,
        "noise_level": effective_noise_level,
        "on_the_fly": on_the_fly,
        "unique_images_only": unique_images_only,
        "seed": seed,
        "noise_after_resize": noise_after_resize,
    }

    train_ds = NoisyCaptionDataset(split_csv_path=TRAIN_CAPTIONS_PATH, **common_kwargs)
    val_ds = NoisyCaptionDataset(split_csv_path=VALIDATION_CAPTIONS_PATH, **common_kwargs)
    test_ds = NoisyCaptionDataset(split_csv_path=TEST_CAPTIONS_PATH, **common_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_noisy_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_noisy_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_noisy_batch,
    )
    return train_loader, val_loader, test_loader


def _mean_psnr_01(dataset: NoisyCaptionDataset, num_samples: int = 20) -> float:
    psnrs: list[float] = []
    limit = min(num_samples, len(dataset))
    for idx in range(limit):
        sample = dataset[idx]
        clean_01 = denormalize_image(sample["clean_image"]).numpy()
        noisy_01 = denormalize_image(sample["noisy_image"]).numpy()
        mse = float(np.mean((clean_01 - noisy_01) ** 2))
        psnrs.append(10.0 * math.log10(1.0 / max(mse, 1e-12)))
    return float(np.mean(psnrs))


def run_self_test(noise_level: str = "25", batch_size: int = 4) -> None:
    """Chạy kiểm thử nhanh toàn bộ Data Pipeline Giai đoạn 1."""
    print(f"=== PHASE 1 DATA PIPELINE SELF-TEST (noise_level={noise_level}) ===")
    for split_name, split_path in (
        ("train", TRAIN_CAPTIONS_PATH),
        ("validation", VALIDATION_CAPTIONS_PATH),
        ("test", TEST_CAPTIONS_PATH),
    ):
        ds_full = NoisyCaptionDataset(split_csv_path=split_path, noise_level=noise_level)
        ds_uniq = NoisyCaptionDataset(
            split_csv_path=split_path, noise_level=noise_level, unique_images_only=True
        )
        print(f"  {split_name:10s}: {len(ds_uniq):>5} unique images | {len(ds_full):>6} caption pairs")

    for lvl in (None, 15, 25, 50, "random"):
        ds_check = NoisyCaptionDataset(split_csv_path=TEST_CAPTIONS_PATH, noise_level=lvl)
        s0 = ds_check[0]
        assert s0["clean_image"].shape == (3, IMAGE_SIZE[1], IMAGE_SIZE[0])
        assert s0["noisy_image"].shape == (3, IMAGE_SIZE[1], IMAGE_SIZE[0])
        assert (s0["noisy_image"] - s0["clean_image"] - s0["noise_target"]).abs().max().item() == 0.0

    ds_alias = NoisyCaptionDataset(split_csv_path=TEST_CAPTIONS_PATH, noiseL=noise_level)
    assert ds_alias[0]["clean_image"].shape == (3, IMAGE_SIZE[1], IMAGE_SIZE[0])

    tr_loader, va_loader, te_loader = create_dataloaders(
        batch_size=batch_size, noise_level=noise_level
    )
    batch = next(iter(tr_loader))
    row_img, prev_ids, targets = build_teacher_forcing_targets(batch["caption_tokens"])
    print(
        f"  DataLoader batch={tuple(batch['noisy_image'].shape)} | "
        f"teacher_forcing rows={row_img.numel()} | prev_ids={tuple(prev_ids.shape)}"
    )

    ds_disk = NoisyCaptionDataset(
        split_csv_path=TEST_CAPTIONS_PATH,
        noise_level=noise_level,
        unique_images_only=True,
        noise_after_resize=False,
    )
    ds_resized = NoisyCaptionDataset(
        split_csv_path=TEST_CAPTIONS_PATH,
        noise_level=noise_level,
        unique_images_only=True,
        noise_after_resize=True,
    )
    psnr_disk = _mean_psnr_01(ds_disk, num_samples=20)
    psnr_resized = _mean_psnr_01(ds_resized, num_samples=20)
    print(f"  Mean PSNR (20 test images, default disk)      : {psnr_disk:.2f} dB")
    print(f"  Mean PSNR (20 test images, noise_after_resize): {psnr_resized:.2f} dB")
    assert len(va_loader) > 0 and len(te_loader) > 0 and targets.numel() == row_img.numel()
    print("SELF-TEST PASSED!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test NoisyCaptionDataset and DataLoader for Phase 1."
    )
    parser.add_argument(
        "--self_test",
        "--self_test.",
        action="store_true",
        help="Run quick Phase 1 Data Pipeline self-test (default when running file directly).",
    )
    parser.add_argument("--noise_level", "--noiseL", dest="noise_level", default="25")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    run_self_test(noise_level=args.noise_level, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
