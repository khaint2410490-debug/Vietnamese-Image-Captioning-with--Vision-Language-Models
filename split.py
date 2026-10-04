"""
split.py

Chia file captions.csv thành 3 tập: train, validation, test (mặc định 0.8 / 0.1 / 0.1).

Chia THEO ẢNH, không chia theo caption: một ảnh thường có nhiều caption, nên toàn bộ
caption của cùng một ảnh luôn nằm chung một tập. Nếu chia theo caption, cùng một ảnh có
thể vừa xuất hiện trong train vừa xuất hiện trong test, khiến kết quả đánh giá cao giả tạo.

Định dạng file vào (có header):
    image,caption
    1000268201_693b08cb0e.jpg,"Một đứa trẻ mặc váy hồng đang leo lên một bộ cầu thang theo lối vào ."

output: train.csv, validation.csv, test.csv.
"""

import csv
import random
from pathlib import Path

from project_paths import CAPTIONS_PATH, IMAGES_DIR

DEFAULT_CAP_PATH = CAPTIONS_PATH

def read_captions(captions_path: str = DEFAULT_CAP_PATH) -> dict[str, list[str]]:
    """Đọc captions.csv, gom các caption theo ảnh: {tên ảnh: [caption, ...]}."""
    grouped: dict[str, list[str]] = {}
    with open(captions_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not {"image", "caption"} <= set(reader.fieldnames):
            raise ValueError("File caption cần có header với hai cột: image,caption")
        for row in reader:
            image = (row["image"] or "").strip()
            caption = (row["caption"] or "").strip()
            if image and caption:
                grouped.setdefault(image, []).append(caption)
    if not grouped:
        raise RuntimeError("Không đọc được cặp (ảnh, caption) nào từ file")
    return grouped

def split_images(
    images: list[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Xáo trộn danh sách ảnh (cố định theo seed) rồi cắt thành train / validation / test."""
    images = sorted(images)                       
    """sắp xếp trước để kết quả không phụ thuộc thứ tự file"""
    random.Random(seed).shuffle(images)

    n = len(images)
    n_val = max(1, round(n * val_ratio)) if val_ratio > 0 else 0
    n_test = max(1, round(n * test_ratio)) if test_ratio > 0 else 0
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError(f"Chỉ có {n} ảnh, không đủ để chia thành 3 tập")

    train = images[:n_train]
    val = images[n_train : n_train + n_val]
    test = images[n_train + n_val :]
    return train, val, test

def write_split(path: Path, images: list[str], grouped: dict[str, list[str]]) -> int:
    """Ghi các ảnh trong một tập (kèm toàn bộ caption của chúng) ra file .csv. Trả về số caption."""
    count = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["image", "caption"])
        for image in images:
            for caption in grouped[image]:
                writer.writerow([image, caption])
                count += 1
    return count

def split_dataset(
    captions_path: str | Path = CAPTIONS_PATH,
    output_dir: str | Path | None = None,
    images_dir: str | Path | None = IMAGES_DIR,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Path, Path, Path]:
    """Chia captions theo ảnh, ghi CSV và trả về đường dẫn train/validation/test."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio phải bằng 1")

    captions_path = Path(captions_path)
    grouped = read_captions(str(captions_path))

    if images_dir is not None:
        images_dir = Path(images_dir)
        missing = [name for name in grouped if not (images_dir / name).is_file()]
        for name in missing:
            del grouped[name]
        if missing:
            print(f"Cảnh báo: bỏ qua {len(missing)} ảnh không tồn tại trong {images_dir}")
        if not grouped:
            raise RuntimeError("Không còn ảnh nào hợp lệ để chia")

    train, val, test = split_images(list(grouped), val_ratio, test_ratio, seed)

    # Kiểm tra: không có ảnh nào nằm ở hai tập
    assert not (set(train) & set(val)) and not (set(train) & set(test)) and not (set(val) & set(test))

    output_dir = Path(output_dir) if output_dir else captions_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    total_captions = sum(len(c) for c in grouped.values())
    print(f"Tổng: {len(grouped)} ảnh, {total_captions} caption")
    paths = []
    for name, images in (("train", train), ("validation", val), ("test", test)):
        path = output_dir / f"{name}.csv"
        n_captions = write_split(path, images, grouped)
        print(f"  {name:<10}: {len(images):>6} ảnh, {n_captions:>7} caption  -> {path}")
        paths.append(path)
    return paths[0], paths[1], paths[2]
