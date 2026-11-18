# Vietnamese Noisy Image Captioning

Dự án nghiên cứu sinh chú thích tiếng Việt cho ảnh bị nhiễu Gaussian. Hệ thống
dùng CNN ResNet-18 để mã hóa ảnh và decoder Attention + LSTM để sinh caption.
Ngoài mô hình captioning nền huấn luyện trên ảnh sạch, dự án có hai cách xử lý
ảnh nhiễu để so sánh.

## Các pipeline

1. **Baseline:** ảnh sạch → mô hình captioning đã huấn luyện trên ảnh sạch.
2. **DnCNN (hai giai đoạn):** ảnh nhiễu → DnCNN dự đoán phần nhiễu → ảnh đã
   khử nhiễu → mô hình captioning baseline.
3. **Fine-tune end-to-end:** ảnh nhiễu → mô hình captioning baseline được
   fine-tune trực tiếp với ảnh nhiễu; không có bước khử nhiễu riêng.

Baseline là điểm khởi đầu cho phương án fine-tune end-to-end. Checkpoint của
captioning và vocabulary phải cùng một lần huấn luyện vì các ID token trong
vocabulary gắn với embedding của mô hình.

## Yêu cầu và cài đặt

- Python hỗ trợ bởi phiên bản PyTorch/torchvision bạn cài.
- Cài PyTorch và torchvision phù hợp CPU hoặc CUDA của máy.
- Các thư viện Python còn lại được khai báo trong `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

`scikit-image` được dùng để tính PSNR/SSIM trong đánh giá DnCNN; thiếu thư viện
này sẽ khiến import module đánh giá hoặc huấn luyện DnCNN thất bại.

## Dữ liệu và artifact

- `data/raw/captions.csv`: CSV có header `image,caption`, chứa caption tiếng
  Việt; một ảnh có thể có nhiều caption.
- `data/raw/Images/`: ảnh gốc, tên file khớp với cột `image`.
- `split.py` tạo `data/processed/train.csv`, `validation.csv` và `test.csv`,
  chia theo ảnh để caption của cùng ảnh không rơi vào nhiều tập.
- `preprocessing/noise_generator.py` tạo ảnh nhiễu ở
  `data/processed/NoisyImages_sigma{level}/` và file ánh xạ
  `data/processed/noisy_image_map.csv`. Nhiễu được cộng trên pixel RGB [0,255],
  sau đó ảnh được lưu JPEG.
- `data/vocabulary/tokenizer.json`: vocabulary dùng chung với mô hình captioning.
- `data/processed/caption_weights.pt`: checkpoint baseline.
- `data/processed/dncnn_weights.pt`: checkpoint DnCNN.
- `data/processed/end2end_caption_weights_sigma{level}.pt`: checkpoint
  end-to-end theo mức nhiễu.

Các checkpoint và dữ liệu sinh ra cần được giữ cùng vocabulary, split và mức
nhiễu tương ứng. Không dùng tập test để huấn luyện hay chọn checkpoint.

## Quy trình huấn luyện

Chạy các lệnh từ thư mục gốc repository. Cần có dữ liệu ảnh và caption trước.

### 1. Huấn luyện baseline trên ảnh sạch

```powershell
python main.py
```

Lệnh này tạo lại các split theo ảnh, xây vocabulary từ tập train và huấn luyện
mô hình captioning baseline. Các tham số như epoch, batch size và learning rate
được cấu hình ở đầu `main.py`.

### 2. Sinh ảnh nhiễu

Chạy riêng cho mỗi mức nhiễu cần thực nghiệm:

```powershell
python -m preprocessing.noise_generator --noise_level 25
```

Các mức hỗ trợ được dự định dùng là `15`, `25`, `50` hoặc `random`. Lệnh tạo
ảnh nhiễu cùng tên file với ảnh sạch trong `data/processed/NoisyImages_sigma25/`.

### 3. Huấn luyện DnCNN (phương án khử nhiễu)

```powershell
python -m training.train_dncnn --noise_level 25
```

DnCNN học dự đoán residual `noisy - clean`, dùng loss bình phương residual,
AdamW, scheduler và early stopping theo PSNR validation. Ảnh đầu vào và mục
tiêu DnCNN được biểu diễn trong miền [0,1]. Mức nhiễu cố định dùng mặc định
17 lớp; `random` dùng 20 lớp.

### 4. Fine-tune captioning end-to-end

```powershell
python -m training.train_end2end --noise_level 25
```

Lệnh nạp baseline và vocabulary đã tạo ở bước 1, rồi fine-tune trên ảnh nhiễu
đã tạo ở bước 2. Mặc định mở khóa hai stage cuối của ResNet-18 ngay từ epoch
đầu; learning rate của encoder nhỏ hơn learning rate decoder. Có thể chạy thử
với ít mẫu:

```powershell
python -m training.train_end2end --noise_level 25 --limit 64 --epochs 1
```

Đầu ra mặc định là
`data/processed/end2end_caption_weights_sigma25.pt`. Muốn chạy lại ở mức nhiễu
khác, lặp lại bước sinh dữ liệu và huấn luyện với cùng `--noise_level`.

## Benchmark

`evaluation/benchmark.v2.py` chạy inference thực tế trên các ảnh trong
`data/processed/test.csv`, đối chiếu nhiều caption tham chiếu trên mỗi ảnh và
lưu từng ảnh ra `data/processed/benchmark_comparison.csv`. Đo lường:

- BLEU-1..4 theo cách tách token tiếng Việt trong dự án.
- Thời gian inference cho baseline, pipeline DnCNN và end-to-end.
- PSNR/SSIM giữa ảnh sạch và ảnh do DnCNN khôi phục.

Sinh dữ liệu nhiễu và huấn luyện DnCNN trước khi benchmark:

```powershell
python -m preprocessing.noise_generator --noise_level 25
python -m training.train_dncnn --noise_level 25
python evaluation/benchmark.v2.py --noise_level 25
```

Benchmark tìm checkpoint end-to-end tại
`data/processed/end2end_caption_weights_sigma25.pt`. Nếu chưa có checkpoint,
Pipeline 2 được ghi để trống và được thông báo là bỏ qua; truyền
`--end2end_weights <đường_dẫn>` để chỉ định checkpoint khác. Ví dụ chạy nhanh
trên một ảnh:

```powershell
python evaluation/benchmark.v2.py --noise_level 25 --limit 1
```

Có thể tùy chỉnh đường dẫn bằng `--captions`, `--images-dir`, `--noisy-dir`,
`--baseline-weights`, `--dncnn-weights`, `--end2end-weights`, `--vocab-path`
và `--output`; các cờ đường dẫn cũ dùng dấu gạch dưới vẫn được hỗ trợ. Chọn
thiết bị với `--device auto|cpu|cuda`. `--limit` phải là số nguyên dương,
`--max-length` phải từ 2 trở lên, còn `--noise-level` nhận `15`, `25`, `50`
hoặc `random`. Nếu thiếu ảnh sạch hoặc ảnh nhiễu trong tập test, benchmark dừng
và báo các tên ảnh bị thiếu thay vì âm thầm loại bỏ mẫu.

PSNR/SSIM nhận ảnh dạng [0,1] hoặc [0,255] và chọn `data_range` tương ứng.
`evaluation/benchmark.v1.py` là bản benchmark phác thảo; dùng `benchmark.v2.py`
cho kết quả inference trên mô hình và dữ liệu thực.

## Cấu trúc mã nguồn

| Thư mục/tệp | Vai trò |
| --- | --- |
| `preprocessing/noise_generator.py` | Tạo ảnh nhiễu Gaussian có seed và map file |
| `preprocessing/dataset.py` | Nạp cặp ảnh sạch/nhiễu, residual và caption |
| `preprocessing/image_preprocessing.py` | Resize và chuẩn hóa ảnh cho CNN |
| `model/denoising/dncnn.py` | DnCNN dự đoán nhiễu residual |
| `model/cnn/cnn_encoder.py` | ResNet/encoder trích xuất đặc trưng ảnh |
| `model/captioning/` | Embedding, spatial attention, LSTM và sinh caption |
| `training/train.py` | Vòng lặp train captioning, validation và checkpoint |
| `training/train_dncnn.py` | Huấn luyện và lưu DnCNN |
| `training/train_end2end.py` | Fine-tune captioning trên ảnh nhiễu |
| `evaluation/denoise_metrics.py` | PSNR/SSIM cho tensor trong [0,1] |
| `evaluation/benchmark.v2.py` | So sánh inference ba pipeline trên tập test |
| `project_paths.py` | Đường dẫn artifact dùng chung trong dự án |

`main.py` hiện huấn luyện baseline trên ảnh sạch; các lệnh huấn luyện DnCNN,
fine-tune ảnh nhiễu và benchmark được chạy riêng như hướng dẫn ở trên.
