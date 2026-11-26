# Vietnamese Noisy Image Captioning

Dự án nghiên cứu sinh chú thích tiếng Việt cho ảnh bị nhiễu Gaussian. Hệ thống
dùng CNN ResNet-18 để mã hóa ảnh và decoder Attention + LSTM để sinh caption.
Ngoài mô hình captioning nền huấn luyện trên ảnh sạch, dự án có hai cách xử lý
ảnh nhiễu để so sánh.

## Các pipeline

1. **Baseline:** ảnh sạch → mô hình captioning đã huấn luyện trên ảnh sạch.
2. **DnCNN (hai giai đoạn):** ảnh nhiễu → DnCNN dự đoán phần nhiễu → ảnh đã
   khử nhiễu → mô hình captioning baseline.
3. **Captioner train trực tiếp trên ảnh nhiễu:** ảnh nhiễu → mô hình
   CNN + decoder được huấn luyện trực tiếp trên ảnh nhiễu; không có bước khử
   nhiễu riêng.

Baseline được dùng cho phương án DnCNN. Phương án captioner trên ảnh nhiễu
khởi tạo decoder mới và dùng CNN pretrained trên ImageNet; nó không nạp
checkpoint captioning baseline. Cả hai captioner dùng cùng vocabulary để so
sánh trên cùng không gian token.

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

- Dataset Flickr8k: [Kaggle – adityajn105/flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k).
- `data/raw/captions.csv`: CSV có header `image,caption`, chứa caption tiếng
  Việt; một ảnh có thể có nhiều caption.
- `data/raw/Images/`: ảnh gốc, tên file khớp với cột `image`.
- `split.py` tạo `data/processed/train.csv`, `validation.csv` và `test.csv`,
  chia theo ảnh để caption của cùng ảnh không rơi vào nhiều tập.
- `preprocessing/noise_generator.py` tạo ảnh nhiễu ở
  `data/processed/NoisyImages_sigma{level}/` và file ánh xạ riêng theo mức nhiễu
  `data/processed/noisy_image_map_noise_{level}.csv`. Nhiễu được cộng trên pixel
  RGB [0,255], sau đó ảnh được lưu JPEG.
- `data/vocabulary/tokenizer.json`: vocabulary dùng chung với mô hình captioning.
- `data/processed/caption_weights.pt`: checkpoint baseline.
- `data/processed/dncnn_weights_noise_{level}.pt`: checkpoint DnCNN theo mức nhiễu.
- `data/processed/end2end_caption_weights_sigma{level}.pt`: checkpoint
  end-to-end theo mức nhiễu.

Các checkpoint và dữ liệu sinh ra cần được giữ cùng vocabulary, split và mức
nhiễu tương ứng. Không dùng tập test để huấn luyện hay chọn checkpoint.

## Quy trình huấn luyện

Chuẩn bị `data/raw/captions.csv` và ảnh trong `data/raw/Images/`, sau đó chỉ cần
chạy từ thư mục gốc:

```powershell
python main.py
```

`main.py` thực hiện tuần tự toàn bộ quy trình:

1. Chia dữ liệu theo ảnh, xây vocabulary và train baseline captioning trên ảnh
   sạch. Baseline làm captioner sau DnCNN ở phương án 1 và là mốc đánh giá.
2. Tạo ảnh nhiễu Gaussian tại
   `data/processed/NoisyImages_sigma{NOISE_LEVEL}/`.
3. **Phương án 1:** train DnCNN trên cặp ảnh sạch/nhiễu, lưu
   `data/processed/dncnn_weights_noise_{NOISE_LEVEL}.pt`. Khi inference, ảnh nhiễu qua DnCNN rồi
   captioner baseline sinh caption.
4. **Phương án 2:** khởi tạo captioner mới, dùng CNN pretrained trên ImageNet,
   rồi train trực tiếp trên ảnh nhiễu (mở khóa các stage cuối ResNet-18);
   checkpoint được lưu tại
   `data/processed/end2end_caption_weights_sigma{NOISE_LEVEL}.pt`.

Mức nhiễu và thông số train được đặt ở đầu `main.py`: `NOISE_LEVEL`,
`EPOCHS`, `DNCNN_EPOCHS`, `END2END_EPOCHS`, batch size và learning rate. Mức
nhiễu cố định thường dùng là `15`, `25`, `50`; giá trị `random` lấy sigma riêng
cho từng ảnh. Đặt `TRAINING_LIMIT` thành số nguyên nhỏ để chạy thử nhanh; để
`None` khi train toàn bộ dữ liệu. Mỗi lần chạy `main.py` sẽ train lại baseline
và cả hai phương án.

Dòng lệnh riêng của `noise_generator.py`, `train_dncnn.py` và
`train_end2end.py` không còn cần dùng; `main.py` gọi trực tiếp các hàm này.

## Benchmark

`evaluation/benchmark.v2.py` chạy inference thực tế trên các ảnh trong
`data/processed/test.csv`, đối chiếu nhiều caption tham chiếu trên mỗi ảnh và
lưu từng ảnh ra `data/processed/benchmark_comparison.csv`. Đo lường:

- BLEU-1..4 theo cách tách token tiếng Việt trong dự án.
- Thời gian inference cho baseline, pipeline DnCNN và end-to-end.
- PSNR/SSIM giữa ảnh sạch và ảnh do DnCNN khôi phục.

Hai benchmark không có parser/CLI và không được gọi khi train. Các hàm chạy
benchmark dùng cấu hình mặc định trong file; benchmark v2 nhận mức nhiễu từ
hằng `NOISE_LEVEL`, các giới hạn mẫu và độ dài caption từ các hằng ở đầu file.
Benchmark v2 bỏ qua Pipeline 2 nếu checkpoint end-to-end tương ứng chưa có.
Thiếu ảnh sạch hoặc nhiễu trong tập test sẽ báo lỗi thay vì âm thầm bỏ mẫu.

PSNR/SSIM nhận ảnh dạng [0,1] hoặc [0,255] và chọn `data_range` tương ứng.
`evaluation/benchmark.v1.py` đo baseline; `benchmark.v2.py` so sánh cả ba
pipeline.

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
| `training/train_end2end.py` | Train captioner trực tiếp trên ảnh nhiễu |
| `evaluation/denoise_metrics.py` | PSNR/SSIM cho tensor trong [0,1] |
| `evaluation/benchmark.v2.py` | So sánh inference ba pipeline trên tập test |
| `project_paths.py` | Đường dẫn artifact dùng chung trong dự án |

`main.py` là entrypoint huấn luyện duy nhất cho baseline và cả hai phương án
noisy-captioning. Benchmark là tiện ích đánh giá riêng, không chạy trong quá
trình huấn luyện.
