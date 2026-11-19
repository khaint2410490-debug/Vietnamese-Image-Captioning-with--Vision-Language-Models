# Cấu trúc dự án

Tài liệu này mô tả các file mã nguồn và cấu hình chính theo thứ tự workflow.
Các ảnh trong dataset, file cache `__pycache__` và checkpoint sinh ra trong lúc
chạy không được liệt kê từng file vì số lượng lớn và có thể thay đổi.

## Workflow tổng quát

1. Chuẩn bị ảnh và caption trong `data/raw/`.
2. `split.py` chia dữ liệu theo ảnh thành train, validation và test.
3. `preprocessing/` xử lý caption, vocabulary, ảnh và sinh ảnh nhiễu.
4. `model/` định nghĩa CNN encoder, DnCNN và các thành phần captioner.
5. `training/` huấn luyện baseline, DnCNN và captioner trực tiếp trên ảnh nhiễu.
6. `evaluation/` sinh caption, đánh giá chất lượng và benchmark các pipeline.
7. `tests/` kiểm tra các hành vi quan trọng của workflow.

Khi chạy `python main.py`, luồng huấn luyện được điều phối trong `main.py`.
Các benchmark và đánh giá trên tập test là tác vụ riêng, không tự chạy trong
quá trình train.

## Thư mục gốc

| File | Nhiệm vụ | Input | Output | Trong pipeline |
| --- | --- | --- | --- | --- |
| `main.py` | Entrypoint và cấu hình thí nghiệm; điều phối train baseline, DnCNN và captioner ảnh nhiễu. | Caption CSV, ảnh sạch, cấu hình, split/checkpoint cần thiết. | Các split, vocabulary, ảnh nhiễu và checkpoint của các model. | Bắt đầu workflow huấn luyện; gọi `split.py`, `preprocessing/`, `training/`. |
| `split.py` | Chia dữ liệu theo ảnh để tránh cùng một ảnh lọt vào nhiều tập. | `captions.csv`, thư mục ảnh, tỉ lệ chia và seed. | `train.csv`, `validation.csv`, `test.csv`. | Chạy trước khi train và đánh giá. |
| `project_paths.py` | Tập trung các đường dẫn dự án và tạo đường dẫn checkpoint theo mức nhiễu. | Không có input runtime; lấy thư mục dự án từ vị trí file, nhận mức sigma khi gọi helper. | Các hằng `Path` và checkpoint path như `dncnn_weights_noise_50.pt`. | Được import để định vị data, vocabulary, checkpoint và report. |
| `requirements.txt` | Khai báo thư viện Python cần cài. | Package manager và môi trường Python. | Các dependencies được cài trong môi trường. | Cài trước khi chạy train, inference hoặc evaluation. |
| `.gitignore` | Chọn file/thư mục Git bỏ qua. | Đường dẫn trong working tree. | Quyết định file nào không được Git theo dõi. | Áp dụng xuyên suốt quá trình phát triển. |
| `README.md` | Hướng dẫn cài đặt, dữ liệu và cách chạy dự án. | Thông tin mô tả dự án. | Tài liệu hướng dẫn người dùng. | Đọc trước khi thiết lập và chạy workflow. |
| `structure.md` | Bản đồ file theo folder, kèm input/output/pipeline. | Cấu trúc source và workflow của dự án. | Tài liệu tham khảo này. | Dùng để tra cứu nơi đặt hoặc nơi gọi từng chức năng. |

## Dữ liệu: `data/`

| Đường dẫn | Nhiệm vụ | Input | Output | Trong pipeline |
| --- | --- | --- | --- | --- |
| `data/raw/captions.csv` | Caption nguồn, cột `image` và `caption`. | Dataset caption gốc. | CSV nguồn được các bước chia dữ liệu và sinh nhiễu đọc. | Điểm vào dữ liệu văn bản. |
| `data/raw/Images/` | Ảnh gốc có tên khớp với cột `image`. | Dataset ảnh gốc. | File ảnh sạch cho preprocessing và huấn luyện. | Điểm vào dữ liệu hình ảnh. |
| `data/raw/captions.txt` | File văn bản đang có trong workspace; không phải input mặc định của workflow hiện tại. | File văn bản có sẵn. | Không có output từ workflow chính. | Chỉ dùng nếu một cấu hình hoặc công cụ khác trỏ rõ tới file này. |
| `data/processed/train.csv` | Danh sách ảnh-caption dùng để train. | `captions.csv`, ảnh và cấu hình split. | Các cặp train. | Đầu vào train captioner và DnCNN. |
| `data/processed/validation.csv` | Dữ liệu theo dõi validation. | `captions.csv`, ảnh và cấu hình split. | Các cặp validation. | Chọn checkpoint và early stopping. |
| `data/processed/test.csv` | Dữ liệu đánh giá cuối. | `captions.csv`, ảnh và cấu hình split. | Các cặp test. | Đầu vào evaluation/benchmark; không dùng để train. |
| `data/processed/NoisyImages_*/` | Ảnh nhiễu theo mức sigma. | Ảnh sạch, sigma, seed và tùy chọn noise generation. | Các ảnh nhiễu có cùng tên tương ứng ảnh sạch. | Dùng cho DnCNN, captioner ảnh nhiễu và benchmark. |
| `data/processed/noisy_image_map*.csv` | Map ảnh sạch/nhiễu và thông tin sigma/chế độ. | Danh sách ảnh và kết quả sinh nhiễu. | CSV ánh xạ. | Theo dõi nguồn gốc ảnh nhiễu và mức nhiễu. |
| `data/processed/*.pt` | Checkpoint captioner hoặc DnCNN, có mức nhiễu trong tên file. | Trọng số model trong quá trình train. | Ví dụ `dncnn_weights_noise_50.pt`, `end2end_caption_weights_sigma50.pt`. | Được nạp khi inference hoặc benchmark; cần đúng vocabulary và mức nhiễu. |
| `data/processed/evaluation.csv` | Kết quả caption theo ảnh. | Dự đoán model và caption tham chiếu. | CSV chứa caption và điểm đánh giá từng ảnh. | Artifact của đánh giá caption thông thường. |
| `data/vocabulary/tokenizer.json` | Vocabulary dùng chung với captioner. | Caption train hoặc vocabulary đã tạo. | Ánh xạ token ↔ ID được lưu. | Dùng khi mã hóa caption, tạo model và giải mã dự đoán. |

## Tiền xử lý: `preprocessing/`

Các module này chuẩn bị đầu vào cho pipeline captioning và denoising.

| File | Nhiệm vụ | Input | Output | Trong pipeline |
| --- | --- | --- | --- | --- |
| `text_preprocessing.py` | Chuẩn hóa Unicode, chữ thường, khoảng trắng; tách caption và thêm token đặc biệt. | Chuỗi caption. | Danh sách token gồm `<start>` và `<end>`. | Được tokenizer dùng trước khi mã hóa hoặc tính điểm caption. |
| `tokenizer.py` | Tạo vocabulary và chuyển đổi giữa token với ID. | Caption/token; tùy chọn file vocabulary. | Token IDs, token đã giải mã hoặc `tokenizer.json`. | Dùng trong train, validation và inference. |
| `image_preprocessing.py` | Mở ảnh, đổi kích thước, chuẩn hóa ImageNet và chuyển thành tensor. | Đường dẫn ảnh RGB. | Tensor ảnh dạng `(1, 3, H, W)`. | Tiền xử lý ảnh đưa vào captioner. |
| `noise_generator.py` | Sinh AWGN cố định/ngẫu nhiên, hỗ trợ seed, worker, resize và đồng bộ thư mục. | CSV caption, ảnh sạch và cấu hình nhiễu. | Thư mục ảnh nhiễu, map CSV và đường dẫn thư mục trả về. | Được gọi trước khi train hai nhánh noisy-captioning và benchmark. |
| `dataset.py` | Nạp cặp sạch/nhiễu, tính residual, chuẩn bị tensor và caption cho DnCNN. | Split CSV, thư mục ảnh, tùy chọn tokenizer/mức nhiễu. | Mỗi sample gồm ảnh sạch, ảnh nhiễu, residual, caption và token. | Data source cho `training/train_dncnn.py`. |

## Mô hình: `model/`

### Encoder và denoiser

| File | Nhiệm vụ | Input | Output | Trong pipeline |
| --- | --- | --- | --- | --- |
| `cnn/cnn_encoder.py` | CNN encoder dựa trên backbone ảnh; hỗ trợ đóng băng/mở khóa stage. | Tensor ảnh `(B, 3, H, W)`. | Feature map không gian hoặc feature vector. | Nhận ảnh đã preprocess và cung cấp đặc trưng cho decoder. |
| `denoising/dncnn.py` | Dự đoán residual nhiễu bằng mạng tích chập. | Tensor ảnh nhiễu. | Tensor nhiễu dự đoán; `denoise()` trả ảnh đã khử nhiễu. | DnCNN train trên cặp sạch/nhiễu, rồi xử lý ảnh trước captioner baseline. |

### Captioner: `model/captioning/`

| File | Nhiệm vụ | Input | Output | Trong pipeline |
| --- | --- | --- | --- | --- |
| `captioning/embedding.py` | Chuyển token ID thành vector embedding và tạo mask padding. | Tensor ID `(B, T)`. | Embedding `(B, T, embed_dim)` và mask. | Chuẩn bị từ trước đó cho decoder. |
| `captioning/decoder.py` | Kết hợp embedding từ trước, image feature và attention để dự đoán từ kế tiếp. | Feature map ảnh và ID các từ đã có. | Logits/xác suất vocabulary hoặc token ID kế tiếp. | Được `ImageCaptioningModel` dùng khi train và sinh caption. |
| `captioning/image_captioning.py` | Ghép CNN encoder và decoder; cung cấp API train, sinh caption, save/load weights. | Ảnh/tensor cùng caption token hoặc prefix. | Feature, dự đoán token/caption; checkpoint khi lưu. | Model hoàn chỉnh được trainer và inference sử dụng. |

### Thành phần neural network: `model/neural_network/`

| File | Nhiệm vụ | Input | Output | Trong pipeline |
| --- | --- | --- | --- | --- |
| `neural_network/activation.py` | Các hàm activation và tra activation theo tên. | Tensor hoặc tên activation. | Tensor sau activation hoặc hàm activation. | Được layer/decoder dùng khi tính toán. |
| `neural_network/layers.py` | Dense layer tùy biến và khởi tạo trọng số. | Tensor đầu vào có số chiều phù hợp. | Tensor qua phép chiếu tuyến tính và activation. | Thành phần tuyến tính của decoder. |
| `neural_network/feedforward.py` | Soft attention và decoder tuần tự LSTM/GRU. | Feature map ảnh, embedding từ và trạng thái recurrent. | Logits/xác suất token, attention weights và trạng thái mới. | Decoder quét từng bước để sinh hoặc học caption. |

## Huấn luyện: `training/`

| File | Nhiệm vụ | Input | Output | Trong pipeline |
| --- | --- | --- | --- | --- |
| `train.py` | Vòng train captioner: batch, loss, validation, early stopping và checkpoint. | Model, sample `(image_path, token_ids)`, tham số train và validation samples. | History theo epoch và checkpoint có validation tốt nhất. | Dùng cho baseline và làm lõi train captioner ảnh nhiễu. |
| `train_dncnn.py` | Huấn luyện DnCNN trên ảnh sạch/nhiễu; theo dõi loss, PSNR, SSIM. | Split CSV, thư mục sạch/nhiễu và tham số train. | History metric và checkpoint DnCNN tốt nhất. | Nhánh denoise trước captioner baseline. |
| `train_end2end.py` | Khởi tạo captioner với CNN pretrained và train trực tiếp trên ảnh nhiễu. | Noisy image directory, train/validation CSV, vocabulary và cấu hình. | History và checkpoint captioner noisy. | Nhánh captioning không đi qua DnCNN. |
| `loss.py` | Cross-entropy cho probability hoặc logits; hỗ trợ bỏ qua `<pad>`. | Logits/probabilities và target token IDs. | Scalar loss. | Loss function được vòng train captioner sử dụng. |
| `optimizer.py` | Tiện ích Gradient Descent đơn giản. | Weight, gradient và learning rate. | Weight đã cập nhật. | Công cụ cấp thấp; không phải optimizer chính của các trainer hiện tại. |
| `backpropagation.py` | Tiện ích gọi autograd và phân tích gradient. | Scalar loss và model. | Gradient theo tên tham số, hướng cập nhật hoặc gradient norm. | Công cụ cấp thấp; các trainer chính gọi PyTorch autograd trực tiếp. |

`train.py` và `train_dncnn.py` dùng optimizer của PyTorch cho các vòng train
tương ứng; `optimizer.py` và `backpropagation.py` là các tiện ích riêng, không
phải entrypoint huấn luyện chính.

## Suy luận và đánh giá: `evaluation/`

| File | Nhiệm vụ | Input | Output | Trong pipeline |
| --- | --- | --- | --- | --- |
| `evaluation/generate_caption.py` | Nạp model/vocabulary và cung cấp API sinh caption; có CLI cho một ảnh. | Checkpoint, vocabulary, ảnh và cấu hình decode. | Model đã nạp hoặc caption dạng chuỗi. | Dùng cho suy luận và được benchmark gọi để nạp model. |
| `evaluation/evaluation.py` | Chấm caption bằng BLEU-1..4, loss và perplexity; lưu điểm từng ảnh. | Caption tham chiếu, ảnh test, model và cấu hình. | Dictionary metric, kết quả theo ảnh và `evaluation.csv`. | Đánh giá caption tổng quát sau train. |
| `evaluation/denoise_metrics.py` | Tính PSNR/SSIM cho ảnh sạch và ảnh khử nhiễu. | Batch tensor sạch và khử nhiễu trong `[0, 1]`. | Dictionary gồm PSNR và SSIM trung bình. | Dùng trong validation của DnCNN. |
| `evaluation/benchmark.v1.py` | Đo tốc độ suy luận và BLEU baseline trên tập test. | Checkpoint baseline, vocabulary, ảnh và caption test. | Số ảnh, latency, throughput và BLEU-1..4. | Benchmark baseline độc lập. |
| `evaluation/benchmark.v2.py` | So sánh baseline, DnCNN + baseline và captioner noisy. | Checkpoint, vocabulary, test CSV, ảnh sạch/nhiễu và cấu hình. | Kết quả theo ảnh trong CSV, BLEU/thời gian và PSNR/SSIM cho DnCNN. | Benchmark so sánh các pipeline sau khi train. |
| `evaluation/__init__.py` | Đánh dấu package đánh giá. | Không có input runtime. | Namespace `evaluation`. | Cho phép import các module đánh giá. |

## Kiểm thử: `tests/`

| File | Nhiệm vụ | Input | Output | Trong pipeline |
| --- | --- | --- | --- | --- |
| `tests/test_main_workflow.py` | Kiểm tra thứ tự workflow và tham số truyền giữa các bước. | Hàm `main()`/workflow với trainer và generator được mock. | Các assertion pass hoặc lỗi test. | Regression test cho orchestration. |
| `tests/test_noise_generator.py` | Kiểm tra CSV BOM, sinh ảnh và lỗi header. | CSV/ảnh tạm trong thư mục temporary. | Ảnh nhiễu/map hoặc lỗi được kiểm tra. | Regression test cho noise generation. |
| `tests/test_train_end2end.py` | Kiểm tra captioner noisy nhận đúng samples và không nạp checkpoint baseline. | Dữ liệu tạm; model, tokenizer và train được mock. | Các assertion pass hoặc lỗi test. | Regression test cho Phương án 2. |
| `tests/test_benchmark_v2.py` | Kiểm tra metrics, giới hạn mẫu, validate input và API benchmark. | Mảng ảnh, tham số và dữ liệu test tạm. | Kết quả metric hoặc lỗi đúng loại. | Regression test cho benchmark v2. |

## Thứ tự chạy thường dùng

1. Đặt `captions.csv` và ảnh trong `data/raw/`.
2. Chạy `python main.py` để chia dữ liệu, train baseline, tạo ảnh nhiễu, train DnCNN và train Phương án 2.
3. Chạy các tiện ích trong `evaluation/` riêng khi muốn sinh caption, chấm điểm tập test hoặc so sánh benchmark.
4. Chạy `python -m unittest discover -s tests -v` để kiểm tra các workflow.
