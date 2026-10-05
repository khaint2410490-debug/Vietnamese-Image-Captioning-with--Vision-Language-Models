# Vietnamese Image Captioning with Vision-Language Models

## Mục đích

Dự án xây dựng mô hình sinh chú thích tiếng Việt cho ảnh. Mô hình nhận một ảnh,
trích xuất đặc trưng thị giác bằng CNN pretrained, sau đó sinh caption từng token
một bằng decoder có attention.

Tên dự án có cụm “Vision-Language Models”, nhưng kiến trúc hiện tại không dùng một
mô hình vision-language đóng gói sẵn như CLIP. Đây là pipeline image captioning
tự huấn luyện, ghép CNN encoder với bộ giải mã ngôn ngữ.

Database sử dụng: https://www.kaggle.com/datasets/adityajn105/flickr8k

## Chức năng chính

- Tiền xử lý ảnh và văn bản caption.
- Chuyển caption thành chuỗi token ID và lưu từ điển.
- Chia dữ liệu train, validation và test theo ảnh để tránh một ảnh xuất hiện ở
  nhiều tập.
- Huấn luyện CNN encoder và decoder attention bằng loss dự đoán token tiếp theo.
- Fine-tune một số stage cuối của backbone, đánh giá validation loss, early
  stopping và lưu checkpoint tốt nhất.
- Tải checkpoint cùng từ điển để sinh caption cho một ảnh.

## Cấu trúc và thứ tự pipeline

Thứ tự xử lý chính khi chạy `main.py`:

1. `split.py` chia `data/raw/captions.csv` thành train, validation và test.
2. `training/train.py` đọc các tập train/validation.
3. `preprocessing/tokenizer.py` xây từ điển từ caption train; sau đó caption được
   mã hóa thành token ID.
4. `model/captioning/image_captioning.py` kết nối CNN encoder với decoder.
5. `training/train.py` huấn luyện, tính validation loss và lưu checkpoint.
6. `evaluation/generate_caption.py` có thể tải từ điển và checkpoint để suy luận
   trên ảnh mới.

Luồng sinh caption là:

`Ảnh -> tiền xử lý -> CNN encoder -> feature map không gian -> attention + LSTM ->
token kế tiếp -> caption tiếng Việt`

## Vai trò của từng file

### Dữ liệu và artifact

- `data/raw/captions.csv`: dữ liệu caption tiếng Việt dùng làm đầu vào mặc định.
  Header là `image,caption`; một ảnh có thể có nhiều caption.
- `data/raw/captions.txt`: dữ liệu caption nguồn tiếng Anh để đối chiếu; pipeline
  `main.py` hiện không đọc file này.
- `data/raw/Images/`: ảnh đầu vào, tham chiếu bằng tên file trong cột `image`.
- `data/processed/train.csv`, `validation.csv`, `test.csv`: các split được
  `split.py` sinh ra. Các file có thể bị Git ignore.
- `data/vocabulary/tokenizer.json`: từ điển token-ID được `main.py` xây từ train
  split và lưu lại.
- `data/processed/caption_weights.pt`: weights của checkpoint được `train.py`
  lưu. Khi validation được bật, đây là checkpoint có validation loss tốt nhất.

Checkpoint và tokenizer phải thuộc cùng một lần huấn luyện/cùng một từ điển; nếu
khác nhau, kích thước hoặc ID token có thể không khớp khi nạp model.

### `preprocessing/`

#### `preprocessing/image_preprocessing.py`

- **Vai trò:** Đọc và chuẩn hóa ảnh trước khi đưa vào encoder.
- **Input:** Đường dẫn tới ảnh `.jpg`/`.jpeg`.
- **Output:** Tensor ảnh RGB đã resize và chuẩn hóa ImageNet, shape
  `(1, 3, 224, 224)` theo mặc định.
- **Pipeline:** Kiểm tra đường dẫn và phần mở rộng -> mở ảnh bằng Pillow -> đổi
  sang RGB -> resize -> chuyển pixel sang `float32` và chuẩn hóa theo mean/std
  ImageNet -> đổi layout sang `(C, H, W)` và thêm chiều batch.
- **Kiến thức:** Xử lý ảnh, chuẩn hóa dữ liệu, NumPy, Pillow và tensor PyTorch.

#### `preprocessing/text_preprocessing.py`

- **Vai trò:** Chuẩn hóa và tách caption thành các token nhất quán.
- **Input:** Caption dạng chuỗi.
- **Output:** Danh sách token chữ thường, có `<start>` ở đầu và `<end>` ở cuối.
- **Pipeline:** Chuẩn hóa Unicode NFC -> chuyển chữ thường -> gộp khoảng trắng ->
  tách từ và dấu câu -> thêm token bắt đầu/kết thúc.
- **Kiến thức:** tokenization và token đặc biệt.

#### `preprocessing/tokenizer.py`

- **Vai trò:** Xây dựng từ điển, mã hóa caption thành ID, giải mã ID thành token,
  và lưu/nạp từ điển.
- **Input:** Caption hoặc token, ngưỡng tần suất từ, danh sách token ID, hoặc file
  JSON từ điển.
- **Output:** Mapping token-ID, chuỗi ID, danh sách token, hoặc file
  `data/vocabulary/tokenizer.json`.
- **Pipeline:** Khi tạo từ điển, áp dụng `preprocess_caption`, đếm tần suất, giữ
  token đủ ngưỡng và gán ID. Khi mã hóa, từ ngoài từ điển thành `<unk>`; có thể
  cắt chuỗi và giữ `<end>` hoặc đệm bằng `<pad>`.
- **Kiến thức:** token-to-ID, tần suất token, padding và serialization
  JSON.

### `model/cnn/`

#### `model/cnn/__init__.py`

#### `model/cnn/cnn_encoder.py`

- **Vai trò:** Dùng backbone CNN của `torchvision` để trích xuất đặc trưng ảnh.
  Mặc định mô hình tổng sử dụng ResNet-18 pretrained.
- **Input:** Tensor ảnh `(B, 3, H, W)` đã chuẩn hóa.
- **Output:** `extract_feature_map` trả feature map backbone; `forward_spatial`
  chiếu từng vị trí không gian thành feature map `(B, feature_dim, H', W')` cho
  attention; `forward` trả vector ảnh đã pooling.
- **Pipeline:** Chọn backbone ResNet hoặc EfficientNet được hỗ trợ -> chạy ảnh
  qua các stage CNN -> lấy feature map -> chiếu đặc trưng bằng lớp tuyến tính.
  Backbone có thể bị đóng băng ban đầu và mở khóa các stage cuối để fine-tune.
- **Kiến thức:** CNN pretrained, transfer learning, feature map không gian,
  global average pooling, projection layer, BatchNorm và fine-tuning.

### `model/neural_network/`

#### `model/neural_network/activation.py`

- **Vai trò:** Cung cấp các hàm activation và tra cứu activation theo tên.
- **Input:** Tensor logits/giá trị tuyến tính; tên activation khi gọi
  `get_activation`.
- **Output:** Tensor sau sigmoid, ReLU, softmax hoặc identity.
- **Pipeline:** `DenseLayer` lấy activation từ `get_activation`; decoder dùng
  softmax khi cần chuyển logits thành xác suất.
- **Kiến thức:** Hàm kích hoạt, phân phối xác suất, ổn định số học khi tính
  softmax.

#### `model/neural_network/layers.py`

- **Vai trò:** Cài đặt lớp fully connected `DenseLayer`.
- **Input:** Tensor đầu vào `(B, in_features)`.
- **Output:** Tensor `(B, out_features)` sau phép tuyến tính và activation.
- **Pipeline:** Tính `x @ weight + bias` -> áp dụng activation. Khởi tạo weight
  bằng He cho ReLU hoặc Xavier cho activation khác.
- **Kiến thức:** Dense layer, tham số học được, phép nhân ma trận, bias và khởi
  tạo trọng số.

#### `model/neural_network/feedforward.py`

- **Vai trò:** Cài đặt Bahdanau-style spatial attention và decoder tuần tự dựa
  trên LSTMCell hoặc GRUCell.
- **Input:** Feature map ảnh và embedding của các từ trước đó.
- **Output:** Logits/xác suất token ở từng bước và trọng số attention trên các
  vùng ảnh.
- **Pipeline:** Làm phẳng feature map thành các vị trí ảnh -> tính attention
  theo trạng thái ẩn hiện tại -> ghép context ảnh với embedding từ -> cập nhật
  trạng thái LSTM/GRU -> dự đoán token. Khi sinh, bắt đầu bằng token BOS và lặp
  đến EOS hoặc giới hạn độ dài.
- **Kiến thức:** Attention, RNN tuần tự, LSTM/GRU, teacher forcing, greedy
  decoding và regularization attention.

### `model/captioning/`

#### `model/captioning/embedding.py`

- **Vai trò:** Ánh xạ token ID thành vector embedding có thể học được.
- **Input:** Tensor token ID `(B, T)` và cấu hình kích thước từ điển/embedding.
- **Output:** Tensor embedding `(B, T, embed_dim)`; hàm `mask` tạo mask phân biệt
  token thật với `<pad>`.
- **Pipeline:** Khởi tạo bảng `nn.Embedding` theo kích thước tokenizer -> kiểm tra
  token ID nằm trong miền hợp lệ -> tra embedding và áp dụng dropout nếu được cấu
  hình.
- **Kiến thức:** Word embedding, padding index, dropout và tensor embedding.

#### `model/captioning/decoder.py`

- **Vai trò:** Kết hợp WordEmbedding với AttentionDecoder; cung cấp dự đoán
  phân phối token khi train và chọn token tiếp theo khi inference.
- **Input:** Feature map ảnh, các token trước đó và tùy chọn greedy/temperature/
  top-k khi sinh.
- **Output:** Logits hoặc xác suất token kế tiếp; hoặc ID token được chọn.
- **Pipeline:** Mã hóa token ID thành embedding -> gọi decoder attention tuần tự
  -> lấy logits tại token cuối cùng không phải padding. Khi sinh, không cho phép
  chọn `<pad>`, `<unk>` và `<start>`.
- **Kiến thức:** Decoder ngôn ngữ, phân phối softmax, mask padding, greedy
  decoding và sampling.

#### `model/captioning/image_captioning.py`

- **Vai trò:** Mô hình tổng hợp điều phối CNN encoder và decoder, dùng chung cho
  huấn luyện, sinh caption, lưu và nạp trọng số.
- **Input:** Tokenizer và cấu hình mô hình; khi forward nhận tensor ảnh cùng
  token ID; khi generate nhận đường dẫn ảnh hoặc tensor.
- **Output:** Logits/xác suất từ tiếp theo; caption dạng danh sách token hoặc văn
  bản; checkpoint state dictionary.
- **Pipeline:** Ảnh -> CNN encoder spatial feature map -> decoder attention;
  trong inference lặp dự đoán token cho đến EOS/độ dài tối đa. Checkpoint được
  ghi qua file tạm rồi thay thế đường dẫn đích.
- **Kiến thức:** `torch.nn.Module`, composition của mô hình, inference không
  gradient, trạng thái train/eval, greedy/sampling và serialization checkpoint.

### `evaluation/`

#### `evaluation/generate_caption.py`

- **Vai trò:** Nạp từ điển và checkpoint tương ứng, rồi cung cấp hàm sinh caption
  cho ảnh mới. File không dịch caption; ngôn ngữ đầu ra phụ thuộc dữ liệu train.
- **Input:** Đường dẫn checkpoint, từ điển, cấu hình kiến trúc và đường dẫn ảnh.
- **Output:** Mô hình ở chế độ eval hoặc caption dạng chuỗi.
- **Pipeline:** Kiểm tra file vocabulary/checkpoint -> kiểm tra checkpoint có cũ
  hơn dữ liệu/từ điển không -> tạo kiến trúc khớp cấu hình train -> nạp weights
  -> gọi `generate_caption_text`.
- **Kiến thức:** Inference, checkpoint compatibility, greedy decoding và
  temperature/top-k sampling.

#### `evaluation/evaluation.py`

- **Vai trò:** Tự nạp checkpoint và tokenizer theo đường dẫn dự án, sinh caption
  cho từng ảnh trong `data/processed/test.csv`, rồi chấm BLEU-1..4 tổng thể và
  theo từng ảnh; đồng thời tính loss và perplexity trên caption tham chiếu.
- **Chạy:** Từ thư mục gốc dự án, chạy `python -m evaluation.evaluation`.
- **Input:** `data/processed/test.csv`, ảnh trong `data/raw/Images`,
  `data/vocabulary/tokenizer.json` và `data/processed/caption_weights.pt`.
- **Output:** BLEU tổng thể, loss/perplexity và BLEU-1..4 cùng caption dự đoán
  cho từng ảnh, in ra terminal; CSV kết quả tại
  `data/processed/evaluation.csv` (được ghi đè sau mỗi lần chạy).
- **Kiến thức:** Corpus BLEU, sinh caption tự hồi quy, teacher forcing,
  cross-entropy và perplexity.

### `training/`

#### `training/train.py`

- **Vai trò:** Đọc caption, mã hóa dữ liệu, tạo batch, tính loss, huấn luyện,
  validation, early stopping và lưu checkpoint.
- **Input:** File CSV/JSON caption, thư mục ảnh, tokenizer, cấu hình train, mẫu
  train/validation đã mã hóa.
- **Output:** Lịch sử loss theo epoch, log quá trình huấn luyện và file weights
  tại `data/processed/caption_weights.pt` theo cấu hình mặc định.
- **Pipeline:** Đọc và kiểm tra dữ liệu -> tạo batch ảnh/token mục tiêu -> encode
  ảnh và dự đoán token kế tiếp bằng teacher forcing -> tính categorical
  cross-entropy -> gọi `loss.backward()` và `AdamW.step()` -> tính validation
  loss -> lưu checkpoint tốt nhất và dừng sớm nếu cần. Sau mỗi epoch, sinh
  caption thử cho một ảnh ngẫu nhiên trong tập train.
- **Kiến thức:** Mini-batch training, teacher forcing, cross-entropy,
  backpropagation/autograd, AdamW, fine-tuning, validation loss, early stopping
  và checkpointing.
- **Lưu ý:** Mã hiện tại dùng `loss.backward()` và `torch.optim.AdamW`; không gọi
  hai utility `training/backpropagation.py` và `training/optimizer.py`.

#### `training/loss.py`

- **Vai trò:** Tính categorical cross-entropy từ xác suất hoặc logits, hỗ trợ
  bỏ qua token padding.
- **Input:** Tensor scores `(B, vocab_size)` và token ID đúng `(B,)`.
- **Output:** Scalar loss trung bình.
- **Pipeline:** Kiểm tra shape -> xử lý `ignore_index` nếu có -> lấy log-probability
  của token mục tiêu -> tính trung bình trên token hợp lệ.
- **Kiến thức:** Cross-entropy, log-softmax/logsumexp và numerical stability.

#### `training/backpropagation.py`

- **Vai trò:** Utility tách riêng để gọi autograd và trả gradient dưới dạng
  dictionary; có thêm helper xóa gradient, lấy hướng gradient và tính norm.
- **Input:** Scalar loss và `nn.Module`.
- **Output:** Dictionary tên tham số -> tensor gradient; hoặc hướng/norm gradient.
- **Pipeline:** Kiểm tra loss -> xóa gradient cũ -> gọi `loss.backward()` -> sao
  chép gradient của tham số có thể train.
- **Kiến thức:** Chain rule, automatic differentiation và gradient descent.
- **Lưu ý:** Không được `training/train.py` sử dụng trong pipeline hiện tại.

#### `training/optimizer.py`

- **Vai trò:** Utility Gradient Descent thủ công để cập nhật tham số từ
  dictionary gradient.
- **Input:** Weight, gradient, learning rate; hoặc network và dictionary gradient.
- **Output:** Tensor weight mới hoặc cập nhật tham số network tại chỗ.
- **Pipeline:** Tính `weight - learning_rate * gradient` cho từng tham số tương
  ứng.
- **Kiến thức:** Gradient Descent và cập nhật tham số.
- **Lưu ý:** Không được `training/train.py` sử dụng; pipeline đang dùng AdamW của
  PyTorch.

### Các file ở thư mục gốc

#### `main.py`

- **Vai trò:** Điểm điều khiển chính cho pipeline huấn luyện.
- **Input:** Các hằng cấu hình trong file, `data/raw/captions.csv`, thư mục ảnh
  và các module của dự án.
- **Output:** Các CSV train/validation/test, file tokenizer, checkpoint model
  và log huấn luyện. Test split được tạo nhưng hiện không dùng trong train.
- **Pipeline:** Đặt seed và chọn CPU/GPU -> chia dữ liệu -> nạp train/validation
  -> xây/lưu tokenizer -> mã hóa caption -> tạo mô hình pretrained -> gọi `train`.
  Khi chạy trực tiếp file, khối `if __name__ == "__main__"` gọi `main()`.
- **Kiến thức:** Điều phối pipeline, cấu hình thí nghiệm, seed ngẫu nhiên, chọn
  thiết bị và transfer learning.

#### `project_paths.py`

- **Vai trò:** Tập trung các đường dẫn mặc định của project.
- **Input:** Vị trí của file Python này trên filesystem.
- **Output:** Các hằng `Path` như `CAPTIONS_PATH`, `IMAGES_DIR`,
  `PROCESSED_DATA_DIR`, `VOCAB_PATH` và `WEIGHTS_PATH`.
- **Pipeline:** Xác định project root từ `__file__`, sau đó ghép đường dẫn cho
  data/raw, data/processed, vocabulary và weights.
- **Kiến thức:** `pathlib.Path` và đường dẫn tương đối theo vị trí dự án.

#### `split.py`

- **Vai trò:** Chia dữ liệu caption thành train/validation/test theo ảnh.
- **Input:** CSV có hai cột `image,caption`, thư mục ảnh tùy chọn, tỉ lệ chia và
  seed.
- **Output:** Ba file `train.csv`, `validation.csv`, `test.csv` trong
  `data/processed/` theo cấu hình mặc định.
- **Pipeline:** Đọc CSV và gom caption theo ảnh -> bỏ qua ảnh thiếu nếu có cung
  cấp thư mục ảnh -> xáo trộn danh sách ảnh theo seed -> ghi toàn bộ caption của
  mỗi ảnh vào duy nhất một split.
- **Kiến thức:** Chia tập dữ liệu, kiểm soát random seed, chống data leakage và
  CSV I/O.
