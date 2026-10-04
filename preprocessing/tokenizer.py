"""
tokenizer.py

Chuyển token chữ (kết quả của text_preprocessing.preprocess_caption) thành token ID
để model xử lý, và ngược lại.

    ["<start>", "một", "con", "chó", "<end>"]  <-->  [2, 4, 5, 6, 3]
"""

import json
from collections import Counter
from pathlib import Path

from project_paths import VOCAB_PATH
from .text_preprocessing import END_TOKEN, START_TOKEN, preprocess_caption

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

# Thứ tự cố định để ID của token đặc biệt luôn giống nhau: pad=0, unk=1, start=2, end=3
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN]

# Đường dẫn mặc định của file từ điển (dùng khi save()/load() không truyền path)
DEFAULT_VOCAB_PATH = VOCAB_PATH

class Tokenizer:
    def __init__(self):
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._init_special_tokens()

    # Vocabulary
    def _init_special_tokens(self) -> None:
        self.token_to_id = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    @property
    def start_id(self) -> int:
        return self.token_to_id[START_TOKEN]

    @property
    def end_id(self) -> int:
        return self.token_to_id[END_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def build_vocab(self, captions: list[str], min_freq: int = 1) -> None:
        """
        Xây dựng từ điển từ danh sách caption thuần văn bản.
        Chỉ giữ những token xuất hiện ít nhất `min_freq` lần;
        các token còn lại sẽ được ánh xạ thành <unk> khi encode.
        """
        counter = Counter()
        for caption in captions:
            counter.update(preprocess_caption(caption))

        self._init_special_tokens()
        # Sắp xếp theo tần suất giảm dần, cùng tần suất thì theo thứ tự chữ cái
        for token, freq in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
            if freq >= min_freq and token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

    # Encode / Decode
    def tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """Token chữ -> token ID (token lạ sẽ thành <unk>)."""
        return [self.token_to_id.get(tok, self.unk_id) for tok in tokens]

    def ids_to_tokens(self, ids: list[int]) -> list[str]:
        """Token ID -> token chữ."""
        return [self.id_to_token.get(int(i), UNK_TOKEN) for i in ids]

    def encode(self, caption, max_length: int | None = None) -> list[int]:
        """
        Chuyển caption thành list token ID.

        - caption: chuỗi văn bản thuần, hoặc list token đã qua preprocess_caption.
        - max_length: nếu có, caption sẽ được cắt hoặc đệm <pad> cho đủ độ dài này.
          Khi cắt, token <end> vẫn được giữ ở cuối.
        """
        tokens = preprocess_caption(caption) if isinstance(caption, str) else list(caption)
        ids = self.tokens_to_ids(tokens)

        if max_length is not None:
            if max_length < 2:
                raise ValueError("max_length phải >= 2 để chứa <start> và <end>")
            if len(ids) > max_length:
                ids = ids[: max_length - 1] + [self.end_id]
            else:
                ids = ids + [self.pad_id] * (max_length - len(ids))
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> list[str]:
        """
        Chuyển token ID về token chữ.
        Nếu skip_special_tokens=True, bỏ <pad>, <start>, <end> và dừng khi gặp <end>.
        """
        tokens = []
        for i in ids:
            token = self.id_to_token.get(int(i), UNK_TOKEN)
            if skip_special_tokens:
                if token == END_TOKEN:
                    break
                if token in (PAD_TOKEN, START_TOKEN):
                    continue
            tokens.append(token)
        return tokens

    # lưu vào đường dẫn, nếu không có thì đường dẫn mặc định luôn
    def save(self, path: str | None = None) -> None:
        """Lưu từ điển ra file JSON, nếu không truyền path thì dùng đường dẫn mặc định."""
        path = Path(path) if path is not None else DEFAULT_VOCAB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | None = None) -> "Tokenizer":
        """Tải từ điển đã lưu từ file JSON, nếu không truyền path thì dùng đường dẫn mặc định."""
        path = Path(path) if path is not None else DEFAULT_VOCAB_PATH
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file từ điển: {path}")

        with open(path, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)

        tokenizer = cls()
        tokenizer.token_to_id = {tok: int(idx) for tok, idx in token_to_id.items()}
        tokenizer.id_to_token = {idx: tok for tok, idx in tokenizer.token_to_id.items()}
        return tokenizer