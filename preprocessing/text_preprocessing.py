"""
text_preprocessing.py

Input : caption dạng văn bản thuần túy, ví dụ "Một con chó đang chạy."
Output: list token đã làm sạch, ví dụ
        ["<start>", "một", "con", "chó", "đang", "chạy", ".", "<end>"]
"""

import re
import unicodedata

START_TOKEN = "<start>"
END_TOKEN = "<end>"

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def normalize_unicode(text: str) -> str:
    """Chuẩn hóa Unicode về dạng NFC để dấu tiếng Việt luôn đồng nhất."""
    return unicodedata.normalize("NFC", text)

def lowercase(text: str) -> str:
    """Chuyển toàn bộ văn bản về chữ thường."""
    return text.lower()

def remove_extra_spaces(text: str) -> str:
    """Gộp nhiều khoảng trắng/xuống dòng liên tiếp thành một khoảng trắng."""
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text: str) -> list[str]:
    """Tách văn bản thành các từ và dấu câu."""
    return _TOKEN_PATTERN.findall(text)

def add_special_tokens(tokens: list[str]) -> list[str]:
    """Thêm <start> vào đầu và <end> vào cuối."""
    return [START_TOKEN, *tokens, END_TOKEN]

def preprocess_caption(caption: str) -> list[str]:
    """
    Chạy toàn bộ pipeline:
    Caption -> Chuẩn hóa Unicode -> Lowercase -> Xóa khoảng trắng thừa
            -> Tokenize -> Thêm <start>/<end>
    """
    if not isinstance(caption, str):
        raise TypeError(f"caption phải là str, nhận được: {type(caption).__name__}")

    caption = normalize_unicode(caption)
    caption = lowercase(caption)
    caption = remove_extra_spaces(caption)
    tokens = tokenize(caption)
    return add_special_tokens(tokens)