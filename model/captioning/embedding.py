"""
embedding.py

Chuyển token ID (do tokenizer.py tạo ra và lưu trong vocab.json) thành vector
embedding để decoder.py sử dụng.

    [2, 4, 5]  -->  3 vector, mỗi vector có `embed_dim` chiều
    (B, T)     -->  (B, T, embed_dim)

Mỗi token ID tương ứng với một hàng trong bảng embedding (vocab_size x embed_dim).
Các vector này là tham số học được, được cập nhật trong quá trình huấn luyện.
"""

import torch
import torch.nn as nn

from preprocessing.tokenizer import Tokenizer


class WordEmbedding(nn.Module):
    """
    Bảng embedding: token ID --> vector.

    Args:
        vocab_size: kích thước từ điển (số hàng của bảng embedding).
        embed_dim : số chiều của mỗi vector.
        pad_id    : ID của <pad>. Vector của <pad> luôn bằng 0 và không được cập nhật.
        dropout   : tỉ lệ dropout áp dụng lên embedding (0 để tắt).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        pad_id: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # Tạo embedding từ tokenizer / từ điển đã lưu
    @classmethod
    def from_tokenizer(
        cls, tokenizer: Tokenizer, embed_dim: int, dropout: float = 0.0
    ) -> "WordEmbedding":
        """Tạo embedding có kích thước khớp với tokenizer."""
        return cls(
            vocab_size=tokenizer.vocab_size,
            embed_dim=embed_dim,
            pad_id=tokenizer.pad_id,
            dropout=dropout,
        )

    @classmethod
    def from_vocab_file(
        cls, vocab_path: str, embed_dim: int, dropout: float = 0.0
    ) -> "WordEmbedding":
        """Tạo embedding từ file từ điển đã lưu (vocab.json)."""
        return cls.from_tokenizer(Tokenizer.load(vocab_path), embed_dim, dropout)

    # Forward
    def forward(self, word_ids: torch.Tensor) -> torch.Tensor:
        """Token ID (B, T) --> embedding (B, T, embed_dim)."""
        word_ids = word_ids.long()
        if word_ids.numel() > 0 and (
            word_ids.min() < 0 or word_ids.max() >= self.vocab_size
        ):
            raise ValueError(
                f"Token ID nằm ngoài khoảng [0, {self.vocab_size - 1}]. "
                "Hãy kiểm tra tokenizer và từ điển đang dùng."
            )
        return self.dropout(self.embedding(word_ids))

    def mask(self, word_ids: torch.Tensor) -> torch.Tensor:
        """Mask (B, T): 1 cho token thật, 0 cho <pad>."""
        return (word_ids != self.pad_id).float()