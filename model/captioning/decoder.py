"""
decoder.py

Decoder dự đoán từ tiếp theo dựa trên image feature và các từ đã có trước đó.

    previous word IDs --> WordEmbedding --┐
                                          ├--> FeedForwardNetwork --> Softmax --> next token ID
    image feature ------------------------┘

Hai chế độ sử dụng:
    - forward(...)         : trả về phân phối xác suất (hoặc logits) trên toàn từ điển.
                             Dùng khi huấn luyện.
    - predict_next_id(...) : chọn ra một token ID. Dùng khi sinh caption.
"""

import torch
import torch.nn as nn

from ..neural_network.activation import softmax
from ..neural_network.feedforward import AttentionDecoder
from preprocessing.tokenizer import Tokenizer
from .embedding import WordEmbedding


class Decoder(nn.Module):
    """
    Args:
        tokenizer        : tokenizer đã build_vocab (hoặc đã load từ vocab.json).
        image_feature_dim: số kênh feature map ảnh (= feature_dim của CNNEncoder).
        word_embed_dim   : số chiều embedding của từ.
        hidden_dim       : kích thước trạng thái ẩn của bộ giải mã tuần tự.
        attn_dim         : kích thước trung gian của attention.
        dropout          : tỉ lệ dropout trong bộ giải mã.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        image_feature_dim: int = 256,
        word_embed_dim: int = 256,
        hidden_dim: int = 512,
        attn_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = tokenizer.vocab_size
        self.pad_id = tokenizer.pad_id
        self.unk_id = tokenizer.unk_id
        self.start_id = tokenizer.start_id
        self.end_id = tokenizer.end_id

        self.embedding = WordEmbedding.from_tokenizer(tokenizer, word_embed_dim)
        self.network = AttentionDecoder(
            vocab_size=self.vocab_size,
            encoder_dim=image_feature_dim,
            word_embed_dim=word_embed_dim,
            hidden_dim=hidden_dim,
            attn_dim=attn_dim,
            dropout=dropout,
        )

        # Các token không bao giờ được chọn làm từ tiếp theo khi sinh caption
        self.register_buffer(
            "banned_ids",
            torch.tensor([self.pad_id, self.unk_id, self.start_id], dtype=torch.long),
            persistent=False,
        )

    # ------------------------------------------------------------------ #
    # Huấn luyện: phân phối trên toàn từ điển
    # ------------------------------------------------------------------ #
    def forward(
        self,
        image_feature_map: torch.Tensor,
        prev_word_ids: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            image_feature_map: (B, C, H, W), spatial features from the image encoder.
            prev_word_ids: (B, T) token ID của các từ trước đó (có thể chứa <pad>).
            return_logits: True để lấy logits (dùng với nn.CrossEntropyLoss khi huấn luyện),
                           False để lấy xác suất sau softmax.
        Returns:
            Logits hoặc xác suất của token kế tiếp, shape (B, vocab_size).
        """
        if prev_word_ids.dim() == 1:
            prev_word_ids = prev_word_ids.unsqueeze(0)
        word_embedding = self.embedding(prev_word_ids)
        logits, _ = self.network(
            image_feature_map, word_embedding, return_logits=True
        )
        lengths = self.embedding.mask(prev_word_ids).sum(dim=1).long().clamp(min=1)
        last_logits = logits[
            torch.arange(logits.size(0), device=logits.device), lengths - 1
        ]
        return last_logits if return_logits else softmax(last_logits, dim=-1)

    # ------------------------------------------------------------------ #
    # Sinh caption: chọn token ID tiếp theo
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_next_id(
        self,
        image_feature_map: torch.Tensor,
        prev_word_ids: torch.Tensor,
        greedy: bool = True,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Dự đoán token ID của từ tiếp theo.

        Args:
            greedy     : True  -> chọn token có xác suất cao nhất.
                         False -> lấy mẫu ngẫu nhiên theo phân phối xác suất.
            temperature: chỉ dùng khi greedy=False. Nhỏ hơn 1 thì chắc chắn hơn,
                         lớn hơn 1 thì đa dạng hơn.
            top_k      : chỉ dùng khi greedy=False. Chỉ lấy mẫu trong k token xác suất cao nhất.
        Returns:
            Tensor (B,) chứa token ID dự đoán.
        """
        logits = self.forward(image_feature_map, prev_word_ids, return_logits=True)
        logits = logits.clone()
        logits[:, self.banned_ids] = float("-inf")           # không chọn <pad>, <unk>, <start>

        if greedy:
            return logits.argmax(dim=-1)

        if temperature <= 0:
            raise ValueError("temperature phải > 0")
        logits = logits / temperature
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            kth_value = torch.topk(logits, k, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < kth_value, float("-inf"))

        probs = softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(1)