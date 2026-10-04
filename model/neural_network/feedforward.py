"""
feedforward.py

Luồng chính của neural network (decoder sinh caption, giữ thứ tự từ):

    Feature map (B, C, H', W') --> flatten --> (B, N, C)      N = H' * W'
                                       |
    Prev. word (t) ----------------> [Attention] --> context_t
    (embedding)                        |
                  concat(embedding_t, context_t) --> LSTM/GRU cell --> h_t
                                                                        |
                                              Dense (linear) --> Softmax
                                                                        v
                                         Probability distribution over vocabulary

Khác với bản cũ (lấy trung bình embedding của các từ trước đó nên mất thứ tự),
trạng thái ẩn h_t của LSTM/GRU được cập nhật tuần tự theo từng từ, vì vậy
"a dog chases a cat" và "a cat chases a dog" cho ra hai trạng thái khác nhau.
Attention cho phép mỗi bước nhìn vào những vùng ảnh liên quan nhất.

Input:
    feature_map         : (B, C, H', W')        - output của CNNEncoder.extract_feature_map
                          (hoặc (B, N, C))
    prev_word_embedding : (B, T, word_embed_dim) - embedding của các từ đầu vào
                          (<bos> w1 ... w_{T-1}), thường là WordEmbedding(captions[:, :-1])
Output:
    logits/xác suất (B, T, vocab_size) và attention alphas (B, T, N).

Huấn luyện (captions: (B, T+1) gồm <bos> ... <eos> <pad>...):

    feature_map = encoder.extract_feature_map(images)
    logits, alphas = decoder(
        feature_map, word_embedding(captions[:, :-1]), return_logits=True
    )
    targets = captions[:, 1:]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=pad_id
    )
    loss = loss + 0.1 * attention_regularization(alphas, word_embedding.mask(targets))

Sinh caption:

    decoder.eval()
    ids, alphas = decoder.generate(
        feature_map, word_embedding, bos_id=..., eos_id=..., pad_id=..., max_len=30
    )
"""

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import softmax
from .layers import DenseLayer


class SoftAttention(nn.Module):
    """
    Attention cộng (Bahdanau): chấm điểm từng vùng ảnh theo trạng thái ẩn hiện tại.

    Args:
        encoder_dim: số kênh C của feature map.
        hidden_dim : kích thước trạng thái ẩn của decoder.
        attn_dim   : kích thước không gian trung gian của attention.
    """

    def __init__(self, encoder_dim: int, hidden_dim: int, attn_dim: int):
        super().__init__()
        self.enc_proj = nn.Linear(encoder_dim, attn_dim)
        self.dec_proj = nn.Linear(hidden_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1)

    def forward(self, feats: torch.Tensor, hidden: torch.Tensor):
        """
        feats : (B, N, C)   hidden : (B, H)
        Trả về context (B, C) và trọng số attention alpha (B, N), tổng mỗi hàng = 1.
        """
        energy = torch.tanh(self.enc_proj(feats) + self.dec_proj(hidden).unsqueeze(1))
        alpha = F.softmax(self.score(energy).squeeze(-1), dim=1)       # (B, N)
        context = (feats * alpha.unsqueeze(-1)).sum(dim=1)             # (B, C)
        return context, alpha


class AttentionDecoder(nn.Module):
    """
    Decoder LSTM/GRU + attention lên feature map, dự đoán từ tiếp theo ở mỗi bước.

    Args:
        vocab_size    : kích thước từ điển (số lớp đầu ra).
        encoder_dim   : số kênh C của feature map (encoder.feature_map_channels).
        word_embed_dim: độ dài vector embedding của từ.
        hidden_dim    : kích thước trạng thái ẩn của LSTM/GRU.
        attn_dim      : kích thước không gian trung gian của attention.
        rnn_type      : "lstm" hoặc "gru".
        dropout       : tỉ lệ dropout trước lớp đầu ra (0 để tắt).
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_dim: int = 512,
        word_embed_dim: int = 256,
        hidden_dim: int = 512,
        attn_dim: int = 256,
        rnn_type: str = "lstm",
        dropout: float = 0.0,
    ):
        super().__init__()
        rnn_type = rnn_type.lower()
        if rnn_type not in ("lstm", "gru"):
            raise ValueError("rnn_type phải là 'lstm' hoặc 'gru'.")

        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.word_embed_dim = word_embed_dim
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type

        self.attention = SoftAttention(encoder_dim, hidden_dim, attn_dim)
        # cổng điều chỉnh mức độ dùng context ở mỗi bước
        self.f_beta = nn.Linear(hidden_dim, encoder_dim)

        cell = nn.LSTMCell if rnn_type == "lstm" else nn.GRUCell
        self.rnn = cell(word_embed_dim + encoder_dim, hidden_dim)

        # khởi tạo trạng thái ẩn từ trung bình các vùng ảnh
        self.init_h = nn.Linear(encoder_dim, hidden_dim)
        self.init_c = nn.Linear(encoder_dim, hidden_dim) if rnn_type == "lstm" else None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Lớp đầu ra: chỉ linear sum (logits), softmax được áp dụng riêng ở forward
        self.output_layer = DenseLayer(hidden_dim, vocab_size, activation="linear")

    # ------------------------------------------------------------------ #
    # Tiện ích
    # ------------------------------------------------------------------ #
    def _flatten(self, feature_map: torch.Tensor) -> torch.Tensor:
        """(B, C, H', W') --> (B, N, C). Cũng nhận sẵn (B, N, C)."""
        if feature_map.dim() == 4:
            feature_map = feature_map.flatten(2).transpose(1, 2)
        if feature_map.dim() != 3 or feature_map.size(-1) != self.encoder_dim:
            raise ValueError(
                f"Feature map phải có {self.encoder_dim} kênh, "
                f"nhận được shape {tuple(feature_map.shape)}"
            )
        return feature_map

    def init_state(self, feats: torch.Tensor):
        """Trạng thái ban đầu từ trung bình các vùng ảnh: (h, c) cho LSTM, (h,) cho GRU."""
        mean = feats.mean(dim=1)
        h = torch.tanh(self.init_h(mean))
        if self.rnn_type == "lstm":
            return (h, torch.tanh(self.init_c(mean)))
        return (h,)

    def step(self, word_emb: torch.Tensor, feats: torch.Tensor, state):
        """
        Một bước giải mã.
        word_emb (B, E), feats (B, N, C), state --> logits (B, V), alpha (B, N), state mới.
        """
        h = state[0]
        context, alpha = self.attention(feats, h)
        context = torch.sigmoid(self.f_beta(h)) * context
        rnn_in = torch.cat([word_emb, context], dim=1)

        if self.rnn_type == "lstm":
            h, c = self.rnn(rnn_in, state)
            state = (h, c)
        else:
            h = self.rnn(rnn_in, h)
            state = (h,)

        logits = self.output_layer(self.dropout(h))
        return logits, alpha, state

    # ------------------------------------------------------------------ #
    # Huấn luyện (teacher forcing)
    # ------------------------------------------------------------------ #
    def forward(
        self,
        feature_map: torch.Tensor,
        prev_word_embedding: torch.Tensor,
        return_logits: bool = False,
    ):
        """
        Args:
            feature_map        : (B, C, H', W') hoặc (B, N, C).
            prev_word_embedding: (B, T, word_embed_dim), embedding của <bos> w1 ... w_{T-1}.
            return_logits      : True để trả về logits (dùng khi huấn luyện với
                                 nn.CrossEntropyLoss, vì loss này đã tự gồm softmax);
                                 False để trả về xác suất.
        Returns:
            (logits hoặc xác suất) (B, T, vocab_size) và alphas (B, T, N).
            Token <pad> ở cuối câu không ảnh hưởng các bước trước nó; hãy bỏ chúng
            khỏi loss bằng ignore_index.
        """
        if prev_word_embedding.dim() != 3:
            raise ValueError(
                "prev_word_embedding phải có shape (B, T, E), "
                f"nhận được {tuple(prev_word_embedding.shape)}"
            )
        if prev_word_embedding.size(-1) != self.word_embed_dim:
            raise ValueError(
                f"Embedding của từ phải có {self.word_embed_dim} chiều, "
                f"nhận được {prev_word_embedding.size(-1)}"
            )

        feats = self._flatten(feature_map)
        state = self.init_state(feats)

        logits_all, alphas_all = [], []
        for t in range(prev_word_embedding.size(1)):
            logits, alpha, state = self.step(prev_word_embedding[:, t], feats, state)
            logits_all.append(logits)
            alphas_all.append(alpha)

        logits = torch.stack(logits_all, dim=1)              # (B, T, vocab_size)
        alphas = torch.stack(alphas_all, dim=1)              # (B, T, N)

        if return_logits:
            return logits, alphas
        return softmax(logits, dim=-1), alphas               # tổng mỗi hàng = 1

    # ------------------------------------------------------------------ #
    # Sinh caption (greedy)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(
        self,
        feature_map: torch.Tensor,
        embed_fn: Callable[[torch.Tensor], torch.Tensor],
        bos_id: int,
        eos_id: int,
        pad_id: int = 0,
        max_len: int = 30,
    ):
        """
        Greedy decoding. Gọi decoder.eval() trước để tắt dropout.

        embed_fn: hàm nhận ID (B, T) trả về embedding (B, T, E), ví dụ WordEmbedding.
        Trả về ids (B, L) (sau <eos> được điền pad_id) và alphas (B, L, N).
        """
        feats = self._flatten(feature_map)
        B = feats.size(0)
        state = self.init_state(feats)

        word = torch.full((B,), bos_id, dtype=torch.long, device=feats.device)
        finished = torch.zeros(B, dtype=torch.bool, device=feats.device)

        ids, alphas_all = [], []
        for _ in range(max_len):
            word_emb = embed_fn(word.unsqueeze(1))[:, 0]     # (B, E)
            logits, alpha, state = self.step(word_emb, feats, state)
            word = logits.argmax(dim=-1).masked_fill(finished, pad_id)
            ids.append(word)
            alphas_all.append(alpha)
            finished = finished | (word == eos_id)
            if finished.all():
                break

        return torch.stack(ids, dim=1), torch.stack(alphas_all, dim=1)


def attention_regularization(
    alphas: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Doubly stochastic regularization (tùy chọn): khuyến khích mỗi vùng ảnh được nhìn
    tới một lượng tương đương trong cả câu, tránh attention dồn vào vài vùng.

    alphas: (B, T, N)   mask: (B, T), 1 cho token thật, 0 cho <pad>.
    Dùng với trọng số nhỏ (khoảng 0.1 - 1.0) cộng vào cross-entropy.
    """
    if mask is not None:
        alphas = alphas * mask.unsqueeze(-1)
    return ((1.0 - alphas.sum(dim=1)) ** 2).mean()