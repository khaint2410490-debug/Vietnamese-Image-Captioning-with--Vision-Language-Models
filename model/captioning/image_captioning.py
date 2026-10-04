"""
image_captioning.py

Ghép toàn bộ các module thành một mô hình sinh caption cho ảnh:

    Image --> CNN Encoder --> Image feature ──┐
                                              ├--> Decoder --> Softmax --> từ tiếp theo
    Caption token --> Word Embedding ─────────┘                               |
              ^                                                               |
              └──────────── thêm từ vừa dự đoán, lặp lại đến khi gặp <end> ───┘

Input : ảnh (đường dẫn .jpg hoặc tensor) + caption token (từ bắt đầu, có thể bỏ trống)
Output: predicted caption (list token hoặc chuỗi văn bản)
"""

import os
import re
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from preprocessing.image_preprocessing import preprocess_image
from preprocessing.text_preprocessing import preprocess_caption
from preprocessing.tokenizer import Tokenizer
from ..cnn.cnn_encoder import CNNEncoder
from .decoder import Decoder

class ImageCaptioningModel(nn.Module):
    """
    Args:
        tokenizer     : tokenizer đã build_vocab (hoặc load từ vocab.json).
        feature_dim   : độ dài image feature vector.
        word_embed_dim: số chiều embedding của từ.
        backbone     : mô hình torchvision dùng làm encoder pretrained.
        pretrained   : nạp trọng số ImageNet cho backbone.
        freeze_backbone: đóng băng backbone lúc khởi tạo để huấn luyện head trước.
        hidden_dim, attn_dim: kích thước trạng thái ẩn và attention của decoder.
        dropout       : tỉ lệ dropout dùng trong encoder head và decoder.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        feature_dim: int = 256,
        word_embed_dim: int = 256,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        hidden_dim: int = 512,
        attn_dim: int = 256,
        dropout: float = 0.0,
        keep_bn_eval: bool = True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = CNNEncoder(
            backbone=backbone,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            feature_dim=feature_dim,
            dropout=dropout,
            keep_bn_eval=keep_bn_eval,
        )
        self.decoder = Decoder(
            tokenizer,
            image_feature_dim=feature_dim,
            word_embed_dim=word_embed_dim,
            hidden_dim=hidden_dim,
            attn_dim=attn_dim,
            dropout=dropout,
        )

    @classmethod
    def from_vocab_file(cls, vocab_path: str, **kwargs) -> "ImageCaptioningModel":
        """Tạo mô hình từ file từ điển đã lưu (vocab.json)."""
        return cls(Tokenizer.load(vocab_path), **kwargs)

    # ------------------------------------------------------------------ #
    # Huấn luyện
    # ------------------------------------------------------------------ #
    def forward(
        self,
        image: torch.Tensor,
        prev_word_ids: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        image tensor (B, 3, H, W) + token ID các từ trước đó (B, T)
        --> phân phối xác suất (hoặc logits) của từ tiếp theo (B, vocab_size).
        """
        image_feature_map = self.encoder.forward_spatial(image)
        return self.decoder(
            image_feature_map, prev_word_ids, return_logits=return_logits
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode images as spatial features for the attention-based decoder."""
        return self.encoder.forward_spatial(image)

    # ------------------------------------------------------------------ #
    # Sinh caption
    # ------------------------------------------------------------------ #
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    def _prepare_image(image) -> torch.Tensor:
        """Đường dẫn .jpg hoặc tensor (3, H, W) / (1, 3, H, W) --> tensor (1, 3, H, W)."""
        if isinstance(image, (str, Path)):
            return preprocess_image(str(image))
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            if image.dim() != 4:
                raise ValueError(
                    f"Tensor ảnh phải có shape (3, H, W) hoặc (1, 3, H, W), nhận được {tuple(image.shape)}"
                )
            return image
        raise TypeError("image phải là đường dẫn file .jpg hoặc torch.Tensor")

    def _prepare_prefix(self, caption_tokens) -> list[int]:
        """
        Chuyển caption token đầu vào thành list token ID bắt đầu bằng <start>
        và không kết thúc bằng <end> (để mô hình viết tiếp được).

        caption_tokens có thể là: None, chuỗi văn bản, list token chữ, hoặc list token ID.
        """
        tk = self.tokenizer
        if caption_tokens is None:
            ids = [tk.start_id]
        elif isinstance(caption_tokens, str):
            ids = tk.tokens_to_ids(preprocess_caption(caption_tokens))
        else:
            items = list(caption_tokens)
            if all(isinstance(t, int) for t in items):
                ids = items
            else:
                ids = tk.tokens_to_ids(items)

        if not ids or ids[0] != tk.start_id:
            ids = [tk.start_id] + ids
        while len(ids) > 1 and ids[-1] in (tk.end_id, tk.pad_id):
            ids.pop()
        return ids

    @torch.no_grad()
    def generate_caption(
        self,
        image,
        caption_tokens=None,
        max_length: int = 20,
        greedy: bool = True,
        temperature: float = 1.0,
        top_k: int | None = None,
        verbose: bool = False,
    ) -> list[str]:
        """
        Sinh caption cho ảnh, bắt đầu từ `caption_tokens` và viết tiếp đến khi gặp <end>
        hoặc đạt `max_length`.

        Args:
            image         : đường dẫn file .jpg hoặc image tensor.
            caption_tokens: phần đầu của caption (chuỗi, list token chữ hoặc list token ID).
                            None nghĩa là bắt đầu từ <start>.
            max_length    : độ dài tối đa của chuỗi token (tính cả <start>).
            greedy, temperature, top_k: cách chọn từ tiếp theo, xem Decoder.predict_next_id.
            verbose       : in token được chọn và xác suất của token đó.
        Returns:
            Predicted caption dạng list token, đã bỏ <start>/<end>/<pad>.
        """
        device = self._device()
        image_tensor = self._prepare_image(image).to(device)
        if image_tensor.size(0) != 1:
            raise ValueError("generate_caption chỉ nhận một ảnh mỗi lần")

        ids = self._prepare_prefix(caption_tokens)

        was_training = self.training
        self.eval()
        try:
            image_feature_map = self.encode_image(image_tensor)
            while len(ids) < max_length:
                prev = torch.tensor([ids], dtype=torch.long, device=device)
                next_id = self.decoder.predict_next_id(
                    image_feature_map,
                    prev,
                    greedy=greedy,
                    temperature=temperature,
                    top_k=top_k,
                ).item()
                if verbose:
                    probs = self.decoder(image_feature_map, prev)
                    word = self.tokenizer.ids_to_tokens([next_id])[0]
                    print(
                        f"  từ thứ {len(ids)}: {word!r}  "
                        f"(p = {probs[0, next_id].item():.3f})"
                    )
                ids.append(next_id)
                if next_id == self.tokenizer.end_id:
                    break
        finally:
            self.train(was_training)

        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @staticmethod
    def tokens_to_text(tokens: list[str], capitalize: bool = True) -> str:
        """["một", "con", "chó", "."] --> "Một con chó."."""
        text = " ".join(tokens)
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        if capitalize and text:
            text = text[0].upper() + text[1:]
        return text

    def generate_caption_text(self, image, caption_tokens=None, **kwargs) -> str:
        """Giống generate_caption nhưng trả về chuỗi văn bản."""
        return self.tokens_to_text(self.generate_caption(image, caption_tokens, **kwargs))

    # ------------------------------------------------------------------ #
    # Lưu / tải trọng số
    # ------------------------------------------------------------------ #
    def save_weights(self, path: str) -> None:
        weights_path = Path(path)
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=weights_path.parent,
            suffix=weights_path.suffix,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        try:
            torch.save(self.state_dict(), temporary_path)
            os.replace(temporary_path, weights_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def load_weights(self, path: str) -> None:
        state = torch.load(path, map_location=self._device())
        self.load_state_dict(state)