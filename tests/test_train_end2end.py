import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from training import train_end2end as train_end2end_module


class TrainNoisyCaptionerTests(unittest.TestCase):
    def test_initializes_new_captioner_and_trains_with_noisy_samples(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            noisy_dir = root / "noisy"
            noisy_dir.mkdir()
            vocab_path = root / "vocab.json"
            vocab_path.touch()

            tokenizer = mock.Mock()
            model = mock.Mock()
            train_samples = [("noisy/image.jpg", [1, 2])]
            validation_samples = [("noisy/validation.jpg", [1, 2])]

            with (
                mock.patch.object(
                    train_end2end_module.Tokenizer, "load", return_value=tokenizer
                ),
                mock.patch.object(
                    train_end2end_module,
                    "load_captions",
                    side_effect=[
                        [("image.jpg", "caption")],
                        [("validation.jpg", "caption")],
                    ],
                ) as load_captions,
                mock.patch.object(
                    train_end2end_module,
                    "encode_captions",
                    side_effect=[train_samples, validation_samples],
                ),
                mock.patch.object(
                    train_end2end_module,
                    "ImageCaptioningModel",
                    return_value=model,
                ) as model_factory,
                mock.patch.object(
                    train_end2end_module, "train", return_value=[{"epoch": 1}]
                ) as train,
                mock.patch("builtins.print"),
            ):
                history = train_end2end_module.train_end2end(
                    noisy_dir=noisy_dir,
                    save_path=root / "captioner.pt",
                    vocab_path=vocab_path,
                    device=torch.device("cpu"),
                )

        self.assertEqual(history, [{"epoch": 1}])
        self.assertEqual(load_captions.call_args_list[0].args[1], str(noisy_dir))
        model_factory.assert_called_once_with(
            tokenizer,
            feature_dim=256,
            word_embed_dim=256,
            pretrained=True,
            freeze_backbone=True,
        )
        model.load_weights.assert_not_called()
        self.assertEqual(train.call_args.kwargs["samples"], train_samples)
        self.assertEqual(train.call_args.kwargs["val_samples"], validation_samples)


if __name__ == "__main__":
    unittest.main()
