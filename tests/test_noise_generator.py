import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

from preprocessing.dataset import NoisyCaptionDataset
from preprocessing.noise_generator import (
    SUPPORTED_DEGRADATION_MODES, add_awgn_to_array, add_fold_line,
    add_gaussian_blur, add_gaussian_noise, add_motion_blur,
    add_random_scratches, adjust_contrast_brightness, apply_degradation,
    generate_noisy_dataset,
)


class NoiseGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(123)
        self.np_rgb = rng.integers(30, 220, size=(32, 40, 3), dtype=np.uint8)
        self.np_gray = self.np_rgb[:, :, 0]
        self.pil_img = Image.fromarray(self.np_rgb, mode="RGB")
        self.t_3d = torch.from_numpy(self.np_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        self.t_4d = torch.stack([self.t_3d, torch.clamp(self.t_3d * 0.8 + 0.1, 0.0, 1.0)], dim=0)

    def test_operators_preserve_formats_and_modify_pixels(self) -> None:
        operators = [
            ("add_gaussian_noise", lambda x: add_gaussian_noise(x, sigma=25.0, seed=7)),
            ("add_gaussian_blur", lambda x: add_gaussian_blur(x, kernel_size=5)),
            ("add_motion_blur", lambda x: add_motion_blur(x, kernel_size=9, seed=7)),
            ("add_random_scratches", lambda x: add_random_scratches(x, seed=7)),
            ("add_fold_line", lambda x: add_fold_line(x, seed=7)),
            ("adjust_contrast_brightness", lambda x: adjust_contrast_brightness(x, seed=7)),
        ]
        for name, fn in operators:
            with self.subTest(operator=name):
                out_pil = fn(self.pil_img)
                self.assertIsInstance(out_pil, Image.Image)
                self.assertEqual(out_pil.size, self.pil_img.size)

                out_rgb, out_gray = fn(self.np_rgb), fn(self.np_gray)
                self.assertIsInstance(out_rgb, np.ndarray)
                self.assertEqual((out_rgb.shape, out_rgb.dtype), (self.np_rgb.shape, np.uint8))
                self.assertFalse(np.array_equal(out_rgb, self.np_rgb))
                self.assertIsInstance(out_gray, np.ndarray)
                self.assertEqual((out_gray.shape, out_gray.dtype), (self.np_gray.shape, np.uint8))

                for tensor in (self.t_3d, self.t_4d):
                    out_t = fn(tensor)
                    self.assertIsInstance(out_t, torch.Tensor)
                    self.assertEqual((out_t.shape, out_t.dtype), (tensor.shape, tensor.dtype))
                    self.assertTrue(0.0 <= float(out_t.min()) and float(out_t.max()) <= 1.0)

    def test_apply_degradation_modes_determinism_and_tiny_image(self) -> None:
        self.assertTrue(torch.allclose(apply_degradation(self.t_4d, mode="clean"), self.t_4d))
        self.assertTrue(torch.allclose(apply_degradation(self.t_4d, mode="gaussian", sigma=0, seed=99), self.t_4d))
        for mode in SUPPORTED_DEGRADATION_MODES:
            with self.subTest(mode=mode):
                out = apply_degradation(self.pil_img, mode=mode, seed=42)
                self.assertIsInstance(out, Image.Image)
                self.assertEqual(out.size, self.pil_img.size)

        run_a = apply_degradation(self.t_4d, mode="combined", seed=2025)
        self.assertTrue(torch.equal(run_a, apply_degradation(self.t_4d, mode="combined", seed=2025)))
        self.assertFalse(torch.equal(run_a, apply_degradation(self.t_4d, mode="combined", seed=2026)))

        tiny_out = apply_degradation(np.full((4, 4, 3), 120, dtype=np.uint8), mode="combined", seed=11)
        self.assertIsInstance(tiny_out, np.ndarray)
        self.assertEqual(tiny_out.shape, (4, 4, 3))

    def test_validation_errors(self) -> None:
        for call in (
            lambda: add_gaussian_noise(self.pil_img, sigma=-1),
            lambda: add_gaussian_blur(self.pil_img, kernel_size=4),
            lambda: apply_degradation(self.pil_img, mode="unknown"),
            lambda: add_gaussian_noise(torch.zeros((3, 8, 8), dtype=torch.uint8)),
            lambda: add_gaussian_noise(self.pil_img, rng=np.random.default_rng(1), seed=1),
            lambda: add_gaussian_noise(np.zeros((8, 8, 3), dtype=bool), sigma=10),
            lambda: add_awgn_to_array(np.zeros((8, 8, 3), dtype=bool), sigma=10),
        ):
            with self.assertRaises(ValueError):
                call()
        with self.assertRaises(TypeError):
            apply_degradation(self.pil_img, mode="combined", unknown_arg=1)

    def test_generate_noisy_dataset_backward_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_dir, out_dir = root / "clean", root / "noisy"
            clean_dir.mkdir()
            self.pil_img.save(clean_dir / "sample.jpg", quality=95)
            csv_path = root / "captions.csv"
            csv_path.write_text("image,caption\nsample.jpg,mot chu cho dang chay\n", encoding="utf-8")

            map_csv = root / "noisy_image_map.csv"
            returned_dir = generate_noisy_dataset(
                captions_csv=csv_path, clean_images_dir=clean_dir, noise_level=15,
                seed=42, output_dir=out_dir, map_path=map_csv, log_every=0,
            )
            self.assertEqual(returned_dir, out_dir)
            self.assertTrue((out_dir / "sample.jpg").is_file())
            self.assertTrue(map_csv.is_file())

    def test_noisy_caption_dataset_modes_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_dir, noisy_dir = root / "clean", root / "pre_noisy"
            clean_dir.mkdir()
            noisy_dir.mkdir()
            clean_arr = np.random.default_rng(7).integers(20, 230, size=(64, 64, 3), dtype=np.uint8)
            Image.fromarray(clean_arr, mode="RGB").save(clean_dir / "img1.jpg", quality=95)
            Image.fromarray(np.full((64, 64, 3), 17, dtype=np.uint8), mode="RGB").save(noisy_dir / "img1.jpg", quality=100)
            split_csv = root / "split.csv"
            split_csv.write_text("image,caption\nimg1.jpg,hai con meo\n", encoding="utf-8")

            ds_default = NoisyCaptionDataset(split_csv, clean_dir, noise_level=25, seed=42)
            first = ds_default[0]
            self.assertTrue(torch.equal(first["noisy"], ds_default[0]["noisy"]))
            self.assertFalse(torch.equal(first["clean"], first["noisy"]))
            self.assertTrue(torch.allclose(first["noise_target"], first["noisy"] - first["clean"]))

            ds_dyn = NoisyCaptionDataset(
                split_csv, clean_dir, noise_level="random", seed=42, degradation_mode="combined", dynamic_noise=True
            )
            dyn_a = ds_dyn[0]
            self.assertTrue(0.0 <= float(dyn_a["noisy"].min()) and float(dyn_a["noisy"].max()) <= 1.0)
            self.assertFalse(torch.equal(dyn_a["noisy"], ds_dyn[0]["noisy"]))

            with mock.patch("preprocessing.dataset.get_worker_info", return_value=mock.Mock(id=0, seed=100)):
                worker_a = ds_dyn[0]
            with mock.patch("preprocessing.dataset.get_worker_info", return_value=mock.Mock(id=1, seed=100)):
                worker_b = ds_dyn[0]
            self.assertFalse(torch.equal(worker_a["noisy"], worker_b["noisy"]))

            ds_pre = NoisyCaptionDataset(split_csv, clean_dir, noisy_images_dir=noisy_dir, degradation_mode="blur", seed=42)
            self.assertTrue(torch.allclose(ds_pre[0]["noisy"], torch.full_like(first["noisy"], 17.0 / 255.0), atol=0.02))

            for bad_kwargs in ({"degradation_mode": "bad_mode"}, {"noise_level": -5}):
                with self.assertRaises(ValueError):
                    NoisyCaptionDataset(split_csv, clean_dir, **bad_kwargs)


if __name__ == "__main__":
    unittest.main()
