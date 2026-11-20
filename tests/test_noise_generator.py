import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

from preprocessing.dataset import NoisyCaptionDataset
from preprocessing.noise_generator import (
    SUPPORTED_DEGRADATION_MODES,
    add_awgn_to_array,
    add_fold_line,
    add_gaussian_blur,
    add_gaussian_noise,
    add_motion_blur,
    add_random_scratches,
    adjust_contrast_brightness,
    apply_degradation,
    generate_noisy_dataset,
)


class NoiseGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(123)
        self.np_rgb = rng.integers(30, 220, size=(32, 40, 3), dtype=np.uint8)
        self.np_gray = self.np_rgb[:, :, 0]
        self.pil_img = Image.fromarray(self.np_rgb, mode="RGB")
        self.t_3d = (
            torch.from_numpy(self.np_rgb.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .contiguous()
        )
        self.t_4d = torch.stack(
            [self.t_3d, torch.clamp(self.t_3d * 0.8 + 0.1, 0.0, 1.0)],
            dim=0,
        )

    def test_all_degradation_operators_preserve_formats_and_modify_pixels(self) -> None:
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

                out_np = fn(self.np_rgb)
                self.assertIsInstance(out_np, np.ndarray)
                self.assertEqual(out_np.shape, self.np_rgb.shape)
                self.assertEqual(out_np.dtype, np.uint8)
                self.assertFalse(np.array_equal(out_np, self.np_rgb))

                out_gray = fn(self.np_gray)
                self.assertIsInstance(out_gray, np.ndarray)
                self.assertEqual(out_gray.shape, self.np_gray.shape)
                self.assertEqual(out_gray.dtype, np.uint8)

                out_t3 = fn(self.t_3d)
                self.assertIsInstance(out_t3, torch.Tensor)
                self.assertEqual(out_t3.shape, self.t_3d.shape)
                self.assertEqual(out_t3.dtype, self.t_3d.dtype)
                self.assertGreaterEqual(float(out_t3.min()), 0.0)
                self.assertLessEqual(float(out_t3.max()), 1.0)

                out_t4 = fn(self.t_4d)
                self.assertIsInstance(out_t4, torch.Tensor)
                self.assertEqual(out_t4.shape, self.t_4d.shape)
                self.assertEqual(out_t4.dtype, self.t_4d.dtype)
                self.assertGreaterEqual(float(out_t4.min()), 0.0)
                self.assertLessEqual(float(out_t4.max()), 1.0)

    def test_apply_degradation_modes_passthrough_determinism_and_tiny_image(self) -> None:
        clean_out = apply_degradation(self.t_4d, mode="clean")
        self.assertTrue(torch.allclose(clean_out, self.t_4d))

        zero_noise = apply_degradation(self.t_4d, mode="gaussian", sigma=0, seed=99)
        self.assertTrue(torch.allclose(zero_noise, self.t_4d))

        for mode in SUPPORTED_DEGRADATION_MODES:
            with self.subTest(mode=mode):
                out = apply_degradation(self.pil_img, mode=mode, seed=42)
                self.assertIsInstance(out, Image.Image)
                self.assertEqual(out.size, self.pil_img.size)

        run_a = apply_degradation(self.t_4d, mode="combined", seed=2025)
        run_b = apply_degradation(self.t_4d, mode="combined", seed=2025)
        run_c = apply_degradation(self.t_4d, mode="combined", seed=2026)
        self.assertTrue(torch.equal(run_a, run_b))
        self.assertFalse(torch.equal(run_a, run_c))

        tiny_img = np.full((4, 4, 3), 120, dtype=np.uint8)
        tiny_out = apply_degradation(tiny_img, mode="combined", seed=11)
        self.assertIsInstance(tiny_out, np.ndarray)
        self.assertEqual(tiny_out.shape, (4, 4, 3))

    def test_validation_errors(self) -> None:
        with self.assertRaises(ValueError):
            add_gaussian_noise(self.pil_img, sigma=-1)
        with self.assertRaises(ValueError):
            add_gaussian_blur(self.pil_img, kernel_size=4)
        with self.assertRaises(ValueError):
            apply_degradation(self.pil_img, mode="unknown")
        with self.assertRaises(ValueError):
            add_gaussian_noise(torch.zeros((3, 8, 8), dtype=torch.uint8))
        with self.assertRaises(TypeError):
            apply_degradation(self.pil_img, mode="combined", unknown_arg=1)
        with self.assertRaises(ValueError):
            add_gaussian_noise(self.pil_img, rng=np.random.default_rng(1), seed=1)
        with self.assertRaises(ValueError):
            add_gaussian_noise(np.zeros((8, 8, 3), dtype=bool), sigma=10)
        with self.assertRaises(ValueError):
            add_awgn_to_array(np.zeros((8, 8, 3), dtype=bool), sigma=10)

    def test_generate_noisy_dataset_backward_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_dir = root / "clean"
            out_dir = root / "noisy"
            clean_dir.mkdir()
            self.pil_img.save(clean_dir / "sample.jpg", quality=95)

            csv_path = root / "captions.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["image", "caption"])
                writer.writerow(["sample.jpg", "mot chu cho dang chay"])

            map_csv = root / "noisy_image_map.csv"
            returned_dir = generate_noisy_dataset(
                captions_csv=csv_path,
                clean_images_dir=clean_dir,
                noise_level=15,
                seed=42,
                output_dir=out_dir,
                map_path=map_csv,
                log_every=0,
            )

            self.assertEqual(returned_dir, out_dir)
            self.assertTrue((out_dir / "sample.jpg").is_file())
            self.assertTrue(map_csv.is_file())


class NoisyCaptionDatasetTests(unittest.TestCase):
    def _create_sample_workspace(self, root: Path, *, with_pregenerated: bool = False) -> tuple[Path, Path, Path | None]:
        clean_dir = root / "clean"
        clean_dir.mkdir()
        img = Image.fromarray(
            np.random.default_rng(7).integers(20, 230, size=(64, 64, 3), dtype=np.uint8),
            mode="RGB",
        )
        img.save(clean_dir / "img1.jpg", quality=95)

        split_csv = root / "split.csv"
        with split_csv.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["image", "caption"])
            writer.writerow(["img1.jpg", "hai con meo"])

        noisy_dir: Path | None = None
        if with_pregenerated:
            noisy_dir = root / "pre_noisy"
            noisy_dir.mkdir()
            pre_img = Image.fromarray(np.full((64, 64, 3), 17, dtype=np.uint8), mode="RGB")
            pre_img.save(noisy_dir / "img1.jpg", quality=100)

        return split_csv, clean_dir, noisy_dir

    def test_default_gaussian_mode_deterministic_and_residual_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            split_csv, clean_dir, _ = self._create_sample_workspace(Path(tmp))
            dataset = NoisyCaptionDataset(
                split_csv_path=split_csv,
                clean_images_dir=clean_dir,
                noise_level=25,
                seed=42,
            )

            first = dataset[0]
            second = dataset[0]
            self.assertTrue(torch.equal(first["noisy"], second["noisy"]))
            self.assertFalse(torch.equal(first["clean"], first["noisy"]))
            self.assertTrue(
                torch.allclose(first["noise_target"], first["noisy"] - first["clean"])
            )

    def test_combined_dynamic_noise_varies_across_reads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            split_csv, clean_dir, _ = self._create_sample_workspace(Path(tmp))
            dataset = NoisyCaptionDataset(
                split_csv_path=split_csv,
                clean_images_dir=clean_dir,
                noise_level="random",
                seed=42,
                degradation_mode="combined",
                dynamic_noise=True,
            )

            first = dataset[0]
            second = dataset[0]
            self.assertGreaterEqual(float(first["noisy"].min()), 0.0)
            self.assertLessEqual(float(first["noisy"].max()), 1.0)
            self.assertFalse(torch.equal(first["noisy"], second["noisy"]))

    def test_pregenerated_noisy_dir_takes_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            split_csv, clean_dir, noisy_dir = self._create_sample_workspace(
                Path(tmp), with_pregenerated=True
            )
            dataset = NoisyCaptionDataset(
                split_csv_path=split_csv,
                clean_images_dir=clean_dir,
                noisy_images_dir=noisy_dir,
                degradation_mode="blur",
                seed=42,
            )

            sample = dataset[0]
            expected_val = 17.0 / 255.0
            self.assertTrue(torch.allclose(sample["noisy"], torch.full_like(sample["noisy"], expected_val), atol=0.02))

    def test_invalid_degradation_mode_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            split_csv, clean_dir, _ = self._create_sample_workspace(Path(tmp))
            with self.assertRaises(ValueError):
                NoisyCaptionDataset(
                    split_csv_path=split_csv,
                    clean_images_dir=clean_dir,
                    degradation_mode="bad_mode",
                )

    def test_negative_noise_level_raises_in_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            split_csv, clean_dir, _ = self._create_sample_workspace(Path(tmp))
            with self.assertRaises(ValueError):
                NoisyCaptionDataset(
                    split_csv_path=split_csv,
                    clean_images_dir=clean_dir,
                    noise_level=-5,
                )

    def test_dynamic_noise_uses_distinct_rng_per_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            split_csv, clean_dir, _ = self._create_sample_workspace(Path(tmp))
            dataset = NoisyCaptionDataset(
                split_csv_path=split_csv,
                clean_images_dir=clean_dir,
                noise_level=25,
                seed=42,
                dynamic_noise=True,
            )

            worker_0 = mock.Mock(id=0, seed=100)
            worker_1 = mock.Mock(id=1, seed=100)
            with mock.patch("preprocessing.dataset.get_worker_info", return_value=worker_0):
                sample_0 = dataset[0]
            with mock.patch("preprocessing.dataset.get_worker_info", return_value=worker_1):
                sample_1 = dataset[0]

            self.assertFalse(torch.equal(sample_0["noisy"], sample_1["noisy"]))


if __name__ == "__main__":
    unittest.main()
