import importlib.util
import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "benchmark.v2.py"
SPEC = importlib.util.spec_from_file_location("benchmark_v2", MODULE_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class BenchmarkMetricsTests(unittest.TestCase):
    def test_benchmark_entrypoints_take_no_arguments(self):
        self.assertEqual(
            list(inspect.signature(benchmark.run_benchmark).parameters), []
        )

        module_path = MODULE_PATH.with_name("benchmark.v1.py")
        spec = importlib.util.spec_from_file_location("benchmark_v1", module_path)
        benchmark_v1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(benchmark_v1)
        self.assertEqual(
            list(inspect.signature(benchmark_v1.run_inference_benchmark).parameters),
            [],
        )

    def test_run_benchmark_rejects_invalid_arguments_before_io(self):
        with self.assertRaisesRegex(ValueError, "noise_level"):
            benchmark._run_benchmark(noise_level="35")
        with self.assertRaisesRegex(ValueError, "max_length"):
            benchmark._run_benchmark(max_length=1)

    def test_run_benchmark_prevents_output_inside_input_directories(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            images_dir = root / "images"
            noisy_dir = root / "noisy"
            images_dir.mkdir()
            noisy_dir.mkdir()

            with self.assertRaisesRegex(ValueError, "thư mục ảnh"):
                benchmark._run_benchmark(
                    images_dir=images_dir,
                    noisy_dir=noisy_dir,
                    output_path=images_dir / "result.csv",
                )

    def test_image_quality_uses_matching_ranges_for_float_and_byte_images(self):
        clean = np.linspace(0.0, 1.0, 8 * 8 * 3, dtype=np.float32).reshape(8, 8, 3)
        denoised = np.clip(clean + 0.05, 0.0, 1.0)

        normalized_scores = benchmark.calculate_image_quality(clean, denoised)
        byte_scores = benchmark.calculate_image_quality(
            np.round(clean * 255).astype(np.uint8),
            np.round(denoised * 255).astype(np.uint8),
        )

        self.assertAlmostEqual(normalized_scores[0], byte_scores[0], places=1)
        self.assertAlmostEqual(normalized_scores[1], byte_scores[1], places=2)

    def test_dark_byte_images_still_use_byte_range(self):
        byte_scores = benchmark.calculate_image_quality(
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.ones((8, 8, 3), dtype=np.uint8),
        )
        normalized_scores = benchmark.calculate_image_quality(
            np.zeros((8, 8, 3), dtype=np.float32),
            np.full((8, 8, 3), 1 / 255, dtype=np.float32),
        )

        self.assertAlmostEqual(byte_scores[0], normalized_scores[0], places=2)
        self.assertAlmostEqual(byte_scores[1], normalized_scores[1], places=2)

    def test_image_quality_rejects_mismatched_shapes_and_invalid_ranges(self):
        image = np.zeros((8, 8, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            benchmark.calculate_image_quality(image, np.zeros((7, 8, 3)))
        with self.assertRaises(ValueError):
            benchmark.calculate_image_quality(image - 0.1, image)

    def test_limit_checks_only_requested_test_images(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            clean_dir = root / "clean"
            noisy_dir = root / "noisy"
            clean_dir.mkdir()
            noisy_dir.mkdir()
            (clean_dir / "first.jpg").touch()
            (noisy_dir / "first.jpg").touch()

            captions_path = root / "test.csv"
            captions_path.write_text(
                "image,caption\nfirst.jpg,caption one\nsecond.jpg,caption two\n",
                encoding="utf-8",
            )

            samples = benchmark._read_test_samples(
                captions_path, clean_dir, noisy_dir, limit=1
            )

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0][0], "first.jpg")


if __name__ == "__main__":
    unittest.main()
