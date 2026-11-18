import io
import unittest
from unittest import mock

import main


class ReconfigurableOutput(io.StringIO):
    def reconfigure(self, **kwargs):
        pass


class MainWorkflowTests(unittest.TestCase):
    def test_main_runs_baseline_then_noisy_workflows(self):
        steps = []
        with (
            mock.patch.object(
                main, "train_baseline", side_effect=lambda device: steps.append("baseline")
            ),
            mock.patch.object(
                main,
                "train_noisy_approaches",
                side_effect=lambda device: steps.append("noisy"),
            ),
            mock.patch.object(main.torch.cuda, "is_available", return_value=False),
            mock.patch.object(main.sys, "stdout", ReconfigurableOutput()),
        ):
            main.main()

        self.assertEqual(steps, ["baseline", "noisy"])

    def test_noisy_workflow_generates_data_then_trains_both_approaches(self):
        device = main.torch.device("cpu")
        noisy_dir = main.PROCESSED_DATA_DIR / "test-noisy"
        events = mock.Mock()
        with (
            mock.patch.object(
                main, "generate_noisy_dataset", return_value=noisy_dir
            ) as generate,
            mock.patch.object(main, "train_dncnn") as train_dncnn,
            mock.patch.object(main, "train_end2end") as train_end2end,
            mock.patch.object(main.sys, "stdout", ReconfigurableOutput()),
        ):
            events.attach_mock(generate, "generate")
            events.attach_mock(train_dncnn, "dncnn")
            events.attach_mock(train_end2end, "end2end")
            main.train_noisy_approaches(device)

        self.assertEqual(
            [call[0] for call in events.mock_calls],
            ["generate", "dncnn", "end2end"],
        )
        self.assertEqual(train_dncnn.call_args.kwargs["noisy_images_dir"], noisy_dir)
        self.assertEqual(train_end2end.call_args.kwargs["noisy_dir"], noisy_dir)
        self.assertNotIn("init_weights", train_end2end.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
