import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from scripts.experiments.p1_gain_dose_response_20261016 import pilot, wav2lip_seeded_inference


class PilotContractsTest(unittest.TestCase):
    def test_eight_render_conditions_include_shared_zero_alias(self):
        plan = pilot.condition_plan(6.0)
        self.assertEqual(len(plan), 8)
        self.assertEqual([row["condition_id"] for row in plan], [
            "RAW_R1", "RAW_R2", "FIXED_R1", "FIXED_R2",
            "DOSE_M6", "DOSE_M3", "DOSE_P3", "DOSE_P6",
        ])
        self.assertEqual(plan[0]["relative_dose_db"], 0.0)
        self.assertTrue(plan[0]["dose_curve_zero_alias"])
        self.assertEqual([row["relative_dose_db"] for row in plan[4:]], [-6.0, -3.0, 3.0, 6.0])
        self.assertEqual([row["seed"] for row in (plan[0], plan[2], plan[4], plan[5], plan[6], plan[7])], [2026101601] * 6)
        self.assertEqual(plan[1]["seed"], 2026101602)
        self.assertEqual(plan[3]["seed"], 2026101602)

    def test_float32_scalar_gain_preserves_clock_and_spectrum(self):
        time = np.arange(4096, dtype=np.float32) / 16_000.0
        source = (0.2 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
        gained, receipt = pilot.scaled_float32(source, 10 ** (6 / 20))
        self.assertEqual(gained.shape, source.shape)
        self.assertTrue(receipt["clock_preserved"])
        self.assertTrue(receipt["no_clipping"])
        self.assertLessEqual(receipt["relative_l2_residual"], 1e-7)
        self.assertLessEqual(receipt["inverse_relative_l2_residual"], 1e-7)
        self.assertLessEqual(receipt["normalized_spectrum_shape_max_abs_error"], 1e-7)

    def test_float_wav_round_trip_keeps_every_sample(self):
        source = np.linspace(-0.25, 0.25, 1001, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scaled.wav"
            pilot.write_float_wav(path, source)
            restored, header = pilot.read_float_wav(path)
        np.testing.assert_array_equal(restored, source)
        self.assertEqual(header["sample_rate_hz"], 16_000)
        self.assertEqual(header["frames"], source.size)

    def test_shared_support_uses_one_unpadded_query_set_for_all_cells(self):
        counts = {f"C{index}": 75 for index in range(8)}
        result = pilot.shared_support(80, counts)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["primary_guard20_q_indices"], list(range(20, 55)))
        self.assertEqual(result["full_q_indices"], list(range(15, 65)))
        self.assertTrue(result["no_padding_verified"])
        audio_limited = pilot.shared_support(75, {f"C{index}": 80 for index in range(8)})
        self.assertEqual(audio_limited["full_q_indices"], list(range(15, 60)))
        counts["C7"] = 30
        unsupported = pilot.shared_support(80, counts)
        self.assertEqual(unsupported["status"], "INSUFFICIENT_OR_INVALID_SUPPORT")
        self.assertEqual(unsupported["primary_guard20_q_indices"], [])

    def test_render_attempt_dirs_create_speaker_log_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "videos" / "S0102" / "RAW_R1.mp4"
            log = root / "logs" / "wav2lip" / "S0102" / "RAW_R1.log"
            work = root / "wav2lip_work" / "S0102" / "RAW_R1"

            pilot.prepare_render_attempt_dirs(video, log, work)

            self.assertTrue(video.parent.is_dir())
            self.assertTrue(log.parent.is_dir())
            self.assertTrue(work.is_dir())
            self.assertTrue((work / "temp").is_dir())

    def test_seeded_inference_configures_python_numpy_torch_and_cudnn(self):
        seed = 2026101601
        cudnn = SimpleNamespace(benchmark=True, deterministic=False)
        cuda = SimpleNamespace(manual_seed_all=Mock(), is_available=Mock(return_value=True))
        torch = SimpleNamespace(
            __version__="test-torch",
            manual_seed=Mock(),
            cuda=cuda,
            backends=SimpleNamespace(cudnn=cudnn),
            are_deterministic_algorithms_enabled=Mock(return_value=False),
        )
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        previous_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        try:
            with patch.dict(os.environ, {"PYTHONHASHSEED": str(seed)}, clear=False):
                with patch.dict(sys.modules, {"torch": torch}):
                    receipt = wav2lip_seeded_inference.seed_runtime(seed)
            torch.manual_seed.assert_called_once_with(seed)
            cuda.manual_seed_all.assert_called_once_with(seed)
            self.assertEqual(receipt["python_random_seed"], seed)
            self.assertEqual(receipt["numpy_seed"], seed)
            self.assertEqual(receipt["torch_seed"], seed)
            self.assertEqual(receipt["cuda_seed_all"], seed)
            self.assertEqual(receipt["cublas_workspace_config"], ":4096:8")
            self.assertFalse(receipt["cudnn_benchmark"])
            self.assertTrue(receipt["cudnn_deterministic"])
            self.assertEqual(receipt["pythonhashseed"], str(seed))
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            if previous_cublas is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_cublas

    def test_curve_metrics_are_median_minus_minimum_and_zero_column(self):
        curve = np.arange(31, dtype=np.float64)
        result = pilot.curve_metrics(curve, 5)
        self.assertEqual(result["B"], 15.0)
        self.assertEqual(result["Dmin"], 0.0)
        self.assertEqual(result["D0"], 15.0)
        self.assertEqual(result["C"], 15.0)
        self.assertEqual(result["anchor_D"], 5.0)


if __name__ == "__main__":
    unittest.main()
