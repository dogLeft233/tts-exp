"""Focused tests for the source-only snapshot mask/gate diagnostic math."""
import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "experiments"
    / "cross_tfg_audio_head_20261003"
    / "analyze_fit_snapshot_mask_gates_attempt05.py"
)
SPEC = importlib.util.spec_from_file_location("mask_gate_diagnostic_test_subject", MODULE_PATH)
SUBJECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUBJECT)


class _FakeHead:
    def __init__(self, torch):
        self.band_basis = torch.eye(4, dtype=torch.float32)
        self.n_fft = 6
        self.mask_smoothing_frequency_bins = 1
        self.mask_smoothing_time_frames = 1
        self.mask_smoothing_pad_mode = "replicate"
        self.frequency_common_mode_policy = "retain_reference_coded"
        self.max_abs_mask_db = 3.0


class _IdentitySmoother:
    @staticmethod
    def _smooth_mask_axis(values, _width, _axis, _mode):
        return values


class MaskGateDiagnosticTests(unittest.TestCase):
    def test_gate_dc_and_centered_traces_keep_invalid_positions(self):
        output = {
            "valid_transition_mask": np.asarray([False, True, True, False]),
            "positive_flux_gate": np.asarray([0.0, 0.2, 0.8, 0.0]),
            "negative_flux_gate": np.asarray([0.0, 0.7, 0.3, 0.0]),
            "positive_relative_flux": np.asarray([0.0, 0.1, 0.9, 0.0]),
            "negative_relative_flux": np.asarray([0.0, 0.9, 0.1, 0.0]),
            "positive_flux_threshold": np.asarray(0.4),
            "negative_flux_threshold": np.asarray(0.6),
            "positive_flux_valid_transition_count": 2,
            "negative_flux_valid_transition_count": 2,
            "positive_flux_has_valid_support": True,
            "negative_flux_has_valid_support": True,
            "active_frame_mask": np.asarray([True, True, True, False]),
            "stft_frame_centers_source_samples": np.asarray([0, 128, 256, 384]),
        }
        result = SUBJECT._gate_diagnostics(output)
        positive = result["summary"]["positive"]
        self.assertAlmostEqual(positive["gate_dc_mean_all_stft_frames"], 0.25)
        self.assertAlmostEqual(positive["gate_mean_valid_transitions"], 0.5)
        self.assertEqual(positive["valid_transition_count_recomputed"], 2)
        self.assertAlmostEqual(
            result["summary"]["polarity_asymmetry_on_valid_source_transitions"]["positive_minus_negative_gate_mean"],
            0.0,
        )
        np.testing.assert_allclose(
            result["arrays"]["positive_gate_centered_all_frames"],
            [-0.25, -0.05, 0.55, -0.25],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            result["arrays"]["positive_gate_centered_valid_transitions"],
            [0.0, -0.3, 0.3, 0.0],
            atol=1e-7,
        )

    def test_component_decomposition_uses_one_observed_global_scale(self):
        import torch

        head = _FakeHead(torch)
        weights = torch.tensor([[0.2, -0.1, 0.3, 0.0],
                                [0.4, 0.2, -0.2, 0.1],
                                [-0.2, 0.3, 0.1, -0.4]], dtype=torch.float32)
        positive = torch.tensor([0.25, 0.75], dtype=torch.float32)
        negative = torch.tensor([0.5, 0.1], dtype=torch.float32)
        raw_components = torch.stack((
            (head.band_basis.T @ weights[0].reshape(4, 1)).expand(-1, 2),
            (head.band_basis.T @ weights[1].reshape(4, 1)) * positive.reshape(1, -1),
            (head.band_basis.T @ weights[2].reshape(4, 1)) * negative.reshape(1, -1),
        ))
        raw = raw_components.sum(dim=0)
        premax = float(raw.abs().amax().item())
        extent = premax / head.max_abs_mask_db
        scale = math.tanh(extent) / extent if extent else 1.0
        projected = raw * scale
        output = {
            "reference_coded_band_weights_db": weights.numpy(),
            "positive_flux_gate": positive.numpy(),
            "negative_flux_gate": negative.numpy(),
            "raw_log_magnitude_mask_db": raw.numpy(),
            "smoothed_log_magnitude_mask_db": raw.numpy(),
            "frequency_common_mode_before_policy_db": np.zeros((1, 2), dtype=np.float32),
            "policy_mask_db_before_smooth_bound": raw.numpy(),
            "projected_log_magnitude_mask_db": projected.numpy(),
            "smooth_projection_scale": scale,
            "preprojection_max_abs_mask_db": premax,
            "postprojection_max_abs_mask_db": float(projected.abs().amax().item()),
            "valid_transition_mask": np.asarray([True, True]),
            "positive_relative_flux": np.asarray([0.1, 0.9], dtype=np.float32),
            "negative_relative_flux": np.asarray([0.9, 0.1], dtype=np.float32),
        }
        result = SUBJECT._component_decomposition(head, _IdentitySmoother, output, torch)
        self.assertTrue(result["checks"]["component_decomposition_uses_observed_single_global_scale"])
        self.assertLess(result["checks"]["projected_component_sum_max_abs_error"], 1e-6)
        self.assertLess(result["checks"]["global_projection_scale_formula_abs_error"], 1e-7)
        self.assertEqual(result["arrays"]["four_band_preprojection_mean_db_by_frame"].shape, (4, 2))
        self.assertEqual(result["arrays"]["four_band_projected_component_mean_db_by_frame"].shape, (3, 4, 2))
        self.assertEqual(result["arrays"]["four_band_postprojection_mean_gain_by_frame"].shape, (4, 2))
        self.assertEqual(len(result["band_statistics"]["postprojection"]["per_band"]), 4)
        self.assertLess(result["checks"]["preprojection_dc_plus_centered_max_abs_error"], 1e-7)
        self.assertLess(result["checks"]["postprojection_dc_plus_centered_max_abs_error"], 1e-7)
        self.assertIn("gain_vs_positive_relative_flux", result["band_statistics"]["postprojection"]["per_band"][0])

    def test_component_decomposition_rejects_mismatched_raw_mask(self):
        import torch

        head = _FakeHead(torch)
        output = {
            "reference_coded_band_weights_db": np.zeros((3, 4), dtype=np.float32),
            "positive_flux_gate": np.zeros(2, dtype=np.float32),
            "negative_flux_gate": np.zeros(2, dtype=np.float32),
            "raw_log_magnitude_mask_db": np.ones((4, 2), dtype=np.float32),
        }
        with self.assertRaises(SUBJECT.DiagnosticError):
            SUBJECT._component_decomposition(head, _IdentitySmoother, output, torch)

    def test_common_mode_policy_must_reconstruct_exact_preprojection_mask(self):
        import torch

        head = _FakeHead(torch)
        weights = torch.zeros((3, 4), dtype=torch.float32)
        gates = torch.zeros(2, dtype=torch.float32)
        raw = torch.zeros((4, 2), dtype=torch.float32)
        output = {
            "reference_coded_band_weights_db": weights.numpy(),
            "positive_flux_gate": gates.numpy(),
            "negative_flux_gate": gates.numpy(),
            "raw_log_magnitude_mask_db": raw.numpy(),
            "smoothed_log_magnitude_mask_db": raw.numpy(),
            "frequency_common_mode_before_policy_db": np.zeros((1, 2), dtype=np.float32),
            "policy_mask_db_before_smooth_bound": np.ones((4, 2), dtype=np.float32),
        }
        with self.assertRaises(SUBJECT.DiagnosticError):
            SUBJECT._component_decomposition(head, _IdentitySmoother, output, torch)


if __name__ == "__main__":
    unittest.main()
