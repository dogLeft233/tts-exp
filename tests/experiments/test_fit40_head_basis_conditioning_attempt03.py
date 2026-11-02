"""Synthetic engineering tests plus a pinned small-input V3 derivative check.

The finite-difference fixture is not Fit-cohort evidence and performs no
audio/video scoring or model selection.
"""
from __future__ import annotations

import importlib.util
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/fit40_head_basis_conditioning_attempt03.py"
SPEC = importlib.util.spec_from_file_location("fit40_head_basis_conditioning_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GateSummaryTests(unittest.TestCase):
    def test_population_mean_std_cv_and_dc_fraction(self) -> None:
        summary = MODULE.population_summary([0.0, 0.5, 1.0])
        self.assertAlmostEqual(summary["mean"], 0.5)
        self.assertAlmostEqual(summary["population_std"], np.sqrt(1.0 / 6.0))
        self.assertAlmostEqual(summary["coefficient_of_variation"], np.sqrt(2.0 / 3.0))
        dc = MODULE._dc_dominance(summary)
        self.assertAlmostEqual(dc["dc_energy_fraction"], 0.6)
        self.assertAlmostEqual(dc["variance_to_dc_energy"], 2.0 / 3.0)

    def test_correlation_and_constant_gate_behavior(self) -> None:
        self.assertAlmostEqual(MODULE.population_correlation([0, 1, 2], [2, 1, 0]), -1.0)
        self.assertIsNone(MODULE.population_correlation([1, 1, 1], [0, 1, 2]))


class EmpiricalDesignTests(unittest.TestCase):
    @staticmethod
    def identity_smoother(values: np.ndarray) -> np.ndarray:
        return values

    def test_nine_column_order_and_zero_reference_band(self) -> None:
        basis = np.eye(4, dtype=np.float32)
        positive = np.asarray([0.25, 0.75], dtype=np.float32)
        negative = np.asarray([0.5, 0.5], dtype=np.float32)
        design, labels = MODULE.build_preprojection_design(
            basis, 3, positive, negative, self.identity_smoother)
        self.assertEqual(design.shape, (8, 9))
        self.assertEqual([(row["component"], row["band_index"]) for row in labels], [
            ("static", 0), ("static", 1), ("static", 2),
            ("positive_flux", 0), ("positive_flux", 1), ("positive_flux", 2),
            ("negative_flux", 0), ("negative_flux", 1), ("negative_flux", 2),
        ])
        # Each trainable column is its one-hot physical band. The fixed
        # reference band has coefficient zero and is not subtracted.
        self.assertEqual(design[0, 0], 1.0)
        self.assertEqual(design[0, 3], 0.25)
        self.assertEqual(design[0, 6], 0.5)
        self.assertEqual(design[6, 0], 0.0)
        self.assertEqual(design[6, 3], 0.0)
        self.assertEqual(design[6, 6], 0.0)
        self.assertEqual(design[1, 3], 0.75)
        self.assertEqual(design[1, 6], 0.5)
        self.assertEqual(design[7, 3], 0.0)
        self.assertEqual(design[7, 6], 0.0)

    def test_constant_gates_reduce_empirical_rank_without_dropping_parameters(self) -> None:
        basis = np.eye(4, dtype=np.float32)
        design, labels = MODULE.build_preprojection_design(
            basis, 3, [1.0, 1.0], [2.0, 2.0], self.identity_smoother)
        self.assertEqual(len(labels), 9)
        summary = MODULE.design_gram_summary(design.astype(np.float64).T @ design.astype(np.float64))
        self.assertEqual(summary["effective_rank"], 3)
        self.assertEqual(summary["normalized_effective_rank"], 3)
        self.assertTrue(all(value >= 0.0 for value in summary["singular_values"]))

    def test_invalid_frequency_partition_and_clock_mismatch_reject(self) -> None:
        with self.assertRaises(MODULE.ConditioningError):
            MODULE.build_preprojection_design(
                np.full((4, 3), 0.25, dtype=np.float32), 3, [0.1], [0.2], self.identity_smoother)
        with self.assertRaises(MODULE.ConditioningError):
            MODULE.build_preprojection_design(
                np.eye(4, dtype=np.float32), 3, [0.1, 0.2], [0.2], self.identity_smoother)


class FrozenV3DerivativeTests(unittest.TestCase):
    def test_actual_v3_autograd_and_finite_difference_match_design(self) -> None:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        source_path = ROOT / "scripts/audio_enhancement/shared_recipient_transition_head_v3.py"
        source_raw = source_path.read_bytes()
        self.assertEqual(len(source_raw), 32233)
        self.assertEqual(hashlib.sha256(source_raw).hexdigest(),
                         "77acdff03cca22261852f48364cbc75263db58a14cb565c08aa0d268afca6e87")
        snapshot_path = ROOT / ".codex/scratch/fit40_training_snapshots_attempt05/legacy_native_attempt11/snapshot_step120.json"
        snapshot_raw = snapshot_path.read_bytes()
        self.assertEqual(len(snapshot_raw), 9223)
        self.assertEqual(hashlib.sha256(snapshot_raw).hexdigest(),
                         "9519c18b8c95c8291d70871a452417f4fb1cd361828284ad39feaf9cb1b178e1")
        snapshot = json.loads(snapshot_raw.decode("utf-8"))

        head_spec = importlib.util.spec_from_file_location("fit40_basis_actual_v3", source_path)
        self.assertIsNotNone(head_spec)
        self.assertIsNotNone(head_spec.loader)
        head_module = importlib.util.module_from_spec(head_spec)
        head_spec.loader.exec_module(head_module)
        head = head_module.SharedRecipientTransitionHeadV3(**snapshot["head_constructor"])
        theta = torch.as_tensor(snapshot["parameter"]["values"], dtype=torch.float32)
        with torch.no_grad():
            head.band_contrasts_db.copy_(theta)
        head.eval()

        time = torch.arange(4096, dtype=torch.float32) / 16000.0
        source = (0.070 * torch.sin(2.0 * torch.pi * 220.0 * time)
                  + 0.035 * torch.sin(2.0 * torch.pi * 1030.0 * time + 0.2)
                  + 0.018 * torch.sin(2.0 * torch.pi * 4100.0 * time + 0.6))
        baseline = head._rms_peak_baseline(source)[0]
        result = head(source, baseline)
        mask = result["smoothed_log_magnitude_mask_db"]
        positive = result["positive_flux_gate"].detach().cpu().numpy()
        negative = result["negative_flux_gate"].detach().cpu().numpy()

        def exact_smooth(plane: np.ndarray) -> np.ndarray:
            value = torch.as_tensor(np.ascontiguousarray(plane, dtype=np.float32))
            value = head_module._smooth_mask_axis(
                value, head.mask_smoothing_frequency_bins, "frequency", head.mask_smoothing_pad_mode)
            value = head_module._smooth_mask_axis(
                value, head.mask_smoothing_time_frames, "time", head.mask_smoothing_pad_mode)
            return value.detach().cpu().numpy()

        design, _labels = MODULE.build_preprojection_design(
            head.band_basis.detach().cpu().numpy(), head.reference_band_index,
            positive, negative, exact_smooth)
        probe = torch.linspace(-0.75, 0.9, mask.numel(), dtype=torch.float32).reshape(mask.shape)
        scalar = torch.sum(mask * probe)
        gradient = torch.autograd.grad(scalar, head.band_contrasts_db)[0].detach().cpu().numpy().reshape(9)
        expected = design.astype(np.float64).T @ probe.detach().cpu().numpy().reshape(-1).astype(np.float64)
        np.testing.assert_allclose(gradient, expected, rtol=2e-5, atol=2e-5)

        def projected_mask(theta_values: np.ndarray) -> float:
            with torch.no_grad():
                head.band_contrasts_db.copy_(torch.as_tensor(theta_values.reshape(3, 3), dtype=torch.float32))
                current = head(source, baseline)["smoothed_log_magnitude_mask_db"]
                return float(torch.sum(current * probe).item())

        theta_np = theta.detach().cpu().numpy().astype(np.float32).reshape(9)
        epsilon = 1e-2
        finite_difference = []
        for index in range(9):
            plus = theta_np.copy()
            minus = theta_np.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            finite_difference.append((projected_mask(plus) - projected_mask(minus)) / (2.0 * epsilon))
        np.testing.assert_allclose(np.asarray(finite_difference), gradient, rtol=5e-3, atol=2e-4)
        with torch.no_grad():
            head.band_contrasts_db.copy_(theta)


if __name__ == "__main__":
    unittest.main()
