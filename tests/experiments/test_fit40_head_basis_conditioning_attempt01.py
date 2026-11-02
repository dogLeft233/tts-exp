"""Synthetic engineering tests for the Fit-only basis diagnostic helpers."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/fit40_head_basis_conditioning_attempt01.py"
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

    def test_nine_column_order_and_reference_contrasts(self) -> None:
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
        # Frequency row 0 is band 0; the fixed reference band is row 3.
        self.assertEqual(design[0, 0], 1.0)
        self.assertEqual(design[0, 3], 0.25)
        self.assertEqual(design[0, 6], 0.5)
        self.assertEqual(design[6, 0], -1.0)
        self.assertEqual(design[6, 3], -0.25)
        self.assertEqual(design[6, 6], -0.5)

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


if __name__ == "__main__":
    unittest.main()
