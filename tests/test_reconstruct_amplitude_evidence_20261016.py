from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.experiments import reconstruct_amplitude_evidence_20261016 as analysis


class AmplitudeEvidenceUnitTests(unittest.TestCase):
    def test_rms_curve_decomposition_uses_median_min_zero_and_anchor(self) -> None:
        curve = [float(i) for i in range(31)]
        result = analysis.derive_rms_decomposition(curve)
        self.assertEqual(result["B"], 15.0)
        self.assertEqual(result["D"], 0.0)
        self.assertEqual(result["D0"], 15.0)
        self.assertEqual(result["D_anchor"], 18.0)
        self.assertEqual(result["best_lag"], -15)
        self.assertEqual(result["C"], 15.0)
        self.assertEqual(result["C_anchor"], -3.0)
        self.assertEqual(result["search_gain"], 15.0)

    def test_rms_curve_decomposition_rejects_bad_width_or_nonfinite(self) -> None:
        with self.assertRaises(ValueError):
            analysis.derive_rms_decomposition([1.0, 2.0])
        with self.assertRaises(ValueError):
            analysis.derive_rms_decomposition([float("nan")] * 31)

    def test_common_support_requires_all_q_cells_and_identical_frame_indices(self) -> None:
        cells = {q: {"indices": [4, 5, 6], "support_n": 3} for q in analysis.CELLS}
        self.assertEqual(analysis.validate_common_support(cells), (4, 5, 6))
        cells["q01"] = {"indices": [4, 5, 7], "support_n": 3}
        with self.assertRaises(ValueError):
            analysis.validate_common_support(cells, "synthetic mismatch")
        with self.assertRaises(ValueError):
            analysis.validate_common_support({"q00": cells["q00"]}, "missing q")

    def test_equal_speaker_curve_mean_refuses_missing_clip_and_speaker(self) -> None:
        values = {"S1": [1.0, 3.0], "S2": [5.0, 7.0]}
        self.assertEqual(analysis.equal_speaker_curve_mean(values, {"S1", "S2"}), 4.0)
        with self.assertRaises(ValueError):
            analysis.equal_speaker_curve_mean({"S1": [1.0, 3.0], "S2": [5.0]}, {"S1", "S2"})
        with self.assertRaises(ValueError):
            analysis.equal_speaker_curve_mean({"S1": [1.0, 3.0]}, {"S1", "S2"})

    def test_q_cell_semantics_remain_separate_between_experiments(self) -> None:
        self.assertEqual(analysis.MEL_Q_LABELS["q10"], "interior-only")
        self.assertEqual(analysis.MEL_Q_LABELS["q01"], "boundary-only")
        self.assertEqual(analysis.RMS_Q_LABELS["q10"], "direction-only")
        self.assertEqual(analysis.RMS_Q_LABELS["q01"], "radius-only")

    def test_forest_selection_has_four_primary_interventions_and_secondary_joint(self) -> None:
        mel_contrasts = ("interior_q10_minus_q00", "boundary_q01_minus_q00", "joint_q11_minus_q00")
        rms_contrasts = ("direction_q10_minus_q00", "radius_q01_minus_q00", "joint_q11_minus_q00")

        def rows(contrasts: tuple[str, ...], estimand: str | None) -> list[dict]:
            return [
                {
                    "contrast": contrast, "geometry": "raw", "support": "guard20", "metric": "C",
                    **({"estimand": estimand} if estimand is not None else {}),
                    "equal_speaker_mean": 0.1, "ci99_lower": 0.01, "ci99_upper": 0.2,
                    "complete_speakers": 40, "speaker_count": 40,
                }
                for contrast in contrasts
            ]

        selected = analysis.forest_rows(rows(mel_contrasts, None), rows(rms_contrasts, "observed"))
        self.assertEqual(sum(p["priority"] == "primary" for p in selected), 4)
        self.assertEqual(sum(p["priority"] == "secondary" for p in selected), 2)
        self.assertTrue(all(p["ci99_lower"] == 0.01 and p["ci99_upper"] == 0.2 for p in selected))


if __name__ == "__main__":
    unittest.main()
