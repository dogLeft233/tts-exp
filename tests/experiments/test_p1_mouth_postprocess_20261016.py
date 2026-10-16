from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/experiments/p1_gain_dose_response_20261016/mouth_postprocess.py"
SPEC = importlib.util.spec_from_file_location("p1_mouth_postprocess", SOURCE)
assert SPEC is not None and SPEC.loader is not None
mouth = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mouth
SPEC.loader.exec_module(mouth)


class MouthPostprocessTests(unittest.TestCase):
    def landmarks(self, upper_y: float = 0.40, lower_y: float = 0.60) -> list[list[float]]:
        points = [[0.0, 0.0, 0.0] for _ in range(478)]
        points[13] = [0.50, upper_y, 0.0]
        points[14] = [0.50, lower_y, 0.0]
        points[33] = [0.25, 0.50, 0.0]
        points[263] = [0.75, 0.50, 0.0]
        return points

    def test_render_success_status_accepts_receipt_labels_only(self) -> None:
        self.assertTrue(mouth._is_render_success({"status": "PASS"}))
        self.assertTrue(mouth._is_render_success({"status": "RENDERED"}))
        self.assertFalse(mouth._is_render_success({"status": "FAILED"}))

    def test_pixel_space_aperture_uses_lip_gap_over_eye_distance(self) -> None:
        result = mouth.opening_geometry(self.landmarks(), 1920, 1080)
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.opening_ratio or 0.0, 216 / 960, places=12)

    def test_required_nonfinite_or_degenerate_eye_points_are_invalid(self) -> None:
        points = self.landmarks()
        points[13][1] = math.nan
        self.assertEqual(mouth.opening_geometry(points, 1920, 1080).reason, "required_landmark_missing_or_nonfinite")
        points = self.landmarks()
        points[263] = list(points[33])
        self.assertEqual(mouth.opening_geometry(points, 1920, 1080).reason, "eye_distance_not_positive")

    def test_common_mask_is_intersection_of_all_eight_cells(self) -> None:
        masks = [[True, True, False, True] for _ in mouth.CONDITION_IDS]
        masks[3][1] = False
        self.assertEqual(mouth.common_valid_mask(masks), [True, False, False, True])
        with self.assertRaises(ValueError):
            mouth.common_valid_mask(masks[:-1])

    def test_absolute_speed_uses_only_valid_original_adjacent_pairs(self) -> None:
        speed, valid = mouth.adjacent_speed([0.08, 0.04, 0.12, 0.16], [True, True, False, True])
        self.assertEqual(valid, [True, False, False])
        self.assertAlmostEqual(speed[0] or 0.0, 1.0)
        self.assertGreaterEqual(speed[0] or 0.0, 0.0)
        self.assertEqual(speed[1:], [None, None])

    def test_gap_is_not_bridged_for_velocity(self) -> None:
        speed, valid = mouth.adjacent_speed([0.0, None, 0.10], [True, False, True])
        self.assertEqual(valid, [False, False])
        self.assertEqual(speed, [None, None])

    def test_frozen_mouth_gate_is_inclusive_and_does_not_change_score_rows(self) -> None:
        mask = [True] * 27 + [False] * 3
        result = mouth.qualification(frame_count=30, common_mask=mask, metadata_aligned=True)
        self.assertEqual(result["status"], "PASS_MOUTH_MEASUREMENT_QUALIFICATION")
        self.assertEqual(result["common_valid_frames"], 27)
        self.assertEqual(result["common_adjacent_valid_edges"], 26)
        self.assertFalse(result["score_rows_affected"])

    def test_low_coverage_or_clock_failure_is_unmeasurable(self) -> None:
        low = mouth.qualification(frame_count=30, common_mask=[True] * 26 + [False] * 4, metadata_aligned=True)
        self.assertEqual(low["status"], "UNMEASURABLE")
        no_clock = mouth.qualification(frame_count=30, common_mask=[True] * 30, metadata_aligned=False)
        self.assertEqual(no_clock["status"], "UNMEASURABLE")
        self.assertFalse(no_clock["score_rows_affected"])


if __name__ == "__main__":
    unittest.main()
