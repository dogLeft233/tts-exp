from __future__ import annotations

import unittest

from scripts.experiments.p1_gain_dose_response_20261016.render_p1_gain_summary import (
    DOSE_CONTRAST_IDS,
    build_payload,
    render_svg,
)


class P1GainSummaryRenderTests(unittest.TestCase):
    def _fixture(self):
        planned = [f"S{i:04d}" for i in range(102, 112)] + [f"S{i:04d}" for i in range(144, 154)]
        summary = {
            "planned_speakers": planned,
            "n_planned_speakers": 20,
            "protocol_sha256": "a" * 64,
            "input_sha256": "b" * 64,
            "primary_fixed_minus_raw": {
                "status": "WITHHELD_INCOMPLETE_PLANNED_DENOMINATOR",
                "n_planned": 20,
                "n_complete": 19,
            },
            "dose_family_vs_raw_r1": {
                "status": "COMPUTABLE",
                "n_planned": 20,
                "n_complete": 20,
                "contrasts": {
                    dose_id: {
                        "n": 20,
                        "mean": (i - 1.5) / 10,
                        "simultaneous_ci95": [(i - 2.0) / 10, (i - 1.0) / 10],
                    }
                    for i, dose_id in enumerate(DOSE_CONTRAST_IDS)
                },
            },
        }
        rows = []
        for i, speaker_id in enumerate(planned):
            delta = (0.1 if i < 8 else (-0.05 if i < 18 else 0.0)) if i < 19 else None
            row = {
                "speaker_id": speaker_id,
                "primary_complete": "True" if delta is not None else "False",
                "primary_delta_C": "" if delta is None else str(delta),
            }
            for j, (_, field) in enumerate((
                ("−6 dB", "DOSE_M6_dose_delta_C"),
                ("−3 dB", "DOSE_M3_dose_delta_C"),
                ("+3 dB", "DOSE_P3_dose_delta_C"),
                ("+6 dB", "DOSE_P6_dose_delta_C"),
            )):
                row[field] = str((j - 1.5) / 10 + i / 1000)
            rows.append(row)
        return summary, rows, planned

    def test_sign_counts_keep_missing_in_planned_denominator_and_family_ci_unchanged(self):
        summary, rows, _ = self._fixture()
        payload = build_payload(summary, rows)
        self.assertEqual(payload["counts"], {"higher": 8, "lower": 10, "zero": 1, "missing": 1})
        self.assertEqual(payload["n_planned"], 20)
        self.assertEqual(payload["n_primary_complete"], 19)
        self.assertEqual(payload["dose_intervals"]["DOSE_M6"]["ci"], (-0.2, -0.1))
        self.assertEqual(payload["dose_intervals"]["DOSE_P6"]["mean"], 0.15)

    def test_svg_shows_all_speakers_reference_and_scope_captions(self):
        summary, rows, planned = self._fixture()
        svg = render_svg(build_payload(summary, rows))
        for speaker_id in planned:
            self.assertIn(speaker_id, svg)
        self.assertIn("exact ΔC=0 1/19", svg)
        self.assertIn("missing 1/20 planned", svg)
        self.assertIn("exact zero reference · no CI", svg)
        self.assertIn("not a fifth contrast", svg)
        self.assertIn("No best dose is selected", svg)
        self.assertIn("FIXED’s measured per-sentence RMS dBFS is reported separately", svg)

    def test_rejects_mismatch_between_completion_flag_and_delta(self):
        summary, rows, _ = self._fixture()
        rows[0]["primary_complete"] = "False"
        with self.assertRaisesRegex(ValueError, "primary_complete and primary_delta_C disagree"):
            build_payload(summary, rows)


if __name__ == "__main__":
    unittest.main()
