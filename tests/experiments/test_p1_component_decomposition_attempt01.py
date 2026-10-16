from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.experiments.p1_gain_dose_response_20261016.render_p1_component_decomposition_attempt01 import (
    BOOTSTRAP_DRAWS,
    CURVE_ORDER,
    METRICS,
    SEED,
    compute_decomposition,
    read_curve_summary,
    render_svg,
)


class P1ComponentDecompositionTests(unittest.TestCase):
    def _fixture(self):
        planned = [f"S{i:04d}" for i in range(102, 112)] + [f"S{i:04d}" for i in range(144, 154)]
        condition_rows = []
        primary_rows = []
        c_deltas = []
        dose_c_deltas = {dose_id: [] for dose_id in ("DOSE_M6", "DOSE_M3", "DOSE_P3", "DOSE_P6")}
        for i, speaker_id in enumerate(planned):
            base_b = 0.8 + i * 0.013
            base_dmin = 0.15 + (i % 5) * 0.007
            base_d0 = 0.24 + (i % 7) * 0.008
            base_anchor = 0.18 + (i % 3) * 0.004
            speaker_rows = {}
            for condition_id in ("RAW_R1", "RAW_R2", "FIXED_R1", "FIXED_R2"):
                is_fixed = condition_id.startswith("FIXED")
                repeat = 1 if condition_id.endswith("R2") else 0
                b = base_b + repeat * 0.002 + (0.10 + i * 0.001 if is_fixed else 0.0)
                dmin = base_dmin + repeat * 0.001 + (0.03 + (i % 4) * 0.001 if is_fixed else 0.0)
                d0 = base_d0 + repeat * 0.003 + (0.08 + i * 0.0005 if is_fixed else 0.0)
                anchor_d = base_anchor + repeat * 0.001 + (0.02 + i * 0.0007 if is_fixed else 0.0)
                metrics = {
                    "B": b,
                    "Dmin": dmin,
                    "D0": d0,
                    "anchor_D": anchor_d,
                    "C": b - dmin,
                    "anchor_C": b - anchor_d,
                    "search_gain": d0 - dmin,
                }
                speaker_rows[condition_id] = metrics
                condition_rows.append({"speaker_id": speaker_id, "condition_id": condition_id, "status": "PASS", **{k: str(v) for k, v in metrics.items()}})
            for dose_idx, (dose_id, dose_delta_b, dose_delta_min) in enumerate((
                ("DOSE_M6", -0.06, 0.012),
                ("DOSE_M3", -0.03, 0.008),
                ("DOSE_P3", 0.03, -0.004),
                ("DOSE_P6", 0.06, -0.009),
            )):
                dose_metrics = {
                    "B": speaker_rows["RAW_R1"]["B"] + dose_delta_b + i * 0.0004,
                    "Dmin": speaker_rows["RAW_R1"]["Dmin"] + dose_delta_min + (i % 3) * 0.0002,
                    "D0": speaker_rows["RAW_R1"]["D0"] + dose_delta_b * 0.7 + i * 0.0003,
                    "anchor_D": speaker_rows["RAW_R1"]["anchor_D"] + dose_delta_min * 0.5 + i * 0.0001,
                }
                dose_metrics["C"] = dose_metrics["B"] - dose_metrics["Dmin"]
                dose_metrics["anchor_C"] = dose_metrics["B"] - dose_metrics["anchor_D"]
                dose_metrics["search_gain"] = dose_metrics["D0"] - dose_metrics["Dmin"]
                condition_rows.append({"speaker_id": speaker_id, "condition_id": dose_id, "status": "PASS", **{k: str(v) for k, v in dose_metrics.items()}})
                dose_c_deltas[dose_id].append(dose_metrics["C"] - speaker_rows["RAW_R1"]["C"])
            deltas = {}
            for metric in METRICS:
                raw = (speaker_rows["RAW_R1"][metric] + speaker_rows["RAW_R2"][metric]) / 2
                fixed = (speaker_rows["FIXED_R1"][metric] + speaker_rows["FIXED_R2"][metric]) / 2
                deltas[metric] = fixed - raw
            c_deltas.append(deltas["C"])
            primary_row = {"speaker_id": speaker_id, "primary_complete": "True", "family_complete": "True", "primary_delta_C": str(deltas["C"])}
            for dose_id in dose_c_deltas:
                primary_row[f"{dose_id}_dose_delta_C"] = str(dose_c_deltas[dose_id][-1])
            primary_rows.append(primary_row)

        rng = np.random.Generator(np.random.PCG64(SEED))
        indices = rng.integers(0, len(planned), size=(BOOTSTRAP_DRAWS, len(planned)), endpoint=False)
        boot_means = np.asarray(c_deltas, dtype=np.float64)[indices].mean(axis=1)
        ci = [float(x) for x in np.quantile(boot_means, [0.025, 0.975], method="linear")]
        summary = {
            "planned_speakers": planned,
            "n_planned_speakers": 20,
            "protocol_sha256": "a" * 64,
            "input_sha256": "b" * 64,
            "primary_fixed_minus_raw": {
                "status": "COMPUTABLE",
                "n_planned": 20,
                "n_complete": 20,
                "bootstrap_seed": SEED,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "mean": float(np.mean(c_deltas)),
                "ci95": ci,
            },
            "n_family_complete": 20,
            "dose_family_vs_raw_r1": {
                "status": "COMPUTABLE",
                "n_planned": 20,
                "n_complete": 20,
                "contrasts": {
                    dose_id: {
                        "n": 20,
                        "mean": float(np.mean(values)),
                        "simultaneous_ci95": [float(np.mean(values) - 0.025), float(np.mean(values) + 0.025)],
                    }
                    for dose_id, values in dose_c_deltas.items()
                },
            },
            "curve_groups": list(CURVE_ORDER),
        }
        return summary, condition_rows, primary_rows

    def _curve_csv(self, path: Path):
        fields = ("curve_id", "lag", "n_speakers", "n_planned", "mean_distance", "ci95_low_pointwise", "ci95_high_pointwise", "interval")
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for group_idx, group in enumerate(CURVE_ORDER):
                for lag in range(-15, 16):
                    mean = 0.3 + group_idx * 0.01 + abs(lag) * 0.002
                    writer.writerow({
                        "curve_id": group,
                        "lag": lag,
                        "n_speakers": 20,
                        "n_planned": 20,
                        "mean_distance": mean,
                        "ci95_low_pointwise": mean - 0.02,
                        "ci95_high_pointwise": mean + 0.02,
                        "interval": "descriptive speaker-bootstrap pointwise 95%; not simultaneous",
                    })

    def test_component_effects_use_repeat_means_and_reuse_frozen_c_interval(self):
        summary, conditions, primary = self._fixture()
        result = compute_decomposition(summary, conditions, primary)
        first = result["speaker_components"][0]
        self.assertAlmostEqual(first["delta_C"], first["delta_B"] - first["delta_Dmin"], places=10)
        self.assertAlmostEqual(first["delta_anchor_C"], first["delta_B"] - first["delta_anchor_D"], places=10)
        self.assertAlmostEqual(first["delta_search_gain"], first["delta_D0"] - first["delta_Dmin"], places=10)
        self.assertEqual(result["component_summary"]["C"]["ci95"], summary["primary_fixed_minus_raw"]["ci95"])
        self.assertEqual(result["component_summary"]["C"]["ci_source"], "analysis_summary.json:primary_fixed_minus_raw.ci95")
        for dose_id in ("DOSE_M6", "DOSE_M3", "DOSE_P3", "DOSE_P6"):
            self.assertEqual(result["dose_component_summary"][dose_id]["C"]["ci95"], summary["dose_family_vs_raw_r1"]["contrasts"][dose_id]["simultaneous_ci95"])
            self.assertEqual(result["dose_component_summary"][dose_id]["C"]["ci_type"], "simultaneous_max_t_family")
            self.assertEqual(result["dose_component_summary"][dose_id]["B"]["ci_type"], "pointwise_descriptive")
        self.assertEqual(result["bootstrap"]["seed"], 20261016)
        self.assertEqual(result["bootstrap"]["draws"], 20_000)
        for metric in METRICS[1:]:
            self.assertIsNotNone(result["component_summary"][metric]["ci95"])
            self.assertIn("descriptive point interval only", result["component_summary"][metric]["interval"])

    def test_curve_summary_reuses_all_31_groups_and_caption_limits(self):
        summary, conditions, primary = self._fixture()
        decomposition = compute_decomposition(summary, conditions, primary)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "distance_curve_summary.csv"
            self._curve_csv(csv_path)
            curves = read_curve_summary(csv_path, expected_groups=list(CURVE_ORDER), n_planned=20)
        self.assertEqual(list(curves), list(CURVE_ORDER))
        self.assertEqual(len(curves["RAW_R1"]), 31)
        self.assertAlmostEqual(curves["RAW_R1"][0]["mean_distance"], 0.33)
        svg = render_svg(decomposition, curves)
        self.assertEqual(svg.count("<polyline "), 7)
        self.assertEqual(svg.count("<polygon "), 7)
        self.assertIn("distance_curve_summary.csv", svg)
        self.assertIn("not mediator shares", svg)
        self.assertIn("A rise in B alone is not evidence of gaming or lower audio quality", svg)

    def test_rejects_missing_component_on_pass_condition_row(self):
        summary, conditions, primary = self._fixture()
        conditions[0]["B"] = ""
        with self.assertRaisesRegex(ValueError, "B is blank on a PASS row"):
            compute_decomposition(summary, conditions, primary)


if __name__ == "__main__":
    unittest.main()
