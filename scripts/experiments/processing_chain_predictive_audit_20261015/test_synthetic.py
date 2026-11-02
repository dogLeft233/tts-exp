from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from . import modeling
from .cli import _resolve_bound_inputs, _select_smoke_rows, _sha256
from .plots import plot_analysis
from .synthetic_fixture import make_tables, make_test_configuration


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "runs/processing_chain_predictive_audit_20261015_attempt01/schema_design_v1.json"
ADDENDUM_PATH = REPO_ROOT / "runs/processing_chain_predictive_audit_20261015_attempt01/schema_addendum_v1.json"


class PredictiveAuditSyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        cls.addendum = json.loads(ADDENDUM_PATH.read_text(encoding="utf-8"))
        cls.assets, cls.features, cls.scores = make_tables(cls.schema, cls.addendum)
        cls.test_schema, cls.test_addendum, cls.protocol = make_test_configuration(
            cls.schema, cls.addendum
        )

    def test_full_synthetic_pipeline_and_grouped_lodo(self) -> None:
        results, l1_oof, l2_oof = modeling.run_analysis(
            self.test_schema,
            self.test_addendum,
            self.protocol,
            self.assets.copy(),
            self.features.copy(),
            self.scores.copy(),
            allow_pending_for_synthetic=True,
        )
        l1 = results["l1"]["generator"]
        self.assertEqual(l1["status"], "PASS")
        self.assertEqual(len(l1["family_incremental"]), 7)
        self.assertEqual(l1["relative_silence_sensitivity"]["status"], "PASS")
        self.assertIn("missingness_only", l1["models"])
        self.assertEqual(l1["cross_domain_lodo"]["status"], "PASS")
        self.assertEqual(len(l1["cross_domain_lodo"]["domains"]), 2)

        self.assertEqual(len(results["l2"]), 1)
        l2 = next(iter(results["l2"].values()))
        self.assertEqual(l2["status"], "PASS")
        self.assertEqual(len(l2["family_incremental"]), 7)
        self.assertEqual(l2["relative_silence_sensitivity"]["status"], "PASS")
        self.assertIn("missingness_only_plus_context", l2["models"])
        self.assertEqual(l2["cross_domain_lodo"]["status"], "PASS")
        self.assertIn("B", l2["auxiliary_score_diagnostics"])
        self.assertTrue(len(l1_oof) > 0)
        self.assertTrue(len(l2_oof) > 0)
        self.assertTrue(set(l1_oof["source_group"]).issubset(set(l2_oof["source_group"])))
        self.assertIn("silence_leading_s_relative", l1["feature_diagnostics"])

    def test_engineering_smoke_uses_frozen_subset_without_inference(self) -> None:
        selected_pairs = [
            f"fixture_A_{group:02d}_{pair:02d}"
            for group in range(6)
            for pair in range(2)
        ]
        selected_groups = [f"fixture_A_source_{group:02d}" for group in range(6)]
        protocol = {
            "status": "FROZEN",
            "cohort_binding": {"primary": {
                "cohort_id": "fixture_A",
                "source_layer": "generator",
                "score_contract": "fixture_contract",
                "contract_fingerprint": "fixture_contract_fp",
            }},
            "execution": {
                "stage_authorized": "smoke_only",
                "full_analysis_authorized": False,
                "smoke_pair_uids": selected_pairs,
                "smoke_source_groups": selected_groups,
                "smoke_audio_count": 24,
            },
            "feature_variant_mapping": self.test_addendum["feature_variant_mapping"],
        }
        assets, features, scores, selection = _select_smoke_rows(
            protocol, self.assets, self.features, self.scores
        )
        self.assertEqual(selection["assets"], 24)
        self.assertEqual(selection["features"], 24)
        self.assertEqual(selection["scores"], 12)
        results, l1_oof, l2_oof = modeling.run_analysis(
            self.test_schema, self.test_addendum, protocol,
            assets, features, scores, engineering_only=True,
        )
        self.assertEqual(results["status"], "ENGINEERING_ONLY_SMOKE")
        l1 = results["l1"]["generator"]
        self.assertEqual(l1["status"], "PASS_ENGINEERING_ONLY")
        self.assertEqual(len(l1["models"]), 21)
        self.assertEqual(l1["bootstrap"]["status"], "NOT_RUN_ENGINEERING_ONLY")
        self.assertEqual(l1["permutation"]["status"], "NOT_RUN_ENGINEERING_ONLY")
        self.assertTrue(all(
            row["stable_label"] == "NOT_EVALUATED_ENGINEERING_ONLY"
            for row in l1["family_incremental"].values()
        ))
        l2 = next(iter(results["l2"].values()))
        self.assertEqual(l2["status"], "PASS_ENGINEERING_ONLY")
        self.assertEqual(l2["relative_silence_sensitivity"]["status"], "PASS")
        self.assertEqual(l2["bootstrap"]["status"], "NOT_RUN_ENGINEERING_ONLY")
        self.assertEqual(l2["permutation"]["status"], "NOT_RUN_ENGINEERING_ONLY")
        self.assertEqual(len(l1_oof), 24 * 21)
        self.assertGreater(len(l2_oof), 0)

    def test_root_smoke_input_binding_resolves_all_pins(self) -> None:
        with tempfile.TemporaryDirectory(prefix="processing-chain-input-binding-") as temp_dir:
            root = Path(temp_dir)
            paths = {name: root / name for name in ("assets.json", "features.csv", "scores.csv", "extractor.py")}
            for path in paths.values():
                path.write_text(f"fixture:{path.name}\n", encoding="utf-8")
            primary = {
                "asset_inventory_path": "assets.json",
                "asset_inventory_sha256": _sha256(paths["assets.json"]),
                "paired_scores_path": "scores.csv",
                "paired_scores_sha256": _sha256(paths["scores.csv"]),
            }
            protocol = {
                "status": "FROZEN",
                "project_root": str(root),
                "cohort_binding": {"primary": primary},
            }
            protocol_path = root / "protocol.json"
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            features_manifest_path = root / "features_manifest.json"
            features_manifest_path.write_text(json.dumps({
                "features_csv": {"path": "features.csv", "sha256": _sha256(paths["features.csv"])}
            }), encoding="utf-8")
            binding = {
                "status": "BOUND",
                "stage": "smoke",
                "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
                "inputs": {
                    "assets_json": {"path": "assets.json", "sha256": _sha256(paths["assets.json"])},
                    "features_csv": {"path": "features.csv", "sha256": _sha256(paths["features.csv"])},
                    "paired_scores_csv": {"path": "scores.csv", "sha256": _sha256(paths["scores.csv"])},
                },
                "features_manifest": {"path": "features_manifest.json", "sha256": _sha256(features_manifest_path)},
                "sources": {"feature_extractor": {"path": "extractor.py", "sha256": _sha256(paths["extractor.py"])}},
            }
            resolved, pins = _resolve_bound_inputs(
                protocol, protocol_path,
                {"assets_json": str(paths["assets.json"]), "features_csv": str(paths["features.csv"]),
                 "paired_scores_csv": str(paths["scores.csv"])},
                input_binding=binding,
            )
            self.assertEqual(resolved["features_csv"], paths["features.csv"].resolve())
            self.assertEqual(pins["features_manifest"]["sha256"], _sha256(features_manifest_path))
            self.assertEqual(pins["producer_sources"]["feature_extractor"]["sha256"], _sha256(paths["extractor.py"]))

    def test_relative_alternates_are_required_and_retained_for_l2(self) -> None:
        damaged = self.features.drop(columns=["silence_leading_s_relative"])
        with self.assertRaises(modeling.AuditInputError):
            modeling.validate_input_tables(
                self.schema, self.assets, damaged, self.scores,
                relative_fields=self.test_addendum["feature_variant_mapping"]["relative_silence"],
            )
        joined = modeling.join_assets_features(self.schema, self.assets, self.features)
        frame, summary = modeling.build_l2_frame(
            self.schema, self.scores, joined, "fixture_contract", "fixture_contract_fp", "generator",
            relative_fields=self.test_addendum["feature_variant_mapping"]["relative_silence"],
        )
        self.assertGreater(summary["eligible_score_rows"], 0)
        self.assertIn("natural__silence_leading_s_relative", frame.columns)
        self.assertIn("tts__silence_leading_s_relative", frame.columns)

    def test_train_only_scaling_and_pending_protocol_guard(self) -> None:
        transformer = modeling.FoldTransformer(["x"], [])
        train = pd.DataFrame({"x": [0.0, 1.0, np.nan]})
        test = pd.DataFrame({"x": [1_000_000.0]})
        _, x_test = transformer.fit_transform_pair(train, test)
        self.assertAlmostEqual(transformer._means["x"], 0.5)
        expected = (1_000_000.0 - 0.5) / float(np.std([0.0, 1.0, 0.5], ddof=0))
        self.assertAlmostEqual(float(x_test[0, 0]), expected)
        with self.assertRaises(modeling.AuditInputError):
            modeling._validate_protocol_is_final({"status": "FROZEN_SCHEMA_AND_METHODS_COHORT_BINDING_PENDING"})

    def test_plot_outputs_are_descriptive_and_deterministic(self) -> None:
        fake = {
            "l1": {"generator": {
                "bootstrap": {"seven_family_adjusted_coverage": 0.9928571428571429},
                "family_incremental": {"energy": {
                    "point_gain": 0.1,
                    "seven_family_adjusted_interval": {"intervals": {"0.992857142857": [0.01, 0.2]}},
                }},
            }},
            "l2": {},
        }
        with tempfile.TemporaryDirectory(prefix="processing-chain-audit-plot-") as temp_dir:
            out = Path(temp_dir)
            artifacts = plot_analysis(fake, out)
            self.assertEqual(len(artifacts), 1)
            self.assertTrue((out / "l1_family_increments.png").is_file())
            self.assertTrue((out / "index.json").is_file())


if __name__ == "__main__":
    unittest.main()
