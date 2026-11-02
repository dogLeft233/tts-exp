"""Synthetic CPU-only checks for Ditto SPSA plan descriptor/protocol binding.

These fixtures test schema wiring only; they are not model outputs or scores.
"""
from __future__ import annotations

import importlib.util
import copy
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PATH = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/ditto_spsa_stagea_adapter_attempt04.py"
sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("ditto_spsa_adapter04_under_test", ADAPTER_PATH)
assert SPEC is not None and SPEC.loader is not None
ADAPTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ADAPTER
SPEC.loader.exec_module(ADAPTER)


def composed_and_source_plans() -> tuple[dict, dict]:
    # The runtime descriptor and scientific/scoring protocol are deliberately
    # different objects in the actual producer schema.
    descriptor = {
        "renderer_path": "/models/ditto/renderer.py",
        "weights_root": "/models/ditto/weights",
        "strict_bootstrap_sha256": "a" * 64,
    }
    protocol_generator = {
        "model_id": "wav2lip_gan_bac9a81e",
        "score_audio_preprocessing_sha256": "b" * 64,
        "source_video_role": "shared static portrait reference",
    }
    execution = {"runtime": "pinned Ditto execution record"}
    scientific_input_pins_sha256 = "c" * 64
    source_plan = {
        "models": [{"model_id": ADAPTER.DITTO_MODEL_ID, "generator": descriptor}],
        "protocol": {"generator": protocol_generator},
        "execution": execution,
        "scientific_input_pins_sha256": scientific_input_pins_sha256,
    }
    family_plan = {
        "models": [{"model_id": ADAPTER.DITTO_MODEL_ID, "generator": descriptor}],
        "protocol": {"generators_by_model": {ADAPTER.DITTO_MODEL_ID: protocol_generator}},
        "execution": {"source_model_execution": {ADAPTER.DITTO_MODEL_ID: execution}},
        "scientific_input_pins_sha256": scientific_input_pins_sha256,
    }
    return family_plan, source_plan


class DittoDescriptorProtocolBindingTests(unittest.TestCase):
    def test_runtime_descriptor_and_source_protocol_are_bound_separately(self) -> None:
        family_plan, source_plan = composed_and_source_plans()
        self.assertNotEqual(family_plan["models"][0]["generator"], source_plan["protocol"]["generator"])
        ADAPTER._validate_composed_source_plan(family_plan, source_plan)

    def test_runtime_descriptor_must_match_source_model_descriptor(self) -> None:
        family_plan, source_plan = composed_and_source_plans()
        family_plan["models"][0]["generator"] = dict(source_plan["protocol"]["generator"])
        with self.assertRaises(ADAPTER.DittoSPSAAdapterError):
            ADAPTER._validate_composed_source_plan(family_plan, source_plan)

    def test_scoring_protocol_must_match_source_plan_protocol(self) -> None:
        family_plan, source_plan = composed_and_source_plans()
        family_plan["protocol"]["generators_by_model"][ADAPTER.DITTO_MODEL_ID] = dict(
            family_plan["models"][0]["generator"]
        )
        with self.assertRaises(ADAPTER.DittoSPSAAdapterError):
            ADAPTER._validate_composed_source_plan(family_plan, source_plan)


class SharedApplicationValidatorSourceTests(unittest.TestCase):
    def test_exact_common_validator_source_loads_from_staged_project(self) -> None:
        source = ROOT / ADAPTER.APPLICATION_VALIDATOR_REL
        expected = source.read_bytes()
        self.assertEqual(len(expected), ADAPTER.APPLICATION_VALIDATOR_BYTES)
        import hashlib
        self.assertEqual(hashlib.sha256(expected).hexdigest(), ADAPTER.APPLICATION_VALIDATOR_SHA256)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / ADAPTER.APPLICATION_VALIDATOR_REL
            target.parent.mkdir(parents=True)
            target.write_bytes(expected)
            try:
                module, pin = ADAPTER._load_application_validator(root)
                self.assertEqual(pin["bytes"], len(expected))
                self.assertEqual(pin["sha256"], ADAPTER.APPLICATION_VALIDATOR_SHA256)
                self.assertTrue(callable(module.validate_snapshot_point))
                self.assertTrue(callable(module.validate_application_lineage))
                target.write_bytes(expected + b"\n")
                with self.assertRaises(ADAPTER.DittoSPSAAdapterError):
                    ADAPTER._load_application_validator(root)
            finally:
                sys.modules.pop(ADAPTER.APPLICATION_VALIDATOR_MODULE, None)


class SharedApplicationValidatorAdapterWiringTests(unittest.TestCase):
    class Core:
        def _reference(self, natural_ref):
            class Samples:
                size = natural_ref["sample_count"]

            return (Path(natural_ref["path"]).resolve(strict=True), Samples(), {
                "file_sha256": natural_ref["file_sha256"],
                "pcm_sha256": natural_ref["pcm_sha256"],
            })

    class Validator:
        def __init__(self, config, config_sha):
            self.config = config
            self.config_sha = config_sha
            self.point_calls = []
            self.lineage_calls = []

        def validate_snapshot_point(self, **kwargs):
            self.point_calls.append(kwargs)
            return {"config": self.config, "config_sha256": self.config_sha, "snapshot": {}}

        def validate_application_lineage(self, **kwargs):
            self.lineage_calls.append(kwargs)
            row = kwargs["candidate_row"]
            return {"generator_condition_input": row["generator_condition_input"],
                    "sample_count": row["natural_pcm"]["sample_count"]}

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.natural_root = self.root / "natural"
        self.natural_root.mkdir()
        self.samples = [f"sample-{index:02d}" for index in range(8)]
        self.config = {"method": "fit40_spsa_proposal", "sample_ids": self.samples}
        self.plan = {"execution": {"spsa_execution_scope": {}}}
        self.method = {"method_id": "proposal-method"}
        self.core = self.Core()
        self.validator = self.Validator(self.config, "c" * 64)
        self.adapter = object.__new__(ADAPTER.DittoSPSAStageAAdapter)
        self.adapter.plan = self.plan
        self.adapter.identity_rows = {}
        self.adapter.core = self.core
        self.adapter.application_validator = self.validator
        self.adapter.snapshot_project_root = self.root
        self.rows = []
        for sample_id in self.samples:
            source = self.natural_root / f"{sample_id}.wav"
            source.write_bytes(b"fixture source only")
            natural = {"path": str(source), "file_sha256": "a" * 64, "pcm_sha256": "b" * 64,
                       "sample_count": 62639, "sample_rate_hz": 16000, "channels": 1}
            parent = {"row_id": f"parent-{sample_id}", "split": "fit", "sample_id": sample_id,
                      "speaker_id": f"speaker-{sample_id}", "model_id": ADAPTER.DITTO_MODEL_ID,
                      "face_id": "portrait", "seed": 20261003,
                      "natural_pcm": natural, "score_audio": natural,
                      "ditto_native_frame_binding": {"sample_count": 62639, "sample_rate_hz": 16000,
                          "fps_num": 25, "fps_den": 1, "expected_frame_count": 98}}
            candidate = copy.deepcopy(parent)
            candidate.update({
                "row_id": f"candidate-{sample_id}",
                "method_id": self.method["method_id"],
                "method_config_sha256": "c" * 64,
                "spsa_binding": {"method_id": self.method["method_id"]},
                "spsa_point_binding_sha256": "d" * 64,
                "generator_condition_input": {"path": f"/generated/{sample_id}.wav"},
                "ditto_native_frame_binding": {"sample_count": 62639, "sample_rate_hz": 16000,
                    "fps_num": 25, "fps_den": 1, "expected_frame_count": 98},
            })
            self.adapter.identity_rows[ADAPTER._identity_key(parent)] = parent
            self.rows.append(candidate)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_shared_point_and_lineage_validation_precede_ditto_hooks(self) -> None:
        parents, input_refs, config, config_sha = self.adapter._validate_rows(self.rows, self.method)
        self.assertEqual(len(self.validator.point_calls), 1)
        self.assertEqual(self.validator.point_calls[0]["project_root"], self.root)
        self.assertEqual(len(self.validator.lineage_calls), 8)
        self.assertTrue(all(call["snapshot_point"]["config"] is self.config
                            for call in self.validator.lineage_calls))
        self.assertEqual(len(parents), 8)
        self.assertEqual(len(input_refs), 8)
        self.assertEqual(config, self.config)
        self.assertEqual(config_sha, "c" * 64)

    def test_native_clock_gate_rejects_wrong_ceil_frame_count(self) -> None:
        self.rows[0]["ditto_native_frame_binding"]["expected_frame_count"] = 97
        with self.assertRaises(ADAPTER.DittoSPSAAdapterError):
            self.adapter._validate_rows(self.rows, self.method)

if __name__ == "__main__":
    unittest.main()
