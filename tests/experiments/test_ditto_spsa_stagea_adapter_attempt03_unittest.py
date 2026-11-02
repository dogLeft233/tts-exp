"""Synthetic CPU-only checks for Ditto SPSA plan descriptor/protocol binding.

These fixtures test schema wiring only; they are not model outputs or scores.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PATH = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/ditto_spsa_stagea_adapter_attempt03.py"
sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("ditto_spsa_adapter03_under_test", ADAPTER_PATH)
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


if __name__ == "__main__":
    unittest.main()
