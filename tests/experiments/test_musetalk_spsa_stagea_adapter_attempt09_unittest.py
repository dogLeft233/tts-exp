from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PATH = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/musetalk_spsa_stagea_adapter_attempt09.py"
QC_PATH = ROOT / ".codex/scratch/musetalk_spsa_stagea_adapter_attempt09/actual_pre_render_validation_attempt01/run_output_attempt01/actual_prerender_qc_attempt01.json"
EXPECTED_ADAPTER = (76887, "2301a0817a3f94d5130c221d1141247b502bc14975aa93c89006cc73eedb0b67")
EXPECTED_HELPER = (66046, "989b28c6239c2770bee6799afa85f0b0f0b48ec39bdd584cf68c035c464f1ec4")

def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")

def _self_hash(value, field="self_sha256") -> str:
    return _sha(_canonical({key: item for key, item in value.items() if key != field}))

def _load_adapter():
    spec = importlib.util.spec_from_file_location("musetalk_spsa_stagea_adapter_attempt09_unittest", ADAPTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Muse adapter attempt09")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

class MuseSPSAAdapterAttempt09Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = _load_adapter()

    def test_exact_helper_and_adapter_source_pins(self):
        raw = ADAPTER_PATH.read_bytes()
        self.assertEqual((len(raw), _sha(raw)), EXPECTED_ADAPTER)
        helper = self.adapter.shared_application
        helper_raw = Path(helper.__file__).read_bytes()
        self.assertEqual((len(helper_raw), _sha(helper_raw)), EXPECTED_HELPER)
        self.adapter._verify_shared_helper_pin(helper)
        for name in ("validate_snapshot_point", "validate_shared_application_lineage",
                     "validate_model_projection_lineage", "project_shared_application_to_model_row"):
            self.assertTrue(callable(getattr(helper, name, None)), name)

    def test_native_frame_floor_contract(self):
        for sample_count, expected in ((1, 0), (639, 0), (640, 1), (62639, 97), (64000, 100)):
            self.assertEqual(self.adapter.expected_native_frame_count(sample_count), expected)

    def test_actual_producer_shaped_pre_render_receipt(self):
        if not QC_PATH.is_file():
            self.skipTest("actual CPU pre-render replay receipt has not been produced")
        raw = QC_PATH.read_bytes()
        packet = json.loads(raw.decode("utf-8"))
        self.assertEqual(packet["status"], "PASS_ACTUAL_PROPOSAL8_CENTER40_AND_RESEALED_NEGATIVE_GATES")
        self.assertEqual(packet["self_sha256"], _self_hash(packet))
        self.assertFalse(packet["render_or_score_performed"])
        self.assertFalse(packet["optimizer_created_or_stepped"])
        self.assertFalse(packet["gpu_used"])
        phases = {item["phase"]: item for item in packet["tests"] if item["phase"] in {"proposal", "center"}}
        self.assertEqual(phases["proposal"]["rows"], 8)
        self.assertEqual(phases["center"]["rows"], 40)
        self.assertEqual(phases["proposal"]["renderer_or_score_calls"], [])
        self.assertEqual(phases["center"]["renderer_or_score_calls"], [])
        rejected = {item["case"] for item in packet["tests"] if item.get("phase") == "proposal_negative"}
        self.assertEqual(rejected, {"wrong_theta_binding", "wrong_model_context", "resealed_wrong_theta",
                                    "resealed_wrong_context", "resealed_wrong_model", "resealed_wrong_source_clock"})
        self.assertEqual(len(raw), 4141)
        self.assertEqual(_sha(raw), "a89b7eae955aed44264461433a832ce42fbdf3b96f90193b60b1141f61bf12c2")

if __name__ == "__main__":
    unittest.main()
