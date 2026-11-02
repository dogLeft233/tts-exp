from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ".codex/scratch/fit40_queue_progress_index_attempt07/inspect_fit40_terminal_prefixes_attempt07.py"
SPEC = importlib.util.spec_from_file_location("fit40_terminal_prefixes_attempt07", SOURCE)
assert SPEC is not None and SPEC.loader is not None
INSPECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSPECTOR)
BUILDER_SOURCE = ROOT / ".codex/scratch/fit40_queue_progress_index_attempt08/build_fit40_queue_progress_index_attempt08.py"
BUILDER_SPEC = importlib.util.spec_from_file_location("fit40_queue_index_builder_attempt08", BUILDER_SOURCE)
assert BUILDER_SPEC is not None and BUILDER_SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(BUILDER_SPEC)
BUILDER_SPEC.loader.exec_module(BUILDER)


class LaunchReceiptProducerSchemaTests(unittest.TestCase):
    def test_builder_emits_frozen_controller_index_schema(self) -> None:
        self.assertEqual(BUILDER.INDEX_SCHEMA, "fit40_historical_terminal_prefix_index_v1")

    def test_w2l22_receipt_uses_top_level_remote_file_pins(self) -> None:
        pins = {
            label: dict(value)
            for label, value in BUILDER.EXPECTED_W2L22_SOURCE_FILE_PINS.items()
        }
        outputs = {
            "run_status": {"path": "/out/control/run_status.json", "bytes": 990, "sha256": "a" * 64},
            "result_manifest": {"path": "/out/result_manifest.json", "bytes": 2650, "sha256": "b" * 64},
            "results_jsonl": {"path": "/out/results.jsonl", "bytes": 310393, "sha256": "c" * 64},
        }
        pins.update(outputs)
        doc = {
            "remote_file_pins": pins,
            "remote_readback": {
                "run_status_bytes": 990, "run_status_raw_sha256": "a" * 64,
                "result_manifest_bytes": 2650, "result_manifest_raw_sha256": "b" * 64,
                "results_jsonl_bytes": 310393, "results_jsonl_raw_sha256": "c" * 64,
            },
        }
        self.assertEqual(BUILDER.validate_w2l22_remote_pins(doc, outputs), pins)

    def test_w2l22_nested_remote_file_pins_are_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            BUILDER.validate_w2l22_remote_pins(
                {"remote_readback": {"remote_file_pins": {}}}, {})

    def test_muse_uses_binding_and_intent_nested_cpu_probe(self) -> None:
        spec = INSPECTOR.MODELS["musetalk_fixed_portrait_attempt16"]
        self.assertEqual(INSPECTOR._launch_binding_ref_key(spec), "binding")
        launch = {"cpu_preflight_receipt": {"wrong": True}}
        intent = {"cpu_probe_receipt": {"path": "/muse/probe.json", "self_sha256": "a" * 64}}
        ref, expected_self = INSPECTOR._cpu_preflight_reference(
            "musetalk_fixed_portrait_attempt16", launch, intent)
        self.assertEqual(ref, intent["cpu_probe_receipt"])
        self.assertEqual(expected_self, "a" * 64)

    def test_ditto_uses_runtime_binding_and_top_level_preflight(self) -> None:
        spec = INSPECTOR.MODELS["ditto_hubert_fixed_portrait_attempt18"]
        self.assertEqual(INSPECTOR._launch_binding_ref_key(spec), "runtime_binding")
        ref = {"path": "/ditto/preflight.json", "bytes": 10, "sha256": "b" * 64}
        launch = {"cpu_preflight_receipt": ref,
                  "cpu_preflight_receipt_self_sha256": "c" * 64}
        observed, expected_self = INSPECTOR._cpu_preflight_reference(
            "ditto_hubert_fixed_portrait_attempt18", launch, None)
        self.assertEqual(observed, ref)
        self.assertEqual(expected_self, "c" * 64)

    def test_unknown_binding_reference_label_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            INSPECTOR._launch_binding_ref_key({"binding_ref_key": "model_binding_guess"})


class ProcessIdentityObservationTests(unittest.TestCase):
    @staticmethod
    def write_stat(root: Path, pid: int, ticks: int | None) -> None:
        proc = root / str(pid)
        proc.mkdir(parents=True, exist_ok=True)
        if ticks is None:
            (proc / "stat").write_text("not a proc stat record", encoding="ascii")
            return
        fields = ["S"] + ["0"] * 18 + [str(ticks)]
        (proc / "stat").write_text(f"{pid} (python child) " + " ".join(fields), encoding="ascii")

    def test_absent_identity_is_not_converted_into_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            value = INSPECTOR.observe_process_identity(123, 456, Path(temp))
        self.assertFalse(value["present"])
        self.assertFalse(value["alive_with_same_identity"])
        self.assertNotIn("exit_code", value)

    def test_live_and_reused_pid_are_distinguished(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_stat(root, 123, 456)
            live = INSPECTOR.observe_process_identity(123, 456, root)
            self.write_stat(root, 124, 789)
            reused = INSPECTOR.observe_process_identity(124, 456, root)
        self.assertTrue(live["alive_with_same_identity"])
        self.assertFalse(live["pid_reused"])
        self.assertTrue(reused["pid_reused"])
        self.assertFalse(reused["alive_with_same_identity"])

    def test_malformed_present_process_identity_is_unreadable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_stat(root, 123, None)
            value = INSPECTOR.observe_process_identity(123, 456, root)
        self.assertTrue(value["present"])
        self.assertTrue(value["identity_unreadable"])


if __name__ == "__main__":
    unittest.main()
