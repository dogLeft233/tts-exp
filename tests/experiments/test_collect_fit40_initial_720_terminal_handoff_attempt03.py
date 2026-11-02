from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


MODULE_PATH = (Path(__file__).resolve().parents[2] / "scripts" / "experiments" /
               "cross_tfg_audio_head_20261003" /
               "collect_fit40_initial_720_terminal_handoff_attempt03.py")
SPEC = importlib.util.spec_from_file_location("collect_fit40_handoff_attempt03", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
handoff = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(handoff)


class CollectorEvidenceHelpersTest(unittest.TestCase):
    def test_historical_child_exit_remains_unknown(self) -> None:
        self.assertTrue(handoff._historical_rc_is_unknown({
            "historical_exit_code_available": False,
            "historical_exit_code": None,
        }))
        for invalid in (
            {"historical_exit_code_available": True, "historical_exit_code": 0},
            {"historical_exit_code_available": False, "historical_exit_code": 0},
            {"historical_exit_code_available": False, "historical_exit_code": False},
            {"historical_exit_code_available": 0, "historical_exit_code": None},
        ):
            self.assertFalse(handoff._historical_rc_is_unknown(invalid), invalid)

    def test_block_evidence_must_stay_under_exact_output_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            leaf = root / "block"
            leaf.mkdir()
            expected = leaf / "results.jsonl"
            expected.write_text("{}\n", encoding="utf-8")
            self.assertEqual(handoff._require_inside_output_leaf(expected, leaf, "ledger"), expected)
            outside = root / "outside.jsonl"
            outside.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(handoff.HandoffError):
                handoff._require_inside_output_leaf(outside, leaf, "ledger")
            with self.assertRaises(handoff.HandoffError):
                handoff._require_inside_output_leaf(leaf / ".." / "outside.jsonl", leaf, "ledger")

    def test_block_evidence_rejects_symlinked_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            leaf = root / "block"
            leaf.mkdir()
            target = root / "target.jsonl"
            target.write_text("{}\n", encoding="utf-8")
            link = leaf / "results.jsonl"
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation is unavailable on this host")
            with self.assertRaises(handoff.HandoffError):
                handoff._require_inside_output_leaf(link, leaf, "ledger")

    def test_coordinator_source_and_config_files_are_full_byte_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "queue_coordinator_attempt01.py"
            config_path = root / "queue_coordinator_config_attempt01.json"
            source_raw = b"reviewed supervisor source\n"
            config = {"schema": "fit40_queue_detached_coordinator_config_v1", "nonce": "test-only"}
            config_raw = (json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n").encode()
            source.write_bytes(source_raw)
            config_path.write_bytes(config_raw)
            source_pin = {"path": str(source.resolve()), "bytes": len(source_raw),
                          "sha256": hashlib.sha256(source_raw).hexdigest()}
            config_pin = {"path": str(config_path.resolve()), "bytes": len(config_raw),
                          "sha256": hashlib.sha256(config_raw).hexdigest()}
            launch = {"coordinator_source": source_pin, "config": config_pin}
            with mock.patch.object(handoff, "COORDINATOR_SOURCE_BYTES", len(source_raw)), \
                    mock.patch.object(handoff, "COORDINATOR_SOURCE_SHA256", source_pin["sha256"]), \
                    mock.patch.object(handoff, "COORDINATOR_CONFIG_BYTES", len(config_raw)), \
                    mock.patch.object(handoff, "COORDINATOR_CONFIG_SHA256", config_pin["sha256"]):
                observed_source, observed_config, observed_value = handoff._verify_coordinator_source_config_files(
                    root, launch
                )
                self.assertEqual(observed_source, source_pin)
                self.assertEqual(observed_config, config_pin)
                self.assertEqual(observed_value, config)
                tampered = {**launch, "config": {**config_pin, "sha256": "0" * 64}}
                with self.assertRaises(handoff.HandoffError):
                    handoff._verify_coordinator_source_config_files(root, tampered)

    def test_config_object_must_match_exact_invocation(self) -> None:
        expected = {"entry_argv": ["python", "-B", "entry.py"], "cuda": "disabled"}
        handoff._require_exact_mapping(expected, expected, "config")
        with self.assertRaises(handoff.HandoffError):
            handoff._require_exact_mapping({**expected, "cuda": "0"}, expected, "config")
        with self.assertRaises(handoff.HandoffError):
            handoff._require_exact_mapping(None, expected, "config")

    def test_pinned_json_uses_named_self_field(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.json"
            value = {"schema": "test-plan", "plan_sha256": ""}
            value["plan_sha256"] = handoff.self_hash(value, "plan_sha256")
            raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
            path.write_bytes(raw)
            parsed, pin, _ = handoff._pinned_json(
                str(path), len(raw), hashlib.sha256(raw).hexdigest(),
                value["plan_sha256"], "test plan", self_field="plan_sha256",
            )
            self.assertEqual(parsed, value)
            self.assertEqual(pin["bytes"], len(raw))
            with self.assertRaises(handoff.HandoffError):
                handoff._pinned_json(
                    str(path), len(raw), hashlib.sha256(raw).hexdigest(),
                    value["plan_sha256"], "test plan", self_field="self_sha256",
                )


if __name__ == "__main__":
    unittest.main()
