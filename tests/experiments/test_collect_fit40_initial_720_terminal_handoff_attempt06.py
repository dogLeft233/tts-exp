from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest import mock

MODULE_PATH = (Path(__file__).resolve().parents[2] / "scripts" / "experiments" /
               "cross_tfg_audio_head_20261003" /
               "collect_fit40_initial_720_terminal_handoff_attempt06.py")
SPEC = importlib.util.spec_from_file_location("collect_fit40_handoff_attempt06", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
handoff = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(handoff)


class LivePassEventProofTest(unittest.TestCase):
    def setUp(self) -> None:
        self.claim = {"claim_nonce": "diagnostic-only"}
        self.process = {"path": "/queue/receipt.json", "bytes": 900, "sha256": "a" * 64}
        self.output = {
            "run_status": {"path": "/output/control/run_status.json", "bytes": 1000, "sha256": "b" * 64},
            "result_manifest": {"path": "/output/result_manifest.json", "bytes": 2000, "sha256": "c" * 64},
            "results_jsonl": {"path": "/output/results.jsonl", "bytes": 3000, "sha256": "d" * 64},
            "result_rows": 6,
            "status_counts": {"FAILED": 0, "INCOMPLETE": 0, "PASS": 6},
        }
        self.strict = {"manifest_self_hash_verified": True, "rows": [{"status": "PASS"}] * 6}
        self.controller = mock.Mock()
        self.controller._verify_pass_outputs.return_value = dict(self.output)
        self.queue_runner = mock.Mock()
        self.queue_runner._verify_strict_terminal_bindings.return_value = self.strict

    def test_reconstruction_matches_actual_finish_claim_proof_shape(self) -> None:
        # finish_claim creates process_receipt first, then merges the output
        # proof and the strict per-row proof. Attempt05 omitted that first pin.
        expected = {"process_receipt": self.process, **self.output,
                    "strict_terminal_bindings": self.strict}
        observed = handoff._verify_live_pass_event_output_pins(
            self.controller, self.queue_runner, self.claim, {"exit_code": 0},
            Path("/output"), self.process, expected,
        )
        self.assertEqual(observed, expected)
        self.controller._verify_pass_outputs.assert_called_once_with(
            self.claim, {"exit_code": 0}, Path("/output")
        )
        self.queue_runner._verify_strict_terminal_bindings.assert_called_once_with(
            self.controller, self.claim, {"exit_code": 0}, Path("/output")
        )

    def test_event_must_bind_the_exact_process_receipt_pin(self) -> None:
        expected = {"process_receipt": self.process, **self.output,
                    "strict_terminal_bindings": self.strict}
        missing = dict(expected)
        missing.pop("process_receipt")
        changed = {**expected, "process_receipt": {**self.process, "sha256": "e" * 64}}
        for event in (missing, changed):
            with self.subTest(event=event):
                with self.assertRaises(handoff.HandoffError):
                    handoff._verify_live_pass_event_output_pins(
                        self.controller, self.queue_runner, self.claim, {"exit_code": 0},
                        Path("/output"), self.process, event,
                    )


if __name__ == "__main__":
    unittest.main()