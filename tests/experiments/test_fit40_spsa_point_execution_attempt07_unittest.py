from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SOURCE = (Path(__file__).resolve().parents[2] / "scripts" / "experiments"
          / "cross_tfg_audio_head_20261003" / "fit40_spsa_point_execution_attempt07.py")
SPEC = importlib.util.spec_from_file_location("point_execution_attempt07_under_test", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load point execution attempt07")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EmittedPinIdentityTests(unittest.TestCase):
    def test_strips_parsed_packet_from_canonical_candidate_pin(self) -> None:
        pin = {
            "path": "/tmp/candidate.json", "bytes": 12,
            "sha256": "a" * 64, "self_sha256": "b" * 64,
            "packet": {"schema": "fit40_spsa_shared_candidate_rows_v1"},
        }
        self.assertEqual(
            MODULE._emitted_pin_identity(pin, "candidate"),
            {key: pin[key] for key in ("path", "bytes", "sha256", "self_sha256")},
        )

    def test_requires_verified_packet_and_complete_file_pin(self) -> None:
        with self.assertRaises(MODULE.PointExecutionError):
            MODULE._emitted_pin_identity({"path": "/tmp/candidate.json"}, "candidate")
        with self.assertRaises(MODULE.PointExecutionError):
            MODULE._emitted_pin_identity({
                "path": "/tmp/candidate.json", "bytes": 12,
                "sha256": "a" * 64, "self_sha256": "b" * 64,
            }, "candidate")


if __name__ == "__main__":
    unittest.main()
