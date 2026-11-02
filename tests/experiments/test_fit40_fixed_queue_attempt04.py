from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.experiments.cross_tfg_audio_head_20261003 import run_fit40_fixed_queue_attempt04 as queue


class QueueAttempt04GateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resources = {
            "gpu_uuid": "GPU-d12b3309-c299-da83-064d-a3cd619bd560",
            "gpu_name": "Tesla V100-SXM2-16GB",
            "min_gpu_free_mib": 10000,
            "min_effective_ram_available_bytes": 8 * 1024**3,
            "output_mount_target": "/dev/shm",
            "output_reserve_bytes": 5 * 1024**3,
            "cgroup": {
                "version": "v2",
                "identity": {"mount_target": "/sys/fs/cgroup", "mount_root": "/"},
            },
            "capacity_basis": {"status": "TEST_ONLY"},
        }
        self.snapshot = {
            "host": "v100-test-host",
            "gpus": [{
                "index": 0,
                "uuid": self.resources["gpu_uuid"],
                "name": self.resources["gpu_name"],
                "memory_total_mib": 16384,
                "memory_free_mib": 16140,
            }],
            "ram": {
                "available_bytes": 16 * 1024**3,
                "cgroup": {
                    "version": "v2",
                    "identity": {
                        "mount_target": "/sys/fs/cgroup", "mount_root": "/",
                        "membership_path": "/user.slice/session-test.scope",
                        "leaf_path": "/user.slice/session-test.scope",
                    },
                    "finite_limit": False,
                    "available_bytes": None,
                },
            },
            "shm": {"available_bytes": 16 * 1024**3},
            "output_disk": {"target": "/dev/shm", "available_bytes": 16 * 1024**3},
        }
        self.helper_pin = {"path": "/tmp/resource.py", "bytes": 10, "sha256": "a" * 64}

    def test_full_live_probe_passes_and_seals_observations(self) -> None:
        record = queue._resource_gate_record(
            queue.MODEL_ORDER[0], 126, "b" * 64, "c" * 64, self.helper_pin,
            self.resources, self.snapshot, 10 * 1024**3,
        )
        self.assertEqual(record["status"], "PASS")
        self.assertEqual(record["model_id"], queue.MODEL_ORDER[0])
        self.assertEqual(record["block_start_index"], 126)
        self.assertEqual(record["observed"]["gpu"]["memory_free_mib"], 16140)
        self.assertEqual(len(record["evidence_raw_sha256"]), 64)

    def test_resource_probe_rejects_gpu_ram_shm_and_mount_failures(self) -> None:
        changed = dict(self.snapshot)
        changed["gpus"] = [{**self.snapshot["gpus"][0], "uuid": "GPU-other"}]
        changed["ram"] = {**self.snapshot["ram"], "available_bytes": 1}
        changed["shm"] = {"available_bytes": 1}
        changed["output_disk"] = {"target": "/home", "available_bytes": 1}
        record = queue._resource_gate_record(
            queue.MODEL_ORDER[0], 126, "b" * 64, "c" * 64, self.helper_pin,
            self.resources, changed, 10 * 1024**3,
        )
        self.assertEqual(record["status"], "FAILED_RESOURCE_GATE")
        self.assertGreaterEqual(len(record["failure_reasons"]), 4)

    def test_finite_cgroup_uses_current_availability_and_stable_mount(self) -> None:
        changed = {
            **self.snapshot,
            "ram": {
                **self.snapshot["ram"],
                "cgroup": {
                    **self.snapshot["ram"]["cgroup"],
                    "finite_limit": True,
                    "available_bytes": 1024,
                    "identity": {
                        **self.snapshot["ram"]["cgroup"]["identity"],
                        "membership_path": "/user.slice/session-new.scope",
                        "leaf_path": "/user.slice/session-new.scope",
                    },
                },
            },
        }
        record = queue._resource_gate_record(
            queue.MODEL_ORDER[0], 126, "b" * 64, "c" * 64, self.helper_pin,
            self.resources, changed, 10 * 1024**3,
        )
        self.assertEqual(record["status"], "FAILED_RESOURCE_GATE")
        self.assertTrue(any("cgroup memory" in item for item in record["failure_reasons"]))

    def test_absent_leaf_is_model_and_block_bound_and_create_only(self) -> None:
        model_id = queue.MODEL_ORDER[0]
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "blocks"
            base.mkdir()
            with mock.patch.dict(queue.OUTPUT_ROOT_OVERRIDES, {model_id: str(base)}):
                proof = queue.build_absent_leaf_proof(model_id, 126, "d" * 64, base)
                self.assertIsNotNone(proof)
                assert proof is not None
                self.assertEqual(proof["output_leaf"], str(base / "block_021_attempt01"))
                self.assertFalse((base / "block_021_attempt01").exists())
                (base / "block_021_attempt01").mkdir()
                self.assertIsNone(queue.build_absent_leaf_proof(model_id, 126, "d" * 64, base))

    def test_namespace_cannot_be_overridden(self) -> None:
        model_id = queue.MODEL_ORDER[0]
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(queue.QueueError):
                queue.build_absent_leaf_proof(model_id, 126, "d" * 64, Path(temp))


if __name__ == "__main__":
    unittest.main()
