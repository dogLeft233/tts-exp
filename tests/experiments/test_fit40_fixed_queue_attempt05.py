from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.experiments.cross_tfg_audio_head_20261003 import run_fit40_fixed_queue_attempt05 as queue


class QueueAttempt05GateTests(unittest.TestCase):
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

    def test_complete_requires_all_three_full_offsets_and_no_paused_model(self) -> None:
        full = {
            "next_start_by_model": {model: 240 for model in queue.MODEL_ORDER},
            "paused_models": {}, "active_claim": None,
        }
        self.assertEqual(queue.classify_queue_run(full, []), "PASS_QUEUE_COMPLETE")
        paused = {**full, "paused_models": {queue.MODEL_ORDER[0]: "row failure"}}
        self.assertEqual(queue.classify_queue_run(paused, []), "PAUSED_WITH_MODEL_FAILURES")

    def test_failure_and_no_ready_are_never_reported_complete(self) -> None:
        partial = {
            "next_start_by_model": {model: 6 for model in queue.MODEL_ORDER},
            "paused_models": {queue.MODEL_ORDER[0]: "failed"}, "active_claim": None,
        }
        failed_event = [{"event": "BLOCK_PAUSED_FAILURE"}]
        self.assertEqual(queue.classify_queue_run(partial, failed_event), "PAUSED_AFTER_BLOCK_FAILURE")
        no_ready = {**partial, "paused_models": {}}
        self.assertEqual(queue.classify_queue_run(no_ready, []), "NO_READY_BLOCK")

    def _strict_fixture(self, root: Path) -> tuple[object, dict[str, object], Path]:
        model_id = queue.MODEL_ORDER[0]
        output = root / "output"
        (output / "control").mkdir(parents=True)
        natural = {"pcm_sha256": "a" * 64}
        face = {"file_sha256": "b" * 64}
        video = {"file_sha256": None}
        planned_rows = []
        actual_rows = []
        for index, method_id in enumerate(queue.METHODS):
            row_id = f"row-{index:03d}"
            planned_rows.append({
                "row_id": row_id, "model_id": model_id, "method_id": method_id,
                "sample_id": "clip-00", "speaker_id": "speaker-00", "face_id": "face-00",
                "split": "fit", "seed": 20261003, "source_media_kind": "static_portrait",
                "source_video": video, "source_face": face, "natural_pcm": natural,
                "score_audio": {"pcm_sha256": "a" * 64},
                "method_config_sha256": "c" * 64, "checkpoint_sha256": None,
                "audio_frontend": "test-frontend",
            })
            actual_rows.append({
                "row_id": row_id, "status": "PASS", "model_id": model_id,
                "method_id": method_id, "sample_id": "clip-00", "speaker_id": "speaker-00",
                "face_id": "face-00", "split": "fit", "source_media_kind": "static_portrait",
                "source_video_sha256": None, "source_face_sha256": "b" * 64,
                "natural_pcm_sha256": "a" * 64, "score_audio_sha256": "a" * 64,
                "method_config_sha256": "c" * 64, "checkpoint_sha256": None,
                "seed": 20261003, "audio_frontend": "test-frontend",
            })
        plan_path = "/tmp/w2l/project/plan.json"
        rows_path = "/tmp/w2l/project/rows.jsonl"
        plan = {
            "run_id": "run-test", "plan_sha256": "d" * 64,
            "execution": {"w2l_runtime_binding": {"path": "/tmp/w2l/project/control/binding.json"}},
            "models": [{"audio_frontend": "test-frontend"}],
        }
        spec = type("Spec", (), {
            "model_id": model_id,
            "plan": type("Pin", (), {"path": ".codex/plan.json", "sha256": "e" * 64})(),
            "planned_rows": type("Pin", (), {"path": ".codex/rows.jsonl", "sha256": "f" * 64})(),
        })()
        bundle = type("Bundle", (), {"plan": plan, "spec": spec, "rows": tuple(planned_rows)})()
        controller = type("Controller", (), {"bundles": {model_id: bundle}})()
        status = {
            "run_id": "run-test", "plan_path": plan_path, "plan_raw_sha256": "e" * 64,
            "planned_rows_path": rows_path, "planned_rows_sha256": "f" * 64,
            "result_manifest_path": str(output / "result_manifest.json"),
        }
        manifest = {"run_id": "run-test", "plan_sha256": "d" * 64,
                    "plan_raw_sha256": "e" * 64, "planned_rows_sha256": "f" * 64,
                    "manifest_sha256": ""}
        manifest["manifest_sha256"] = queue.controller_module_self_hash(manifest, "manifest_sha256")
        (output / "control" / "run_status.json").write_text(queue.json.dumps(status), encoding="utf-8")
        (output / "result_manifest.json").write_text(queue.json.dumps(manifest), encoding="utf-8")
        (output / "results.jsonl").write_text(
            "".join(queue.json.dumps(row) + "\n" for row in actual_rows), encoding="utf-8"
        )
        return controller, {"model_id": model_id, "block_start_index": 0}, output

    def test_strict_terminal_check_binds_all_planned_row_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            controller, claim, output = self._strict_fixture(Path(temp))
            proof = queue._verify_strict_terminal_bindings(controller, claim, {}, output)
            self.assertTrue(proof["manifest_self_hash_verified"])
            self.assertEqual(len(proof["six_rows_full_identity_verified"]), 6)

    def test_strict_terminal_check_rejects_row_speaker_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            controller, claim, output = self._strict_fixture(Path(temp))
            ledger = output / "results.jsonl"
            row = queue.json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
            row["speaker_id"] = "wrong-speaker"
            ledger.write_text(queue.json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(queue.QueueError):
                queue._verify_strict_terminal_bindings(controller, claim, {}, output)

    def test_strict_terminal_check_rejects_plan_path_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            controller, claim, output = self._strict_fixture(Path(temp))
            status_path = output / "control" / "run_status.json"
            status = queue.json.loads(status_path.read_text(encoding="utf-8"))
            status["planned_rows_path"] = "/tmp/w2l/project/other.jsonl"
            status_path.write_text(queue.json.dumps(status), encoding="utf-8")
            with self.assertRaises(queue.QueueError):
                queue._verify_strict_terminal_bindings(controller, claim, {}, output)


if __name__ == "__main__":
    unittest.main()
