"""CPU-only end-to-end and fail-closed checks for snapshot application."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import importlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PLAN_PATH = (REPO_ROOT / ".codex/scratch/v100_musetalk_fit40_240_attempt01"
             / "remote_readback_attempt02/project/musetalk_fit40_240_plan.json")
ROWS_PATH = PLAN_PATH.with_name("planned_rows.jsonl")
ORIGINAL_APPLICATION_RECEIPT = (
    REPO_ROOT / ".codex/scratch/fit40_snapshot_audio_application_attempt04"
    / "application_output/application_receipt.json"
)
PAYLOAD_ROOT = (REPO_ROOT / ".codex/scratch/v100_fit_stagea_formal_w2l_fit_slice_attempt08"
                / "project_payload_attempt05")
MODULE_PATH = REPO_ROOT / "scripts/audio_enhancement/fit40_snapshot_spsa_application_attempt05.py"
SPEC = importlib.util.spec_from_file_location("snapshot_spsa_application_attempt05_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
BUILDER = importlib.import_module(
    "scripts.experiments.cross_tfg_audio_head_20261003.prepare_fit40_spsa_stagea_packets_attempt04"
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _with_self_hash(packet: dict[str, object]) -> dict[str, object]:
    packet.pop("self_sha256", None)
    packet["self_sha256"] = _sha(_canonical(packet))
    return packet


class SnapshotApplicationRoundTripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import numpy as np
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        cls.np = np
        cls.temp = tempfile.TemporaryDirectory(prefix="fit40_snapshot_apply_attempt05_")
        cls.root = Path(cls.temp.name).resolve()
        cls.plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        cls.snapshot_key = "snapshot_step_120"
        cls.index_key = "snapshot_index"
        cls.sample_id = "BAC009S0052W0121"
        cls.natural_key = f"natural_pcm_{cls.sample_id}"
        cls.seed_output_key = f"snapshot_pcm_120_{cls.sample_id}"
        cls._stage_pin(cls.snapshot_key)
        cls._stage_pin(cls.index_key)
        cls._stage_pin(cls.natural_key)
        cls._stage_pin(cls.seed_output_key)
        for rel in (
            "scripts/audio_enhancement/__init__.py",
            "scripts/audio_enhancement/audio_io.py",
            "scripts/audio_enhancement/baselines.py",
            "scripts/audio_enhancement/pipeline.py",
            "scripts/audio_enhancement/overdub.py",
            "scripts/audio_enhancement/shared_recipient_transition_head_v3.py",
            "scripts/experiments/cross_tfg_audio_head_20261003/fit40_training_snapshots_attempt02.py",
        ):
            src = REPO_ROOT / rel
            dest = cls.root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copyfile(src, dest)
        cls.parent_row = next(
            json.loads(line) for line in ROWS_PATH.read_text(encoding="utf-8").splitlines() if line
            if (lambda row: row.get("sample_id") == cls.sample_id
                and row.get("method_id") == "identity"
                and row.get("model_id") == "musetalk_fixed_portrait_attempt16"
                and row.get("split") == "fit")(json.loads(line))
        )
        cls.parent_row = copy.deepcopy(cls.parent_row)
        natural_rel = cls.plan["scientific_input_pins"][cls.natural_key]["path"]
        cls.parent_row["natural_pcm"]["path"] = str(cls.root / natural_rel)
        cls.parent_row["score_audio"] = copy.deepcopy(cls.parent_row["natural_pcm"])

        snapshot = json.loads((cls.root / cls.plan["scientific_input_pins"][cls.snapshot_key]["path"])
                              .read_text(encoding="utf-8"))
        theta = [float(value) for row in snapshot["parameter"]["values"] for value in row]
        source_theta = list(theta)
        artifact_sha = snapshot["snapshot_sha256"]
        point_id = "fit40_spsa_DIAGNOSTIC_SEED_REPLAY_ONLY_attempt05"
        point_identity = {
            "purpose": "test_only_seed_replay_no_spsa_initialization",
            "point_id": point_id,
            "seed_artifact": artifact_sha,
            "source_step": 120,
            "sample_id": cls.sample_id,
        }
        config = {
            "method": "fit40_spsa_proposal",
            "schema": "fit40_spsa_proposal_transform_v1",
            "diagnostic_only": True,
            "diagnostic_purpose": "exercise_actual_snapshot_application_roundtrip_only",
            "proposal_id": "diagnostic_seed_replay_only_no_optimizer",
            "pair_id": "diagnostic_only_not_an_spsa_pair",
            "iteration": 0,
            "side": "plus",
            "method_id": point_id,
            "theta": theta,
            "theta_sha256": _sha(_canonical(theta)),
            "artifact_sha256": artifact_sha,
            "source_snapshot_artifact_sha256": artifact_sha,
            "binding_sha256": _sha(_canonical(point_identity)),
            "input_pins_sha256": cls.plan["scientific_input_pins_sha256"],
            "batch_id": "diagnostic_test_only_single_sample",
            "sample_ids": [cls.sample_id],
            "direction": [0] * 9,
            "c_k": 1.0,
            "source_seed_method_id": snapshot["method_id"],
            "source_snapshot_step": 120,
            "source_theta": source_theta,
            "source_theta_sha256": _sha(_canonical(source_theta)),
            "source_snapshot_record": dict(cls.plan["scientific_input_pins"][cls.snapshot_key]),
            "source_snapshot_index": dict(cls.plan["scientific_input_pins"][cls.index_key]),
            "source_snapshot_record_raw_sha256": cls.plan["scientific_input_pins"][cls.snapshot_key]["sha256"],
            "source_snapshot_index_raw_sha256": cls.plan["scientific_input_pins"][cls.index_key]["sha256"],
            "head_family": snapshot["training_family"],
        }
        method = {
            "method_id": point_id,
            "kind": "fit40_spsa_proposal",
            "config": config,
            "config_sha256": _sha(_canonical(config)),
            "checkpoint_sha256": None,
        }
        # Build the diagnostic-only, one-sample derived plan with the actual
        # packet-builder function. Its roster contains only this point method;
        # the seed snapshot is authenticated by its pinned record/index/theta.
        base_rows = {}
        for line in ROWS_PATH.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            base_rows[(row["sample_id"], row["model_id"], row["method_id"])] = row
        base_raw = PLAN_PATH.read_bytes()
        cls.plan = BUILDER._execution_plan(
            cls.plan,
            {"path": str(PLAN_PATH), "bytes": len(base_raw), "sha256": _sha(base_raw)},
            base_rows,
            method,
            [cls.sample_id],
            {"run_id": "fit40_spsa_DIAGNOSTIC_SEED_REPLAY_ONLY_attempt05"},
        )
        if ([item.get("method_id") for item in cls.plan["methods"]] != [point_id]
                or any(item.get("method_id") == snapshot["method_id"] for item in cls.plan["methods"])):
            raise AssertionError("derived diagnostic plan unexpectedly retained its legacy seed method")
        cls.method = method
        cls.snapshot_point = MODULE.validate_snapshot_point(
            project_root=cls.root, plan=cls.plan, method=cls.method, parent_row=cls.parent_row
        )
        output_dir = cls.root / "diagnostic_outputs"
        output_dir.mkdir()
        cls.output_path = output_dir / "seed120_cpu_replay.wav"
        cls.receipt_path = output_dir / "seed120_cpu_replay_receipt.json"
        cls.application = MODULE.apply_snapshot_theta(
            project_root=cls.root, plan=cls.plan, method=cls.method,
            parent_row=cls.parent_row, output_wav_path=cls.output_path,
            receipt_path=cls.receipt_path, device="cpu",
        )
        cls.candidate_row = copy.deepcopy(cls.parent_row)
        cls.candidate_row.update({
            "row_id": f"{cls.parent_row['row_id']}__{point_id}",
            "method_id": point_id,
            "method_config_sha256": method["config_sha256"],
            "score_audio": copy.deepcopy(cls.parent_row["natural_pcm"]),
            "spsa_point_binding_sha256": config["binding_sha256"],
            "spsa_binding": copy.deepcopy(cls.application["spsa_binding"]),
            "generator_condition_input": copy.deepcopy(cls.application["generator_condition_input"]),
        })

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    @classmethod
    def _stage_pin(cls, key: str) -> None:
        pin = cls.plan["scientific_input_pins"][key]
        source = PAYLOAD_ROOT / pin["path"]
        raw = source.read_bytes()
        if len(raw) != pin["bytes"] or _sha(raw) != pin["sha256"]:
            raise AssertionError(f"source payload does not match actual pin {key}")
        dest = cls.root / pin["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)

    def _validate(self, method=None, candidate=None) -> dict[str, object]:
        return MODULE.validate_application_lineage(
            project_root=self.root, plan=self.plan, method=method or self.method,
            parent_row=self.parent_row, candidate_row=candidate or self.candidate_row,
            snapshot_point=self.snapshot_point,
        )

    def test_actual_seed_application_round_trips_and_matches_full_pcm_clock(self) -> None:
        result = self._validate()
        self.assertEqual(result["natural_pcm_sha256"], "f4e611e9f816665ad7c4dce4980f461c4d9af4fe6fe9678c8f756bd03fa6b36c")
        self.assertEqual(result["generator_condition_input"]["sample_count"], 62639)
        self.assertEqual(result["application_packet"]["optimizer_created_or_stepped"], False)
        self.assertEqual(result["application_packet"]["render_or_score_performed"], False)
        self.assertEqual(result["application_packet"]["runtime"]["device"], "cpu")
        ref_path = self.root / self.plan["scientific_input_pins"][self.seed_output_key]["path"]
        audio_module = importlib.import_module("scripts.audio_enhancement.audio_io")
        self.assertEqual(Path(audio_module.__file__).resolve(),
                         (self.root / "scripts/audio_enhancement/audio_io.py").resolve())
        actual, actual_info = audio_module.read_canonical_wav(self.output_path, target_rate=16000)
        frozen, frozen_info = audio_module.read_canonical_wav(ref_path, target_rate=16000)
        self.assertEqual(actual.size, frozen.size)
        self.assertTrue(self.np.isfinite(actual).all())
        max_abs_delta = float(self.np.max(self.np.abs(actual.astype("float64") - frozen.astype("float64"))))
        self.assertLessEqual(max_abs_delta, 1e-5)
        cpu_receipt_path = Path(self.application["application_receipt"]["path"])
        cpu_receipt_raw = cpu_receipt_path.read_bytes()
        original_receipt_raw = ORIGINAL_APPLICATION_RECEIPT.read_bytes()
        original_receipt = json.loads(original_receipt_raw.decode("utf-8"))
        print("SEED_REPLAY_EVIDENCE " + json.dumps({
            "status": "PASS_TEST_ONLY_ACTUAL_SEED_REPLAY_NO_SPSA_INITIALIZATION",
            "snapshot_raw_sha256": self.snapshot_point["record_pin"]["sha256"],
            "snapshot_self_sha256": self.snapshot_point["snapshot"]["snapshot_sha256"],
            "source_optimizer_step": self.snapshot_point["snapshot"]["source_optimizer_step"],
            "head_implementation": self.snapshot_point["head_pin"],
            "head_constructor_sha256": self.snapshot_point["snapshot"]["head_constructor_sha256"],
            "natural_file_sha256": self.application["application_packet"]["input_original_file"]["sha256"],
            "natural_pcm_sha256": self.application["application_packet"]["canonical_original_pcm"]["pcm_float32_sha256"],
            "sample_count": self.application["application_packet"]["source_clock"]["sample_count"],
            "cpu_application_receipt": {
                "path": str(cpu_receipt_path), "bytes": len(cpu_receipt_raw),
                "sha256": _sha(cpu_receipt_raw),
                "self_sha256": self.application["application_receipt"]["self_sha256"],
            },
            "cpu_runtime": self.application["application_packet"]["runtime"],
            "cpu_output_wav": self.application["application_packet"]["output_generator_drive_wav"],
            "cpu_output_pcm_sha256": actual_info.pcm_float32_sha256,
            "original_gpu_application_receipt": {
                "path": str(ORIGINAL_APPLICATION_RECEIPT), "bytes": len(original_receipt_raw),
                "sha256": _sha(original_receipt_raw),
                "self_sha256": original_receipt.get("self_sha256"),
                "runtime": original_receipt.get("runtime"),
            },
            "original_gpu_output": {
                **self.plan["scientific_input_pins"][self.seed_output_key],
                "pcm_float32_sha256": frozen_info.pcm_float32_sha256,
                "sample_count": int(frozen_info.frames), "sample_rate_hz": 16000,
                "channels": 1,
            },
            "max_abs_pcm_delta_vs_original_gpu_output": max_abs_delta,
            "pcm_bitwise_equal": actual_info.pcm_float32_sha256 == frozen_info.pcm_float32_sha256,
            "source_clock_preserved": self.application["application_packet"]["source_clock"]["preserved"],
        }, sort_keys=True, allow_nan=False))

    def test_rejects_mismatched_theta_binding(self) -> None:
        candidate = copy.deepcopy(self.candidate_row)
        candidate["spsa_binding"]["theta"][0] += 0.25
        with self.assertRaisesRegex(MODULE.ApplicationError, "SPSA binding"):
            self._validate(candidate=candidate)

    def test_rejects_corrupt_method_config_digest(self) -> None:
        method = copy.deepcopy(self.method)
        method["config"]["theta"][0] += 0.25
        with self.assertRaisesRegex(MODULE.ApplicationError, "canonical hash|validated snapshot point"):
            self._validate(method=method)

    def test_rejects_receipt_pin_mismatch(self) -> None:
        candidate = copy.deepcopy(self.candidate_row)
        candidate["spsa_binding"]["application_receipt_raw_sha256"] = "0" * 64
        with self.assertRaisesRegex(MODULE.ApplicationError, "receipt"):
            self._validate(candidate=candidate)

    def test_rejects_candidate_pcm_pin_mismatch(self) -> None:
        candidate = copy.deepcopy(self.candidate_row)
        candidate["generator_condition_input"]["pcm_sha256"] = "0" * 64
        with self.assertRaisesRegex(MODULE.ApplicationError, "candidate generator WAV"):
            self._validate(candidate=candidate)

    def test_rejects_sample_count_change(self) -> None:
        candidate = copy.deepcopy(self.candidate_row)
        candidate["generator_condition_input"]["sample_count"] += 1
        with self.assertRaisesRegex(MODULE.ApplicationError, "full-length 16 kHz mono float32 clock"):
            self._validate(candidate=candidate)

    def test_rejects_out_of_bound_projection_with_rehashed_receipt(self) -> None:
        candidate = copy.deepcopy(self.candidate_row)
        original_pin = candidate["spsa_binding"]["application_receipt"]
        packet = json.loads(Path(original_pin["path"]).read_text(encoding="utf-8"))
        packet["projection_evidence"]["postprojection_max_abs_mask_db"] = 3.25
        _with_self_hash(packet)
        receipt_path = self.root / "diagnostic_outputs/projection_out_of_bound_receipt.json"
        raw = json.dumps(packet, sort_keys=True, ensure_ascii=False, indent=2,
                         allow_nan=False).encode("utf-8") + b"\n"
        receipt_path.write_bytes(raw)
        new_pin = {"path": str(receipt_path), "bytes": len(raw), "sha256": _sha(raw),
                   "self_sha256": packet["self_sha256"]}
        candidate["spsa_binding"]["application_receipt"] = new_pin
        candidate["spsa_binding"]["application_receipt_raw_sha256"] = new_pin["sha256"]
        candidate["spsa_binding"]["application_receipt_self_sha256"] = new_pin["self_sha256"]
        candidate["generator_condition_input"]["application_receipt_raw_sha256"] = new_pin["sha256"]
        candidate["generator_condition_input"]["application_receipt_self_sha256"] = new_pin["self_sha256"]
        with self.assertRaisesRegex(MODULE.ApplicationError, "projection|mask"):
            self._validate(candidate=candidate)


if __name__ == "__main__":
    unittest.main()
