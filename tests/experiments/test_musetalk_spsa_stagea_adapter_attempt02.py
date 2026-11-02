from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from scripts.audio_enhancement.audio_io import read_canonical_wav, write_float32_wav
from scripts.experiments.cross_tfg_audio_head_20261003.musetalk_spsa_stagea_adapter_attempt01 import (
    EXPECTED_SCIENTIFIC_INPUT_PINS_SHA256,
    MuseSPSAAdapterError,
    MuseSPSAStageAAdapter,
    expected_native_frame_count,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _self_hash(value: dict[str, Any], key: str) -> dict[str, Any]:
    value[key] = _sha(_canonical({name: item for name, item in value.items() if name != key}))
    return value


class FakeCore:
    def __init__(self, module_path: Path, receipt_dir: Path):
        self.__file__ = str(module_path)
        self.receipt_dir = receipt_dir
        self.generator_calls = 0
        self._run_generator = self._original_generator
        self._frame_clock = self._original_frame_clock

    def _original_generator(self, **kwargs: Any):
        raise AssertionError("the fake renderer should replace this placeholder")

    @staticmethod
    def _original_frame_clock(path, ffprobe, fps_num, fps_den):
        frames = list(range((16000 * fps_num) // (16000 * fps_den)))
        return frames, {"avg_frame_rate": f"{fps_num}/{fps_den}"}

    @staticmethod
    def _reference(ref):
        path = Path(ref["path"]).resolve(strict=True)
        samples, info = read_canonical_wav(path)
        actual = {
            "path": str(path), "file_sha256": info.file_sha256,
            "pcm_sha256": info.pcm_float32_sha256, "sample_count": info.frames,
            "sample_rate_hz": 16000, "channels": 1, "subtype": "IEEE_FLOAT_32_LE",
        }
        return path, samples, actual

    def _render_and_score(self, row, model, method, natural_path, natural_samples,
                          natural_ref, row_root, args, *, qc_repeat=False):
        assert method["config"]["method"] == "precomputed_fit_snapshot_waveform"
        assert method["config_sha256"] == row["method_config_sha256"]
        assert method["method_id"] == row["method_id"]
        row_root.mkdir(parents=True, exist_ok=False)
        (row_root / "audio").mkdir()
        (row_root / "render").mkdir()
        video = row_root / "render" / "generated_video.mp4"
        video.write_bytes(b"fake silent video")
        self.generator_calls += 1
        self._run_generator(row=row, model=model, audio_path=Path(row["generator_condition_input"]["path"]),
                            output_video=video, work=row_root, args=args)
        frames, _ = self._frame_clock(row_root / "render" / "silent_video.mkv", "ffprobe", 25, 1)
        n = len(frames)
        arrays = {
            "query_frame_index": np.arange(n, dtype="<i4"),
            "query_timestamp_num": np.arange(n, dtype="<i8"),
            "query_timestamp_den": np.full(n, 25, dtype="<i8"),
            "track_frame_index": np.arange(n, dtype="<i4"),
            "valid_query_mask": np.asarray([index == 0 for index in range(n)], dtype=np.bool_),
            "distance_by_query_lag": np.zeros((n, 31), dtype="<f4"),
            "lag_offsets_frames": np.arange(-15, 16, dtype="<i4"),
        }
        return {
            "row_id": row["row_id"], "method_id": row["method_id"], "status": "PASS",
            "score_audio_sha256": natural_ref["pcm_sha256"],
            "score": {"status": "PASS"}, "_internal_arrays": arrays,
            "_internal_video_path": str(video),
        }


class FakeOverlay:
    def __init__(self, module_path: Path, receipt_dir: Path):
        self.__file__ = str(module_path)
        self.receipt_dir = receipt_dir
        self.gate_calls = 0
        self.validated_rows = []

    def _capacity_observation(self, plan, binding, helper, output_root, first_remaining_row,
                              *, whole_stage):
        self.gate_calls += 1
        assert whole_stage is False
        assert first_remaining_row == 0
        assert len(plan["rows"]) > 0
        return {"status": "PASS", "remaining_rows": len(plan["rows"]), "reserve_bytes": 5 * 1024**3}

    def _validate_runtime_evidence(self, core, cross, *, row, model, audio_path,
                                   output_video, work, args, generator, prior_receipt_names):
        assert model["model_id"] == "musetalk_fixed_portrait_attempt16"
        assert generator == model["generator"]
        self.validated_rows.append(row["row_id"])
        next_pid = 100000 + 2 * len(self.validated_rows)
        for pid in (next_pid, next_pid + 1):
            (self.receipt_dir / f"{pid}.json").write_text("{}", encoding="utf-8")
        return {"returncode": 0, "adapter_receipt": {"path": "fake"}}


class MuseSPSAAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.samples = np.linspace(-0.05, 0.05, 16000, dtype=np.float32)
        natural_info = write_float32_wav(self.root / "natural.wav", self.samples,
                                         sample_rate=16000, peak_ceiling=None, overwrite=False)
        candidate_info = write_float32_wav(self.root / "candidate.wav", self.samples * 0.9,
                                           sample_rate=16000, peak_ceiling=None, overwrite=False)
        self.natural_ref = {
            "path": str(self.root / "natural.wav"), "file_sha256": natural_info.file_sha256,
            "pcm_sha256": natural_info.pcm_float32_sha256, "sample_count": natural_info.frames,
            "sample_rate_hz": 16000, "channels": 1, "subtype": "IEEE_FLOAT_32_LE",
        }
        self.candidate_ref = {
            "path": str(self.root / "candidate.wav"), "bytes": (self.root / "candidate.wav").stat().st_size,
            "file_sha256": candidate_info.file_sha256, "pcm_sha256": candidate_info.pcm_float32_sha256,
            "sample_count": candidate_info.frames, "sample_rate_hz": 16000, "channels": 1,
            "encoding": "FLOAT_32",
        }
        self.module_paths = {}
        for name, content in (("stagea.py", b"stagea frozen source"),
                              ("overlay.py", b"muse overlay frozen source"),
                              ("cross.py", b"cross runner frozen source"),
                              ("helper.py", b"resource helper frozen source")):
            path = self.root / name
            path.write_bytes(content)
            self.module_paths[name] = path
        self.model_id = "musetalk_fixed_portrait_attempt16"
        self.generator = {"renderer_path": "/frozen/renderer.py", "renderer_sha256": "a" * 64}
        self.model = {"model_id": self.model_id, "generator": self.generator,
                      "audio_frontend": "musetalk_whisper_fixed_portrait_haar_expanded_box_fp16"}
        self.receipt_dir = self.root / "strict-receipts"
        self.receipt_dir.mkdir()
        self.output_root = self.root / "out"
        self.output_root.mkdir()

        self.helper_ref = self._module_ref(self.module_paths["helper.py"])
        self.binding = {
            "schema": "cross_tfg_fit40_musetalk_runtime_binding_v1",
            "status": "PASS_ACTUAL_V100_MUSETALK_ROW_EVIDENCE_AND_MEASURED_ESTIMATE_WITH_LIVE_GATES",
            "model_id": self.model_id,
            "run_id": "fit40_musetalk240_test",
            "scientific_input_pins_sha256": EXPECTED_SCIENTIFIC_INPUT_PINS_SHA256,
            "resources": {"resource_helper": self.helper_ref},
        }
        _self_hash(self.binding, "binding_sha256")
        self.binding_raw = _canonical(self.binding) + b"\n"
        self.binding_path = self.root / "runtime-binding.json"
        self.binding_path.write_bytes(self.binding_raw)

        self.plan = self._make_plan()
        self.plan["plan_sha256"] = _sha(_canonical({k: v for k, v in self.plan.items() if k != "plan_sha256"}))
        self.core = FakeCore(self.module_paths["stagea.py"], self.receipt_dir)
        self.overlay = FakeOverlay(self.module_paths["overlay.py"], self.receipt_dir)
        self.cross = SimpleNamespace(__file__=str(self.module_paths["cross.py"]))
        self.helper = SimpleNamespace(__file__=str(self.module_paths["helper.py"]))
        self.args = SimpleNamespace(project_root=str(self.root), ffmpeg="ffmpeg", ffprobe="ffprobe")
        self._set_environment()

    def tearDown(self):
        import os
        for key in ("CROSS_TFG_CAPACITY_STAGE", "CROSS_TFG_RUNTIME_BINDING_PATH",
                    "CROSS_TFG_RUNTIME_BINDING_RAW_SHA256", "CROSS_TFG_DETERMINISM_RECEIPT_DIR"):
            os.environ.pop(key, None)
        self.temp.cleanup()

    def _module_ref(self, path: Path):
        raw = path.read_bytes()
        return {"path": str(path.resolve()), "bytes": len(raw), "sha256": _sha(raw)}

    def _set_environment(self):
        import os
        os.environ["CROSS_TFG_CAPACITY_STAGE"] = "fit40_musetalk240"
        os.environ["CROSS_TFG_RUNTIME_BINDING_PATH"] = str(self.binding_path)
        os.environ["CROSS_TFG_RUNTIME_BINDING_RAW_SHA256"] = _sha(self.binding_raw)
        os.environ["CROSS_TFG_DETERMINISM_RECEIPT_DIR"] = str(self.receipt_dir.resolve())

    def _make_plan(self):
        method_ids = ["identity", "rms_fixed", "snapshot030", "snapshot060", "snapshot090", "snapshot120"]
        methods = []
        for method_id in method_ids:
            config = {"method": "identity" if method_id == "identity" else "fixture", "method_id": method_id}
            methods.append({"method_id": method_id, "config": config,
                            "config_sha256": _sha(_canonical(config)), "kind": method_id,
                            "checkpoint_sha256": None})
        rows = []
        for index in range(40):
            sample = f"clip-{index:02d}"
            speaker = f"speaker-{index // 2:02d}"
            for method in methods:
                method_id = method["method_id"]
                row = {
                    "row_id": f"parent-{sample}-{method_id}", "split": "fit", "sample_id": sample,
                    "speaker_id": speaker, "model_id": self.model_id,
                    "audio_frontend": self.model["audio_frontend"], "model_role": "candidate",
                    "face_id": "fixed-face", "source_media_kind": "static_portrait", "seed": 20261003,
                    "method_id": method_id, "method_config_sha256": method["config_sha256"],
                    "checkpoint_sha256": None, "natural_pcm": dict(self.natural_ref),
                    "score_audio": dict(self.natural_ref), "source_face": {"path": str(self.root / "face.png"), "file_sha256": "b" * 64},
                    "source_video": {"path": None, "file_sha256": None},
                    "generation_settings": {"seed": 20261003, "fps_num": 25, "fps_den": 1},
                    "musetalk_native_frame_binding": {"sample_count": 16000,
                                                        "expected_frame_count": 25,
                                                        "fps_num": 25, "fps_den": 1},
                }
                rows.append(row)
        binding_ref = {"path": str(self.binding_path), "bytes": len(self.binding_raw),
                       "raw_sha256": _sha(self.binding_raw), "status": self.binding["status"]}
        return {
            "schema_version": "cross-tfg-audio-eval-v1", "status": "FROZEN",
            "run_id": self.binding["run_id"],
            "scientific_input_pins_sha256": EXPECTED_SCIENTIFIC_INPUT_PINS_SHA256,
            "models": [self.model], "methods": methods, "rows": rows,
            "protocol": {"generator": self.generator},
            "execution": {
                "dev_or_confirm_access": False, "candidate_selection": "not performed",
                "capacity_stage": "fit40_musetalk240", "planned_rows": 240,
                "model_ids": [self.model_id],
                "scientific_input_pins_sha256": EXPECTED_SCIENTIFIC_INPUT_PINS_SHA256,
                "muse_runtime_binding": binding_ref,
                "stagea_runner": self._module_ref(self.module_paths["stagea.py"]),
                "musetalk_block_overlay": self._module_ref(self.module_paths["overlay.py"]),
                "cross_frontend_runner": self._module_ref(self.module_paths["cross.py"]),
            },
            "plan_sha256": "",
        }

    def _method_and_rows(self, count=8):
        sample_ids = [f"clip-{index:02d}" for index in range(count)]
        config = {
            "method": "fit40_spsa_proposal", "schema": "fit40_spsa_proposal_transform_v1",
            "method_id": "fit40_d0_spsa_plus_k01", "proposal_id": "proposal-plus",
            "pair_id": "pair-01", "iteration": 0, "side": "plus", "theta": [0.01] * 9,
            "theta_sha256": _sha(_canonical([0.01] * 9)), "artifact_sha256": "c" * 64,
            "binding_sha256": "d" * 64, "input_pins_sha256": "e" * 64,
            "batch_id": "batch-01", "sample_ids": sample_ids,
            "direction": [1] * 9, "c_k": 0.25,
        }
        method = {"method_id": config["method_id"], "kind": config["method"],
                  "config": config, "config_sha256": _sha(_canonical(config)),
                  "checkpoint_sha256": None}
        rows = []
        for sample in sample_ids:
            parent = self._parent_row(sample)
            app_pin, app_hash = self._write_application_receipt(parent)
            spsa_binding = {
                "schema": "fit40_spsa_stagea_row_binding_v1", "parent_row_id": parent["row_id"],
                "seed": parent["seed"], "method_id": method["method_id"],
                "method_config_sha256": method["config_sha256"], "proposal_id": config["proposal_id"],
                "pair_id": config["pair_id"], "iteration": config["iteration"], "side": config["side"],
                "batch_id": config["batch_id"], "c_k": config["c_k"],
                "theta": list(config["theta"]), "theta_sha256": config["theta_sha256"],
                "artifact_sha256": config["artifact_sha256"], "binding_sha256": config["binding_sha256"],
                "input_pins_sha256": config["input_pins_sha256"],
                "checkpoint_parameter_sha256": "f" * 64, "gate_mode": "d0",
                "application_receipt": app_pin,
                "application_receipt_raw_sha256": app_hash["raw"],
                "application_receipt_self_sha256": app_hash["self"],
            }
            generator_input = {
                **self.candidate_ref, "proposal_id": config["proposal_id"], "pair_id": config["pair_id"],
                "side": config["side"], "method_id": config["method_id"],
                "method_config_sha256": method["config_sha256"], "artifact_sha256": config["artifact_sha256"],
                "binding_sha256": config["binding_sha256"], "input_pins_sha256": config["input_pins_sha256"],
                "theta_sha256": config["theta_sha256"], "parent_row_id": parent["row_id"],
                "seed": parent["seed"], "application_receipt_raw_sha256": app_hash["raw"],
                "application_receipt_self_sha256": app_hash["self"],
            }
            row = dict(parent)
            row.update({
                "row_id": f"candidate-{sample}-{method['method_id']}",
                "method_id": method["method_id"], "method_config_sha256": method["config_sha256"],
                "checkpoint_sha256": None, "spsa_point_binding_sha256": config["binding_sha256"],
                "spsa_binding": spsa_binding, "generator_condition_input": generator_input,
            })
            rows.append(row)
        return method, rows

    def _parent_row(self, sample):
        return next(row for row in self.plan["rows"]
                    if row["sample_id"] == sample and row["method_id"] == "identity")

    def _write_application_receipt(self, parent):
        output_path = Path(self.candidate_ref["path"])
        packet = {
            "schema": "cross_tfg_d0_gatedc_generator_drive_result_v1",
            "status": "PASS_D0_GENERATOR_DRIVE_ONLY", "profile_role": "UNSELECTED_FIT_CHECKPOINT_INFERENCE",
            "input_original_file": {"sha256": self.natural_ref["file_sha256"]},
            "canonical_generator_input": {"pcm_float32_sha256": self.natural_ref["pcm_sha256"]},
            "output_generator_drive_wav": {
                "path": str(output_path), "bytes": self.candidate_ref["bytes"],
                "sha256": self.candidate_ref["file_sha256"],
                "pcm_float32_sha256": self.candidate_ref["pcm_sha256"],
                "sample_count": self.candidate_ref["sample_count"],
                "sample_rate_hz": 16000, "channels": 1, "encoding": "FLOAT_32",
            },
            "output_pcm_float32_sha256": self.candidate_ref["pcm_sha256"],
            "output_sample_count": self.candidate_ref["sample_count"],
            "source_clock_preserved_after_canonicalization": True,
            "render_or_score_performed": False, "optimizer_created_or_stepped": False,
            "checkpoint_parameter_sha256": "f" * 64, "gate_mode": "d0",
        }
        packet["self_sha256"] = _sha(_canonical(packet))
        receipt_path = self.root / f"application-{parent['sample_id']}.json"
        raw = _canonical(packet) + b"\n"
        receipt_path.write_bytes(raw)
        pin = {"path": str(receipt_path), "bytes": len(raw), "sha256": _sha(raw),
               "self_sha256": packet["self_sha256"]}
        return pin, {"raw": pin["sha256"], "self": pin["self_sha256"]}

    def _adapter(self):
        return MuseSPSAStageAAdapter(
            core=self.core, muse_overlay=self.overlay, cross_frontend=self.cross,
            parent_plan=self.plan, runtime_binding=self.binding,
            runtime_binding_raw=self.binding_raw, runtime_binding_path=self.binding_path,
            resource_helper=self.helper, args=self.args, strict_receipt_dir=self.receipt_dir,
        )

    def test_native_frame_floor_and_invalid_inputs(self):
        self.assertEqual(expected_native_frame_count(62639), 97)
        self.assertEqual(expected_native_frame_count(640), 1)
        self.assertEqual(expected_native_frame_count(639), 0)
        for bad in (0, -1, True, 1.5):
            with self.assertRaises(MuseSPSAAdapterError):
                expected_native_frame_count(bad)

    def test_proposal_renders_eight_rows_through_direct_stagea_call(self):
        method, rows = self._method_and_rows()
        adapter = self._adapter()
        original_generator = self.core._run_generator
        result = adapter.render_subset(candidate_rows=rows, method=method, output_root=self.output_root)
        self.assertEqual(len(result), 8)
        self.assertEqual(self.core.generator_calls, 8)
        self.assertEqual(self.overlay.gate_calls, 8)
        self.assertEqual(len(self.overlay.validated_rows), 8)
        self.assertIs(self.core._run_generator, original_generator)
        self.assertEqual(result[0]["official_array_summary"], {"query_count": 25, "valid_query_count": 1})
        self.assertEqual(result[0]["row"]["score_audio_sha256"], self.natural_ref["pcm_sha256"])
        self.assertEqual(result[0]["row"]["parent_identity_row_id"], rows[0]["spsa_binding"]["parent_row_id"])

    def test_bad_application_receipt_pin_fails_before_generator_call(self):
        method, rows = self._method_and_rows()
        rows[0]["generator_condition_input"]["application_receipt_raw_sha256"] = "0" * 64
        adapter = self._adapter()
        with self.assertRaises(MuseSPSAAdapterError):
            adapter.render_subset(candidate_rows=rows, method=method, output_root=self.output_root)
        self.assertEqual(self.core.generator_calls, 0)

    def test_wrong_identity_profile_fails_before_generator_call(self):
        method, rows = self._method_and_rows()
        rows[0]["audio_frontend"] = "other-profile"
        adapter = self._adapter()
        with self.assertRaises(MuseSPSAAdapterError):
            adapter.render_subset(candidate_rows=rows, method=method, output_root=self.output_root)
        self.assertEqual(self.core.generator_calls, 0)

    def test_wrong_subset_size_or_reordered_samples_is_rejected(self):
        method, rows = self._method_and_rows()
        adapter = self._adapter()
        with self.assertRaises(MuseSPSAAdapterError):
            adapter.render_subset(candidate_rows=rows[:7], method=method, output_root=self.output_root)
        self.assertEqual(self.core.generator_calls, 0)
        rows[0], rows[1] = rows[1], rows[0]
        with self.assertRaises(MuseSPSAAdapterError):
            adapter.render_subset(candidate_rows=rows, method=method, output_root=self.output_root)
        self.assertEqual(self.core.generator_calls, 0)


if __name__ == "__main__":
    unittest.main()
